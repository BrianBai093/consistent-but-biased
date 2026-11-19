"""Run one published, text-only case through the department workflow.

The hospital runner (``run_department_experiment``) starts from photographed
pages and films. A case taken from the literature -- a PMC-Patients summary,
say -- is a single English narrative with no test sheets and no films, so it
enters differently and then rejoins the same protocol:

- the record is the text itself; there is no OCR and no test transcription;
- a model (DeepSeek) cuts the report down to what was known before the
  treatment decision. A case report states the decision inside the sentences
  that present the patient, so it may cut inside a sentence; every answer is
  checked without a model to be a deletion-only edit of the source, and a
  refused answer is asked for again with the reason
  (``services/case_report_filter``);
- nobody signs the result. It is approved automatically, and the run records
  that, the removed passages, and every keyword-screen hit left in the kept
  text, so an analysis can audit or exclude a case;
- the radiologist, which has no films to read, reports the imaging as the
  filtered record describes it, and says that it has not seen the films.

Everything from stage I onward -- the roles, the frozen baseline, the three
arms, the observer -- is the hospital runner's own code.

The source is published, so it is not patient data in the sense the hospital
cases are, but it is also likely to be in a model's training data. A model
that recognises a famous case report is remembering an answer, not reaching
one; read results from these cases with that in mind.

**This spends money**, though much less than a hospital case: one filter call
(more if an answer is refused), five stage I-II turns, then the arms. Run
``--dry-run`` first.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from agents.case_report_filter import CaseReportFilterAgent
from config import Settings
from department.factory import create_department
from department.state import DEPARTMENT_WORKFLOW_VERSION, MAX_NEGOTIATION_ROUNDS
from schemas import DirectorStyle, DocumentPage
from services.baseline_store import TextCaseBaselineStore
from services.case_report_filter import (
    FilterRecord,
    RemovedPassageRecord,
    render_filter_review,
)
from services.llm_client import DeepSeekTextClient
from services.record_redaction import RedactionCategory
from tools.run_department_experiment import (
    PERSONA_PLACEMENTS,
    Prepared,
    _close,
    attempt_arms,
    comparison,
    prompt_fingerprints,
    report_name,
    role_styles,
    run_arms,
    screen,
)

logger = logging.getLogger("department.text_case")

LANGUAGE = "en"
PAGE_ID = "case-page-001"
CASE_SOURCE = "text"
PREPARATION_METHOD = "llm_filter"
#: What the anaesthetist is told in place of a test transcript. A published
#: summary carries whatever laboratory values the authors chose to report inside
#: the narrative; there is no separate sheet, and saying so is better than an
#: empty string that reads as "no tests were done".
NO_TEST_SHEET_MD = (
    "No separate laboratory report exists for this case. It is a published case "
    "report; any laboratory values the authors reported are in the case record."
)
#: Stored where the summary would be. The summary exists to brief the
#: radiologist before it reads films; with no films, the radiologist reads the
#: filtered record itself and no role reads a summary.
NO_SUMMARY_MD = (
    "Not generated: a text-only case has no films, so no role reads a summary."
)
DEFAULT_PMC_CSV = Path(".data/pmc_patients/PMC-Patients.csv")
FILTER_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / (
    CaseReportFilterAgent.prompt_filename
)
#: Carried into every summary: the source is reusable for research under this
#: licence, with attribution and not commercially.
PMC_DATASET = {
    "name": "PMC-Patients",
    "source": "https://huggingface.co/datasets/zhengyun21/PMC-Patients",
    "license": "CC BY-NC-SA 4.0",
}


@dataclass(frozen=True, slots=True)
class TextCase:
    """What the arms and the stores read from a case, for a text-only one."""

    case_id: str
    case_pages: list[DocumentPage]
    test_pages: list[DocumentPage]
    images: list
    artifact_dir: str


def read_pmc_patient(csv_path: Path, uid: str) -> tuple[str, str]:
    """The patient summary of one PMC-Patients row, and its PMID for citation.

    The article title is left behind on purpose: "Surgical treatment for ..."
    is the answer in five words. The PMID goes to the summary, never to a model.
    """

    csv.field_size_limit(sys.maxsize)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["patient_uid"] == uid:
                return normalise(row["patient"]), row["PMID"]
    raise KeyError(f"no PMC-Patients row with patient_uid {uid!r}")


def normalise(text: str) -> str:
    """The dataset stores paragraph breaks as a literal backslash-n."""

    return text.replace("\\n", "\n").strip() + "\n"


def materialise(text: str, *, settings: Settings, case_id: str, thread_id: str) -> TextCase:
    """Write the record where an upload would land, so the store can hash it."""

    root = Path(settings.upload_dir) / thread_id
    pages = root / "source" / "case_pages"
    pages.mkdir(parents=True, exist_ok=True)
    path = pages / f"{PAGE_ID}.txt"
    path.write_text(text, encoding="utf-8")
    return TextCase(
        case_id=case_id,
        case_pages=[
            DocumentPage(
                page_id=PAGE_ID,
                page_number=1,
                path_or_url=str(path),
                original_filename=f"{PAGE_ID}.txt",
            )
        ],
        test_pages=[],
        images=[],
        artifact_dir=str(root / "derived"),
    )


def record_markdown(text: str) -> str:
    """The record as the transcribers would have written it: one marked page."""

    return f"<!-- source: {PAGE_ID} -->\n\n{text}"


async def prepare_text(
    case: TextCase,
    *,
    store: TextCaseBaselineStore,
    case_preop_md: str,
    redaction_sha: str,
) -> Prepared:
    """Freeze the cleaned record. There is no test sheet and, here, no summary.

    No model call: see ``NO_SUMMARY_MD``.
    """

    stored = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    if (
        stored is not None
        and stored.redaction_sha == redaction_sha
        # The store strips surrounding whitespace on write, as every record
        # model does; compared raw, a record ending in a newline never matched.
        and stored.case_preop_md.strip() == case_preop_md.strip()
    ):
        return Prepared(
            case_preop_md=stored.case_preop_md,
            tests_preop_md=stored.tests_preop_md,
            case_summary_md=stored.case_summary_md,
            warnings=list(stored.preparation_warnings),
            reused=True,
        )
    await store.store_preparation(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md=case_preop_md,
        tests_preop_md=NO_TEST_SHEET_MD,
        case_summary_md=NO_SUMMARY_MD,
        preparation_warnings=[],
        redaction_sha=redaction_sha,
    )
    return Prepared(
        case_preop_md=case_preop_md,
        tests_preop_md=NO_TEST_SHEET_MD,
        case_summary_md=NO_SUMMARY_MD,
    )


def series_dir(out: Path, series: str | None) -> Path:
    """Where a series writes: the main runs at the top, each series beneath."""

    return out / f"series-{series}" if series else out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_design(series: str | None, persona_on: str, negotiation: bool) -> str | None:
    """Why this combination may not run, or None.

    An A4 arm writes its report under the same file name as the main arm of its
    style, so outside a series of its own it would overwrite main data.
    """

    try:
        role_styles(DirectorStyle.NEUTRAL, persona_on, negotiation)
    except ValueError as exc:
        return str(exc)
    if (persona_on != "both" or not negotiation) and not series:
        return "--persona-on and --no-negotiation need a --series of their own"
    return None


async def filter_record(
    case: TextCase,
    text: str,
    *,
    settings: Settings,
    store: TextCaseBaselineStore,
    out: Path,
    redo: bool = False,
) -> tuple[FilterRecord, int]:
    """The accepted filter of this record, made once and then read from disk.

    A second answer would differ from the first, and the arms of a case must
    all read the same text, so a filter on file is reused unless ``redo``.
    """

    digest = sha256(text)
    record = await store.load_filter(case_pages=case.case_pages, case_id=case.case_id)
    calls = 0
    if record is None or record.source_sha256 != digest or redo:
        config = settings.require_deepseek()
        client = DeepSeekTextClient(config, thinking=settings.deepseek_thinking)
        try:
            result, passages = await CaseReportFilterAgent(client).run(record=text)
        finally:
            await _close(client)
        calls = len(result.all_usages)
        record = FilterRecord(
            source_sha256=digest,
            preoperative_text=result.output.preoperative_text,
            removals=list(result.output.removals),
            removed_passages=[RemovedPassageRecord(**asdict(item)) for item in passages],
            provider=client.provider,
            model=config.model,
            attempts=calls,
            created_at=datetime.now(UTC),
        )
        await store.store_filter(record, case_pages=case.case_pages, case_id=case.case_id)

    hidden = out / "analysis_only"
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "case_record.raw.txt").write_text(text, encoding="utf-8")
    (hidden / "removed_passages.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in record.removed_passages],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return record, calls


def text_prompt_fingerprints(registry, filter_prompt: str) -> dict[str, str]:
    """The hospital runner's fingerprints, plus the two prompts only this path reads."""

    def digest(text: str) -> str:
        return sha256(text)[:16]

    fingerprints = prompt_fingerprints(registry)
    fingerprints["radiologist:record_only"] = digest(registry.radiologist.record_only_prompt)
    fingerprints["case_report_filter"] = digest(filter_prompt)
    return fingerprints


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case-file", type=Path, help="A plain-text case record.")
    source.add_argument(
        "--pmc-uid",
        help=f"patient_uid of a PMC-Patients row, read from --pmc-csv ({DEFAULT_PMC_CSV}).",
    )
    parser.add_argument("--pmc-csv", type=Path, default=DEFAULT_PMC_CSV)
    parser.add_argument(
        "--case-id",
        required=True,
        help="A label of your choosing; it is stored with the results.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--redo-filter",
        action="store_true",
        help="Ask the model for a new filter instead of reading the one on file.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--only-style",
        action="append",
        choices=[style.value for style in DirectorStyle],
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--series",
        help=(
            "Name a robustness series (another model, another wording). Its arms "
            "are written under <out>/series-<name>/ on threads of their own, and "
            "read the same frozen filter and baseline as the main runs."
        ),
    )
    parser.add_argument(
        "--persona-on",
        choices=PERSONA_PLACEMENTS,
        default="both",
        help=(
            "Which role reads the style under test; the other reads the neutral "
            "prefix (amendment A4). Needs --series."
        ),
    )
    parser.add_argument(
        "--no-negotiation",
        action="store_true",
        help=(
            "The director rules once, alone, after stage II: no deputy turn and "
            "no advocate reply (amendment A4). Needs --series."
        ),
    )
    args = parser.parse_args(argv)
    negotiation = not args.no_negotiation
    refusal = check_design(args.series, args.persona_on, negotiation)
    if refusal:
        parser.error(refusal)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    pmid = None
    if args.pmc_uid:
        text, pmid = read_pmc_patient(args.pmc_csv, args.pmc_uid)
    else:
        text = normalise(args.case_file.read_text(encoding="utf-8"))

    started = datetime.now(UTC)
    out = series_dir(args.out, args.series) / args.case_id
    out.mkdir(parents=True, exist_ok=True)
    case = materialise(
        text,
        settings=settings,
        case_id=args.case_id,
        thread_id=f"{args.case_id}-{started:%Y%m%d%H%M%S}",
    )
    store = TextCaseBaselineStore(settings.baseline_store_dir)
    raw_findings = screen("case-record-raw", record_markdown(text), out, language=LANGUAGE)
    logger.info(
        "Record: %d characters; the screen flags %d page line(s) before filtering "
        "(expected: a case report states its treatment).",
        len(text),
        len(raw_findings),
    )
    if args.dry_run:
        logger.info(
            "A live run would ask the filter once (DeepSeek), then make at most %d "
            "department turns.",
            5 + (3 * MAX_NEGOTIATION_ROUNDS if negotiation else 1) * len(DirectorStyle),
        )
        return 0

    try:
        filtered, filter_calls = await filter_record(
            case,
            text,
            settings=settings,
            store=store,
            out=out,
            redo=args.redo_filter,
        )
    except Exception as exc:  # noqa: BLE001 - reported, and the batch moves on
        logger.error("The filter failed: %s: %s", type(exc).__name__, exc)
        return 2
    case_preop_md = record_markdown(filtered.preoperative_text + "\n")
    hits = screen("case-record-filtered", case_preop_md, out, language=LANGUAGE)
    (out / "case_preop.md").write_text(case_preop_md, encoding="utf-8")
    (out / "filter-review.md").write_text(
        render_filter_review(
            case_id=args.case_id,
            passages=filtered.removed_passages,
            removals=[item.model_dump(mode="json") for item in filtered.removals],
            screen_findings=hits,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Filter: %d passage(s) removed, %d of %d characters kept (%d call(s) this run); "
        "approved automatically",
        len(filtered.removed_passages),
        len(filtered.preoperative_text),
        len(text),
        filter_calls,
    )
    for line in hits:
        logger.warning("  left in, unreviewed: %s", line)

    preparation_sha = sha256(f"{PREPARATION_METHOD}:{filtered.preoperative_text}")
    prepared = await prepare_text(
        case,
        store=store,
        case_preop_md=case_preop_md,
        redaction_sha=preparation_sha,
    )
    logger.info("Preparation %s", "reused" if prepared.reused else "frozen")
    if args.prepare_only:
        logger.info("Frozen. Read %s, then run again without --prepare-only.", out)
        return 0

    styles = [
        style
        for style in DirectorStyle
        if not args.only_style or style.value in args.only_style
    ]
    registry, assignments = create_department(settings)
    try:
        rows, reports, failed = await run_arms(
            styles=styles,
            case=case,
            prepared=prepared,
            settings=settings,
            registry=registry,
            assignments=assignments,
            baseline_store=store,
            out=out,
            attempt=args.attempt,
            thread_suffix=f"-{preparation_sha[:8]}"
            + (f"-{args.series}" if args.series else ""),
            persona_on=args.persona_on,
            negotiation=negotiation,
        )
    finally:
        await _close(*(getattr(agent, "client", None) for agent in registry.all_agents()))
    if not rows:
        logger.error("No arm completed; nothing to compare.")
        return 1

    filter_prompt = FILTER_PROMPT.read_text(encoding="utf-8").strip()
    summary = {
        "case_id": args.case_id,
        "case_source": CASE_SOURCE,
        "series": args.series,
        "dataset": PMC_DATASET if args.pmc_uid else None,
        "pmc_patient_uid": args.pmc_uid,
        "pmid": pmid,
        "workflow_version": DEPARTMENT_WORKFLOW_VERSION,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "sources": {"case_pages": 1, "test_pages": 0, "images": 0},
        # The radiologist read prose, not films. Stated here so that no table
        # pools this case's imaging fields with a hospital case's.
        "imaging_source": "record_text",
        "preparation": {"method": PREPARATION_METHOD, "reused": prepared.reused},
        "filter": {
            "approved_by": "automatic",
            "provider": filtered.provider,
            "model": filtered.model,
            "attempts": filtered.attempts,
            "source_characters": len(text),
            "kept_characters": len(filtered.preoperative_text),
            "removed_passages": len(filtered.removed_passages),
            "by_category": {
                category.value: sum(
                    1 for item in filtered.removals if item.category is category
                )
                for category in RedactionCategory
            },
            # Keyword hits nobody looked at. A case with any is the first to
            # audit, and the one to drop if its hits turn out to be the answer.
            "unreviewed_screen_hits": hits,
        },
        "failed_arms": failed,
        "attempt": args.attempt,
        "prompt_fingerprints": text_prompt_fingerprints(registry, filter_prompt),
        "settings": {
            "text_provider": settings.text_provider,
            "image_provider": settings.image_provider,
            "deepseek_thinking": settings.deepseek_thinking,
            "department_role_models": settings.department_role_models,
            "department_style_variant": settings.department_style_variant,
            "max_negotiation_rounds": MAX_NEGOTIATION_ROUNDS,
            "persona_on": args.persona_on,
            "negotiation": negotiation,
        },
        **comparison(*attempt_arms(out, args.attempt)),
    }
    summary_path = out / report_name("summary", args.attempt)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)
    if failed:
        logger.error("Arms that did not finish: %s", ", ".join(failed))
        return 1
    if not summary["validity"]["one_baseline"]:
        logger.error("The arms did not share one baseline; the comparison is invalid.")
        return 1
    if not summary["validity"]["identical_baseline_transcript"]:
        logger.error("The arms did not read the same evidence; the comparison is invalid.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
