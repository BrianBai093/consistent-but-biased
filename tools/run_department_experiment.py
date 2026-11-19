"""Run one case through the department workflow, once per decision style.

One case is one experimental unit: stages I and II are prepared and frozen once,
then the negotiation runs three times, once per ``DirectorStyle``, over that one
frozen baseline. Everything each run produced is written out, because the point
of the exercise is the comparison between the three arms, and a number nobody
can trace back to a statement is not evidence.

**This spends money.** Every arm calls a provider. Run ``--dry-run`` first: it
does the free half -- ingestion, page slicing and the deterministic OCR of the
case record -- and then reports what the paid half would cost in calls.

Four things this runner must get right, each of them learned from a defect:

1. It records which model served which role. Nothing else populates that, and
   without it a role whose model changed between batches is undetectable.
2. It never writes the raw state. The state carries real filesystem paths and
   original filenames; the report carries identifiers and counts instead.
3. It runs the arms strictly in sequence. Three concurrent arms would all miss
   the baseline and each generate its own, which is the one thing the design
   exists to prevent.
4. It gives each arm its own thread. A finished arm is finished: re-driving one
   is refused by the transcript reducer rather than silently appended to.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from pydantic import BaseModel
from starlette.datastructures import Headers, UploadFile

from agents.record_redactor import RecordRedactorAgent
from agents.preparation_agents import (
    CaseDocumentTranscriber,
    CaseSummaryAgent,
    TestDocumentAgent,
)
from api_schemas import ImageUploadMetadata
from config import Settings
from department.factory import create_department
from department.graph import create_department_graph, department_input, thread_config
from department.observer import ObserverMetrics, observe
from department.state import DEPARTMENT_WORKFLOW_VERSION, MAX_NEGOTIATION_ROUNDS
from schemas import AgentRole, CaseInput, DirectorStyle, Modality
from services.agent_factory import image_client, text_client
from services.baseline_store import BaselineStore
from services.checkpoint import open_sqlite_checkpointer
from services.ocr_engine import RapidOcr
from services.record_redaction import (
    RedactionCategory,
    RedactionPlan,
    RedactionRecord,
    Segment,
    apply,
    flagged_sentences,
    parse_numbers,
    removed_sentences,
    render_review,
    segment,
    source_sha256,
    units,
)
from services.record_screen import screen_markdown, summarise
from services.upload_store import IMAGE_TYPES, UploadStore

logger = logging.getLogger("department.experiment")

#: What the upload path accepts. The suffix is lower-cased first, because the
#: source films are named ".JPG".
PAGE_SUFFIXES = frozenset(IMAGE_TYPES)

#: See the note in ``prepare``. Deliberately local to this runner.
DOCUMENT_RETRIES = 2

#: The source folders are named by hand and the wording varies between cases
#: ("入院常规检验及检查" and "入院常规检查及检验" are both in use), so a folder is
#: classified by what it contains rather than matched literally.
CASE_FOLDER_MARKS = ("基础资料",)
TEST_FOLDER_MARKS = ("检验", "检查")
IMAGE_FOLDER_MARKS = ("术前",)
#: Postoperative material is the answer to the question being asked. It states
#: which operation was actually performed, so it is analysis-layer evidence and
#: must never reach an agent.
EXCLUDED_FOLDER_MARKS = ("术后",)
#: Documents that must never be read, whatever folder they are filed in. The
#: operative note (手术) and the discharge summary (出院) state the operation that
#: was performed. This is a rule set by the study's clinician, and it sits
#: upstream of the leak screen on purpose: nothing -- not --skip-leak-screen,
#: not --drop-leaking-pages -- can put these files back in.
EXCLUDED_FILE_MARKS = ("手术", "出院")

MODALITY_MARKS: tuple[tuple[str, Modality], ...] = (
    ("MRI", Modality.MRI),
    ("CT", Modality.CT),
    ("X-RAY", Modality.XRAY),
    ("XRAY", Modality.XRAY),
    ("X线", Modality.XRAY),
)


@dataclass(frozen=True, slots=True)
class SourceFiles:
    case_pages: list[Path]
    test_pages: list[Path]
    images: list[Path]
    excluded: list[Path]
    #: Folder name -> the category it was read as. Printed, so a misrouted
    #: folder is visible before any money is spent rather than inferred later
    #: from a baseline that looks wrong.
    routing: dict[str, str]


@dataclass(slots=True)
class Prepared:
    case_preop_md: str
    tests_preop_md: str
    case_summary_md: str
    warnings: list[str] = field(default_factory=list)
    reused: bool = False
    model_calls: int = 0


class AmbiguousFolder(ValueError):
    """A folder whose name fits more than one category, or none."""


def _category(folder: str) -> str:
    matches = {
        category
        for category, marks in (
            ("excluded", EXCLUDED_FOLDER_MARKS),
            ("case_pages", CASE_FOLDER_MARKS),
            ("images", IMAGE_FOLDER_MARKS),
            ("test_pages", TEST_FOLDER_MARKS),
        )
        if any(mark in folder for mark in marks)
    }
    # Postoperative material wins outright: it is the answer to the question
    # being asked, whatever else the folder name says.
    if "excluded" in matches:
        return "excluded"
    if len(matches) != 1:
        raise AmbiguousFolder(
            f"Cannot tell what {folder!r} holds (matched: {sorted(matches) or 'nothing'}). "
            "Rename the folder or extend the marks in this module. Guessing here changes "
            "the frozen baseline that all three arms then argue over."
        )
    return matches.pop()


def collect_sources(case_dir: Path) -> SourceFiles:
    """Sort a case folder into the three inputs, and drop what must not be read.

    Classification is by the folder a file sits in, not by a substring of its
    whole path, and a folder that fits two categories stops the run instead of
    taking the first match. "术前检查" fits both "术前" and "检查": sending it to
    the radiologist as film while it disappears from the test pages would be
    silent, and it would change the evidence every arm shares.
    """

    buckets: dict[str, list[Path]] = {
        "case_pages": [],
        "test_pages": [],
        "images": [],
        "excluded": [],
    }
    routing: dict[str, str] = {}
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PAGE_SUFFIXES:
            continue
        if any(mark in path.stem for mark in EXCLUDED_FILE_MARKS):
            buckets["excluded"].append(path)
            continue
        folder = path.parent.name
        if folder not in routing:
            routing[folder] = _category(folder)
        buckets[routing[folder]].append(path)
    return SourceFiles(
        case_pages=buckets["case_pages"],
        test_pages=buckets["test_pages"],
        images=buckets["images"],
        excluded=buckets["excluded"],
        routing=routing,
    )


def _modality(path: Path) -> Modality:
    name = path.name.upper()
    for mark, modality in MODALITY_MARKS:
        if mark in name:
            return modality
    raise ValueError(
        f"Cannot tell the modality of an image from its name: {path.name!r}. "
        "Rename it to carry CT, MRI or X-RAY."
    )


def _upload(path: Path) -> UploadFile:
    """Present a local file the way a browser presents an upload.

    The store checks that the declared content type matches the extension, so a
    local file has to declare one too; it is read from the same table the store
    validates against rather than restated here.
    """

    content_type, _ = IMAGE_TYPES[path.suffix.lower()]
    return UploadFile(
        file=BytesIO(path.read_bytes()),
        filename=path.name,
        headers=Headers({"content-type": content_type}),
    )


async def ingest(
    sources: SourceFiles,
    *,
    case_id: str,
    settings: Settings,
    thread_id: str,
) -> CaseInput:
    """Take the same path an upload does, so slicing and page ids match the API."""

    store = UploadStore(
        settings.upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
        max_images_per_case=settings.max_images_per_case,
        page_slice_max_height=settings.page_slice_max_height,
        page_slice_min_blank_run=settings.page_slice_min_blank_run,
    )
    metadata = [
        ImageUploadMetadata(image_id=f"image-{index:03d}", modality=_modality(path))
        for index, path in enumerate(sources.images, start=1)
    ]
    return await store.save_case_package(
        thread_id=thread_id,
        case_id=case_id,
        image_metadata=metadata,
        case_files=[_upload(path) for path in sources.case_pages],
        test_files=[_upload(path) for path in sources.test_pages],
        image_files=[_upload(path) for path in sources.images],
    )


async def transcribe_case_record(case: CaseInput, settings: Settings) -> tuple[str, list[str]]:
    """Deterministic OCR: no model, no cost, and identical on every run."""

    transcriber = CaseDocumentTranscriber(
        RapidOcr(settings.ocr_model_dir),
        low_confidence_threshold=settings.ocr_low_confidence_threshold,
    )
    result = await transcriber.run(case.case_pages)
    return result.output.markdown, list(result.output.warnings)


def page_filenames(pages: Sequence) -> dict[str, str]:
    """Page id -> the document it came from. Slices keep their source's name."""

    return {page.page_id: page.original_filename for page in pages}


def screen(
    label: str,
    markdown: str,
    out: Path,
    *,
    filenames: dict[str, str] | None = None,
    language: str = "zh",
) -> list[str]:
    """Report decision-revealing content; the lines of the report, or none.

    Before redaction this is informational -- a record written after the fact
    is expected to state the answer. After redaction it is the check for what
    the redaction left behind, which a diff of what was removed cannot show.
    """

    findings = screen_markdown(markdown, language=language)
    if not findings:
        return []
    text = summarise(findings, filenames=filenames)
    (out / f"leak-report-{label}.txt").write_text(text, encoding="utf-8")
    # The findings, not the report's layout: no indentation, no blank lines.
    # They are compared with what a reviewer accepted, and that record is stored.
    return [line.strip() for line in text.splitlines()[1:] if line.strip()]


async def prepare(
    case: CaseInput,
    *,
    settings: Settings,
    store: BaselineStore,
    case_preop_md: str,
    warnings: Sequence[str],
    redaction_sha: str,
) -> Prepared:
    """Transcribe the tests and summarise the cleaned record, once.

    A stored preparation is reused only if it was built from this exact signed
    redaction. A summary written from an earlier version of the record would
    describe a record the arms no longer read.
    """

    stored = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    if (
        stored is not None
        and stored.case_summary_md
        and stored.redaction_sha == redaction_sha
        and stored.case_preop_md == case_preop_md
    ):
        return Prepared(
            case_preop_md=stored.case_preop_md,
            tests_preop_md=stored.tests_preop_md,
            case_summary_md=stored.case_summary_md,
            warnings=list(stored.preparation_warnings),
            reused=True,
        )

    openai_config = settings.require_openai()
    # v5 transcribes documents without retries, and that rule is enforced by a
    # test. It is kept for v5. Here it is not: transcription is sequential and
    # all-or-nothing, and on a four-case batch one page timed out on page 12 of
    # 22 and discarded the eleven pages already paid for. The retry is narrow --
    # ``_invoke`` retries only a timeout, a rate limit or a dropped connection,
    # never a response that arrived -- so nothing that was actually transcribed
    # is ever transcribed a second time and made to disagree with itself.
    document_client = image_client(
        settings,
        openai_config,
        timeout_seconds=settings.document_timeout_seconds,
        max_retries=DOCUMENT_RETRIES,
    )
    summary_client = text_client(settings, openai_config)
    calls = 0
    try:
        tests = await TestDocumentAgent(
            document_client,
            previous_page_tail_chars=settings.document_previous_page_tail_chars,
        ).run(case.test_pages)
        calls += len(tests.all_usages)
        summary = await CaseSummaryAgent(summary_client).run(
            case_preop_md=case_preop_md,
            tests_preop_md=tests.output.markdown,
        )
        calls += len(summary.all_usages)
    finally:
        await _close(document_client, summary_client)
    # Frozen immediately. Transcription is the expensive half and has nothing to
    # do with the arms, so a later run -- or a run stopped here to let someone
    # read it -- must not pay for it twice.
    await store.store_preparation(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md=case_preop_md,
        tests_preop_md=tests.output.markdown,
        case_summary_md=summary.output.markdown,
        preparation_warnings=list(warnings),
        redaction_sha=redaction_sha,
    )
    return Prepared(
        case_preop_md=case_preop_md,
        tests_preop_md=tests.output.markdown,
        case_summary_md=summary.output.markdown,
        warnings=list(warnings),
        model_calls=calls,
    )


class SignatureMismatch(ValueError):
    """An approval that names a different set of deletions than the one on file."""


class UnacceptedScreenFindings(ValueError):
    """A signature over a record the screen still flags, with no judgement on the flags."""


@dataclass(slots=True)
class Redaction:
    record: RedactionRecord
    segments: list[Segment]
    clean_md: str
    #: What the keyword screen still finds in the cleaned record.
    findings: list[str]
    model_calls: int = 0

    @property
    def unaccepted(self) -> list[str]:
        return self.record.unaccepted(self.findings)


PlanFor = Callable[[Sequence[Segment]], Awaitable[tuple[RedactionPlan, int]]]


async def redact(
    *,
    raw_md: str,
    case: CaseInput,
    store: BaselineStore,
    plan_for: PlanFor,
    out: Path,
    redo: bool = False,
    also_remove: Sequence[int] = (),
    restore: Sequence[int] = (),
    approve: str | None = None,
    accept_screen_hits: bool = False,
) -> Redaction:
    """Steps 2-6: number, plan, apply, screen, and take or check the signature.

    The plan is made once and frozen; a later run reads it rather than asking
    again, because a second plan would differ and a signature given on the
    first would no longer describe what is being run.
    """

    identity = {
        "case_pages": case.case_pages,
        "test_pages": case.test_pages,
        "images": case.images,
        "case_id": case.case_id,
    }
    segments = segment(raw_md)
    count = len(units(segments))
    digest = source_sha256(raw_md)
    record = await store.load_redaction(**identity)
    calls = 0
    if record is None or record.source_sha256 != digest or redo:
        plan, calls = await plan_for(segments)
        record = RedactionRecord(
            source_sha256=digest,
            unit_count=count,
            deletions=list(plan.deletions),
        )

    beyond = sorted({n for n in (*also_remove, *restore) if n < 1 or n > count})
    if beyond:
        raise ValueError(f"no such sentence: {beyond} (the record has {count})")
    if also_remove or restore:
        removed = (set(record.reviewer_removed) | set(also_remove)) - set(restore)
        restored = (set(record.reviewer_restored) | set(restore)) - set(also_remove)
        record = record.model_copy(
            update={
                "reviewer_removed": sorted(removed),
                "reviewer_restored": sorted(restored),
                # A correction is a different set of deletions, so whatever was
                # signed before no longer describes it -- nor what was judged
                # about the screen's hits on the old cleaned record.
                "approved_sha": None,
                "approved_at": None,
                "accepted_screen_findings": [],
            }
        )

    clean_md = apply(segments, record.removals())
    findings = screen(
        "case-record-cleaned",
        clean_md,
        out,
        filenames=page_filenames(case.case_pages),
    )
    if approve is not None:
        if approve != record.signature():
            raise SignatureMismatch(
                "the signature does not match the redaction on file -- it was made "
                "for a different set of deletions; read the review again"
            )
        pending = record.unaccepted(findings)
        if pending and not accept_screen_hits:
            raise UnacceptedScreenFindings(
                "the screen still flags the cleaned record; remove what states the "
                "answer, or sign with --accept-screen-hits if none of it does"
            )
        record = record.model_copy(
            update={
                "approved_sha": approve,
                "approved_at": datetime.now(UTC),
                "accepted_screen_findings": [
                    *record.accepted_screen_findings,
                    *pending,
                ],
            }
        )
    await store.store_redaction(record, **identity)

    hidden = out / "analysis_only"
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "case_preop.raw.md").write_text(raw_md, encoding="utf-8")
    (hidden / "removed_sentences.json").write_text(
        json.dumps(removed_sentences(segments, record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "redaction-review.md").write_text(
        render_review(
            case_id=case.case_id,
            segments=segments,
            record=record,
            screen_findings=findings,
            flagged_sentences=flagged_sentences(
                segments,
                record.removals(),
                lambda text: bool(screen_markdown(text)),
            ),
            filenames=page_filenames(case.case_pages),
        ),
        encoding="utf-8",
    )
    (out / "case_preop.md").write_text(clean_md, encoding="utf-8")
    return Redaction(
        record=record,
        segments=segments,
        clean_md=clean_md,
        findings=findings,
        model_calls=calls,
    )


def redactor_for(settings: Settings) -> PlanFor:
    """The paid half of redaction: one text call, made only when no plan is on file."""

    async def plan_for(segments: Sequence[Segment]) -> tuple[RedactionPlan, int]:
        client = text_client(settings, settings.require_openai())
        try:
            result = await RecordRedactorAgent(client).run(segments=segments)
        finally:
            await _close(client)
        return result.output, len(result.all_usages)

    return plan_for


async def _close(*clients) -> None:
    for client in clients:
        closer = getattr(client, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001 - shutdown must not mask a result
                logger.warning("A provider client did not close cleanly")


def _dump(value) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def arm_report(state: dict, metrics: ObserverMetrics, *, thread_id: str) -> dict:
    """Everything this arm produced, and nothing that identifies a file on disk."""

    usages = [
        {
            "role": AgentRole(usage.role).value,
            "stage": usage.stage,
            "provider": usage.usage.provider,
            "model": usage.usage.model,
            "input_tokens": usage.usage.input_tokens,
            "output_tokens": usage.usage.output_tokens,
            "total_tokens": usage.usage.total_tokens,
            "latency_ms": usage.usage.latency_ms,
            "attempts": usage.attempts,
        }
        for usage in state["role_usages"]
    ]
    calls_per_role: dict[str, int] = {}
    for usage in usages:
        calls_per_role[usage["role"]] = calls_per_role.get(usage["role"], 0) + 1
    return {
        "thread_id": thread_id,
        "metrics": metrics.model_dump(mode="json"),
        "role_models": [_dump(item) for item in state["role_models"]],
        "outputs": {
            "resident": _dump(state["resident_report"]),
            "imaging": _dump(state["imaging"]),
            # What each batch of films said before the readings were merged: the
            # only way to see a consolidation that dropped or softened a finding.
            "imaging_batches": [_dump(item) for item in state["imaging_batches"]],
            "neurology": _dump(state["neurology"]),
            "anesthesia": _dump(state["anesthesia"]),
            "physiatrist_opening": _dump(state["physiatrist_opening"]),
            "physiatrist_final": _dump(state["physiatrist_latest"]),
            # One entry per round, because a position withdrawn in round two is
            # the behaviour under study, not a superseded draft.
            "physiatrist_responses": [
                _dump(item) for item in state["physiatrist_responses"]
            ],
            "vice_opinions": [_dump(item) for item in state["vice_opinions"]],
            "director_rulings": [_dump(item) for item in state["director_rulings"]],
        },
        "transcript": [
            {
                "message_id": message.message_id,
                "sequence": message.sequence,
                "turn_number": message.turn_number,
                "speaker_role": message.speaker_role.value,
                "speaker_name": message.speaker_name,
                "characters": len(message.content),
                "content": message.content,
            }
            for message in state["chat_log"]
        ],
        "usage": usages,
        "calls_per_role": calls_per_role,
        # How many tries a role needed before its output satisfied its schema.
        # Note the limit: an attempt the provider rejected outright reports no
        # usage at all, so its tokens are spent and not counted here.
        "attempts_per_role": {
            role: max(item["attempts"] for item in usages if item["role"] == role)
            for role in calls_per_role
        },
    }


#: Which roles read the style under test (amendment A4). ``both`` is the main
#: design; in the other two, the role left out reads the neutral prefix.
PERSONA_PLACEMENTS = ("both", "chair", "deputy")


def role_styles(
    style: DirectorStyle,
    persona_on: str = "both",
    negotiation: bool = True,
) -> tuple[DirectorStyle, DirectorStyle | None]:
    """The director's and the deputy's prefix for the style under test.

    Without negotiation the deputy never speaks, so it reads nothing.
    """

    if persona_on not in PERSONA_PLACEMENTS:
        raise ValueError(f"unknown persona placement {persona_on!r}")
    if persona_on == "deputy" and not negotiation:
        raise ValueError("a deputy-only arm needs the negotiation the deputy speaks in")
    neutral = DirectorStyle.NEUTRAL
    director = style if persona_on in ("both", "chair") else neutral
    if not negotiation:
        return director, None
    return director, style if persona_on in ("both", "deputy") else neutral


async def run_arm(
    *,
    style: DirectorStyle,
    case: CaseInput,
    prepared: Prepared,
    settings: Settings,
    registry,
    assignments,
    baseline_store: BaselineStore,
    checkpoint_path: Path,
    attempt: int = 1,
    thread_suffix: str = "",
    persona_on: str = "both",
    negotiation: bool = True,
) -> tuple[dict, ObserverMetrics]:
    # A finished arm cannot be driven again -- the transcript reducer refuses to
    # blend two discussions into one -- so a retry after a provider failure needs
    # its own thread. The case id is unchanged, so the baseline is still shared
    # and is not paid for a second time.
    # ``thread_suffix`` names the evidence when a case can be prepared more
    # than once under one label -- a published case re-filtered -- so a new
    # preparation never lands on the thread an old one finished.
    thread_id = f"{case.case_id}-{style.value}-{attempt}{thread_suffix}"
    director_style, deputy_style = role_styles(style, persona_on, negotiation)
    async with open_sqlite_checkpointer(checkpoint_path) as saver:
        graph = create_department_graph(
            registry=registry,
            checkpointer=saver,
            baseline_store=baseline_store,
        )
        state = await graph.ainvoke(
            department_input(
                case_id=case.case_id,
                director_style=director_style,
                deputy_style=deputy_style,
                negotiation=negotiation,
                artifact_dir=case.artifact_dir,
                case_pages=case.case_pages,
                test_pages=case.test_pages,
                images=case.images,
                case_preop_md=prepared.case_preop_md,
                tests_preop_md=prepared.tests_preop_md,
                case_summary_md=prepared.case_summary_md,
                preparation_warnings=prepared.warnings,
                role_models=assignments,
            ),
            config=thread_config(thread_id),
        )
    metrics = observe(state)
    report = arm_report(state, metrics, thread_id=thread_id)
    # The style under test. In a deputy-only arm the director's style is
    # neutral, so the metrics alone would not say which arm this is.
    report["arm"] = style.value
    return report, metrics


async def run_arms(
    *,
    styles: Sequence[DirectorStyle],
    case,
    prepared: Prepared,
    settings: Settings,
    registry,
    assignments,
    baseline_store: BaselineStore,
    out: Path,
    attempt: int,
    thread_suffix: str = "",
    persona_on: str = "both",
    negotiation: bool = True,
) -> tuple[list[ObserverMetrics], list[dict], list[str]]:
    """Run the requested arms in order and write one report per arm.

    Sequential on purpose: the first arm freezes the baseline the other two
    read. Run them together and each would generate its own.
    """

    rows: list[ObserverMetrics] = []
    reports: list[dict] = []
    failed: list[str] = []
    for style in DirectorStyle:
        if style not in styles:
            continue
        logger.info("Running the %s arm", style.value)
        try:
            report, metrics = await run_arm(
                style=style,
                case=case,
                prepared=prepared,
                settings=settings,
                registry=registry,
                assignments=assignments,
                baseline_store=baseline_store,
                checkpoint_path=Path(settings.checkpoint_db_path),
                attempt=attempt,
                thread_suffix=thread_suffix,
                persona_on=persona_on,
                negotiation=negotiation,
            )
        except Exception as exc:  # noqa: BLE001 - a failed arm is a result too
            # An arm that dies still cost money, and the arms that already
            # finished are still on disk. Record what happened and keep the
            # rest rather than ending with nothing written.
            logger.error("  %s arm failed: %s", style.value, type(exc).__name__)
            failed.append(style.value)
            (out / f"arm-{style.value}-attempt{attempt}-failed.json").write_text(
                json.dumps(
                    {
                        "director_style": style.value,
                        "persona_on": persona_on,
                        "negotiation": negotiation,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        # One file per attempt: a repeat of the same cell is the evidence of
        # whether the cell is stable, and it must not overwrite the first.
        (out / report_name("arm", attempt, style)).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(metrics)
        reports.append(report)
        logger.info(
            "  %s: %s after %d round(s), %d model calls",
            style.value,
            metrics.decision.value,
            metrics.negotiation_rounds,
            len(report["usage"]),
        )
    return rows, reports, failed


def attempt_arms(out: Path, attempt: int) -> tuple[list[ObserverMetrics], list[dict]]:
    """Every arm of this attempt on disk, in arm order, whenever it was run.

    An arm can be added to an attempt later -- the control arm was, after the
    three styles had finished -- or retried on its own. The summary compares
    all of them; built from one invocation's arms alone, a later single-arm run
    would replace a three-arm summary with a one-arm one.
    """

    rows: list[ObserverMetrics] = []
    reports: list[dict] = []
    for style in DirectorStyle:
        path = out / report_name("arm", attempt, style)
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(ObserverMetrics.model_validate(report["metrics"]))
        reports.append(report)
    return rows, reports


def report_name(kind: str, attempt: int, style: DirectorStyle | None = None) -> str:
    """``arm-avoidant-attempt2.json``, ``summary-attempt2.json``."""

    arm = f"-{style.value}" if style is not None else ""
    return f"{kind}{arm}-attempt{attempt}.json"


def prompt_fingerprints(registry) -> dict[str, str]:
    """A digest of every prompt actually loaded, role prompts and style prefixes.

    Two arms run minutes apart from an edited file would produce a difference
    with no record of why. The digests are of our own prompt text and carry no
    case material.
    """

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    fingerprints: dict[str, str] = {}
    for agent in registry.all_agents():
        fingerprints[agent.role.value] = digest(agent.system_prompt)
        for style, prefix in getattr(agent, "style_prompts", {}).items():
            owner = "deputy-style" if agent.role is AgentRole.VICE_DIRECTOR else "style"
            fingerprints[f"{owner}:{style.value}"] = digest(prefix)
        negotiation = getattr(agent, "negotiation_prompt", None)
        if negotiation:
            # The advocate's negotiation turns read an instruction its frozen
            # opening did not, and a change to it must be visible run by run.
            fingerprints[f"{agent.role.value}:negotiation"] = digest(negotiation)
    return fingerprints


#: The roles that speak only after the baseline is frozen: the first turn of
#: either one ends stage I-II. The director comes first when nobody negotiates.
ARM_ROLES = frozenset({AgentRole.VICE_DIRECTOR.value, AgentRole.DIRECTOR.value})


def baseline_prefix(report: dict) -> list[tuple]:
    """The frozen stage I-II part of one arm's transcript.

    It ends at the first turn that carries a style -- the deputy's, or the
    director's in an arm without negotiation. The identifier is excluded
    because it names the arm by design.
    """

    prefix: list[tuple] = []
    for message in report["transcript"]:
        if message["speaker_role"] in ARM_ROLES:
            break
        prefix.append(
            (
                message["sequence"],
                message["speaker_role"],
                message["speaker_name"],
                message["content"],
            )
        )
    return prefix


def comparison(rows: Sequence[ObserverMetrics], reports: Sequence[dict]) -> dict:
    """The table the experiment exists to produce, plus its own validity check."""

    prefixes = [baseline_prefix(report) for report in reports]
    return {
        "arms": [
            {
                # Reports written before amendment A4 carry no "arm"; their
                # arm was the director's style.
                "arm": report.get("arm", row.director_style.value),
                "director_style": row.director_style.value,
                "deputy_style": row.deputy_style.value if row.deputy_style else None,
                "negotiation": row.negotiation,
                "decision": row.decision.value,
                "surgical_indication": row.surgical_indication.value,
                "workup_would_change_decision_to": (
                    row.workup_would_change_decision_to.value
                    if row.workup_would_change_decision_to
                    else None
                ),
                "surgery_recommended": row.surgery_recommended,
                "surgery_intended": row.surgery_intended,
                "termination_cause": (
                    row.termination_cause.value if row.termination_cause else None
                ),
                "negotiation_rounds": row.negotiation_rounds,
                "turn_count": row.turn_count,
                "total_characters": row.total_characters,
                "director_character_share": row.director_character_share,
                "negotiation_tokens": row.negotiation_tokens,
                "juniors_addressed_declared": row.juniors_addressed_declared,
                "juniors_addressed_detected": row.juniors_addressed_detected,
                "liability_transfer_declared": row.liability_transfer_declared,
                "liability_transfer_detected": row.liability_transfer_detected,
                "family_requests_declared": row.family_requests_declared,
                "tests_ordered_declared": row.tests_ordered_declared,
                "studies_mentioned_detected": row.studies_mentioned_detected,
                "deputy_preference_stated": row.deputy_preference_stated,
                "deputy_disagreements": row.deputy_disagreements,
                "deputy_questions_to_director": row.deputy_questions_to_director,
                "contingency_plan_declared": row.contingency_plan_declared,
                "baseline_conservative_success_rate": (
                    row.baseline_conservative_success_rate
                ),
                "flags_changing_indication": row.flags_changing_indication,
                "flags_changing_plan": row.flags_changing_plan,
                "flags_coexisting": row.flags_coexisting,
                "consensus_declared": row.consensus_declared.value,
                "physiatrist_final_stance": row.physiatrist_final_stance.value,
                "advocate_agrees_with_ruling": row.advocate_agrees_with_ruling,
                "consensus_reached": row.consensus_reached,
            }
            for row, report in zip(rows, reports, strict=True)
        ],
        "validity": {
            # One key across the arms is the evidence that the manipulated
            # variable never reached the evidence.
            "one_baseline": len({row.baseline_key for row in rows}) == 1,
            "baseline_keys": sorted({row.baseline_key or "" for row in rows}),
            "generated_then_reused": [row.baseline_reused for row in rows],
            # What must be identical is the evidence, word for word. Turn counts
            # and character totals must NOT be: the arms are expected to differ
            # there, which is the result being measured.
            "identical_baseline_transcript": all(
                prefix == prefixes[0] for prefix in prefixes
            ),
            "baseline_turns": [len(prefix) for prefix in prefixes],
        },
    }


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument(
        "--case-id",
        required=True,
        help="A label of your choosing. It is stored and must not name a patient.",
    )
    parser.add_argument("--out", type=Path, default=Path(".data/experiments"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ingest, slice and OCR the case record, then stop before paying.",
    )
    parser.add_argument(
        "--approve-redaction",
        metavar="SIGNATURE",
        help=(
            "Sign the redaction after reading redaction-review.md. The signature "
            "names exactly which sentences are removed from exactly which "
            "transcription; any correction makes a new one."
        ),
    )
    parser.add_argument(
        "--also-remove",
        metavar="N,N",
        help="Remove these sentences as well: the model left them in and they state the answer.",
    )
    parser.add_argument(
        "--restore",
        metavar="N,N",
        help="Put these sentences back: the model removed them and they are evidence.",
    )
    parser.add_argument(
        "--accept-screen-hits",
        action="store_true",
        help=(
            "With --approve-redaction: you read the sentences the keyword screen "
            "still flags and none of them states this operation (past surgical "
            "history, say). The judgement is recorded with the signature, and the "
            "later screens of the test pages and the summary stay on."
        ),
    )
    parser.add_argument(
        "--redo-redaction",
        action="store_true",
        help="Ask the model for a new plan instead of reading the one on file.",
    )
    parser.add_argument(
        "--skip-leak-screen",
        action="store_true",
        help=(
            "Run even though the record states what was decided or done. The "
            "result is not an experiment: the department will be reading the "
            "answer rather than reaching one."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Transcribe and summarise, freeze the result, and stop. The first "
            "paid call happens here, so a misconfigured image provider fails on "
            "one page rather than after a dozen films, and the transcription can "
            "be read before the arms argue over it."
        ),
    )
    parser.add_argument(
        "--only-style",
        action="append",
        choices=[style.value for style in DirectorStyle],
        help=(
            "Run only this arm; repeatable. Use it to retry one arm after a "
            "provider failure without re-running, or re-paying for, the others."
        ),
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help=(
            "Attempt number, folded into the thread id. A finished arm cannot be "
            "driven again, so a retry needs a new attempt; the baseline is keyed "
            "on the case and is still shared."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    try:
        sources = collect_sources(args.case_dir)
    except AmbiguousFolder as exc:
        logger.error("%s", exc)
        return 2
    for folder, category in sorted(sources.routing.items()):
        logger.info("  %-28s -> %s", folder, category)
    if not (sources.case_pages and sources.test_pages and sources.images):
        logger.error(
            "Need case-record pages, test pages and preoperative images; found %d/%d/%d",
            len(sources.case_pages),
            len(sources.test_pages),
            len(sources.images),
        )
        return 2
    logger.info(
        "Sources: %d case pages, %d test pages, %d images, %d excluded as postoperative "
        "or unclassified",
        len(sources.case_pages),
        len(sources.test_pages),
        len(sources.images),
        len(sources.excluded),
    )

    started = datetime.now(UTC)
    out = args.out / args.case_id
    out.mkdir(parents=True, exist_ok=True)
    case = await ingest(
        sources,
        case_id=args.case_id,
        settings=settings,
        thread_id=f"{args.case_id}-{started:%Y%m%d%H%M%S}",
    )
    logger.info(
        "Ingested: %d case pages after slicing, %d test pages, %d images",
        len(case.case_pages),
        len(case.test_pages),
        len(case.images),
    )

    baseline_store = BaselineStore(settings.baseline_store_dir)
    # Step 2: the case record is OCR'd locally, which costs nothing. The raw
    # transcription states the answer -- that is what the redaction is for -- so
    # it is kept apart from anything an agent reads.
    raw_md, warnings = await transcribe_case_record(case, settings)
    sentences = len(units(segment(raw_md)))
    logger.info(
        "OCR produced %d characters in %d sentences from %d pages, %d low-confidence warnings",
        len(raw_md),
        sentences,
        len(case.case_pages),
        len(warnings),
    )
    raw_findings = screen(
        "case-record-raw",
        raw_md,
        out,
        filenames=page_filenames(case.case_pages),
    )
    logger.info(
        "Before redaction the keyword screen finds %d decision-revealing phrase lines "
        "(expected: the record was written after the fact).",
        len(raw_findings),
    )
    if args.dry_run:
        logger.info(
            "A live run would ask the redactor once, stop for your review, then call "
            "a provider for %d test pages, 1 summary, and at most %d department turns.",
            len(case.test_pages),
            5 + 3 * 2 * len(DirectorStyle),
        )
        return 0

    # Steps 3-6: plan, apply, screen what is left, and stop for a signature.
    try:
        redaction = await redact(
            raw_md=raw_md,
            case=case,
            store=baseline_store,
            plan_for=redactor_for(settings),
            out=out,
            redo=args.redo_redaction,
            also_remove=parse_numbers(args.also_remove),
            restore=parse_numbers(args.restore),
            approve=args.approve_redaction,
            accept_screen_hits=args.accept_screen_hits,
        )
    except SignatureMismatch as exc:
        logger.error("%s", exc)
        return 4
    except UnacceptedScreenFindings as exc:
        logger.error("%s", exc)
        return 3
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    record = redaction.record
    logger.info(
        "Redaction: %d of %d sentences removed (%d model call(s)); review at %s",
        len(record.removals()),
        record.unit_count,
        redaction.model_calls,
        out / "redaction-review.md",
    )
    if redaction.unaccepted and not args.skip_leak_screen:
        logger.error(
            "The screen still flags the cleaned record. The flagged sentences are "
            "listed in the review: remove the ones that state this operation with "
            "--also-remove N,N, or, if none does, sign with --accept-screen-hits."
        )
        for line in redaction.unaccepted:
            logger.error("%s", line)
        if not record.approved:
            logger.info("Signature for the current redaction: %s", record.signature())
        return 3
    if not record.approved:
        logger.info(
            "Waiting for your review. Read %s, correct it with --also-remove / "
            "--restore if needed, then run again with --approve-redaction %s",
            out / "redaction-review.md",
            record.signature(),
        )
        return 4
    logger.info("Redaction signed %s at %s", record.signature()[:12], record.approved_at)

    styles = [
        style
        for style in DirectorStyle
        if not args.only_style or style.value in args.only_style
    ]
    prepared = await prepare(
        case,
        settings=settings,
        store=baseline_store,
        case_preop_md=redaction.clean_md,
        warnings=warnings,
        redaction_sha=record.signature(),
    )
    (out / "case_preop.md").write_text(prepared.case_preop_md, encoding="utf-8")
    (out / "tests_preop.md").write_text(prepared.tests_preop_md, encoding="utf-8")
    (out / "case_summary.md").write_text(prepared.case_summary_md, encoding="utf-8")
    logger.info(
        "Preparation %s (%d model calls), %d transcription warnings",
        "reused" if prepared.reused else "generated",
        prepared.model_calls,
        len(prepared.warnings),
    )
    for warning in prepared.warnings:
        logger.info("  %s", warning)
    transcribed_leaks = screen(
        "test-pages",
        prepared.tests_preop_md,
        out,
        filenames=page_filenames(case.test_pages),
    ) + screen("case-summary", prepared.case_summary_md, out)
    if transcribed_leaks and not args.skip_leak_screen:
        logger.error("Stopping before the arms run; the transcription states the answer.")
        for line in transcribed_leaks:
            logger.error("%s", line)
        return 3
    if args.prepare_only:
        logger.info("Frozen. Read %s, then run again without --prepare-only.", out)
        return 0

    registry, assignments = create_department(settings)
    try:
        rows, reports, failed = await run_arms(
            styles=styles,
            case=case,
            prepared=prepared,
            settings=settings,
            registry=registry,
            assignments=assignments,
            baseline_store=baseline_store,
            out=out,
            attempt=args.attempt,
        )
    finally:
        await _close(*(getattr(agent, "client", None) for agent in registry.all_agents()))

    if not rows:
        logger.error("No arm completed; nothing to compare.")
        return 1

    summary = {
        "case_id": args.case_id,
        "workflow_version": DEPARTMENT_WORKFLOW_VERSION,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "sources": {
            "case_pages": len(case.case_pages),
            "test_pages": len(case.test_pages),
            "images": len(case.images),
            "excluded_files": len(sources.excluded),
        },
        "preparation": {
            "reused": prepared.reused,
            "model_calls": prepared.model_calls,
            "warnings": prepared.warnings,
        },
        "redaction": {
            "signature": record.signature(),
            "approved_at": record.approved_at.isoformat() if record.approved_at else None,
            "sentences": record.unit_count,
            "removed": len(record.removals()),
            "by_category": {
                category.value: sum(
                    1
                    for item in record.deletions
                    if item.category is category
                    and item.unit in record.removals()
                )
                for category in RedactionCategory
            },
            "reviewer_removed": record.reviewer_removed,
            "reviewer_restored": record.reviewer_restored,
        },
        "failed_arms": failed,
        "attempt": args.attempt,
        # A prompt edited between arms would otherwise be undetectable, which is
        # the drift the per-role model record exists to catch.
        "prompt_fingerprints": prompt_fingerprints(registry),
        "settings": {
            "text_provider": settings.text_provider,
            "image_provider": settings.image_provider,
            "openai_image_detail": settings.openai_image_detail,
            "deepseek_thinking": settings.deepseek_thinking,
            "radiology_batch_max_images": settings.radiology_batch_max_images,
            "radiology_batch_concurrency": settings.radiology_batch_concurrency,
            "max_negotiation_rounds": MAX_NEGOTIATION_ROUNDS,
            "page_slice_max_height": settings.page_slice_max_height,
            "ocr_low_confidence_threshold": settings.ocr_low_confidence_threshold,
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
