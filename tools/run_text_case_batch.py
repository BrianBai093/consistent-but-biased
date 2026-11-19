"""Run every case of a published-case test set through the department workflow.

Each case goes through ``run_text_case_experiment`` exactly as it would on its
own, one case after another. A case whose summary for this attempt already
exists is skipped, so a batch stopped halfway -- a provider outage, a closed
laptop -- resumes where it stopped instead of paying again.

The manifest carries the real decision for each case. That is the answer: this
script passes nothing but the ``patient_uid`` to the runner, and the labels
appear only in the batch report written after the runs, beside the results
they are to be compared with.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from schemas import DirectorStyle
from tools import run_text_case_experiment as runner
from tools.run_department_experiment import report_name

logger = logging.getLogger("department.text_batch")

DEFAULT_MANIFEST = Path("data/test_sets/pmc_dcm_v1.json")


def case_id_for(uid: str) -> str:
    return f"pmc-{uid}"


def arm_results(summary_path: Path) -> dict | None:
    """The per-arm outcome of one finished case, or None if it has not finished."""

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _results(summary)
    except (OSError, ValueError, KeyError):
        # A summary this batch cannot read is reported as unfinished rather
        # than ending the batch -- and losing the report -- after the runs.
        return None


def _results(summary: dict) -> dict:
    return {
        "arms": {
            arm.get("arm", arm["director_style"]): {
                "decision": arm["decision"],
                "surgical_indication": arm["surgical_indication"],
                "termination_cause": arm["termination_cause"],
                "negotiation_rounds": arm["negotiation_rounds"],
            }
            for arm in summary["arms"]
        },
        "valid": summary["validity"]["one_baseline"]
        and summary["validity"]["identical_baseline_transcript"],
        "filter_attempts": summary["filter"]["attempts"],
        "removed_passages": summary["filter"]["removed_passages"],
        "unreviewed_screen_hits": len(summary["filter"]["unreviewed_screen_hits"]),
        "failed_arms": summary["failed_arms"],
    }


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pmc-csv", type=Path, default=runner.DEFAULT_PMC_CSV)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        metavar="PATIENT_UID",
        help="Run only this case; repeatable.",
    )
    parser.add_argument(
        "--only-style",
        action="append",
        choices=[style.value for style in DirectorStyle],
        help="Run only this arm in every case; repeatable. Arms already on disk "
        "for the attempt are kept and joined into the case's summary.",
    )
    parser.add_argument("--series", help="Run a robustness series; see the case runner.")
    parser.add_argument(
        "--persona-on",
        choices=runner.PERSONA_PLACEMENTS,
        default="both",
        help="Which role reads the style under test; see the case runner.",
    )
    parser.add_argument(
        "--no-negotiation",
        action="store_true",
        help="The director rules alone after stage II; see the case runner.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run a case even if its summary for this attempt exists.",
    )
    args = parser.parse_args(argv)
    refusal = runner.check_design(args.series, args.persona_on, not args.no_negotiation)
    if refusal:
        parser.error(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [
        case
        for case in manifest["cases"]
        if not args.only or case["patient_uid"] in args.only
    ]
    out = runner.series_dir(args.out, args.series)
    styles = [
        style
        for style in DirectorStyle
        if not args.only_style or style.value in args.only_style
    ]
    started = datetime.now(UTC)
    rows = []
    for index, case in enumerate(cases, start=1):
        uid = case["patient_uid"]
        case_id = case_id_for(uid)
        summary_path = out / case_id / report_name("summary", args.attempt)
        logger.info("[%d/%d] %s", index, len(cases), case_id)
        missing = [
            style
            for style in styles
            if not (out / case_id / report_name("arm", args.attempt, style)).exists()
        ]
        if not args.dry_run and not args.force and not missing:
            logger.info("  every requested arm finished for attempt %d; skipped", args.attempt)
            code = 0
        else:
            command = [
                "--pmc-uid", uid,
                "--pmc-csv", str(args.pmc_csv),
                "--case-id", case_id,
                "--out", str(args.out),
                "--attempt", str(args.attempt),
            ]
            if args.dry_run:
                command.append("--dry-run")
            if args.series:
                command.extend(["--series", args.series])
            if args.persona_on != "both":
                command.extend(["--persona-on", args.persona_on])
            if args.no_negotiation:
                command.append("--no-negotiation")
            for style in missing if not args.force else styles:
                command.extend(["--only-style", style.value])
            try:
                code = await runner.main(command)
            except Exception as exc:  # noqa: BLE001 - one case must not end the batch
                logger.error("  %s failed: %s: %s", case_id, type(exc).__name__, exc)
                code = -1
            if not args.dry_run and missing and all(
                (out / case_id / f"arm-{style.value}-attempt{args.attempt}-failed.json").exists()
                for style in missing
            ):
                # Every arm of a case failing at once is the provider refusing,
                # not the case: an exhausted balance answered 402 to all 56 arms
                # of the 14 cases that followed it. Stop instead of recording
                # the same refusal for every case left.
                logger.error(
                    "  every arm of %s failed; stopping the batch. Fix the provider "
                    "and run again -- finished cases are skipped.",
                    case_id,
                )
                rows.append(
                    {
                        "patient_uid": uid,
                        "case_id": case_id,
                        "group": case["group"],
                        "exit_code": code,
                        "real_decision": case["real_decision"],
                        "result": None,
                    }
                )
                break
        rows.append(
            {
                "patient_uid": uid,
                "case_id": case_id,
                "group": case["group"],
                "exit_code": code,
                # Analysis layer: the answer, next to what the arms concluded.
                "real_decision": case["real_decision"],
                "result": None if args.dry_run else arm_results(summary_path),
            }
        )
    if args.dry_run:
        logger.info("Dry run: %d case(s) ingested and segmented; nothing paid.", len(rows))
        return 0 if all(row["exit_code"] == 0 for row in rows) else 1

    report = {
        "manifest": manifest["name"],
        "attempt": args.attempt,
        "series": args.series,
        "persona_on": args.persona_on,
        "negotiation": not args.no_negotiation,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "cases": rows,
    }
    path = out / f"batch-{manifest['name']}-attempt{args.attempt}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finished = sum(1 for row in rows if row["result"] is not None)
    logger.info("Wrote %s: %d of %d case(s) finished", path, finished, len(rows))
    return 0 if finished == len(rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
