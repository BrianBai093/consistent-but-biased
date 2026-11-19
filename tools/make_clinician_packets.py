"""Build the three packets milestone M7 asks clinicians to complete.

1. **Labels**: every case's full published summary (the answer included), with
   blank fields for the group and the real decision. Two clinicians fill a copy
   each, independently; their agreement is reported.
2. **Filter audit**: for the replication subset, the full summary beside what
   the department was shown, with each removed passage listed. The clinician
   marks anything left in that states the decision, and anything removed that
   was a preoperative finding.
3. **Blinded rating**: for the replication subset, the filtered record and the
   four arms' final rulings in a random order under neutral letters. The key
   that maps letters to arms is written to a separate file the raters never
   receive.

Nothing here calls a model. The output is for people, not for the experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Sequence

from tools.run_text_case_experiment import DEFAULT_PMC_CSV, read_pmc_patient

ARMS = ("neutral", "democratic", "autocratic", "avoidant")
SEED = 20261012


def replication_subset(manifest: dict) -> list[str]:
    """Plan v1.0: the first 4 / 3 / 3 cases of each group, in manifest order."""

    quota = {"borderline_or_not_operated": 4, "hard_diagnosis": 3, "clear_indication": 3}
    chosen = []
    for case in manifest["cases"]:
        if quota.get(case["group"], 0) > 0:
            chosen.append(case["patient_uid"])
            quota[case["group"]] -= 1
    return chosen


def ruling_text(report: dict) -> str:
    ruling = report["outputs"]["director_rulings"][-1]
    plan = ruling.get("surgical_plan")
    lines = [f"Decision: {ruling['decision']}", "", ruling["public_message"]]
    if plan:
        lines += ["", f"Surgical plan: {plan}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/test_sets/pmc_dcm_v1.json"))
    parser.add_argument("--pmc-csv", type=Path, default=DEFAULT_PMC_CSV)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("output/clinician_packets"))
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subset = replication_subset(manifest)
    labels = args.out / "1_labels"
    audit = args.out / "2_filter_audit"
    rating = args.out / "3_blinded_rating"
    key_dir = args.out / "KEY_do_not_send"
    for folder in (labels, audit, rating, key_dir):
        folder.mkdir(parents=True, exist_ok=True)

    with (labels / "labels_form.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patient_uid",
                "pmid",
                "is_adult_degenerative_cervical_myelopathy(yes/no)",
                "group(borderline_or_not_operated/hard_diagnosis/clear_indication)",
                "real_decision(operated/operated_after_delay/not_operated)",
                "notes",
            ]
        )
        for case in manifest["cases"]:
            text, pmid = read_pmc_patient(args.pmc_csv, case["patient_uid"])
            (labels / f"{case['patient_uid']}.txt").write_text(text, encoding="utf-8")
            writer.writerow([case["patient_uid"], pmid, "", "", "", ""])

    rng = random.Random(SEED)
    key_rows = []
    with (audit / "filter_audit_form.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patient_uid",
                "decision_left_in(yes/no)",
                "what_was_left_in",
                "preoperative_finding_removed(yes/no)",
                "what_was_removed",
            ]
        )
        for uid in subset:
            folder = args.runs / f"pmc-{uid}"
            text, _ = read_pmc_patient(args.pmc_csv, uid)
            kept = (folder / "case_preop.md").read_text(encoding="utf-8")
            removed = json.loads(
                (folder / "analysis_only" / "removed_passages.json").read_text(encoding="utf-8")
            )
            body = [
                f"# {uid}",
                "",
                "## A. Full published summary",
                "",
                text,
                "## B. What the department was shown",
                "",
                kept.split("-->", 1)[-1].strip(),
                "",
                "## C. Removed passages",
                "",
                *[f"{n}. …{item['before']} **[removed] {item['text']}** {item['after']}…"
                  for n, item in enumerate(removed, start=1)],
            ]
            (audit / f"{uid}.md").write_text("\n".join(body) + "\n", encoding="utf-8")
            writer.writerow([uid, "", "", "", ""])

            reports = {
                arm: json.loads(
                    (folder / f"arm-{arm}-attempt{args.attempt}.json").read_text(encoding="utf-8")
                )
                for arm in ARMS
            }
            order = list(ARMS)
            rng.shuffle(order)
            sheet = [
                f"# {uid}",
                "",
                "## Case as presented to the department",
                "",
                kept.split("-->", 1)[-1].strip(),
                "",
            ]
            for letter, arm in zip("ABCD", order):
                sheet += [f"## Ruling {letter}", "", ruling_text(reports[arm]), ""]
                key_rows.append([uid, letter, arm])
            (rating / f"{uid}.md").write_text("\n".join(sheet), encoding="utf-8")

    with (rating / "rating_form.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patient_uid",
                "ruling(A-D)",
                "appropriateness(1-5)",
                "would_you_decide_the_same(yes/no)",
                "notes",
            ]
        )
        for uid in subset:
            for letter in "ABCD":
                writer.writerow([uid, letter, "", "", ""])
    with (key_dir / "rating_key.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_uid", "ruling", "arm"])
        writer.writerows(key_rows)
    print(f"Wrote packets for {len(manifest['cases'])} cases ({len(subset)} audited/rated) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
