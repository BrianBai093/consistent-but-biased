"""The analysis the experiment plan fixes for M10 and M11 (§11-§13), before any run.

Both take the chair apart from something the main design changed with it:

- **M10** from the deputy. For the autocratic and avoidant styles, a 2x2 of the
  director's prefix (neutral or the style) by the deputy's. Two cells are the
  main runs (neutral; both carry the style); two are new (chair only, deputy
  only). H8a-b test the chair-only cell against neutral, one-sided in the main
  analysis's direction; H8c-d test the deputy-only cell, two-sided; Holm over
  the four. The decomposition -- each role's share of the main effect, and the
  interaction -- is descriptive, with case-level bootstrap intervals.
- **M11** from the negotiation. The director rules once, alone, on the frozen
  stage I-II record: neutral, autocratic and avoidant. H9a-b test each style
  against the neutral director alone (one-sided, Holm over two). The
  interaction compares the effect alone with the effect after negotiation, per
  case: first against the chair-only cell of M10 (the persona sits on the chair
  in both; only the negotiation differs), then against the main runs.

Every new run must have read the main runs' frozen baseline -- the same key and
the same stage I-II statements. A case where one did not is left out and
named, never repaired.

Statistics are the main analysis's own (``analyze_text_experiment``): per-case
shares of runs, an exact sign-flip permutation test, a case-level bootstrap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

from tools.analyze_text_experiment import (
    BOOTSTRAP_RESAMPLES,
    FAMILY_TRANSFER,
    SEED,
    Case,
    Run,
    bootstrap_ci,
    holm,
    load,
    share,
    sign_flip_p,
)
from tools.run_department_experiment import baseline_prefix

STYLES = ("autocratic", "avoidant")
#: The main analysis's direction for each style, which H8a-b and H9a-b keep.
DIRECTION = {"autocratic": "greater", "avoidant": "less"}
MAIN_ATTEMPTS = (1, 2, 3, 4)
#: Where each new cell is written, and which arms it holds.
SERIES = {
    "chair_only": ("series-chair-only", STYLES),
    "deputy_only": ("series-deputy-only", STYLES),
    "alone": ("series-chair-alone", ("neutral", *STYLES)),
}
NEW_ATTEMPTS = (1,)


def operate(run: Run) -> bool:
    return run.decision == "operate"


# ------------------------------------------------------------------ data ---

Cells = dict[str, dict[str, list[Run]]]


def cells_by_case(out: Path, manifest: dict) -> tuple[dict[str, Cells], dict[str, Case]]:
    """Per case, every cell's runs: ``main``, ``chair_only``, ``deputy_only``, ``alone``."""

    main = {case.uid: case for case in load(out, manifest, MAIN_ATTEMPTS)}
    result: dict[str, Cells] = {uid: {"main": case.runs} for uid, case in main.items()}
    for cell, (folder, _) in SERIES.items():
        for case in load(out / folder, manifest, NEW_ATTEMPTS):
            result[case.uid][cell] = case.runs
    return result, main


def validity(out: Path, manifest: dict) -> dict[str, list[str]]:
    """The new runs that did not read the main runs' frozen evidence, per case."""

    problems: dict[str, list[str]] = {}
    for item in manifest["cases"]:
        folder = f"pmc-{item['patient_uid']}"
        keys, prefixes = set(), []
        for path in sorted((out / folder).glob("arm-*-attempt*.json")):
            if path.name.endswith("-failed.json"):
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            keys.add(report["metrics"]["baseline_key"])
            prefixes.append(baseline_prefix(report))
        for cell, (series, arms) in SERIES.items():
            for arm in arms:
                for attempt in NEW_ATTEMPTS:
                    path = out / series / folder / f"arm-{arm}-attempt{attempt}.json"
                    if not path.exists():
                        continue
                    report = json.loads(path.read_text(encoding="utf-8"))
                    label = f"{cell}:{arm}:attempt{attempt}"
                    if len(keys) != 1 or report["metrics"]["baseline_key"] not in keys:
                        problems.setdefault(item["patient_uid"], []).append(
                            f"{label}: another baseline"
                        )
                    elif prefixes and baseline_prefix(report) != prefixes[0]:
                        problems.setdefault(item["patient_uid"], []).append(
                            f"{label}: other stage I-II statements"
                        )
    return problems


def runs(cells: Cells, cell: str, arm: str) -> list[Run]:
    return cells.get(cell, {}).get(arm, [])


# ------------------------------------------------------------ statistics ---


def ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    seed: int = SEED,
) -> tuple[float, float] | None:
    """Case-level bootstrap interval for mean(numerator) / mean(denominator).

    None when the denominator can be zero within the resamples: a share of an
    effect that may not exist is not a number.
    """

    rng = random.Random(seed)
    n = len(numerators)
    ratios = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picks = [rng.randrange(n) for _ in range(n)]
        denominator = sum(denominators[i] for i in picks)
        if denominator == 0:
            return None
        ratios.append(sum(numerators[i] for i in picks) / denominator)
    ratios.sort()
    return (
        ratios[int(0.025 * BOOTSTRAP_RESAMPLES)],
        ratios[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    )


def describe(differences: Sequence[Fraction], alternative: str | None = None) -> dict:
    values = [float(value) for value in differences]
    row = {
        "cases": len(values),
        "mean_difference": sum(values) / len(values),
        "ci95": bootstrap_ci(values),
        "cases_higher": sum(value > 0 for value in values),
        "cases_lower": sum(value < 0 for value in values),
    }
    if alternative is not None:
        row["alternative"] = alternative
        row["p"] = sign_flip_p(differences, alternative)
    return row


def difference(
    cases: Sequence[Cells],
    a: tuple[str, str],
    b: tuple[str, str],
    outcome: Callable[[Run], bool] = operate,
) -> list[Fraction]:
    """Per case, share in cell ``a`` minus share in cell ``b``; cells are (cell, arm)."""

    return [share(runs(c, *a), outcome) - share(runs(c, *b), outcome) for c in cases]


def complete(by_case: dict[str, Cells], needed: Sequence[tuple[str, str]]) -> list[str]:
    return [uid for uid, cells in by_case.items() if all(runs(cells, *cell) for cell in needed)]


# --------------------------------------------------------------- analysis ---

NEUTRAL = ("main", "neutral")


def m10(by_case: dict[str, Cells]) -> dict:
    needed = [NEUTRAL] + [
        cell
        for style in STYLES
        for cell in (("main", style), ("chair_only", style), ("deputy_only", style))
    ]
    uids = complete(by_case, needed)
    cases = [by_case[uid] for uid in uids]
    if not cases:
        return {"cases": 0}
    tests, raw = {}, {}
    names = {
        ("chair_only", "autocratic"): "H8a",
        ("chair_only", "avoidant"): "H8b",
        ("deputy_only", "autocratic"): "H8c",
        ("deputy_only", "avoidant"): "H8d",
    }
    for (cell, style), name in names.items():
        alternative = DIRECTION[style] if cell == "chair_only" else "two-sided"
        tests[name] = {"cell": cell, "style": style, **describe(
            difference(cases, (cell, style), NEUTRAL), alternative
        )}
        raw[name] = tests[name]["p"]
    for name, value in holm(raw).items():
        tests[name]["p_holm"] = value

    decomposition = {}
    for style in STYLES:
        both = [float(v) for v in difference(cases, ("main", style), NEUTRAL)]
        chair = [float(v) for v in difference(cases, ("chair_only", style), NEUTRAL)]
        deputy = [float(v) for v in difference(cases, ("deputy_only", style), NEUTRAL)]
        interaction = [b - c - d for b, c, d in zip(both, chair, deputy)]
        decomposition[style] = {
            "both_effect": sum(both) / len(both),
            "both_ci95": bootstrap_ci(both),
            "chair_share": sum(chair) / sum(both) if sum(both) else None,
            "chair_share_ci95": ratio_ci(chair, both),
            "deputy_share": sum(deputy) / sum(both) if sum(both) else None,
            "deputy_share_ci95": ratio_ci(deputy, both),
            "interaction": sum(interaction) / len(interaction),
            "interaction_ci95": bootstrap_ci(interaction),
        }
    return {"cases": len(cases), "case_uids": uids, "tests": tests, "decomposition": decomposition}


def m11(by_case: dict[str, Cells]) -> dict:
    needed = [("alone", "neutral")] + [("alone", style) for style in STYLES]
    uids = complete(by_case, needed)
    cases = [by_case[uid] for uid in uids]
    if not cases:
        return {"cases": 0}
    tests, raw = {}, {}
    for style, name in zip(STYLES, ("H9a", "H9b")):
        tests[name] = {"style": style, **describe(
            difference(cases, ("alone", style), ("alone", "neutral")), DIRECTION[style]
        )}
        raw[name] = tests[name]["p"]
    for name, value in holm(raw).items():
        tests[name]["p_holm"] = value

    interaction = {}
    for style in STYLES:
        alone = difference(cases, ("alone", style), ("alone", "neutral"))
        rows = {}
        for against, cell in (("chair_only", "chair_only"), ("main", "main")):
            usable = [
                i for i, c in enumerate(cases) if runs(c, cell, style) and runs(c, *NEUTRAL)
            ]
            if not usable:
                rows[against] = {"cases": 0}
                continue
            negotiated = difference([cases[i] for i in usable], (cell, style), NEUTRAL)
            alone_here = [alone[i] for i in usable]
            rows[against] = {
                "cases": len(usable),
                "effect_with_negotiation": float(sum(negotiated) / len(negotiated)),
                "effect_alone": float(sum(alone_here) / len(alone_here)),
                "alone_minus_negotiated": describe(
                    [a - b for a, b in zip(alone_here, negotiated)], "two-sided"
                ),
                "retained": (
                    float(sum(alone_here) / sum(negotiated)) if sum(negotiated) else None
                ),
                "retained_ci95": ratio_ci(
                    [float(v) for v in alone_here], [float(v) for v in negotiated]
                ),
            }
        interaction[style] = rows
    neutral_shift = describe(difference(cases, ("alone", "neutral"), NEUTRAL))
    return {
        "cases": len(cases),
        "case_uids": uids,
        "tests": tests,
        "interaction": interaction,
        "neutral_alone_minus_neutral_negotiated": neutral_shift,
    }


def rates(by_case: dict[str, Cells]) -> dict:
    table = {}
    cells = [NEUTRAL, *(("main", style) for style in STYLES)] + [
        (cell, arm) for cell, (_, arms) in SERIES.items() for arm in arms
    ]
    for cell, arm in cells:
        pooled = [run for cells_ in by_case.values() for run in runs(cells_, cell, arm)]
        if not pooled:
            continue
        table[f"{cell}:{arm}"] = {
            "runs": len(pooled),
            "operate": float(share(pooled, operate)),
            "additional_workup": float(
                share(pooled, lambda run: run.decision == "additional_workup")
            ),
            "conservative": float(share(pooled, lambda run: run.decision == "conservative")),
            "defer_or_undecided": float(
                share(pooled, lambda run: run.decision in ("defer", "not_yet_decided"))
            ),
            "intended": float(share(pooled, lambda run: run.intended)),
            "indication_yes": float(share(pooled, lambda run: run.indication == "yes")),
            "deputy_disagreements_per_run": sum(run.deputy_disagreements for run in pooled)
            / len(pooled),
        }
    return table


#: Where each cell's reports are read for the exploratory process measures.
PROCESS_CELLS = {
    "main": ("", MAIN_ATTEMPTS, ("neutral", *STYLES)),
    **{cell: (folder, NEW_ATTEMPTS, arms) for cell, (folder, arms) in SERIES.items()},
}


def process(out: Path, manifest: dict, excluded: Sequence[str]) -> dict:
    """Exploratory, added before any run and tested on nothing: means per run.

    The deputy's first turn is read on its own because it comes before any
    ruling: in a chair-only arm it shows the deputy as the neutral arm's, and
    in a deputy-only arm it is the manipulation check of the deputy's prefix.
    """

    table = {}
    for cell, (folder, attempts, arms) in PROCESS_CELLS.items():
        for arm in arms:
            rows = []
            for item in manifest["cases"]:
                if item["patient_uid"] in excluded:
                    continue
                for attempt in attempts:
                    path = out / folder / f"pmc-{item['patient_uid']}" / (
                        f"arm-{arm}-attempt{attempt}.json"
                    )
                    if path.exists():
                        rows.append(json.loads(path.read_text(encoding="utf-8")))
            if not rows:
                continue
            first = [r["outputs"]["vice_opinions"][0] for r in rows if r["outputs"]["vice_opinions"]]
            metrics = [r["metrics"] for r in rows]

            def mean(values):
                values = list(values)
                return sum(values) / len(values) if values else None

            table[f"{cell}:{arm}"] = {
                "runs": len(rows),
                "negotiation_rounds": mean(m["negotiation_rounds"] for m in metrics),
                "tests_ordered": mean(m["tests_ordered_declared"] for m in metrics),
                # The main analysis's definition: the choice or the decision is
                # handed to the family; a detailed consent alone is not.
                "family_involvement": mean(
                    bool(FAMILY_TRANSFER & set(m["family_requests_declared"])) for m in metrics
                ),
                "contingency_plan": mean(m["contingency_plan_declared"] for m in metrics),
                "deputy_first_turn": {
                    "runs": len(first),
                    "preference_stated": mean(bool(o["preferred_option"]) for o in first),
                    "options_proposed": mean(len(o["proposed_options"]) for o in first),
                    "objections": mean(len(o["disagreement_points"]) for o in first),
                    "questions_to_director": mean(
                        len(o["deferred_questions_to_director"]) for o in first
                    ),
                },
            }
    return table


def analyse(out: Path, manifest: dict) -> dict:
    problems = validity(out, manifest)
    by_case, _ = cells_by_case(out, manifest)
    usable = {uid: cells for uid, cells in by_case.items() if uid not in problems}
    return {
        "excluded_for_validity": problems,
        "rates": rates(usable),
        "m10_chair_and_deputy": m10(usable),
        "m11_chair_alone": m11(usable),
        "exploratory_process": process(out, manifest, list(problems)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/test_sets/pmc_dcm_v1.json")
    )
    parser.add_argument("--write", type=Path, help="Write the result as JSON here.")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = analyse(args.out, manifest)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=float)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
