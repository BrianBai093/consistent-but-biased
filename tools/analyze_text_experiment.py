"""The analysis the experiment plan fixes in advance (docs/experiment-plan.zh-CN.md, v1.0).

Reads the arm reports already on disk and computes, with the standard library
only:

- **Primary (H1-H3)**: per case and arm, the share of runs that ruled
  ``operate``; per case, the difference from the neutral arm; an exact
  sign-flip permutation test over the cases (one-sided for H1 and H2,
  two-sided for H3), Holm-adjusted; a case-level bootstrap interval for the
  mean difference.
- **H4**: the same differences for "surgery intended", set beside those for
  "operate".
- **H5**: in each arm, the share of case cells whose runs all agree.
- **Sensitivity**: (0) whether a mixed logistic model is estimable at all --
  complete separation is reported as such, as the plan requires; (1) exact
  McNemar on each cell's majority ruling; (3) the primary test again without
  the cases whose filtered record kept unreviewed keyword hits.
- **Secondary endpoints**: neutral-referenced differences, Benjamini-Hochberg.

Nothing here is tuned to a result. A number that disagrees with an
expectation is reported, not adjusted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Callable, Sequence

ARMS = ("neutral", "democratic", "autocratic", "avoidant")
CONTROL = "neutral"
#: H1-H3, with the alternative the plan fixed in advance.
CONTRASTS = (
    ("H1", "autocratic", "greater"),
    ("H2", "avoidant", "less"),
    ("H3", "democratic", "two-sided"),
)
FAMILY_TRANSFER = {"family_chooses", "decide_after_family_meeting"}
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261012


@dataclass
class Run:
    decision: str
    intended: bool
    indication: str
    family: list[str]
    deputy_disagreements: int
    consensus: bool | None


@dataclass
class Case:
    uid: str
    group: str
    real_decision: str
    unreviewed_hits: int
    runs: dict[str, list[Run]] = field(default_factory=dict)


def load(out: Path, manifest: dict, attempts: Sequence[int]) -> list[Case]:
    cases = []
    for item in manifest["cases"]:
        uid = item["patient_uid"]
        folder = out / f"pmc-{uid}"
        case = Case(uid, item["group"], item["real_decision"], 0)
        for attempt in attempts:
            summary = folder / f"summary-attempt{attempt}.json"
            if summary.exists():
                data = json.loads(summary.read_text(encoding="utf-8"))
                case.unreviewed_hits = len(data["filter"]["unreviewed_screen_hits"])
            for arm in ARMS:
                path = folder / f"arm-{arm}-attempt{attempt}.json"
                if not path.exists():
                    continue
                metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                case.runs.setdefault(arm, []).append(
                    Run(
                        decision=metrics["decision"],
                        intended=bool(metrics["surgery_intended"]),
                        indication=metrics["surgical_indication"],
                        family=list(metrics.get("family_requests_declared") or []),
                        deputy_disagreements=int(metrics.get("deputy_disagreements") or 0),
                        consensus=metrics.get("consensus_reached"),
                    )
                )
        cases.append(case)
    return cases


def share(runs: Sequence[Run], outcome: Callable[[Run], bool]) -> Fraction:
    return Fraction(sum(outcome(run) for run in runs), len(runs))


# ------------------------------------------------------------- statistics ---


def sign_flip_p(differences: Sequence[Fraction], alternative: str) -> float:
    """Exact p of the summed differences under random sign flips.

    The differences are shares of a few runs, so on a common denominator they
    are integers and the null distribution of their sum is built exactly by
    convolution rather than by enumerating 2^n sign patterns.
    """

    denominator = 1
    for value in differences:
        denominator = denominator * value.denominator // _gcd(denominator, value.denominator)
    scaled = [int(value * denominator) for value in differences]
    observed = sum(scaled)
    distribution = {0: 1}
    for value in scaled:
        step: dict[int, int] = {}
        for total, count in distribution.items():
            for signed in (value, -value):
                step[total + signed] = step.get(total + signed, 0) + count
        distribution = step
    total = sum(distribution.values())
    if alternative == "greater":
        hits = sum(count for value, count in distribution.items() if value >= observed)
    elif alternative == "less":
        hits = sum(count for value, count in distribution.items() if value <= observed)
    else:
        hits = sum(count for value, count in distribution.items() if abs(value) >= abs(observed))
    return hits / total


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def bootstrap_ci(values: Sequence[float], *, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[key] = running
    return adjusted


def benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1], reverse=True)
    adjusted, running, m = {}, 1.0, len(ordered)
    for index, (key, value) in enumerate(ordered):
        rank = m - index
        running = min(running, value * m / rank)
        adjusted[key] = running
    return adjusted


def exact_mcnemar(only_style: int, only_control: int) -> float:
    n = only_style + only_control
    if n == 0:
        return 1.0
    k = min(only_style, only_control)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


# --------------------------------------------------------------- analysis ---


def complete(cases: Sequence[Case]) -> list[Case]:
    """Cases with at least one run in every arm; the paired analysis needs all four."""

    return [case for case in cases if all(case.runs.get(arm) for arm in ARMS)]


def paired(
    cases: Sequence[Case],
    arm: str,
    outcome: Callable[[Run], bool],
) -> list[Fraction]:
    return [
        share(case.runs[arm], outcome) - share(case.runs[CONTROL], outcome)
        for case in cases
    ]


def primary(cases: Sequence[Case]) -> dict:
    rows, raw = {}, {}
    for name, arm, alternative in CONTRASTS:
        differences = paired(cases, arm, lambda run: run.decision == "operate")
        raw[name] = sign_flip_p(differences, alternative)
        mean = float(sum(differences) / len(differences))
        rows[name] = {
            "arm": arm,
            "alternative": alternative,
            "mean_difference": mean,
            "ci95": bootstrap_ci([float(value) for value in differences]),
            "p": raw[name],
            "cases_higher": sum(value > 0 for value in differences),
            "cases_lower": sum(value < 0 for value in differences),
        }
    for name, value in holm(raw).items():
        rows[name]["p_holm"] = value
    return rows


def rates(cases: Sequence[Case]) -> dict:
    table = {}
    for arm in ARMS:
        runs = [run for case in cases for run in case.runs.get(arm, [])]
        if not runs:
            continue
        table[arm] = {
            "runs": len(runs),
            "operate": float(share(runs, lambda run: run.decision == "operate")),
            "additional_workup": float(
                share(runs, lambda run: run.decision == "additional_workup")
            ),
            "conservative": float(share(runs, lambda run: run.decision == "conservative")),
            "intended": float(share(runs, lambda run: run.intended)),
            "indication_yes": float(share(runs, lambda run: run.indication == "yes")),
        }
    return table


def intent_vs_timing(cases: Sequence[Case]) -> dict:
    """H4: does a style move "operate" more than it moves "intends to operate"?"""

    result = {}
    for _, arm, _ in CONTRASTS:
        operate = [float(v) for v in paired(cases, arm, lambda run: run.decision == "operate")]
        intended = [float(v) for v in paired(cases, arm, lambda run: run.intended)]
        result[arm] = {
            "operate_difference": sum(operate) / len(operate),
            "intended_difference": sum(intended) / len(intended),
            "gap_ci95": bootstrap_ci(
                [abs(a) - abs(b) for a, b in zip(operate, intended)]
            ),
        }
    return result


def stability(cases: Sequence[Case]) -> dict:
    """H5: the share of cells whose runs all gave the same ruling."""

    result = {}
    for arm in ARMS:
        cells = [case.runs[arm] for case in cases if len(case.runs.get(arm, [])) >= 2]
        if not cells:
            result[arm] = None
            continue
        agree = sum(len({run.decision for run in cell}) == 1 for cell in cells)
        result[arm] = {"cells": len(cells), "all_runs_agree": agree / len(cells)}
    return result


def majority_mcnemar(cases: Sequence[Case]) -> dict:
    def majority(runs: Sequence[Run]) -> bool:
        return sum(run.decision == "operate" for run in runs) * 2 > len(runs)

    result = {}
    for name, arm, _ in CONTRASTS:
        only_style = sum(
            majority(case.runs[arm]) and not majority(case.runs[CONTROL]) for case in cases
        )
        only_control = sum(
            majority(case.runs[CONTROL]) and not majority(case.runs[arm]) for case in cases
        )
        result[name] = {
            "only_style_operates": only_style,
            "only_control_operates": only_control,
            "p": exact_mcnemar(only_style, only_control),
        }
    return result


def separation(cases: Sequence[Case]) -> dict:
    """Whether a logistic model of "operate" on arm can be estimated at all."""

    table = rates(cases)
    separated = [
        arm for arm, row in table.items() if row["operate"] in (0.0, 1.0)
    ]
    return {
        "estimable": not separated,
        "separated_arms": separated,
        "note": (
            "complete separation: the mixed logistic model is not estimable and, "
            "as the plan requires, is reported as such"
            if separated
            else "no separation; fit the model with a statistics package"
        ),
    }


SECONDARY: dict[str, Callable[[Run], float]] = {
    "indication_yes": lambda run: run.indication == "yes",
    "additional_workup": lambda run: run.decision == "additional_workup",
    "family_transfer": lambda run: bool(FAMILY_TRANSFER & set(run.family)),
    "detailed_consent": lambda run: "detailed_consent" in run.family,
    "deputy_disagreements": lambda run: run.deputy_disagreements,
}


def secondary(cases: Sequence[Case]) -> dict:
    rows, raw = {}, {}
    for endpoint, outcome in SECONDARY.items():
        for _, arm, _ in CONTRASTS:
            differences = [
                Fraction(sum(outcome(run) for run in case.runs[arm]), len(case.runs[arm]))
                - Fraction(
                    sum(outcome(run) for run in case.runs[CONTROL]),
                    len(case.runs[CONTROL]),
                )
                for case in cases
            ]
            key = f"{endpoint}:{arm}"
            raw[key] = sign_flip_p(differences, "two-sided")
            rows[key] = {"mean_difference": float(sum(differences) / len(differences))}
    for key, value in benjamini_hochberg(raw).items():
        rows[key]["p"] = raw[key]
        rows[key]["p_bh"] = value
    return rows


def analyse(cases: Sequence[Case]) -> dict:
    usable = complete(cases)
    clean = [case for case in usable if case.unreviewed_hits == 0]
    return {
        "cases_total": len(cases),
        "cases_complete": len(usable),
        "runs_per_arm": {
            arm: sum(len(case.runs.get(arm, [])) for case in usable) for arm in ARMS
        },
        "rates": rates(usable),
        "primary": primary(usable),
        "h4_intent_vs_timing": intent_vs_timing(usable),
        "h5_stability": stability(usable),
        "sensitivity": {
            "mixed_model": separation(usable),
            "majority_mcnemar": majority_mcnemar(usable),
            "without_unreviewed_hits": {
                "cases": len(clean),
                "primary": primary(clean) if len(clean) >= 2 else None,
            },
        },
        "secondary": secondary(usable),
        "by_group": {
            group: rates([case for case in usable if case.group == group])
            for group in sorted({case.group for case in usable})
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/test_sets/pmc_dcm_v1.json")
    )
    parser.add_argument("--attempts", default="1,2,3")
    parser.add_argument("--only", action="append", help="Restrict to these patient_uids.")
    parser.add_argument("--write", type=Path, help="Write the result as JSON here.")
    parser.add_argument(
        "--control-out",
        type=Path,
        help=(
            "Take the neutral arm from this directory instead (with --control-attempts). "
            "For a series whose neutral prefix is byte-identical to the main runs', "
            "so the main runs already are its control (M9)."
        ),
    )
    parser.add_argument("--control-attempts", default="1,2,3,4")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.only:
        manifest = {**manifest, "cases": [c for c in manifest["cases"] if c["patient_uid"] in args.only]}
    attempts = [int(item) for item in args.attempts.split(",")]
    cases = load(args.out, manifest, attempts)
    if args.control_out:
        control_attempts = [int(item) for item in args.control_attempts.split(",")]
        controls = {
            case.uid: case.runs.get(CONTROL, [])
            for case in load(args.control_out, manifest, control_attempts)
        }
        for case in cases:
            case.runs[CONTROL] = controls.get(case.uid, [])
    result = analyse(cases)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=float)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
