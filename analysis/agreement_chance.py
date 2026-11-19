"""Chance-corrected agreement across repeated runs, recomputed from the arm reports on disk.

Exploratory: not in the analysis plan frozen before the repeated runs (v1.0). It asks
whether the higher full agreement of the autocratic and avoidant arms (H5) exceeds what
their skewed ruling distributions would produce by chance.

Per arm, with the 20 cases as subjects and the runs of a cell as raters:
- the share of cases whose runs all gave the same ruling (H5, recomputed);
- the full agreement expected if every run drew its ruling at random with the arm's own
  ruling shares (sum of p_k ** n over the ruling categories, n runs per cell);
- Fleiss' kappa over the five ruling categories, with a case-level bootstrap 95% interval
  (resamples where every sampled ruling is the same leave kappa undefined and are dropped
  and counted), and the bootstrap interval of each arm's kappa minus the neutral arm's,
  resampling the same cases for both;
- the Spearman correlation between each arm's per-case share of rulings to operate now and
  the neutral arm's, with a permutation p value.

Main runs: attempts 1-4 (attempt 3 for 6 cases, attempt 4 for 14), three per cell.
Trait-only runs: series-traits attempts 1-2; for them the neutral arm is cut to its first
two runs so that both have two raters per case.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/test_sets/pmc_dcm_v1.json").read_text())
CATEGORIES = ("operate", "additional_workup", "conservative", "defer", "not_yet_decided")
RESAMPLES = 10_000
SEED = 20261012  # the seed of the main analysis
PERMUTATIONS = 100_000


def load(out, attempts, arms):
    runs = {arm: {} for arm in arms}
    for item in MANIFEST["cases"]:
        uid = item["patient_uid"]
        for attempt in attempts:
            for arm in arms:
                path = out / f"pmc-{uid}" / f"arm-{arm}-attempt{attempt}.json"
                if path.exists():
                    decision = json.loads(path.read_text())["metrics"]["decision"]
                    runs[arm].setdefault(uid, []).append(decision)
    return runs


def counts(cells):
    return [[cell.count(c) for c in CATEGORIES] for cell in cells]


def shares(cells):
    total = sum(len(cell) for cell in cells)
    return [sum(cell.count(c) for cell in cells) / total for c in CATEGORIES]


def fleiss_kappa(cells):
    """None when every rating falls in one category (expected agreement of 1)."""
    n = len(cells[0])
    assert all(len(cell) == n for cell in cells)
    p = shares(cells)
    p_e = sum(x * x for x in p)
    if p_e >= 1 - 1e-12:
        return None
    p_i = [sum(c * (c - 1) for c in row) / (n * (n - 1)) for row in counts(cells)]
    p_bar = sum(p_i) / len(p_i)
    return (p_bar - p_e) / (1 - p_e)


def full_agreement(cells):
    return sum(len(set(cell)) == 1 for cell in cells) / len(cells)


def chance_full_agreement(cells):
    n = len(cells[0])
    return sum(x ** n for x in shares(cells))


def percentile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1  # average rank of a tie
        i = j + 1
    return result


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def spearman(x, y, rng):
    rx, ry = ranks(x), ranks(y)
    rho = pearson(rx, ry)
    if rho is None:
        return None
    ry = list(ry)
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(ry)
        if abs(pearson(rx, ry)) >= abs(rho) - 1e-12:
            hits += 1
    return {"rho": rho, "p_two_sided_permutation": (hits + 1) / (PERMUTATIONS + 1)}


def analyse(runs, arms, control, n_runs):
    uids = [item["patient_uid"] for item in MANIFEST["cases"]]
    cells = {arm: [runs[arm][u][:n_runs] for u in uids] for arm in arms}
    for arm in arms:
        assert all(len(cell) == n_runs for cell in cells[arm]), arm
    rng = random.Random(SEED)
    boot = {arm: [] for arm in arms}
    boot_diff = {arm: [] for arm in arms if arm != control}
    undefined = {arm: 0 for arm in arms}
    for _ in range(RESAMPLES):
        idx = [rng.randrange(len(uids)) for _ in uids]
        k = {arm: fleiss_kappa([cells[arm][i] for i in idx]) for arm in arms}
        for arm in arms:
            if k[arm] is None:
                undefined[arm] += 1
            else:
                boot[arm].append(k[arm])
        for arm in boot_diff:
            if k[arm] is not None and k[control] is not None:
                boot_diff[arm].append(k[arm] - k[control])
    result = {}
    srng = random.Random(SEED)
    for arm in arms:
        operate = [cell.count("operate") / n_runs for cell in cells[arm]]
        neutral_operate = [cell.count("operate") / n_runs for cell in cells[control]]
        row = {
            "runs_per_case": n_runs,
            "ruling_shares": dict(zip(CATEGORIES, shares(cells[arm]))),
            "full_agreement_observed": full_agreement(cells[arm]),
            "full_agreement_expected_by_chance": chance_full_agreement(cells[arm]),
            "fleiss_kappa": fleiss_kappa(cells[arm]),
            "kappa_ci95": [percentile(boot[arm], 0.025), percentile(boot[arm], 0.975)],
            "bootstrap_resamples_kappa_undefined": undefined[arm],
        }
        if arm != control:
            row["kappa_minus_neutral"] = row["fleiss_kappa"] - fleiss_kappa(cells[control])
            row["kappa_minus_neutral_ci95"] = [
                percentile(boot_diff[arm], 0.025),
                percentile(boot_diff[arm], 0.975),
            ]
            row["bootstrap_resamples_difference_used"] = len(boot_diff[arm])
            row["spearman_operate_share_vs_neutral"] = spearman(neutral_operate, operate, srng)
        result[arm] = row
    return result


main_arms = ("neutral", "democratic", "autocratic", "avoidant")
trait_arms = ("democratic", "autocratic", "avoidant")
main = load(ROOT / "runs", [1, 2, 3, 4], main_arms)
traits = load(ROOT / "runs/series-traits", [1, 2], trait_arms)
traits["neutral"] = main["neutral"]  # the trait-only control, cut to its first two runs below
out = {
    "note": __doc__.strip().splitlines()[0]
    + " Exploratory; not in the frozen plan. Seed and resamples as in the main analysis.",
    "resamples": RESAMPLES,
    "seed": SEED,
    "main_three_runs": analyse(main, main_arms, "neutral", 3),
    "traits_two_runs": analyse(traits, ("neutral",) + trait_arms, "neutral", 2),
}
target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis/agreement_chance.json"
target.write_text(json.dumps(out, ensure_ascii=False, indent=1))
for series in ("main_three_runs", "traits_two_runs"):
    print("==", series)
    for arm, row in out[series].items():
        print(" ", arm, json.dumps(row))
