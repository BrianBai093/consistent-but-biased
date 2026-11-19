"""Descriptive process measures per arm, recomputed from the arm reports on disk."""
import json, sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/"data/test_sets/pmc_dcm_v1.json").read_text())
ARMS = ("neutral","democratic","autocratic","avoidant")
FAMILY = {"family_chooses","decide_after_family_meeting"}

def load(out, attempts, arms=ARMS):
    runs = {a: {} for a in arms}   # arm -> uid -> [metrics by attempt order]
    for item in manifest["cases"]:
        uid = item["patient_uid"]
        for att in attempts:
            for a in arms:
                p = out/f"pmc-{uid}"/f"arm-{a}-attempt{att}.json"
                if p.exists():
                    report = json.loads(p.read_text())
                    m = dict(report["metrics"])
                    opinions = report["outputs"].get("vice_opinions") or []
                    # dissent in the deputy's first turn, before any ruling by the director
                    m["first_turn_disagreements"] = len(opinions[0]["disagreement_points"]) if opinions else 0
                    runs[a].setdefault(uid, []).append((att, m))
    return runs

def summarise(runs):
    res = {}
    for a, by_case in runs.items():
        ms = [m for lst in by_case.values() for _, m in lst]
        if not ms: continue
        rounds = [m["negotiation_rounds"] for m in ms]
        two = [lst for lst in by_case.values() if len(lst) >= 2]
        res[a] = {
            "runs": len(ms),
            "cases": len(by_case),
            "mean_rounds": round(mean(rounds), 3),
            "share_ruled_round1": round(sum(r == 1 for r in rounds)/len(rounds), 3),
            "mean_tests_ordered": round(mean(m.get("tests_ordered_declared") or 0 for m in ms), 3),
            "mean_roles_addressed": round(mean(len(m.get("roles_addressed_declared") or []) for m in ms), 3),
            "share_contingency_plan": round(mean(bool(m.get("contingency_plan_declared")) for m in ms), 3),
            "share_family_transfer": round(mean(bool(FAMILY & set(m.get("family_requests_declared") or [])) for m in ms), 3),
            "mean_deputy_disagreements": round(mean(int(m.get("deputy_disagreements") or 0) for m in ms), 3),
            "mean_deputy_rounds": round(mean(m["deputy_rounds"] for m in ms), 3),
            "mean_first_turn_disagreements": round(mean(m["first_turn_disagreements"] for m in ms), 3),
            "first_two_runs_disagree": round(mean(lst[0][1]["decision"] != lst[1][1]["decision"] for lst in two), 3),
            "first_two_runs_cases": len(two),
            "all_runs_agree": round(mean(len({m["decision"] for _, m in lst}) == 1 for lst in by_case.values()), 3),
        }
    return res

def sign_flip_p(diffs):
    """Exact two-sided sign-flip test over the case-level differences."""
    sums = {0.0: 1}
    for d in (d for d in diffs if d != 0):
        new = {}
        for s, c in sums.items():
            for v in (round(s + d, 9), round(s - d, 9)):
                new[v] = new.get(v, 0) + c
        sums = new
    obs = abs(sum(diffs))
    return sum(c for s, c in sums.items() if abs(s) >= obs - 1e-9) / sum(sums.values())

def first_turn_vs_neutral(runs, neutral):
    base = {u: mean(m["first_turn_disagreements"] for _, m in lst) for u, lst in neutral.items()}
    res = {}
    for a, by_case in runs.items():
        if a == "neutral" or not by_case: continue
        diffs = [mean(m["first_turn_disagreements"] for _, m in lst) - base[u] for u, lst in by_case.items()]
        res[a] = {"mean_difference": round(mean(diffs), 3), "cases_higher": sum(d > 0 for d in diffs),
                  "cases_lower": sum(d < 0 for d in diffs), "p_two_sided_exact": round(sign_flip_p(diffs), 4)}
    return res

main = load(ROOT/"runs", [1,2,3,4])
traits = load(ROOT/"runs/series-traits", [1,2], arms=("democratic","autocratic","avoidant"))
out = {"note": "Descriptive process measures recomputed from arm reports (metrics block). Main: attempts 1-4 (attempt 3 for 6 cases, attempt 4 for 14). Traits: series-traits attempts 1-2; its control is the main neutral arm.",
       "main": summarise(main), "traits": summarise(traits),
       "first_turn_dissent_vs_neutral_exploratory": {"main": first_turn_vs_neutral(main, main["neutral"]),
                                                      "traits": first_turn_vs_neutral(traits, main["neutral"])}}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else ROOT/"analysis/process_measures.json", "w"), ensure_ascii=False, indent=1)
for k in ("main","traits","first_turn_dissent_vs_neutral_exploratory"):
    print("==", k)
    for a, v in out[k].items(): print(" ", a, v)
