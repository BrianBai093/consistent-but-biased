"""Secondary endpoint 7: how the director handled the neurologist's red flags.

Plan v1.0 lists the endpoint; plan §10 fixes how it is measured (written before
any run was judged):

``judge``      One call per finished run: a judge that is not told the arm codes
               every flag as acted on, acknowledged, or not addressed. Results
               are cached per run, so a stopped job resumes without paying twice.
``crosscheck`` A second model codes a fixed random sample of runs; agreement is
               reported as Cohen's kappa. A judge that disagrees with another
               model as often as chance is not measuring anything.
``analyse``    Per case and arm, the share of decision-bearing flags
               (``changes_indication`` / ``changes_plan``) not addressed and the
               share acted on; neutral-referenced paired sign-flip tests with
               Holm, case bootstrap intervals. Main wording and trait-only
               wording separately; all flags, coexisting included, as exploratory.

The flags are part of the frozen baseline, so every arm and run of a case faced
the same list.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from agents.flag_handling_judge import FlagHandlingJudge
from config import Settings
from exceptions import ModelRetryExhaustedError
from services.llm_client import DeepSeekTextClient
from services.multimodal_client import OpenAIResponsesClient
from tools.analyze_text_experiment import bootstrap_ci, holm, sign_flip_p

ROOT = Path("runs")
CACHE = ROOT / "flag_judgements"
DECISION_BEARING = {"changes_indication", "changes_plan"}
CONTROL = "neutral"
#: (series directory, attempts) for each wording; the trait-only arms share
#: the main runs' neutral control (plan §9).
SOURCES = {"main": (ROOT, (1, 2, 3, 4)), "traits": (ROOT / "series-traits", (1, 2))}
#: Plan §10: the hypotheses, with the alternative fixed in advance.
CONTRASTS = (
    ("H7a", "autocratic", "not_addressed", "greater"),
    ("H7b", "avoidant", "acted", "greater"),
    ("H7c", "democratic", "not_addressed", "two-sided"),
)
CROSSCHECK_SAMPLE = 40
SEED = 20261013
_ARM = re.compile(r"arm-(\w+)-attempt(\d+)\.json$")


def runs(source: str) -> list[dict]:
    folder, attempts = SOURCES[source]
    found = []
    for path in sorted(folder.glob("pmc-*/arm-*-attempt*.json")):
        match = _ARM.search(path.name)
        if not match or int(match.group(2)) not in attempts:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        flags = [
            {
                "flag_id": f"F{index}",
                "finding": flag["finding"],
                "relevance": flag["relevance"],
            }
            for index, flag in enumerate(report["outputs"]["neurology"]["red_flags"], start=1)
        ]
        rulings = [
            {
                "round": number,
                "decision": ruling["decision"],
                "final_decision": ruling.get("final_decision"),
                "surgical_plan": ruling.get("surgical_plan"),
                "ordered_tests": ruling.get("ordered_tests", []),
                "delegated_tasks": ruling.get("delegated_tasks", []),
                "reason_for_delay": ruling.get("reason_for_delay"),
                "public_message": ruling["public_message"],
            }
            for number, ruling in enumerate(report["outputs"]["director_rulings"], start=1)
        ]
        found.append(
            {
                "key": f"{source}/{path.parent.name}/{match.group(1)}/attempt{match.group(2)}",
                "source": source,
                "case": path.parent.name,
                "arm": match.group(1),
                "flags": flags,
                "rulings": rulings,
            }
        )
    return found


def cache_path(key: str, judge: str) -> Path:
    return CACHE / judge / (key.replace("/", "__") + ".json")


async def judge_runs(items: Sequence[dict], client, judge_name: str, concurrency: int) -> int:
    agent = FlagHandlingJudge(client)
    gate = asyncio.Semaphore(concurrency)
    done = 0
    refused = asyncio.Event()

    async def one(item: dict) -> None:
        nonlocal done
        path = cache_path(item["key"], judge_name)
        if path.exists() or not item["flags"]:
            return
        async with gate:
            if refused.is_set():
                # The provider refused a request outright (an exhausted balance
                # answered 402 to 353 runs in a row). Stop sending; the cache
                # keeps what was judged and the next run resumes from it.
                raise RuntimeError("skipped: the provider refused an earlier request")
            try:
                result = await agent.run(flags=item["flags"], rulings=item["rulings"])
            except ModelRetryExhaustedError:
                refused.set()
                raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": item["key"],
                    "judge": judge_name,
                    "model": getattr(client, "model", None),
                    "judgements": [j.model_dump(mode="json") for j in result.output.judgements],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        done += 1

    results = await asyncio.gather(*(one(item) for item in items), return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    for failure in failures[:5]:
        print(f"judge failure: {type(failure).__name__}: {failure}", file=sys.stderr)
    print(f"{judge_name}: judged {done} run(s), {len(failures)} failure(s)")
    return len(failures)


def load_codes(items: Sequence[dict], judge_name: str) -> dict[str, dict[str, str]]:
    codes = {}
    for item in items:
        path = cache_path(item["key"], judge_name)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            codes[item["key"]] = {j["flag_id"]: j["handling"] for j in data["judgements"]}
    return codes


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float:
    labels = sorted({label for pair in pairs for label in pair})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def shares(item: dict, codes: dict[str, str], relevance: set[str] | None) -> dict | None:
    flags = [f for f in item["flags"] if relevance is None or f["relevance"] in relevance]
    if not flags or not codes:
        return None
    labels = [codes.get(f["flag_id"]) for f in flags]
    if any(label is None for label in labels):
        return None
    return {
        "not_addressed": Fraction(sum(label == "not_addressed" for label in labels), len(labels)),
        "acted": Fraction(sum(label == "acted" for label in labels), len(labels)),
    }


def analyse_source(items: Sequence[dict], codes: dict, relevance: set[str] | None) -> dict:
    cells: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        value = shares(item, codes.get(item["key"], {}), relevance)
        if value is not None:
            cells.setdefault((item["case"], item["arm"]), []).append(value)

    def mean(case: str, arm: str, measure: str) -> Fraction | None:
        values = cells.get((case, arm))
        if not values:
            return None
        return sum((v[measure] for v in values), Fraction(0)) / len(values)

    arms = sorted({arm for _, arm in cells})
    rates = {}
    for arm in arms:
        values = [v for (c, a), vs in cells.items() if a == arm for v in vs]
        rates[arm] = {
            "runs": len(values),
            "not_addressed": float(sum((v["not_addressed"] for v in values), Fraction(0)) / len(values)),
            "acted": float(sum((v["acted"] for v in values), Fraction(0)) / len(values)),
        }
    tests, raw = {}, {}
    for name, arm, measure, alternative in CONTRASTS:
        cases = sorted({c for c, _ in cells})
        differences = [
            mean(case, arm, measure) - mean(case, CONTROL, measure)
            for case in cases
            if mean(case, arm, measure) is not None and mean(case, CONTROL, measure) is not None
        ]
        if not differences:
            continue
        raw[name] = sign_flip_p(differences, alternative)
        tests[name] = {
            "arm": arm,
            "measure": measure,
            "alternative": alternative,
            "cases": len(differences),
            "mean_difference": float(sum(differences) / len(differences)),
            "ci95": bootstrap_ci([float(d) for d in differences]),
            "cases_higher": sum(d > 0 for d in differences),
            "cases_lower": sum(d < 0 for d in differences),
            "p": raw[name],
        }
    for name, value in holm(raw).items():
        tests[name]["p_holm"] = value
    return {"rates": rates, "tests": tests}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["judge", "crosscheck", "analyse"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    settings = Settings()
    main_items = runs("main")
    trait_items = [item for item in runs("traits") if item["arm"] != CONTROL]
    every = main_items + trait_items

    if args.command == "judge":
        client = DeepSeekTextClient(settings.require_deepseek(), thinking=settings.deepseek_thinking)
        return 1 if asyncio.run(judge_runs(every, client, "deepseek", args.concurrency)) else 0

    sample = random.Random(SEED).sample([i for i in every if i["flags"]], CROSSCHECK_SAMPLE)
    if args.command == "crosscheck":
        client = OpenAIResponsesClient(settings.require_openai())
        return 1 if asyncio.run(judge_runs(sample, client, "openai", args.concurrency)) else 0

    primary = load_codes(every, "deepseek")
    second = load_codes(sample, "openai")
    pairs = [
        (primary[key][flag], second[key][flag])
        for key in second
        if key in primary
        for flag in second[key]
        if flag in primary[key]
    ]
    traits_with_control = trait_items + [i for i in main_items if i["arm"] == CONTROL]
    result = {
        "runs_judged": len(primary),
        "runs_total": len([i for i in every if i["flags"]]),
        "agreement": {
            "flags": len(pairs),
            "raw": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
            "cohen_kappa": cohen_kappa(pairs) if pairs else None,
        },
        "decision_bearing": {
            "main": analyse_source(main_items, primary, DECISION_BEARING),
            "traits": analyse_source(traits_with_control, primary, DECISION_BEARING),
        },
        "all_flags_exploratory": {
            "main": analyse_source(main_items, primary, None),
            "traits": analyse_source(traits_with_control, primary, None),
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, default=float)
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
