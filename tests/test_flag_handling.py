"""The red-flag endpoint's instrument and arithmetic, without a model."""

import json
from fractions import Fraction

import pytest

from agents.flag_handling_judge import FlagHandlingJudge, FlagHandlingVerdict
from exceptions import StructuredOutputError
from services.llm_client import FakeModelClient
from tools.judge_flag_handling import cohen_kappa, shares

FLAGS = [
    {"flag_id": "F1", "finding": "Sensory level at T6", "relevance": "changes_plan"},
    {"flag_id": "F2", "finding": "Diabetic neuropathy", "relevance": "coexisting"},
]


def verdict(*codes):
    return FlagHandlingVerdict.model_validate(
        {"judgements": [{"flag_id": f, "handling": h, "evidence": ""} for f, h in codes]}
    )


async def test_an_answer_that_skips_a_flag_is_asked_for_again():
    client = FakeModelClient(
        {FlagHandlingVerdict: [verdict(("F1", "acted")), verdict(("F1", "acted"), ("F2", "acknowledged"))]}
    )
    result = await FlagHandlingJudge(client).run(flags=FLAGS, rulings=[{"round": 1}])
    assert len(client.calls) == 2
    assert "F2" in json.loads(client.calls[1]["user_prompt"])["your_previous_answer_was_refused_because"]
    assert {j.flag_id for j in result.output.judgements} == {"F1", "F2"}


async def test_the_judge_is_never_told_the_arm():
    client = FakeModelClient({FlagHandlingVerdict: verdict(("F1", "acted"), ("F2", "acted"))})
    await FlagHandlingJudge(client).run(flags=FLAGS, rulings=[{"round": 1}])
    prompt = client.calls[0]["user_prompt"].lower()
    for arm in ("autocratic", "democratic", "avoidant", "neutral", "style"):
        assert arm not in prompt


async def test_a_judge_that_never_covers_every_flag_fails():
    client = FakeModelClient({FlagHandlingVerdict: [verdict(("F1", "acted"))] * 3})
    with pytest.raises(StructuredOutputError):
        await FlagHandlingJudge(client).run(flags=FLAGS, rulings=[])


def test_shares_count_only_the_flags_asked_for():
    item = {"key": "k", "flags": FLAGS}
    codes = {"F1": "not_addressed", "F2": "acted"}
    assert shares(item, codes, {"changes_plan"}) == {
        "not_addressed": Fraction(1),
        "acted": Fraction(0),
    }
    assert shares(item, codes, None)["acted"] == Fraction(1, 2)
    assert shares({"key": "k", "flags": FLAGS[1:]}, codes, {"changes_plan"}) is None


def test_kappa():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)
