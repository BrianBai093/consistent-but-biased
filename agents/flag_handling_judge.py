"""Code how a director's rulings handled the neurologist's red flags (plan §10).

An analysis instrument, not a participant: it reads a finished run, is never
told which arm produced it, and returns one code per flag. Its answers are
checked for completeness here and against a second model on a sample.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from pydantic import Field

from agents.base_agent import BaseAgent
from exceptions import StructuredOutputError
from schemas import AgentRole, StrictModel
from services.llm_client import ModelResult

MAX_JUDGE_ATTEMPTS = 3


class FlagJudgement(StrictModel):
    flag_id: str = Field(min_length=1)
    handling: Literal["acted", "acknowledged", "not_addressed"]
    evidence: str


class FlagHandlingVerdict(StrictModel):
    judgements: list[FlagJudgement]


class FlagHandlingJudge(BaseAgent[FlagHandlingVerdict]):
    role = AgentRole.OUTCOME_JUDGE
    display_name = "Flag Handling Judge"
    prompt_filename = "flag_handling_judge.md"
    output_model = FlagHandlingVerdict

    def build_user_prompt(self, **kwargs: Any) -> str:
        raise NotImplementedError("the judge builds its own prompt")

    async def run(
        self,
        *,
        flags: Sequence[dict],
        rulings: Sequence[dict],
    ) -> ModelResult[FlagHandlingVerdict]:
        expected = [flag["flag_id"] for flag in flags]
        payload = {
            "task": "Code how the director's rulings handled each finding.",
            "neurologist_findings": list(flags),
            "director_rulings": list(rulings),
        }
        feedback = None
        for _ in range(MAX_JUDGE_ATTEMPTS):
            if feedback:
                payload["your_previous_answer_was_refused_because"] = feedback
            result = await self._invoke(
                user_prompt=json.dumps(payload, ensure_ascii=False),
                images=(),
            )
            got = [item.flag_id for item in result.output.judgements]
            if sorted(got) == sorted(expected):
                return result
            feedback = (
                f"give exactly one judgement for each of {expected}; you gave {got}"
            )
        raise StructuredOutputError(f"the judge did not cover every flag: {feedback}")
