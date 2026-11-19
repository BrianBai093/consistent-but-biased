"""Find the sentences of a case record that state the answer.

The only agent that reads the unredacted record. It returns sentence numbers,
never text, so everything it can do is visible in the diff a person signs. See
``services/record_redaction.py`` for why.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import Field, create_model

from agents.base_agent import BaseAgent
from schemas import AgentRole, StrictModel
from services.llm_client import ModelResult
from services.record_redaction import (
    RedactionCategory,
    RedactionPlan,
    Segment,
    units,
)


def _bounded_plan(unit_count: int) -> type[StrictModel]:
    """The plan schema with this record's sentence count as its upper bound.

    Validating against it raises the same kind of error as any other schema
    violation, so a number past the end is retried with feedback that names
    the field and the bound, instead of failing after the call.
    """

    deletion = create_model(
        "Deletion",
        __base__=StrictModel,
        unit=(int, Field(ge=1, le=max(unit_count, 1))),
        category=(RedactionCategory, ...),
        reason=(str, Field(min_length=1)),
    )
    return create_model(
        "RedactionPlan",
        __base__=StrictModel,
        deletions=(list[deletion], ...),
    )


class RecordRedactorAgent(BaseAgent[RedactionPlan]):
    role = AgentRole.RECORD_REDACTOR
    display_name = "Record Redactor"
    prompt_filename = "record_redactor.md"
    output_model = RedactionPlan

    def build_user_prompt(self, **kwargs: Any) -> str:
        raise NotImplementedError("the redactor builds its own prompt")

    async def run(self, *, segments: Sequence[Segment]) -> ModelResult[RedactionPlan]:
        numbered = units(segments)
        payload = json.dumps(
            {
                "task": (
                    "Return the numbers of the sentences that state the decision or "
                    "anything after it. Never remove a preoperative finding."
                ),
                "sentences": [
                    {"n": item.number, "page": item.page_id, "text": item.text}
                    for item in numbered
                ],
            },
            ensure_ascii=False,
        )
        bounded = _bounded_plan(len(numbered))

        def within_the_record(plan: RedactionPlan) -> RedactionPlan:
            bounded.model_validate(plan.model_dump(mode="json"))
            return plan

        return await self._invoke(
            user_prompt=payload,
            images=(),
            output_processor=within_the_record,
        )
