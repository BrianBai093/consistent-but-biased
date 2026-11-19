"""Return the preoperative part of a published case report, by deletion only.

The only agent that reads a published case unfiltered. Unlike the hospital
redactor it returns text, because a case report's sentences mix the
presentation with the decision; what keeps that honest is
``services/case_report_filter.check_deletion_only``, which every answer must
pass. A refused answer is asked for again with the reason, a bounded number of
times.
"""

from __future__ import annotations

import json
from typing import Any

from agents.base_agent import BaseAgent
from exceptions import StructuredOutputError
from schemas import AgentRole
from services.case_report_filter import (
    FilteredCaseReport,
    FilterViolation,
    RemovedPassage,
    check_deletion_only,
)
from services.llm_client import ModelResult, aggregate_usage

#: Answers refused by the deletion check before the case is given up on.
MAX_FILTER_ATTEMPTS = 3


class CaseReportFilterAgent(BaseAgent[FilteredCaseReport]):
    role = AgentRole.RECORD_REDACTOR
    display_name = "Case Report Filter"
    prompt_filename = "case_report_filter.md"
    output_model = FilteredCaseReport

    def build_user_prompt(self, **kwargs: Any) -> str:
        raise NotImplementedError("the filter builds its own prompt")

    async def run(
        self, *, record: str
    ) -> tuple[ModelResult[FilteredCaseReport], list[RemovedPassage]]:
        """The accepted answer and the passages it removed, as the check found them."""

        usages = []
        feedback: str | None = None
        for _ in range(MAX_FILTER_ATTEMPTS):
            payload = {
                "task": "Delete everything not known before the treatment decision.",
                "record": record,
            }
            if feedback:
                payload["your_previous_answer_was_refused_because"] = feedback
            result = await self._invoke(
                user_prompt=json.dumps(payload, ensure_ascii=False),
                images=(),
            )
            usages.extend(result.all_usages)
            try:
                passages = check_deletion_only(record, result.output.preoperative_text)
            except FilterViolation as exc:
                feedback = str(exc)
                continue
            return (
                ModelResult(
                    output=result.output,
                    usage=aggregate_usage(usages),
                    usages=tuple(usages),
                    attempt_count=len(usages),
                ),
                passages,
            )
        raise StructuredOutputError(
            f"The case report filter was refused {MAX_FILTER_ATTEMPTS} times; "
            f"last reason: {feedback}"
        )
