"""Spine surgeon agent."""

import json
from typing import Sequence

from pydantic import BaseModel

from agents.base_agent import (
    BaseAgent,
    chat_context,
    document_context,
    report_context,
    structured_context,
)
from schemas import (
    AgentRole,
    ChatMessage,
    OperativePlanDraft,
    RadiologyReport,
    SpineSurgeonOutput,
)
from services.plan_convergence import render_plan_draft


class SpineSurgeonAgent(BaseAgent[SpineSurgeonOutput]):
    role = AgentRole.SPINE_SURGEON
    display_name = "Attending Spine Surgeon"
    prompt_filename = "spine_surgeon.md"
    output_model = SpineSurgeonOutput

    def build_user_prompt(self, *, case_preop_md: str, tests_preop_md: str,
                          radiology_preop_md: str, radiology_report: RadiologyReport | None,
                          chat_log: Sequence[ChatMessage], question: str,
                          prior_contribution: BaseModel | None = None,
                          working_plan: OperativePlanDraft | None = None) -> str:
        return json.dumps({
            "task": "Assess surgical indication and formulate a spine surgery strategy.",
            "source_documents": document_context(case_preop_md=case_preop_md,
                                                   tests_preop_md=tests_preop_md,
                                                   radiology_preop_md=radiology_preop_md),
            "structured_radiology_report": report_context(radiology_report),
            "relevant_public_context": chat_context(chat_log),
            "prior_structured_contribution": structured_context(prior_contribution),
            "shared_operative_working_plan": render_plan_draft(working_plan),
            "moderator_question": question,
        }, ensure_ascii=False)
