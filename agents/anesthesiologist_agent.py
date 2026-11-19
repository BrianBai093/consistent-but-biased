"""Anesthesiology agent."""

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
    AnesthesiologistOutput,
    ChatMessage,
    OperativePlanDraft,
    RadiologyReport,
)
from services.plan_convergence import render_plan_draft


class AnesthesiologistAgent(BaseAgent[AnesthesiologistOutput]):
    role = AgentRole.ANESTHESIOLOGIST
    display_name = "Anesthesiologist"
    prompt_filename = "anesthesiologist.md"
    output_model = AnesthesiologistOutput

    def build_user_prompt(self, *, case_preop_md: str, tests_preop_md: str,
                          radiology_preop_md: str, radiology_report: RadiologyReport | None,
                          chat_log: Sequence[ChatMessage], question: str,
                          prior_contribution: BaseModel | None = None,
                          working_plan: OperativePlanDraft | None = None) -> str:
        return json.dumps({
            "task": "Assess anesthesia readiness and peri-anesthetic risk for spine surgery.",
            "source_documents": document_context(case_preop_md=case_preop_md,
                                                   tests_preop_md=tests_preop_md,
                                                   radiology_preop_md=radiology_preop_md),
            "structured_radiology_report": report_context(radiology_report),
            "relevant_public_context": chat_context(chat_log),
            "prior_structured_contribution": structured_context(prior_contribution),
            "shared_operative_working_plan": render_plan_draft(working_plan),
            "moderator_question": question,
        }, ensure_ascii=False)
