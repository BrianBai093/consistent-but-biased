"""MDT moderator agent."""

import json
from typing import Sequence

from agents.base_agent import BaseAgent, chat_context
from schemas import (
    AgentRole,
    ChatMessage,
    Disagreement,
    ModeratorDecision,
    OperativePlanDraft,
    PendingRequest,
    RadiologyReport,
)
from services.context_policy import compact_routing_chat
from services.llm_client import ModelResult
from services.plan_convergence import compute_field_matrix


class ModeratorAgent(BaseAgent[ModeratorDecision]):
    role = AgentRole.MODERATOR
    display_name = "MDT Moderator"
    prompt_filename = "moderator.md"
    output_model = ModeratorDecision

    def build_user_prompt(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport | None,
        chat_log: Sequence[ChatMessage],
        question: str,
        prior_contribution=None,
        working_plan: OperativePlanDraft | None = None,
    ) -> str:
        return json.dumps(
            {
                "task": "Select the next MDT action without performing specialist analysis.",
                "radiology_status": (
                    {
                        "image_quality": radiology_report.image_quality.value,
                        "confidence": radiology_report.confidence,
                        "missing_views_or_sequences": (
                            radiology_report.missing_views_or_sequences
                        ),
                        "uncertain_findings": radiology_report.uncertain_findings,
                    }
                    if radiology_report is not None
                    else None
                ),
                "latest_specialist_messages": chat_context(
                    compact_routing_chat(chat_log)
                ),
                "operative_field_matrix": compute_field_matrix(working_plan),
                "current_question": question,
            },
            ensure_ascii=False,
        )

    async def route(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport,
        chat_log: Sequence[ChatMessage],
        speaker_counts: dict[str, int],
        eligible_speakers: Sequence[AgentRole],
        pending_requests: Sequence[PendingRequest],
        unresolved_issues: Sequence[str],
        disagreements: Sequence[Disagreement],
        last_specialist: AgentRole | None,
        maximum_turns_per_specialist: int,
        available_image_ids: Sequence[str] = (),
        working_plan: OperativePlanDraft | None = None,
    ) -> ModelResult[ModeratorDecision]:
        report_status = {
            "image_quality": radiology_report.image_quality.value,
            "confidence": radiology_report.confidence,
            "missing_views_or_sequences": radiology_report.missing_views_or_sequences,
            "uncertain_findings": radiology_report.uncertain_findings,
        }
        user_prompt = json.dumps(
            {
                "task": "Select the next valid MDT routing action.",
                "radiology_status": report_status,
                "latest_specialist_messages": chat_context(
                    compact_routing_chat(chat_log)
                ),
                "operative_working_plan": (
                    working_plan.model_dump(mode="json")
                    if working_plan is not None
                    else None
                ),
                "operative_field_matrix": compute_field_matrix(working_plan),
                "speaker_counts": speaker_counts,
                "eligible_speakers": [role.value for role in eligible_speakers],
                "pending_requests": [
                    request.model_dump(mode="json") for request in pending_requests
                ],
                "unresolved_issues": list(unresolved_issues),
                "current_disagreements": [
                    item.model_dump(mode="json") for item in disagreements
                ],
                "last_specialist": last_specialist.value if last_specialist else None,
                "maximum_turns_per_specialist": maximum_turns_per_specialist,
                "available_image_ids_for_targeted_recheck": list(available_image_ids),
            },
            ensure_ascii=False,
        )
        return await self._invoke(user_prompt=user_prompt, images=())
