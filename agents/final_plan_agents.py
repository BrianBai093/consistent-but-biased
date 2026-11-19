"""Critical final-plan synthesis and independent evidence review agents."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import cast

from pydantic import BaseModel

from agents.base_agent import BaseAgent, document_context, report_context
from schemas import (
    AgentRole,
    Disagreement,
    FinalPlan,
    FinalPlanReview,
    MISSING_RECORDS,
    RadiologyReport,
    SpecialistContribution,
    TerminationReason,
)
from services.llm_client import ModelResult


class FinalSynthesisAgent(BaseAgent[FinalPlan]):
    role = AgentRole.MODERATOR
    display_name = "Evidence Synthesis"
    prompt_filename = "final_synthesis.md"
    output_model = FinalPlan

    def __init__(self, client) -> None:
        super().__init__(client)
        self.revision_prompt = self._load_prompt_file("final_revision.md")

    def build_user_prompt(self, **kwargs) -> str:
        return json.dumps(kwargs, ensure_ascii=False)

    async def draft(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport,
        latest_contributions: Sequence[SpecialistContribution],
        unresolved_issues: Sequence[str],
        disagreements: Sequence[Disagreement],
        valid_evidence_ids: Sequence[str],
        force_finalize: bool,
        termination_reason: TerminationReason,
        output_processor: Callable[[BaseModel], BaseModel],
    ) -> ModelResult[FinalPlan]:
        prompt = json.dumps(
            {
                "task": "Draft the evidence-bound final preoperative MDT plan.",
                "source_documents": document_context(
                    case_preop_md=case_preop_md,
                    tests_preop_md=tests_preop_md,
                    radiology_preop_md=radiology_preop_md,
                ),
                "structured_radiology_report": report_context(radiology_report),
                "latest_structured_contributions": [
                    contribution.model_dump(mode="json")
                    for contribution in latest_contributions
                ],
                "workflow_unresolved_issues": list(unresolved_issues),
                "workflow_disagreements": [
                    item.model_dump(mode="json") for item in disagreements
                ],
                "valid_evidence_ids": list(valid_evidence_ids),
                "validation_contract": {
                    "missing_value": MISSING_RECORDS,
                    "critical_operating_field_paths": [
                        "/recommended_surgery/procedure_name",
                        "/recommended_surgery/levels",
                        "/recommended_surgery/approach",
                        "/recommended_surgery/decompression_extent",
                        "/recommended_surgery/fusion_or_fixation",
                    ],
                    "rules": [
                        "Use only identifiers from valid_evidence_ids.",
                        (
                            "Every non-missing critical operating field requires a "
                            "claim_evidence item with the exact field_path and at least "
                            "one valid evidence id."
                        ),
                        (
                            "Fill the recommended_surgery value fields with the Spine "
                            "Surgeon's preferred operative option content. Do not set the "
                            "per-field *_status fields or definitive_surgery_status: the "
                            "workflow assigns them deterministically after independent "
                            "review; default them to withheld."
                        ),
                        (
                            "Preserve Spine Surgeon operative_options in "
                            "conditional_surgical_options without changing procedure or "
                            "approach into a generic phrase."
                        ),
                        (
                            "Each surgical option and approach assessment must use only "
                            "identifiers from valid_evidence_ids."
                        ),
                        (
                            "Limited imaging withholds established levels and "
                            "fusion/fixation fields but keeps concrete values where the "
                            "surgeon supplied them. Nondiagnostic or unknown imaging "
                            "requires very_high uncertainty and confidence <= 0.25."
                        ),
                        (
                            "Do not state that an assessment, clearance, optimization, "
                            "or exclusion is completed unless a documented claim_evidence "
                            "item supports that exact field."
                        ),
                    ],
                },
                "force_finalize": force_finalize,
                "termination_reason": termination_reason.value,
            },
            ensure_ascii=False,
        )
        result = await self._invoke(
            user_prompt=prompt,
            images=(),
            output_processor=output_processor,
        )
        return cast(ModelResult[FinalPlan], result)

    async def revise(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport,
        draft: FinalPlan,
        review: FinalPlanReview,
        output_processor: Callable[[BaseModel], BaseModel],
    ) -> ModelResult[FinalPlan]:
        prompt = json.dumps(
            {
                "task": "Revise the draft exactly once using the independent review.",
                "source_documents": document_context(
                    case_preop_md=case_preop_md,
                    tests_preop_md=tests_preop_md,
                    radiology_preop_md=radiology_preop_md,
                ),
                "structured_radiology_report": report_context(radiology_report),
                "draft": draft.model_dump(mode="json"),
                "independent_review": review.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        result = await self._invoke(
            user_prompt=prompt,
            images=(),
            system_prompt=self.revision_prompt,
            output_processor=output_processor,
        )
        return cast(ModelResult[FinalPlan], result)


class FinalReviewerAgent(BaseAgent[FinalPlanReview]):
    role = AgentRole.MODERATOR
    display_name = "Independent Evidence Reviewer"
    prompt_filename = "final_reviewer.md"
    output_model = FinalPlanReview

    def build_user_prompt(self, **kwargs) -> str:
        return json.dumps(kwargs, ensure_ascii=False)

    async def review(
        self,
        *,
        case_preop_md: str,
        tests_preop_md: str,
        radiology_preop_md: str,
        radiology_report: RadiologyReport,
        draft: FinalPlan,
        valid_evidence_ids: Sequence[str],
        output_processor: Callable[[BaseModel], BaseModel] | None = None,
    ) -> ModelResult[FinalPlanReview]:
        prompt = json.dumps(
            {
                "task": "Independently audit the draft against original preoperative evidence.",
                "source_documents": document_context(
                    case_preop_md=case_preop_md,
                    tests_preop_md=tests_preop_md,
                    radiology_preop_md=radiology_preop_md,
                ),
                "structured_radiology_report": report_context(radiology_report),
                "draft": draft.model_dump(mode="json"),
                "valid_evidence_ids": list(valid_evidence_ids),
            },
            ensure_ascii=False,
        )
        result = await self._invoke(
            user_prompt=prompt,
            images=(),
            output_processor=output_processor,
        )
        return cast(ModelResult[FinalPlanReview], result)
