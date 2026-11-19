"""Neurology: localization and symptom-imaging agreement (protocol Agent 3)."""

from __future__ import annotations

from department.agents.base import DepartmentAgent
from department.schemas import ImagingFindings, NeurologyAssessment, ResidentReport
from schemas import AgentRole


class NeurologyAgent(DepartmentAgent[NeurologyAssessment]):
    role = AgentRole.NEUROLOGIST
    display_name = "Neurologist"
    prompt_filename = "neurologist.md"
    output_model = NeurologyAssessment
    knowledge_base = ("neurologic-localization", "myelopathy-functional-scores")

    async def run(
        self,
        *,
        resident_report: ResidentReport,
        imaging: ImagingFindings,
        case_preop_md: str,
    ):
        return await self._invoke(
            user_prompt=self.payload(
                task="Localize the findings and judge how well imaging explains them.",
                resident_report=resident_report.model_dump(mode="json"),
                imaging_findings=imaging.model_dump(mode="json"),
                case_record_md=case_preop_md,
            ),
            images=(),
        )
