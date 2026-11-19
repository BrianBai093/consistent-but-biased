"""Anesthesia: quantitative perioperative risk, stated neutrally (Agent 4)."""

from __future__ import annotations

from department.agents.base import DepartmentAgent
from department.schemas import AnesthesiaAssessment, ResidentReport
from schemas import AgentRole


class AnesthesiaAgent(DepartmentAgent[AnesthesiaAssessment]):
    role = AgentRole.ANESTHESIOLOGIST
    display_name = "Anesthesiologist"
    prompt_filename = "anesthesiologist.md"
    output_model = AnesthesiaAssessment
    knowledge_base = ("asa-classification", "spine-anesthesia-guidelines")

    async def run(
        self,
        *,
        resident_report: ResidentReport,
        case_preop_md: str,
        tests_preop_md: str,
    ):
        return await self._invoke(
            user_prompt=self.payload(
                task="State the quantitative perioperative anesthesia risk.",
                resident_report=resident_report.model_dump(mode="json"),
                case_record_md=case_preop_md,
                laboratory_md=tests_preop_md,
            ),
            images=(),
        )
