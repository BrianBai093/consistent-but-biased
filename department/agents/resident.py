"""Chief resident: reports the chart, takes no position (protocol Agent 1)."""

from __future__ import annotations

from department.agents.base import DepartmentAgent
from department.schemas import ResidentReport
from schemas import AgentRole


class ResidentAgent(DepartmentAgent[ResidentReport]):
    role = AgentRole.RESIDENT
    display_name = "Chief Resident"
    prompt_filename = "resident.md"
    output_model = ResidentReport
    knowledge_base = ("standardized-case-library",)

    async def run(self, *, case_preop_md: str, tests_preop_md: str):
        return await self._invoke(
            user_prompt=self.payload(
                task="Report this patient's history, examination, and basic results.",
                case_record_md=case_preop_md,
                laboratory_md=tests_preop_md,
            ),
            images=(),
        )
