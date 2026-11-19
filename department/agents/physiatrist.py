"""Rehabilitation medicine: the conservative advocate (protocol Agent 5).

Unlike the other stage I-II roles this one speaks again in every negotiation
round, because "needs strong evidence before conceding" can only be observed if
it is given something to concede to.

The negotiation turns read one more instruction than the opening does. Under
protocol dept-v1 the advocate could concede only to *new* evidence, and after
stage II no new evidence can arrive -- in the first clean batch it said "no new
evidence has been presented" in eleven responses of twelve and never moved. What
the design asks for is *strong* evidence, and in a meeting that means an argument
about a finding already in the record. The opening is untouched, so the frozen
stage I-II statements remain exactly what they were.
"""

from __future__ import annotations

from typing import Sequence

from department.agents.base import DepartmentAgent, transcript
from department.knowledge import KnowledgeSource
from department.schemas import (
    ImagingFindings,
    NeurologyAssessment,
    RehabAssessment,
    ResidentReport,
)
from schemas import AgentRole, ChatMessage
from services.llm_client import StructuredModelClient

NEGOTIATION_PROMPT_FILENAME = "physiatrist_negotiation.md"


class PhysiatristAgent(DepartmentAgent[RehabAssessment]):
    role = AgentRole.PHYSIATRIST
    display_name = "Rehabilitation Physician"
    prompt_filename = "physiatrist.md"
    output_model = RehabAssessment
    knowledge_base = (
        "myelopathy-conservative-rct-meta-analysis",
        "functional-recovery-prediction",
    )

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        knowledge_source: KnowledgeSource | None = None,
    ) -> None:
        super().__init__(client, knowledge_source=knowledge_source)
        self.negotiation_prompt = self.load_department_prompt(NEGOTIATION_PROMPT_FILENAME)

    def negotiation_system_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self.negotiation_prompt}"

    async def run(
        self,
        *,
        resident_report: ResidentReport,
        imaging: ImagingFindings,
        neurology: NeurologyAssessment,
    ):
        """The opening position, before anyone has argued with it."""

        return await self._invoke(
            user_prompt=self.payload(
                task="State the case for conservative management and its preconditions.",
                resident_report=resident_report.model_dump(mode="json"),
                imaging_findings=imaging.model_dump(mode="json"),
                neurology_assessment=neurology.model_dump(mode="json"),
            ),
            images=(),
        )

    async def respond(
        self,
        *,
        prior: RehabAssessment,
        chat_log: Sequence[ChatMessage],
        round_number: int,
    ):
        """Hold or move, and say what moved you."""

        return await self._invoke(
            user_prompt=self.payload(
                task=(
                    "Respond to the discussion. Hold your position unless an argument "
                    "has shown you that a finding already in the record matters more "
                    "than you weighed it; if it moved you, name that finding."
                ),
                your_previous_position=prior.model_dump(mode="json"),
                discussion=transcript(chat_log),
                round_number=round_number,
            ),
            images=(),
            system_prompt=self.negotiation_system_prompt(),
        )
