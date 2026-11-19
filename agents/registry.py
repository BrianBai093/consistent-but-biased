"""Explicit registry for the six independent MDT agent classes."""

from dataclasses import dataclass

from agents.anesthesiologist_agent import AnesthesiologistAgent
from agents.base_agent import BaseAgent
from agents.final_plan_agents import FinalReviewerAgent, FinalSynthesisAgent
from agents.moderator_agent import ModeratorAgent
from agents.neurologist_agent import NeurologistAgent
from agents.perioperative_physician_agent import PerioperativePhysicianAgent
from agents.preparation_agents import (
    CaseDocumentTranscriber,
    CaseSummaryAgent,
    TestDocumentAgent,
)
from agents.radiologist_agent import RadiologistAgent
from agents.spine_surgeon_agent import SpineSurgeonAgent
from schemas import AgentRole


@dataclass(frozen=True, slots=True)
class AgentRegistry:
    case_document: CaseDocumentTranscriber
    test_document: TestDocumentAgent
    case_summary: CaseSummaryAgent
    moderator: ModeratorAgent
    final_synthesis: FinalSynthesisAgent
    final_reviewer: FinalReviewerAgent
    radiologist: RadiologistAgent
    spine_surgeon: SpineSurgeonAgent
    anesthesiologist: AnesthesiologistAgent
    neurologist: NeurologistAgent
    perioperative_physician: PerioperativePhysicianAgent

    def __post_init__(self) -> None:
        expected = {
            "case_document": (self.case_document, CaseDocumentTranscriber),
            "test_document": (self.test_document, TestDocumentAgent),
            "case_summary": (self.case_summary, CaseSummaryAgent),
            "moderator": (self.moderator, ModeratorAgent),
            "final_synthesis": (self.final_synthesis, FinalSynthesisAgent),
            "final_reviewer": (self.final_reviewer, FinalReviewerAgent),
            "radiologist": (self.radiologist, RadiologistAgent),
            "spine_surgeon": (self.spine_surgeon, SpineSurgeonAgent),
            "anesthesiologist": (self.anesthesiologist, AnesthesiologistAgent),
            "neurologist": (self.neurologist, NeurologistAgent),
            "perioperative_physician": (
                self.perioperative_physician,
                PerioperativePhysicianAgent,
            ),
        }
        for name, (agent, agent_type) in expected.items():
            if not isinstance(agent, agent_type):
                raise TypeError(f"{name} must be an independent {agent_type.__name__} instance")

    def specialist(self, role: AgentRole) -> BaseAgent:
        specialists = {
            AgentRole.RADIOLOGIST: self.radiologist,
            AgentRole.SPINE_SURGEON: self.spine_surgeon,
            AgentRole.ANESTHESIOLOGIST: self.anesthesiologist,
            AgentRole.NEUROLOGIST: self.neurologist,
            AgentRole.PERIOPERATIVE_PHYSICIAN: self.perioperative_physician,
        }
        try:
            return specialists[role]
        except KeyError as exc:
            raise ValueError(f"{role.value} is not a specialist role") from exc
