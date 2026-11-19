from agents import (
    AgentRegistry,
    AnesthesiologistAgent,
    CaseDocumentTranscriber,
    CaseSummaryAgent,
    FinalReviewerAgent,
    FinalSynthesisAgent,
    ModeratorAgent,
    NeurologistAgent,
    PerioperativePhysicianAgent,
    RadiologistAgent,
    SpineSurgeonAgent,
    TestDocumentAgent,
)
from schemas import AgentRole, ModeratorAction, ModeratorDecision
from services.llm_client import FakeModelClient
from tests.factories import all_fake_responses, fake_ocr_engine


def ask(role: AgentRole, *, question: str | None = None) -> ModeratorDecision:
    prompt = question or f"Please provide the {role.value} assessment."
    return ModeratorDecision(
        public_message=prompt,
        action=ModeratorAction.ASK_SPECIALIST,
        next_speaker=role,
        question_for_next_speaker=prompt,
        unresolved_issues=[],
        disagreements=[],
        consensus_reached=False,
        routing_reason=f"Route to {role.value}.",
        continue_discussion=True,
        enter_final_summary=False,
        target_image_ids=["mri-1"] if role is AgentRole.RADIOLOGIST else [],
    )


def finalize() -> ModeratorDecision:
    return ModeratorDecision(
        public_message="All required specialist perspectives are available for final synthesis.",
        action=ModeratorAction.FINALIZE,
        next_speaker=None,
        question_for_next_speaker=None,
        unresolved_issues=[],
        disagreements=[],
        consensus_reached=True,
        routing_reason="Minimum specialist coverage is complete.",
        continue_discussion=False,
        enter_final_summary=True,
    )


def registry_with_client(
    *,
    moderator_responses,
    overrides: dict | None = None,
    max_retries: int = 0,
) -> tuple[AgentRegistry, FakeModelClient]:
    responses = all_fake_responses()
    responses[ModeratorDecision] = moderator_responses
    responses.update(overrides or {})
    client = FakeModelClient(responses, max_retries=max_retries)
    registry = AgentRegistry(
        case_document=CaseDocumentTranscriber(fake_ocr_engine()),
        test_document=TestDocumentAgent(client),
        case_summary=CaseSummaryAgent(client),
        moderator=ModeratorAgent(client),
        final_synthesis=FinalSynthesisAgent(client),
        final_reviewer=FinalReviewerAgent(client),
        radiologist=RadiologistAgent(client),
        spine_surgeon=SpineSurgeonAgent(client),
        anesthesiologist=AnesthesiologistAgent(client),
        neurologist=NeurologistAgent(client),
        perioperative_physician=PerioperativePhysicianAgent(client),
    )
    return registry, client
