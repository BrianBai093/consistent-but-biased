import asyncio
import json
from pathlib import Path

import pytest

from graph import create_mdt_graph, opening_input, thread_config
from schemas import (
    AgentQuestion,
    AgentRole,
    AnesthesiologistOutput,
    CaseDocumentMarkdown,
    CaseSummaryMarkdown,
    ModeratorDecision,
    NeurologistOutput,
    PerioperativePhysicianOutput,
    RadiologyAddendum,
    RadiologyReport,
    RunPhase,
    SpineSurgeonOutput,
    TestDocumentMarkdown as LabDocumentMarkdown,
    WorkflowLimits,
)
from routing import DEFAULT_QUESTIONS
from services.checkpoint import open_sqlite_checkpointer
from services.llm_client import FakeModelClient
from tests.dynamic_factories import ask, finalize, registry_with_client
from tests.factories import all_fake_responses, spine_output
from tests.test_opening_graph import case_input


STANDARD_ROUTE = [finalize()]

INITIAL_OUTPUT_ROLES = {
    SpineSurgeonOutput: AgentRole.SPINE_SURGEON,
    NeurologistOutput: AgentRole.NEUROLOGIST,
    AnesthesiologistOutput: AgentRole.ANESTHESIOLOGIST,
    PerioperativePhysicianOutput: AgentRole.PERIOPERATIVE_PHYSICIAN,
}


class IndependentRoundGateClient(FakeModelClient):
    def __init__(self, surgeon: SpineSurgeonOutput) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        responses[SpineSurgeonOutput] = surgeon
        super().__init__(responses)
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()
        self.initial_requests: dict[AgentRole, dict] = {}
        self.completion_order: list[AgentRole] = []

    async def generate_structured(self, **kwargs):
        role = INITIAL_OUTPUT_ROLES.get(kwargs["output_model"])
        if role is not None and role not in self.initial_requests:
            self.initial_requests[role] = kwargs
            if len(self.initial_requests) == len(INITIAL_OUTPUT_ROLES):
                self.all_started.set()
            await self.release.wait()
            delay = {
                AgentRole.SPINE_SURGEON: 0.04,
                AgentRole.NEUROLOGIST: 0.03,
                AgentRole.ANESTHESIOLOGIST: 0.02,
                AgentRole.PERIOPERATIVE_PHYSICIAN: 0.01,
            }[role]
            await asyncio.sleep(delay)
            result = await super().generate_structured(**kwargs)
            self.completion_order.append(role)
            return result
        return await super().generate_structured(**kwargs)


@pytest.mark.asyncio
async def test_initial_text_specialists_are_parallel_isolated_and_deterministic(
    tmp_path: Path,
) -> None:
    cross_specialty_question = "QUESTION THAT MUST NOT LEAK INTO FIRST-ROUND PROMPTS"
    surgeon = spine_output().model_copy(
        update={
            "questions_for_other_agents": [
                AgentQuestion(
                    target=AgentRole.NEUROLOGIST,
                    question=cross_specialty_question,
                    evidence_ids=["case-page-001"],
                )
            ]
        }
    )
    client = IndependentRoundGateClient(surgeon)
    registry, _ = registry_with_client(moderator_responses=STANDARD_ROUTE)
    for agent in (
        registry.case_document,
        registry.test_document,
        registry.case_summary,
        registry.moderator,
        registry.final_synthesis,
        registry.final_reviewer,
        registry.radiologist,
        registry.spine_surgeon,
        registry.anesthesiologist,
        registry.neurologist,
        registry.perioperative_physician,
    ):
        agent.client = client

    async with open_sqlite_checkpointer(tmp_path / "independent-round.sqlite3") as saver:
        graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            interrupt_after=["independent_specialist_round"],
        )
        invocation = asyncio.create_task(
            graph.ainvoke(
                opening_input(case_input()),
                config=thread_config("thread-independent-round"),
            )
        )
        try:
            await asyncio.wait_for(client.all_started.wait(), timeout=1)
            assert not any(
                call["output_model"] is ModeratorDecision for call in client.calls
            )

            payloads = {
                role: json.loads(request["user_prompt"])
                for role, request in client.initial_requests.items()
            }
            source_snapshots = [
                payload["source_documents"] for payload in payloads.values()
            ]
            radiology_snapshots = [
                payload["structured_radiology_report"] for payload in payloads.values()
            ]
            assert all(item == source_snapshots[0] for item in source_snapshots)
            assert all(item == radiology_snapshots[0] for item in radiology_snapshots)
            for payload in payloads.values():
                assert payload["relevant_public_context"] == []
                assert payload["prior_structured_contribution"] is None
                assert cross_specialty_question not in json.dumps(payload)
                assert "path_or_url" not in json.dumps(payload)
                assert "original_filename" not in json.dumps(payload)
            assert {
                role: payload["moderator_question"] for role, payload in payloads.items()
            } == {
                role: DEFAULT_QUESTIONS[role] for role in INITIAL_OUTPUT_ROLES.values()
            }
        finally:
            client.release.set()
        result = await invocation

    assert client.completion_order == [
        AgentRole.PERIOPERATIVE_PHYSICIAN,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.SPINE_SURGEON,
    ]
    assert [message.speaker_role for message in result["chat_log"]][-5:] == [
        AgentRole.MODERATOR,
        AgentRole.SPINE_SURGEON,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    ]
    assert [message.sequence for message in result["chat_log"]] == list(range(1, 8))
    assert result["pending_requests"][0].question == cross_specialty_question
    assert all(
        result["speaker_counts"][role.value] == 1
        for role in INITIAL_OUTPUT_ROLES.values()
    )


@pytest.mark.asyncio
async def test_specialist_can_speak_three_times_but_never_a_fourth(
    tmp_path: Path,
) -> None:
    surgeon_outputs = [
        spine_output().model_copy(update={"public_message": f"Spine opinion {turn}."})
        for turn in range(1, 4)
    ]
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.SPINE_SURGEON),
            finalize(),
        ],
        overrides={SpineSurgeonOutput: surgeon_outputs},
    )
    limits = WorkflowLimits(max_workflow_steps=100, max_no_progress_turns=20)
    async with open_sqlite_checkpointer(tmp_path / "three-turns.sqlite3") as saver:
        graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-three-turns"),
        )

    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 3
    assert sum(result["speaker_counts"].values()) <= 12
    spine_calls = [
        call for call in client.calls if call["output_model"] is SpineSurgeonOutput
    ]
    assert len(spine_calls) == 3
    prompts = [json.loads(call["user_prompt"]) for call in spine_calls]
    assert prompts[0]["relevant_public_context"] == []
    assert prompts[0]["prior_structured_contribution"] is None
    assert prompts[1]["prior_structured_contribution"]["public_message"] == (
        "Spine opinion 1."
    )
    assert prompts[2]["prior_structured_contribution"]["public_message"] == (
        "Spine opinion 2."
    )
    assert all(len(prompt["relevant_public_context"]) <= 4 for prompt in prompts[1:])
    assert all(
        prompt["source_documents"] != prompts[0]["source_documents"]
        for prompt in prompts[1:]
    )


@pytest.mark.asyncio
async def test_radiologist_is_limited_to_initial_review_plus_two_targeted_rechecks(
    tmp_path: Path,
) -> None:
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.RADIOLOGIST),
            ask(AgentRole.RADIOLOGIST),
            ask(AgentRole.RADIOLOGIST),
            finalize(),
        ]
    )
    async with open_sqlite_checkpointer(tmp_path / "three-radiology-turns.sqlite3") as saver:
        graph = create_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-three-radiology-turns"),
        )

    assert result["speaker_counts"][AgentRole.RADIOLOGIST.value] == 3
    assert sum(call["output_model"] is RadiologyReport for call in client.calls) == 1
    assert sum(call["output_model"] is RadiologyAddendum for call in client.calls) == 2


@pytest.mark.asyncio
async def test_unresolved_discussion_uses_default_absolute_limit(tmp_path: Path) -> None:
    registry, _ = registry_with_client(
        moderator_responses={"public_message": "invalid"},
    )
    limits = WorkflowLimits(max_workflow_steps=100, max_no_progress_turns=20)
    async with open_sqlite_checkpointer(tmp_path / "absolute-total.sqlite3") as saver:
        graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-absolute-total"),
        )

    assert limits.max_total_specialist_turns == 20
    assert limits.max_absolute_specialist_turns == 24
    assert sum(result["speaker_counts"].values()) == 24
    assert max(result["speaker_counts"].values()) <= 5


@pytest.mark.asyncio
async def test_complete_dynamic_discussion_reaches_finalization_boundary(tmp_path: Path) -> None:
    registry, client = registry_with_client(moderator_responses=STANDARD_ROUTE)
    async with open_sqlite_checkpointer(tmp_path / "dynamic.sqlite3") as checkpointer:
        graph = create_mdt_graph(agent_registry=registry, checkpointer=checkpointer)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-dynamic"),
        )

    assert result["phase"] is RunPhase.AWAITING_FINALIZATION
    assert result["final_plan"] is None
    assert result["force_finalize"] is False
    assert all(count == 1 for count in result["speaker_counts"].values())
    assert len(result["specialist_contributions"]) == 5
    roles = [message.speaker_role for message in result["chat_log"]]
    assert roles == [
        AgentRole.MODERATOR,
        AgentRole.RADIOLOGIST,
        AgentRole.MODERATOR,
        AgentRole.SPINE_SURGEON,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
        AgentRole.MODERATOR,
    ]
    assert [message.sequence for message in result["chat_log"]] == list(range(1, 9))
    text_calls = [
        call
        for call in client.calls
        if call["output_model"] not in (
            RadiologyReport,
            ModeratorDecision,
            CaseDocumentMarkdown,
            LabDocumentMarkdown,
            CaseSummaryMarkdown,
        )
    ]
    assert text_calls
    assert all(call["images"] == [] for call in text_calls)


@pytest.mark.asyncio
async def test_dynamic_radiologist_is_only_repeat_specialist_receiving_images(
    tmp_path: Path,
) -> None:
    route = [ask(AgentRole.RADIOLOGIST), finalize()]
    registry, client = registry_with_client(moderator_responses=route)
    async with open_sqlite_checkpointer(tmp_path / "radiology-repeat.sqlite3") as checkpointer:
        graph = create_mdt_graph(agent_registry=registry, checkpointer=checkpointer)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-radiology-repeat"),
        )
    assert result["speaker_counts"][AgentRole.RADIOLOGIST.value] == 2
    image_calls = [
        call
        for call in client.calls
        if call["output_model"] in {RadiologyReport, RadiologyAddendum}
    ]
    assert len(image_calls) == 2
    assert image_calls[0]["output_model"] is RadiologyReport
    assert image_calls[1]["output_model"] is RadiologyAddendum
    assert [image.image_id for image in image_calls[1]["images"]] == ["mri-1"]


@pytest.mark.asyncio
async def test_pending_request_returns_to_moderator_before_target(tmp_path: Path) -> None:
    surgeon = spine_output().model_copy(
        update={
            "questions_for_other_agents": [
                AgentQuestion(
                    target=AgentRole.NEUROLOGIST,
                    question="Please confirm symptom-imaging concordance.",
                )
            ],
            "need_retrieval": True,
            "retrieval_query": "Spine surgery indication guideline",
        }
    )
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.ANESTHESIOLOGIST),
        ],
        overrides={SpineSurgeonOutput: surgeon},
    )
    async with open_sqlite_checkpointer(tmp_path / "mentions.sqlite3") as checkpointer:
        graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            interrupt_after=["moderator_route"],
        )
        first_stop = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-mentions"),
        )
        assert first_stop["next_speaker"] is AgentRole.NEUROLOGIST

        second_stop = await graph.ainvoke(
            None,
            config=thread_config("thread-mentions"),
        )

    assert second_stop["next_speaker"] is AgentRole.ANESTHESIOLOGIST
    assert second_stop["pending_requests"] == []
    assert second_stop["retrieval_count"] == 0
    assert second_stop["run_metadata"].retrieval_count == 0
    assert second_stop["retrieved_evidence"] == []
    assert any("retrieval" in issue for issue in second_stop["unresolved_issues"])
    assert [message.speaker_role for message in second_stop["chat_log"]][-2:] == [
        AgentRole.NEUROLOGIST,
        AgentRole.MODERATOR,
    ]
    assert sum(call["output_model"] is NeurologistOutput for call in client.calls) == 2


@pytest.mark.asyncio
async def test_sqlite_resume_preserves_dynamic_idempotency(tmp_path: Path) -> None:
    database = tmp_path / "dynamic-resume.sqlite3"
    config = thread_config("thread-dynamic-resume")
    registry, client = registry_with_client(moderator_responses=STANDARD_ROUTE)
    async with open_sqlite_checkpointer(database) as checkpointer:
        interrupted_graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            interrupt_after=["independent_specialist_round"],
        )
        interrupted = await interrupted_graph.ainvoke(opening_input(case_input()), config=config)
        assert interrupted["phase"] is RunPhase.AWAITING_DYNAMIC_ROUTING
        assert len(interrupted["chat_log"]) == 7
        initial_call_counts = {
            output_model: sum(
                call["output_model"] is output_model for call in client.calls
            )
            for output_model in INITIAL_OUTPUT_ROLES
        }

    async with open_sqlite_checkpointer(database) as checkpointer:
        resumed_graph = create_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        resumed = await resumed_graph.ainvoke(None, config=config)
        calls_after_completion = len(client.calls)
        completed_again = await resumed_graph.ainvoke(None, config=config)

    assert resumed["phase"] is RunPhase.AWAITING_FINALIZATION
    assert len({message.message_id for message in resumed["chat_log"]}) == len(
        resumed["chat_log"]
    )
    assert len(resumed["specialist_contributions"]) == 5
    assert all(count == 1 for count in initial_call_counts.values())
    assert all(
        sum(call["output_model"] is output_model for call in client.calls) == 1
        for output_model in INITIAL_OUTPUT_ROLES
    )
    assert completed_again["chat_log"] == resumed["chat_log"]
    assert len(client.calls) == calls_after_completion


@pytest.mark.asyncio
async def test_continuous_moderator_parse_failure_uses_bounded_fallback(tmp_path: Path) -> None:
    registry, _ = registry_with_client(
        moderator_responses={"public_message": "invalid"},
        max_retries=1,
    )
    async with open_sqlite_checkpointer(tmp_path / "fallback.sqlite3") as checkpointer:
        graph = create_mdt_graph(agent_registry=registry, checkpointer=checkpointer)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-fallback"),
        )
    assert result["phase"] is RunPhase.AWAITING_FINALIZATION
    assert result["force_finalize"] is True
    assert all(count >= 1 for count in result["speaker_counts"].values())
    assert max(result["speaker_counts"].values()) <= 3
    assert sum(result["speaker_counts"].values()) <= 12
