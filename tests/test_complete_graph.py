from pathlib import Path

import pytest

import graph as graph_module
from exceptions import (
    FinalPlanIntegrityError,
    ModelFailureCategory,
    ModelInvocationError,
    ModelStageError,
    ModelTimeoutError,
)
from graph import create_complete_mdt_graph, opening_input, thread_config
from schemas import (
    AgentRole,
    AnesthesiologistOutput,
    DiagnosisEvidence,
    Disagreement,
    FinalPlan,
    MISSING_RECORDS,
    PerioperativePhysicianOutput,
    RadiologyReport,
    RunPhase,
    SpineSurgeonOutput,
    TerminationReason,
    WorkflowLimits,
)
from services.checkpoint import open_sqlite_checkpointer
from tests.dynamic_factories import ask, finalize, registry_with_client
from tests.factories import (
    anesthesia_output,
    final_plan,
    perioperative_output,
)
from tests.test_dynamic_graph import STANDARD_ROUTE
from tests.test_opening_graph import case_input


@pytest.mark.asyncio
async def test_normal_complete_graph_generates_final_plan(tmp_path: Path) -> None:
    registry, _ = registry_with_client(moderator_responses=STANDARD_ROUTE)
    async with open_sqlite_checkpointer(tmp_path / "complete.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-complete"),
        )
    assert result["phase"] is RunPhase.COMPLETED
    assert isinstance(result["final_plan"], FinalPlan)
    assert result["final_plan"].force_finalized is False
    assert result["final_plan"].confidence == 0.9
    assert result["termination_reason"] is TerminationReason.NORMAL_CONSENSUS
    assert result["run_metadata"].completed_at is not None
    assert result["run_metadata"].finalization_fallback_used is False
    assert result["run_metadata"].specialist_message_count <= 8
    assert len(result["run_metadata"].model_calls) <= 45
    assert len(result["chat_log"]) <= 20
    assert result["chat_log"][-1].message_id == "case-opening:moderator-final"
    assert result["chat_log"][-1].sequence == len(result["chat_log"])


@pytest.mark.asyncio
async def test_early_finalize_cannot_bypass_minimum_coverage(tmp_path: Path) -> None:
    route = [
        finalize(),
        ask(AgentRole.NEUROLOGIST),
        ask(AgentRole.ANESTHESIOLOGIST),
        ask(AgentRole.PERIOPERATIVE_PHYSICIAN),
        finalize(),
    ]
    registry, _ = registry_with_client(moderator_responses=route)
    async with open_sqlite_checkpointer(tmp_path / "coverage.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-coverage"),
        )
    assert result["phase"] is RunPhase.COMPLETED
    assert all(count >= 1 for count in result["speaker_counts"].values())
    assert result["force_finalize"] is False


@pytest.mark.asyncio
async def test_all_speaker_limits_force_deterministic_plan(tmp_path: Path) -> None:
    registry, _ = registry_with_client(
        moderator_responses={"public_message": "invalid"},
    )
    limits = WorkflowLimits(
        max_turns_per_specialist=3,
        max_total_specialist_turns=15,
        max_workflow_steps=100,
        max_no_progress_turns=20,
    )
    async with open_sqlite_checkpointer(tmp_path / "all-limits.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-all-limits"),
        )
    assert result["termination_reason"] is TerminationReason.ALL_SPEAKER_LIMITS
    assert all(count == 3 for count in result["speaker_counts"].values())
    assert result["final_plan"].force_finalized is True
    assert result["run_metadata"].finalization_fallback_used is False


@pytest.mark.asyncio
async def test_total_speaker_limit_forces_completion(tmp_path: Path) -> None:
    registry, _ = registry_with_client(
        moderator_responses={"public_message": "invalid"},
    )
    limits = WorkflowLimits(
        max_total_specialist_turns=6,
        max_absolute_specialist_turns=6,
        max_no_progress_turns=20,
    )
    async with open_sqlite_checkpointer(tmp_path / "total-limit.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-total-limit"),
        )
    assert result["termination_reason"] is TerminationReason.TOTAL_SPEAKER_LIMIT
    assert sum(result["speaker_counts"].values()) == 6
    assert result["phase"] is RunPhase.COMPLETED


@pytest.mark.asyncio
async def test_no_progress_and_global_step_limits_force_completion(tmp_path: Path) -> None:
    no_progress_route = [
        *STANDARD_ROUTE[:-1],
        ask(AgentRole.SPINE_SURGEON),
        ask(AgentRole.NEUROLOGIST),
    ]
    registry, _ = registry_with_client(moderator_responses=no_progress_route)
    async with open_sqlite_checkpointer(tmp_path / "no-progress.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        no_progress = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-no-progress"),
        )
    assert no_progress["termination_reason"] is TerminationReason.NO_NEW_INFORMATION
    # v5: the final-position pass replaces the stall with structured stances,
    # so the streak resets; the pass must have completed before the forced close.
    assert no_progress["convergence_pass_done"] is True
    assert {
        item.role for item in no_progress["working_plan"].positions
    } == {
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    }

    registry, _ = registry_with_client(moderator_responses=STANDARD_ROUTE)
    async with open_sqlite_checkpointer(tmp_path / "step-limit.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            limits=WorkflowLimits(max_workflow_steps=5),
        )
        step_limited = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-step-limit"),
        )
    assert step_limited["termination_reason"] is TerminationReason.GLOBAL_STEP_LIMIT
    assert step_limited["phase"] is RunPhase.COMPLETED


@pytest.mark.asyncio
async def test_specialist_api_error_still_generates_complete_fallback(tmp_path: Path) -> None:
    registry, client = registry_with_client(
        moderator_responses=[ask(AgentRole.SPINE_SURGEON)],
    )
    del client.responses[SpineSurgeonOutput]
    async with open_sqlite_checkpointer(tmp_path / "api-error.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-api-error"),
        )
    assert result["termination_reason"] is TerminationReason.SPECIALIST_API_ERROR
    assert result["phase"] is RunPhase.COMPLETED
    assert result["final_plan"].force_finalized is True
    assert result["final_plan"].recommended_surgery.procedure_name == MISSING_RECORDS
    assert result["final_plan"].conditional_surgical_options == []
    assert result["final_plan"].approach_comparison is None
    assert any("spine_surgeon" in issue for issue in result["unresolved_issues"])
    assert any(
        "spine_surgeon" in issue
        for issue in result["final_plan"].final_consensus.remaining_concerns
    )
    failure = result["run_metadata"].model_failures[0]
    assert failure.agent_role is AgentRole.SPINE_SURGEON
    assert failure.outcome == "failed"
    assert failure.category == "MODEL_INVOCATION_FAILED"
    assert failure.attempt_count == 1
    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 0
    assert all(
        result["speaker_counts"][role.value] == 1
        for role in (
            AgentRole.NEUROLOGIST,
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PERIOPERATIVE_PHYSICIAN,
        )
    )
    assert {
        contribution.role for contribution in result["specialist_contributions"]
    } == {
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    }


@pytest.mark.asyncio
async def test_anesthesia_failure_records_four_timeout_attempts_and_stays_out_of_prerequisites(
    tmp_path: Path,
) -> None:
    failures = [ModelTimeoutError("private timeout detail") for _ in range(4)]
    registry, client = registry_with_client(
        moderator_responses=[
            ask(AgentRole.SPINE_SURGEON),
            ask(AgentRole.ANESTHESIOLOGIST),
        ],
        overrides={AnesthesiologistOutput: failures},
        max_retries=3,
    )
    async with open_sqlite_checkpointer(tmp_path / "anesthesia-timeout.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-anesthesia-timeout"),
        )

    anesthesia_calls = [
        call
        for call in client.calls
        if call["output_model"] is AnesthesiologistOutput
    ]
    assert len(anesthesia_calls) == 4
    failure = next(
        item
        for item in result["run_metadata"].model_failures
        if item.agent_role is AgentRole.ANESTHESIOLOGIST
    )
    assert failure.outcome == "failed"
    assert failure.category == "MODEL_TIMEOUT"
    assert failure.attempt_count == 4
    assert failure.max_attempts == 4
    assert failure.attempt_categories == ["MODEL_TIMEOUT"] * 4
    assert "private timeout detail" not in failure.model_dump_json()

    plan = result["final_plan"]
    assert failure.safe_message in plan.final_consensus.remaining_concerns
    assert failure.safe_message not in plan.final_consensus.implementation_prerequisites


@pytest.mark.asyncio
async def test_anesthesia_retry_recovery_is_recorded(tmp_path: Path) -> None:
    scripted = [
        ModelTimeoutError("private timeout detail"),
        ModelTimeoutError("private timeout detail"),
        ModelTimeoutError("private timeout detail"),
        anesthesia_output(),
    ]
    registry, client = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        overrides={AnesthesiologistOutput: scripted},
        max_retries=3,
    )
    async with open_sqlite_checkpointer(tmp_path / "anesthesia-recovered.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-anesthesia-recovered"),
        )

    assert len(
        [
            call
            for call in client.calls
            if call["output_model"] is AnesthesiologistOutput
        ]
    ) == 4
    failure = next(
        item
        for item in result["run_metadata"].model_failures
        if item.agent_role is AgentRole.ANESTHESIOLOGIST
    )
    assert failure.outcome == "recovered"
    assert failure.attempt_count == 4
    assert failure.attempt_categories == ["MODEL_TIMEOUT"] * 3


@pytest.mark.parametrize(
    ("output_type", "delayed_output"),
    [
        (
            AnesthesiologistOutput,
            anesthesia_output().model_copy(
                update={
                    "recommend_delay": True,
                    "delay_reasons": ["Airway assessment is incomplete."],
                }
            ),
        ),
        (
            PerioperativePhysicianOutput,
            perioperative_output().model_copy(
                update={
                    "recommend_delay": True,
                    "delay_reasons": ["Active medical optimization is incomplete."],
                }
            ),
        ),
    ],
    ids=["anesthesia", "perioperative-medicine"],
)
@pytest.mark.asyncio
async def test_delay_recommendation_requires_later_attending_response(
    tmp_path: Path,
    output_type: type,
    delayed_output,
) -> None:
    registry, _ = registry_with_client(
        moderator_responses=[finalize(), finalize()],
        overrides={output_type: delayed_output},
    )
    async with open_sqlite_checkpointer(tmp_path / f"delay-{output_type.__name__}.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config(f"thread-delay-{output_type.__name__}"),
        )

    assert result["termination_reason"] is TerminationReason.NORMAL_CONSENSUS
    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 2
    assert any(
        "preoperative delay recommendation" in message.content
        and message.speaker_role is AgentRole.MODERATOR
        for message in result["chat_log"]
    )


@pytest.mark.asyncio
async def test_attending_response_after_warning_allows_normal_finalization(
    tmp_path: Path,
) -> None:
    delayed = anesthesia_output().model_copy(
        update={
            "recommend_delay": True,
            "delay_reasons": ["Airway assessment is incomplete."],
        }
    )
    route = [ask(AgentRole.SPINE_SURGEON), finalize()]
    registry, _ = registry_with_client(
        moderator_responses=route,
        overrides={AnesthesiologistOutput: delayed},
    )
    async with open_sqlite_checkpointer(tmp_path / "delay-already-answered.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-delay-already-answered"),
        )

    assert result["termination_reason"] is TerminationReason.NORMAL_CONSENSUS
    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 2


@pytest.mark.asyncio
async def test_delay_warning_reserves_one_extra_surgeon_response(
    tmp_path: Path,
) -> None:
    delayed = anesthesia_output().model_copy(
        update={
            "recommend_delay": True,
            "delay_reasons": ["Airway assessment is incomplete."],
        }
    )
    registry, _ = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        overrides={AnesthesiologistOutput: delayed},
    )
    limits = WorkflowLimits(max_turns_per_specialist=1, max_no_progress_turns=20)
    async with open_sqlite_checkpointer(tmp_path / "delay-limit.sqlite3") as saver:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-delay-limit"),
        )

    # v5: the reserved surgeon response still answers the first warning, but the
    # anesthesiologist's final-position turn can restate the delay; when no
    # surgeon budget remains, the run fails closed as an unanswered warning.
    assert result["termination_reason"] is TerminationReason.UNANSWERED_SAFETY_WARNING
    assert result["final_plan"].force_finalized is True
    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 2
    assert any(
        "Unanswered preoperative delay recommendation" in issue
        for issue in result["unresolved_issues"]
    )


@pytest.mark.asyncio
async def test_absolute_limit_still_forces_unanswered_safety_completion(
    tmp_path: Path,
) -> None:
    delayed = anesthesia_output().model_copy(
        update={
            "recommend_delay": True,
            "delay_reasons": ["Airway assessment is incomplete."],
        }
    )
    registry, _ = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        overrides={AnesthesiologistOutput: delayed},
    )
    limits = WorkflowLimits(
        max_turns_per_specialist=1,
        max_total_specialist_turns=5,
        max_absolute_specialist_turns=5,
        max_no_progress_turns=20,
    )
    async with open_sqlite_checkpointer(tmp_path / "delay-absolute.sqlite3") as saver:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=saver,
            limits=limits,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-delay-absolute"),
        )

    assert result["termination_reason"] is TerminationReason.UNANSWERED_SAFETY_WARNING
    assert result["speaker_counts"][AgentRole.SPINE_SURGEON.value] == 1
    assert result["final_plan"].force_finalized is True


@pytest.mark.asyncio
async def test_no_final_surgical_indication_clears_concrete_options(
    tmp_path: Path,
) -> None:
    conflicting = final_plan()
    conflicting = conflicting.model_copy(
        update={
            "surgical_indication": conflicting.surgical_indication.model_copy(
                update={"surgery_recommended": False, "indications": []}
            ),
            "recommended_surgery": conflicting.recommended_surgery.model_copy(
                update={
                    "procedure_name": "Conflicting moderator procedure",
                    "levels": ["C1-C2"],
                    "approach": "Anterior",
                }
            ),
            "alternatives": conflicting.alternatives.model_copy(
                update={"alternative_surgery": ["Conflicting alternative"]}
            ),
            "perioperative_risks": conflicting.perioperative_risks.model_copy(
                update={
                    "asa_classification": "Conflicting ASA",
                    "anesthesia_plan": "Conflicting anesthesia plan",
                }
            ),
            "final_consensus": conflicting.final_consensus.model_copy(
                update={"implementation_prerequisites": ["Moderator prerequisite"]}
            ),
        }
    )
    registry, _ = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        overrides={FinalPlan: conflicting},
    )
    async with open_sqlite_checkpointer(tmp_path / "authority.sqlite3") as saver:
        graph = create_complete_mdt_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-authority"),
        )

    plan = result["final_plan"]
    assert result["run_metadata"].finalization_fallback_used is False
    assert plan.surgical_indication.surgery_recommended is False
    assert plan.recommended_surgery.procedure_name == MISSING_RECORDS
    assert plan.recommended_surgery.levels == []
    assert plan.recommended_surgery.approach == MISSING_RECORDS
    assert plan.conditional_surgical_options == []
    assert plan.alternatives.alternative_surgery == []
    assert plan.perioperative_risks.asa_classification == "Conflicting ASA"
    assert plan.perioperative_risks.anesthesia_plan == "Conflicting anesthesia plan"


@pytest.mark.asyncio
async def test_initial_radiology_api_error_fails_closed(
    tmp_path: Path,
) -> None:
    registry, client = registry_with_client(moderator_responses=STANDARD_ROUTE)
    del client.responses[RadiologyReport]
    async with open_sqlite_checkpointer(tmp_path / "radiology-error.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        with pytest.raises(ModelInvocationError):
            await graph.ainvoke(
                opening_input(case_input()),
                config=thread_config("thread-radiology-error"),
            )


@pytest.mark.asyncio
async def test_unknown_finalizer_evidence_ids_are_removed_without_fallback(
    tmp_path: Path,
) -> None:
    invalid_plan = final_plan().model_copy(
        update={
            "diagnoses": [
                DiagnosisEvidence(
                    diagnosis="L4-L5 disc extrusion",
                    case_evidence=[],
                    neurologic_evidence=[],
                    imaging_evidence=[],
                    guideline_evidence_ids=["E999"],
                )
            ]
        }
    )
    registry, _ = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        overrides={FinalPlan: invalid_plan},
    )
    async with open_sqlite_checkpointer(tmp_path / "bad-evidence.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-bad-evidence"),
        )
    assert result["run_metadata"].finalization_fallback_used is False
    assert all(
        diagnosis.guideline_evidence_ids == []
        for diagnosis in result["final_plan"].diagnoses
    )


@pytest.mark.asyncio
async def test_unresolved_disagreement_is_preserved_without_fallback(tmp_path: Path) -> None:
    disagreement = Disagreement(
        agents=[AgentRole.SPINE_SURGEON, AgentRole.NEUROLOGIST],
        issue="The symptom-imaging concordance remains disputed.",
        positions=["Concordant", "Only partially concordant"],
    )
    final_decision = finalize().model_copy(
        update={
            "disagreements": [disagreement],
            "unresolved_issues": [disagreement.issue],
            "consensus_reached": False,
        }
    )
    registry, _ = registry_with_client(
        moderator_responses=[final_decision],
    )
    async with open_sqlite_checkpointer(tmp_path / "disagreement.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-disagreement"),
        )
    assert result["run_metadata"].finalization_fallback_used is False
    assert result["final_plan"].disagreements[0].resolved is False
    assert disagreement.issue in result["final_plan"].final_consensus.remaining_concerns
    assert result["final_plan"].final_consensus.has_unresolved_disagreement is True


@pytest.mark.asyncio
async def test_final_plan_integrity_failure_retries_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validator = graph_module.validate_final_plan_integrity
    validation_attempts = 0

    def fail_three_times(plan, state):
        nonlocal validation_attempts
        validation_attempts += 1
        if validation_attempts <= 3:
            raise FinalPlanIntegrityError(
                ModelFailureCategory.FINAL_PLAN_OMITTED_ISSUES
            )
        return original_validator(plan, state)

    monkeypatch.setattr(
        graph_module,
        "validate_final_plan_integrity",
        fail_three_times,
    )
    registry, client = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        max_retries=3,
    )
    async with open_sqlite_checkpointer(tmp_path / "final-retry.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-final-retry"),
        )

    final_calls = [
        call for call in client.calls if call["output_model"] is FinalPlan
    ]
    assert len(final_calls) == 4
    assert result["phase"] is RunPhase.COMPLETED
    assert isinstance(result["final_plan"], FinalPlan)
    assert result["run_metadata"].finalization_fallback_used is False
    failure = next(
        item
        for item in result["run_metadata"].model_failures
        if item.stage == "final_plan_generation"
    )
    assert failure.outcome == "recovered"
    assert failure.attempt_count == 4
    assert failure.max_attempts == 4
    assert failure.attempt_categories == ["FINAL_PLAN_OMITTED_ISSUES"] * 3


@pytest.mark.asyncio
async def test_final_plan_integrity_exhaustion_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validator = graph_module.validate_final_plan_integrity
    validation_attempts = 0

    def fail_model_attempts(plan, state):
        nonlocal validation_attempts
        validation_attempts += 1
        if validation_attempts <= 4:
            raise FinalPlanIntegrityError(
                ModelFailureCategory.FINAL_PLAN_DISAGREEMENT_STATUS
            )
        return original_validator(plan, state)

    monkeypatch.setattr(
        graph_module,
        "validate_final_plan_integrity",
        fail_model_attempts,
    )
    registry, client = registry_with_client(
        moderator_responses=STANDARD_ROUTE,
        max_retries=3,
    )
    async with open_sqlite_checkpointer(tmp_path / "final-fallback.sqlite3") as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        with pytest.raises(ModelStageError) as raised:
            await graph.ainvoke(
                opening_input(case_input()),
                config=thread_config("thread-final-fallback"),
            )
        snapshot = await graph.aget_state(
            thread_config("thread-final-fallback")
        )

    final_calls = [
        call for call in client.calls if call["output_model"] is FinalPlan
    ]
    assert len(final_calls) == 4
    assert raised.value.stage == "final_plan_generation"
    assert snapshot.values["final_plan"] is None
    assert snapshot.values["case_preop_md"]
    assert snapshot.values["chat_log"]


@pytest.mark.asyncio
async def test_completed_checkpoint_resume_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "complete-resume.sqlite3"
    config = thread_config("thread-complete-resume")
    registry, client = registry_with_client(moderator_responses=STANDARD_ROUTE)
    async with open_sqlite_checkpointer(database) as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            interrupt_after=["publish_final_plan"],
        )
        completed = await graph.ainvoke(opening_input(case_input()), config=config)
        assert completed["phase"] is RunPhase.COMPLETED
        calls_at_completion = len(client.calls)

    async with open_sqlite_checkpointer(database) as checkpointer:
        graph = create_complete_mdt_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        resumed = await graph.ainvoke(None, config=config)
        snapshot = await graph.aget_state(config)
    assert resumed["chat_log"] == completed["chat_log"]
    assert resumed["final_plan"] == completed["final_plan"]
    assert snapshot.values["final_plan"] == completed["final_plan"]
    assert len(client.calls) == calls_at_completion
