from schemas import (
    AgentRole,
    DefinitiveSurgeryStatus,
    PlanPositionAction,
    SurgeryFieldStatus,
)
from services.plan_convergence import (
    CRITICAL_OPERATIVE_FIELDS,
    apply_position,
    compute_field_matrix,
    derive_definitive_status,
    field_oppositions,
    has_positioned,
    positions_for_option,
    project_plan_draft,
    render_plan_draft,
    unpositioned_roles,
)
from tests.factories import position, spine_output, surgical_option


def test_draft_projects_surgeon_options_with_version_one() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=7)

    assert draft is not None
    assert draft.version == 1
    assert draft.source_contribution_sequence == 7
    assert draft.surgical_indication is True
    assert [option.option_id for option in draft.options] == ["option-1"]
    assert draft.preferred_option_id == "option-1"
    assert draft.approach_comparison is not None


def test_draft_without_indication_is_none() -> None:
    surgeon = spine_output().model_copy(update={"surgical_indication": False})

    assert project_plan_draft(surgeon, source_sequence=1) is None


def test_revision_resets_positions_when_option_content_changes() -> None:
    first = project_plan_draft(spine_output(), source_sequence=1)
    attached = apply_position(
        first,
        position(role=AgentRole.NEUROLOGIST, draft_version=1, message_sequence=3),
    )
    assert len(positions_for_option(attached, "option-1")) == 1

    revised_option = surgical_option().model_copy(
        update={"procedure_name": "Revised procedure"}
    )
    revised = spine_output().model_copy(update={"operative_options": [revised_option]})
    draft_v2 = project_plan_draft(revised, source_sequence=5, previous=attached)

    assert draft_v2.version == 2
    assert positions_for_option(draft_v2, "option-1") == []


def test_unchanged_options_carry_positions_to_next_version() -> None:
    first = project_plan_draft(spine_output(), source_sequence=1)
    attached = apply_position(
        first,
        position(role=AgentRole.NEUROLOGIST, draft_version=1, message_sequence=3),
    )
    same = spine_output().model_copy(
        update={"public_message": "Repeated opinion without option changes."}
    )
    draft_v2 = project_plan_draft(same, source_sequence=5, previous=attached)

    assert draft_v2.version == 2
    carried = positions_for_option(draft_v2, "option-1")
    assert len(carried) == 1
    assert carried[0].draft_version == 2


def test_apply_position_ignores_stale_version_and_unknown_option() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)

    stale = apply_position(
        draft,
        position(
            role=AgentRole.NEUROLOGIST,
            option_id="option-1",
            draft_version=99,
            message_sequence=3,
        ),
    )
    unknown = apply_position(
        draft,
        position(
            role=AgentRole.NEUROLOGIST,
            option_id="option-missing",
            draft_version=1,
            message_sequence=3,
        ),
    )

    assert positions_for_option(stale, "option-1") == []
    assert positions_for_option(unknown, "option-1") == []


def test_apply_position_replaces_same_role_on_same_option() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    first = apply_position(
        draft,
        position(role=AgentRole.NEUROLOGIST, draft_version=1, message_sequence=3),
    )
    second = apply_position(
        first,
        position(
            role=AgentRole.NEUROLOGIST,
            draft_version=1,
            message_sequence=4,
            action=PlanPositionAction.OPPOSE,
            target_fields=["/recommended_surgery/levels"],
            reservations=["Levels are not localized on the supplied images."],
        ),
    )

    positions = positions_for_option(second, "option-1")
    assert len(positions) == 1
    assert positions[0].message_sequence == 4
    assert positions[0].action is PlanPositionAction.OPPOSE


def test_field_matrix_reports_stances_per_field() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    draft = apply_position(
        draft,
        position(
            role=AgentRole.NEUROLOGIST,
            draft_version=1,
            message_sequence=3,
            action=PlanPositionAction.OPPOSE,
            target_fields=["/recommended_surgery/levels"],
            reservations=["Unlocalized levels."],
        ),
    )
    draft = apply_position(
        draft,
        position(
            role=AgentRole.ANESTHESIOLOGIST,
            draft_version=1,
            message_sequence=4,
            action=PlanPositionAction.SUPPORT_WITH_RESERVATIONS,
            target_fields=["/recommended_surgery/fusion_or_fixation"],
            reservations=["Confirm blood-loss planning for fusion."],
        ),
    )

    matrix = compute_field_matrix(draft)
    levels = next(
        row
        for row in matrix
        if row["field"] == "/recommended_surgery/levels"
        and row["option_id"] == "option-1"
    )
    fusion = next(
        row
        for row in matrix
        if row["field"] == "/recommended_surgery/fusion_or_fixation"
        and row["option_id"] == "option-1"
    )

    assert [item["role"] for item in levels["oppositions"]] == ["neurologist"]
    assert [item["role"] for item in fusion["supports_with_reservations"]] == [
        "anesthesiologist"
    ]
    oppositions = field_oppositions(
        draft, "option-1", "/recommended_surgery/levels"
    )
    assert len(oppositions) == 1


def test_position_coverage_helpers() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    draft = apply_position(
        draft,
        position(role=AgentRole.NEUROLOGIST, draft_version=1, message_sequence=3),
    )

    assert has_positioned(AgentRole.NEUROLOGIST, draft) is True
    assert has_positioned(AgentRole.ANESTHESIOLOGIST, draft) is False
    assert unpositioned_roles(
        draft,
        [
            AgentRole.NEUROLOGIST,
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PERIOPERATIVE_PHYSICIAN,
        ],
    ) == [AgentRole.ANESTHESIOLOGIST, AgentRole.PERIOPERATIVE_PHYSICIAN]


def test_render_plan_draft_is_deterministic_and_free_of_paths() -> None:
    draft = project_plan_draft(spine_output(), source_sequence=1)
    draft = apply_position(
        draft,
        position(role=AgentRole.NEUROLOGIST, draft_version=1, message_sequence=3),
    )

    first = render_plan_draft(draft)
    second = render_plan_draft(draft)

    assert first == second
    assert "version 1" in first
    assert "option-1" in first
    assert "neurologist" in first
    assert ".data" not in first
    assert "https://" not in first


def test_derive_definitive_status_from_field_statuses() -> None:
    from tests.factories import final_plan

    plan = final_plan()
    surgery = plan.recommended_surgery

    assert (
        derive_definitive_status(surgery)
        is DefinitiveSurgeryStatus.WITHHELD
    )
    all_established = surgery.model_copy(
        update={
            "procedure_status": SurgeryFieldStatus.ESTABLISHED,
            "levels_status": SurgeryFieldStatus.ESTABLISHED,
            "approach_status": SurgeryFieldStatus.ESTABLISHED,
            "decompression_status": SurgeryFieldStatus.ESTABLISHED,
            "fusion_fixation_status": SurgeryFieldStatus.ESTABLISHED,
        }
    )
    assert (
        derive_definitive_status(all_established)
        is DefinitiveSurgeryStatus.ESTABLISHED
    )
    partial = all_established.model_copy(
        update={"levels_status": SurgeryFieldStatus.CONDITIONAL}
    )
    assert derive_definitive_status(partial) is DefinitiveSurgeryStatus.PARTIAL


def test_critical_operative_fields_cover_all_recommended_surgery_fields() -> None:
    assert len(CRITICAL_OPERATIVE_FIELDS) == 5
    assert "/recommended_surgery/fusion_or_fixation" in CRITICAL_OPERATIVE_FIELDS


def test_draft_and_field_statuses_round_trip_through_checkpoint_serializer() -> None:
    from schemas import FinalPlanReview
    from services.final_plan_review import apply_field_level_status
    from state import create_checkpoint_serializer
    from tests.factories import final_plan, radiology_report

    review = FinalPlanReview(
        verdict="pass",
        definitive_surgery_allowed=True,
        established_field_paths=list(CRITICAL_OPERATIVE_FIELDS),
    )
    draft = project_plan_draft(spine_output(), source_sequence=1)
    draft = apply_position(
        draft,
        position(
            role=AgentRole.NEUROLOGIST,
            draft_version=1,
            message_sequence=3,
        ),
    )
    plan = apply_field_level_status(
        final_plan(),
        review=review,
        report=radiology_report(),
        working_plan=draft,
    )
    serializer = create_checkpoint_serializer()

    restored_plan = serializer.loads_typed(serializer.dumps_typed(plan))
    restored_draft = serializer.loads_typed(serializer.dumps_typed(draft))

    assert restored_plan.recommended_surgery.procedure_status is SurgeryFieldStatus.ESTABLISHED
    assert restored_draft.version == 1
    assert len(restored_draft.positions) == 1
