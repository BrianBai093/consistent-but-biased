import re
from pathlib import Path

import pytest

from config import Settings
from department.agents import (
    AnesthesiaAgent,
    DepartmentRegistry,
    DirectorAgent,
    NeurologyAgent,
    PhysiatristAgent,
    RadiologyAgent,
    ResidentAgent,
    ViceDirectorAgent,
)
from department.factory import create_department
from department.model_config import DEPARTMENT_ROLES, parse_override, resolve_role_models
from department.schemas import (
    AsaGrade,
    ConsensusOutcome,
    CompressionType,
    ConservativeStance,
    DirectorDecision,
    FamilyRequest,
    SurgicalIndication,
    DirectorRuling,
    ImagingFindings,
    LevelStatus,
    NEGATIVE_VALUES,
    RehabAssessment,
    RiskBand,
    SignalChange,
    StenosisGrade,
    SymptomImagingCorrelation,
    UNREADABLE_VALUES,
    normalize_level,
)
from exceptions import ConfigurationError
from schemas import AgentRole, DirectorStyle


def settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "openai-test-key",
        "deepseek_api_key": "deepseek-test-key",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def imaging(**overrides) -> ImagingFindings:
    values = {
        "levels": ["C4-5"],
        "level_status": LevelStatus.REPORTED,
        "compression_type": CompressionType.VENTRAL,
        "stenosis_grade": StenosisGrade.MODERATE,
        "signal_change": SignalChange.EQUIVOCAL,
        "correlation": SymptomImagingCorrelation.MODERATE,
        "public_message": "Ventral compression at C4-5.",
        "confidence": 0.6,
    }
    values.update(overrides)
    return ImagingFindings(**values)


# ------------------------------------------------------- role independence ---


def test_every_role_is_its_own_class_with_its_own_prompt():
    registry, _ = create_department(settings())

    agents = registry.all_agents()
    assert len(agents) == 7
    assert len({type(agent) for agent in agents}) == 7
    assert len({agent.prompt_filename for agent in agents}) == 7
    assert len({agent.role for agent in agents}) == 7


def test_every_role_gets_its_own_model_client():
    """Sharing one client would make "change this role's model" a shared edit."""

    registry, _ = create_department(settings())

    assert len({id(agent.client) for agent in registry.all_agents()}) == 7


def test_a_wrong_type_in_a_role_slot_is_rejected():
    registry, _ = create_department(settings())
    slots = {
        "resident": registry.resident,
        "radiologist": registry.radiologist,
        "neurologist": registry.neurologist,
        "anesthesiologist": registry.anesthesiologist,
        "physiatrist": registry.physiatrist,
        "vice_director": registry.vice_director,
        "director": registry.director,
    }

    with pytest.raises(TypeError):
        DepartmentRegistry(**{**slots, "director": registry.vice_director})


def test_one_object_cannot_fill_two_role_slots():
    """Every slot demands its own class, so reuse is a type error by construction."""

    registry, _ = create_department(settings())
    shared = registry.radiologist

    with pytest.raises(TypeError):
        DepartmentRegistry(
            resident=registry.resident,
            radiologist=shared,
            neurologist=shared,
            anesthesiologist=registry.anesthesiologist,
            physiatrist=registry.physiatrist,
            vice_director=registry.vice_director,
            director=registry.director,
        )


def test_each_role_declares_the_library_it_is_meant_to_consult():
    registry, _ = create_department(settings())

    for agent in registry.all_agents():
        assert agent.knowledge_base, f"{type(agent).__name__} declares no knowledge base"
        assert agent.knowledge_source is None  # retrieval is not built yet


# ------------------------------------------------------------ style prefix ---


def test_the_style_reaches_the_director_and_the_deputy_only():
    """Changing only the director would not reproduce a subordinate's self-censorship."""

    registry, _ = create_department(settings())
    styled = {registry.director, registry.vice_director}

    for agent in registry.all_agents():
        assert hasattr(agent, "styled_system_prompt") is (agent in styled)


def test_each_arm_injects_its_own_prefix_verbatim():
    registry, _ = create_department(settings())

    for agent in (registry.director, registry.vice_director):
        for style, filename in agent.style_prompt_filenames.items():
            expected = agent.load_department_prompt(filename)
            composed = agent.styled_system_prompt(style)
            assert composed.startswith(expected)
            assert composed.endswith(agent.system_prompt)


def test_the_deputy_reads_the_style_from_below_not_from_the_chair():
    """The director's rules are addressed to whoever rules.

    Handed to the deputy unchanged they produced a deputy that ruled: in the
    first real run it reported canvassing two junior colleagues and invented
    one of them, which is the director's prescribed behaviour, not its own.
    """

    registry, _ = create_department(settings())

    assert (
        registry.vice_director.style_prompt_filenames
        != registry.director.style_prompt_filenames
    )
    for style in DirectorStyle:
        deputy = registry.vice_director.style_prompts[style]
        assert deputy != registry.director.style_prompts[style]
        # Whatever else it says, it must not tell the deputy to rule or to
        # gather opinions from juniors.
        assert "you do not rule" in deputy.lower()
        assert "canvass juniors" in deputy.lower()


def test_the_arms_differ_only_in_the_prefix():
    """Incidental wording differences between arms would confound the result."""

    registry, _ = create_department(settings())
    prompts = {
        style: registry.director.styled_system_prompt(style)
        for style in DirectorStyle
    }

    assert len(set(prompts.values())) == len(DirectorStyle)
    for text in prompts.values():
        assert text.endswith(registry.director.system_prompt)


def test_the_control_arm_prescribes_no_conduct():
    """The neutral arm is the comparison; a rule in it would be a fourth style."""

    registry, _ = create_department(settings())
    neutral = registry.director.style_prompts[DirectorStyle.NEUTRAL]
    others = [
        registry.director.style_prompts[style]
        for style in DirectorStyle
        if style is not DirectorStyle.NEUTRAL
    ]
    assert "Rules of conduct" not in neutral and "Forbidden" not in neutral
    assert all("Rules of conduct" in text for text in others)
    assert len(neutral) < min(len(text) for text in others) / 2


def test_the_b_wording_changes_every_prefix_and_nothing_else():
    """M6: the same rules in other words, and only the prefix differs."""

    base = create_department(settings())[0]
    variant = create_department(settings(department_style_variant="b"))[0]
    for agent_a, agent_b in (
        (base.director, variant.director),
        (base.vice_director, variant.vice_director),
    ):
        assert agent_a.system_prompt == agent_b.system_prompt
        for style in DirectorStyle:
            assert agent_a.style_prompts[style] != agent_b.style_prompts[style]
    for style in DirectorStyle:
        deputy = variant.vice_director.style_prompts[style].lower()
        assert "you do not rule" in deputy and "canvass juniors" in deputy


#: Words that would tell a trait-only director which way to decide, or what to
#: do. Their absence is the whole point of the M9 ablation.
TREATMENT_CUES = re.compile(
    r"surg|operat|conservat|non-?operative|decompress|fusion|laminect|test|workup|"
    r"imaging|mri|family|famil|consent|rehabilit|anaesth|anesth|waste|delay|wait|"
    r"objection|junior|canvass|ask\b|rule\b|ruling",
    re.IGNORECASE,
)


def test_the_trait_wording_describes_character_and_prescribes_nothing():
    """M9: if the effect survives, it was not because the prompt told it to."""

    registry = create_department(settings(department_style_variant="t"))[0]
    base = create_department(settings())[0]
    for style in DirectorStyle:
        director = registry.director.style_prompts[style]
        deputy = registry.vice_director.style_prompts[style]
        if style is DirectorStyle.NEUTRAL:
            # The control is the same control, so its data can be shared.
            assert director == base.director.style_prompts[style]
            assert deputy == base.vice_director.style_prompts[style]
            continue
        assert "Rules of conduct" not in director and "Forbidden" not in director
        assert not TREATMENT_CUES.findall(director), TREATMENT_CUES.findall(director)
        # Below the boundary sentence, which every deputy prefix shares.
        description = deputy.split("You do not rule")[0]
        assert not TREATMENT_CUES.findall(description), TREATMENT_CUES.findall(description)
        assert "you do not rule" in deputy.lower() and "canvass juniors" in deputy.lower()


# ---------------------------------------------------------- model selection ---


def test_text_roles_share_one_model_until_one_is_deliberately_changed():
    _, assignments = create_department(settings())
    by_role = {assignment.role: assignment for assignment in assignments}

    text_roles = set(DEPARTMENT_ROLES) - {AgentRole.RADIOLOGIST}
    assert {by_role[role].model for role in text_roles} == {"deepseek-v4-pro"}
    assert len(assignments) == len(DEPARTMENT_ROLES)


def test_the_radiologist_follows_the_image_provider():
    """The stock DeepSeek endpoint is text-only and rejects image parts."""

    _, assignments = create_department(settings())
    by_role = {assignment.role: assignment for assignment in assignments}

    assert by_role[AgentRole.RADIOLOGIST].provider == "openai"


def test_the_radiologist_keeps_the_radiology_timeout_budget():
    registry, _ = create_department(settings())

    assert registry.radiologist.client.timeout_seconds == 300
    assert registry.radiologist.client.max_retries == 1


def test_a_single_role_can_be_moved_to_another_model_and_is_recorded():
    _, assignments = create_department(
        settings(department_role_models={"director": "openai:gpt-5.6-sol"})
    )

    by_role = {assignment.role: assignment for assignment in assignments}
    assert by_role[AgentRole.DIRECTOR].provider == "openai"
    assert by_role[AgentRole.DIRECTOR].model == "gpt-5.6-sol"
    assert by_role[AgentRole.RESIDENT].provider == "deepseek"


def test_a_misspelled_role_override_fails_loudly():
    """Silently ignoring it would leave the role on the default model unnoticed."""

    with pytest.raises(ConfigurationError):
        resolve_role_models(settings(department_role_models={"direcotr": "openai:x"}))


def test_a_role_outside_this_workflow_is_rejected():
    with pytest.raises(ConfigurationError):
        resolve_role_models(
            settings(department_role_models={"spine_surgeon": "openai:x"})
        )


@pytest.mark.parametrize("value", ["openai", "openai:", ":model", "grok:x"])
def test_a_malformed_override_is_rejected(value: str):
    with pytest.raises(ConfigurationError):
        parse_override(value)


# --------------------------------------------------------- imaging schema ---


def test_unreadable_and_negative_are_never_the_same_value():
    """TX (cannot be assessed) and T0 (no evidence) are distinct codes."""

    for enum in (LevelStatus, CompressionType, StenosisGrade, SignalChange):
        values = {member.value for member in enum}
        assert values & UNREADABLE_VALUES, f"{enum.__name__} cannot say 'unreadable'"
        assert values & NEGATIVE_VALUES, f"{enum.__name__} cannot say 'negative'"
        assert not UNREADABLE_VALUES & NEGATIVE_VALUES


def test_ambiguous_correlation_is_worded_apart_from_unreadable_imaging():
    """Ambiguity here is the object of study, not a defect in the films."""

    values = {member.value for member in SymptomImagingCorrelation}
    assert "indeterminate" in values
    assert not values & UNREADABLE_VALUES


def test_faint_cord_signal_has_its_own_grade():
    assert SignalChange.EQUIVOCAL.value not in NEGATIVE_VALUES | UNREADABLE_VALUES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C3/4", "C3-4"),
        ("C3-C4", "C3-4"),
        ("c3-4", "C3-4"),
        (" L4/5 ", "L4-5"),
        ("T12-L1", "T12-L1"),
        ("L5/S1", "L5-S1"),
        ("C7-T1", "C7-T1"),
    ],
)
def test_levels_are_stored_in_one_spelling(raw: str, expected: str):
    assert normalize_level(raw) == expected


@pytest.mark.parametrize("raw", ["C34", "L5", "", "cervical"])
def test_an_unrecognized_level_is_rejected_rather_than_guessed(raw: str):
    with pytest.raises(ValueError):
        normalize_level(raw)


def test_levels_are_normalized_on_the_way_into_the_report():
    assert imaging(levels=["C3/4", "C4-C5"]).levels == ["C3-4", "C4-5"]


def test_reported_levels_and_level_status_cannot_contradict_each_other():
    with pytest.raises(ValueError):
        imaging(levels=[], level_status=LevelStatus.REPORTED)
    with pytest.raises(ValueError):
        imaging(levels=["C4-5"], level_status=LevelStatus.NONDIAGNOSTIC)


def test_an_unreadable_study_is_expressible_in_full():
    report = imaging(
        levels=[],
        level_status=LevelStatus.NONDIAGNOSTIC,
        compression_type=CompressionType.NONDIAGNOSTIC,
        stenosis_grade=StenosisGrade.NONDIAGNOSTIC,
        signal_change=SignalChange.NONDIAGNOSTIC,
        correlation=SymptomImagingCorrelation.INDETERMINATE,
    )

    assert report.level_status is LevelStatus.NONDIAGNOSTIC


# ------------------------------------------------------------ role outputs ---


def test_an_unassessable_patient_does_not_force_an_invented_asa_grade():
    assert AsaGrade.UNASSESSABLE.value == "unassessable"
    assert RiskBand.UNASSESSABLE.value == "unassessable"


def rehab(**overrides) -> RehabAssessment:
    values = {
        "conservative_success_rate": 65,
        "proposed_duration": "3-6 months",
        "preconditions_for_failure": ["progressive myelopathy"],
        "stance": ConservativeStance.HOLD,
        "public_message": "Conservative management first.",
    }
    values.update(overrides)
    return RehabAssessment(**values)


def test_the_physiatrist_cannot_move_without_naming_what_moved_it():
    assert rehab().stance is ConservativeStance.HOLD

    with pytest.raises(ValueError):
        rehab(stance=ConservativeStance.CONCEDE)

    conceded = rehab(
        stance=ConservativeStance.CONCEDE,
        persuaded_by=["documented progressive weakness"],
    )
    assert conceded.persuaded_by == ["documented progressive weakness"]


def ruling(**overrides) -> DirectorRuling:
    values = {
        "decision": DirectorDecision.NOT_YET_DECIDED,
        "surgical_indication": SurgicalIndication.UNCERTAIN,
        "delegated_tasks": [],
        "if_delay": False,
        "has_contingency_plan": False,
        "consensus": ConsensusOutcome.UNCLEAR,
        "addressed_roles": [],
        "ordered_tests": [],
        "family_requests": [],
        "public_message": "Let us hear anesthesia again.",
    }
    values.update(overrides)
    return DirectorRuling(**values)


def test_an_undecided_director_carries_no_final_decision():
    assert ruling().final_decision is None

    with pytest.raises(ValueError):
        ruling(final_decision="operate")


def test_a_ruling_must_say_what_was_decided():
    with pytest.raises(ValueError):
        ruling(decision=DirectorDecision.CONSERVATIVE)

    assert ruling(
        decision=DirectorDecision.CONSERVATIVE,
        final_decision="Conservative management for three months.",
    ).final_decision


def test_an_operative_ruling_requires_a_plan():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.OPERATE,
            surgical_indication=SurgicalIndication.YES,
            final_decision="Operate.",
        )

    assert ruling(
        decision=DirectorDecision.OPERATE,
        surgical_indication=SurgicalIndication.YES,
        final_decision="Operate.",
        surgical_plan="C4-5 anterior decompression and fusion.",
    ).surgical_plan


def test_an_operative_ruling_cannot_say_surgery_is_not_indicated():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.OPERATE,
            surgical_indication=SurgicalIndication.NO,
            final_decision="Operate.",
            surgical_plan="ACDF",
        )


def test_seeking_more_evidence_requires_saying_where_it_would_land():
    """Otherwise "run another test" is a free exit from deciding anything.

    Ordering a test is a legitimate ruling, and it is also the cheapest way to
    satisfy "produce a decision". Naming the branch keeps the outcome variable
    defined for a corpus where most meetings end this way.
    """

    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.ADDITIONAL_WORKUP,
            final_decision="Repeat the MRI first.",
            ordered_tests=["dynamic cervical MRI"],
        )

    conditional = ruling(
        decision=DirectorDecision.ADDITIONAL_WORKUP,
        final_decision="Repeat the MRI first.",
        ordered_tests=["dynamic cervical MRI"],
        workup_would_change_decision_to=DirectorDecision.OPERATE,
    )
    assert conditional.workup_would_change_decision_to is DirectorDecision.OPERATE

    # And a branch belongs only to a ruling that actually seeks evidence.
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.CONSERVATIVE,
            final_decision="Three months of conservative care.",
            workup_would_change_decision_to=DirectorDecision.OPERATE,
        )


def test_a_delay_cannot_be_attached_to_a_decision_that_proceeds_now():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.OPERATE,
            final_decision="Operate.",
            surgical_plan="ACDF",
            if_delay=True,
            reason_for_delay="family undecided",
        )


def test_a_delay_requires_a_stated_reason():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.DEFER,
            final_decision="Defer pending family discussion.",
            if_delay=True,
        )


def test_the_director_records_what_it_actually_did():
    """The observer must never have to infer an action from prose alone."""

    recorded = ruling(
        decision=DirectorDecision.ADDITIONAL_WORKUP,
        workup_would_change_decision_to=DirectorDecision.OPERATE,
        final_decision="Repeat dynamic MRI first.",
        if_delay=True,
        reason_for_delay="imaging is not diagnostic",
        addressed_roles=[AgentRole.VICE_DIRECTOR, AgentRole.PHYSIATRIST],
        ordered_tests=["dynamic cervical MRI"],
        family_requests=[FamilyRequest.FAMILY_CHOOSES],
    )

    assert recorded.ordered_tests == ["dynamic cervical MRI"]
    assert recorded.family_requests == [FamilyRequest.FAMILY_CHOOSES]
    assert AgentRole.PHYSIATRIST in recorded.addressed_roles


def test_the_new_roles_exist_without_disturbing_the_v5_roles():
    assert AgentRole.DIRECTOR.value == "director"
    assert AgentRole.SPINE_SURGEON.value == "spine_surgeon"
    assert {agent.role for agent in create_department(settings())[0].all_agents()} == {
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.VICE_DIRECTOR,
        AgentRole.DIRECTOR,
    }


def test_the_agent_classes_are_importable_independently():
    for agent_type in (
        ResidentAgent,
        RadiologyAgent,
        NeurologyAgent,
        AnesthesiaAgent,
        PhysiatristAgent,
        ViceDirectorAgent,
        DirectorAgent,
    ):
        assert agent_type.prompt_filename.endswith(".md")


def test_every_department_type_is_registered_for_checkpoints():
    """An unlisted type is not rejected on load -- it comes back as a plain dict.

    An identity check against an enum member would then quietly start returning
    False after a resume, which is why this is enforced rather than remembered.
    The scan covers the whole package, not just ``department.schemas``, so that
    a type added to another module is caught too.
    """

    import inspect
    import pkgutil
    from enum import Enum
    from importlib import import_module

    from pydantic import BaseModel

    import department
    from state import CHECKPOINT_SCHEMA_ALLOWLIST

    registered = {
        (module, name)
        for module, name in CHECKPOINT_SCHEMA_ALLOWLIST
        if module.startswith("department")
    }
    modules = [department] + [
        import_module(info.name)
        for info in pkgutil.walk_packages(department.__path__, "department.")
    ]
    defined = {
        (value.__module__, name)
        for module in modules
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and value.__module__.startswith("department.")
        and issubclass(value, (Enum, BaseModel))
    }

    assert defined - registered == set(), "unregistered department types"


def test_the_shared_baseline_survives_a_checkpoint_round_trip():
    """The three arms load one baseline; as a dict it would lose its accessors."""

    from state import CHECKPOINT_SCHEMA_ALLOWLIST

    assert ("schemas", "CaseBaseline") in CHECKPOINT_SCHEMA_ALLOWLIST


def test_ordering_a_workup_ruling_must_name_the_tests():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.ADDITIONAL_WORKUP,
            final_decision="Get more imaging.",
            ordered_tests=[],
        )


def test_a_conservative_ruling_cannot_carry_a_surgical_plan():
    with pytest.raises(ValueError):
        ruling(
            decision=DirectorDecision.CONSERVATIVE,
            final_decision="Conservative management.",
            surgical_plan="ACDF",
        )


def test_the_director_cannot_count_itself_as_a_junior_it_addressed():
    with pytest.raises(ValueError):
        ruling(addressed_roles=[AgentRole.DIRECTOR])
    with pytest.raises(ValueError):
        ruling(addressed_roles=[AgentRole.SPINE_SURGEON])


def test_addressing_the_same_junior_twice_counts_once():
    recorded = ruling(
        addressed_roles=[AgentRole.VICE_DIRECTOR, AgentRole.VICE_DIRECTOR]
    )

    assert recorded.addressed_roles == [AgentRole.VICE_DIRECTOR]


def test_self_reported_behaviour_cannot_be_omitted():
    """The default text provider is given the schema as prose in JSON-object
    mode, so an omitted field would silently become "I did not do that" and bias
    every declared metric downwards, differently per provider."""

    required = {
        "delegated_tasks",
        "if_delay",
        "has_contingency_plan",
        "consensus",
        "addressed_roles",
        "ordered_tests",
        "family_requests",
    }

    for name in required:
        assert DirectorRuling.model_fields[name].is_required(), name


def test_the_opening_conservative_position_needs_no_persuasion():
    """In the frozen baseline nothing has argued with the physiatrist yet."""

    assert rehab(stance=ConservativeStance.CONDITIONAL).persuaded_by == []


def test_a_blank_reason_does_not_count_as_being_persuaded():
    with pytest.raises(ValueError):
        rehab(stance=ConservativeStance.CONCEDE, persuaded_by=["   "])


def test_batch_bounds_are_validated(monkeypatch):
    from department.agents import RadiologyAgent
    from services.llm_client import FakeModelClient

    for bad in ({"batch_max_images": 0}, {"batch_max_bytes": 0}, {"batch_concurrency": 0}):
        with pytest.raises(ValueError):
            RadiologyAgent(FakeModelClient({}), **bad)


def test_every_prompt_states_the_contract_its_schema_enforces():
    """A rule the prompt never states is a retry loop that costs money.

    The default text provider gets the schema as prose in JSON-object mode, so
    nothing stops it producing a value the validators reject. When that happens
    the run pays for the whole prompt, the schema and the transcript again, and
    it happens most often in the arm whose style pushes toward the rejected
    shape -- which makes the failure rate a function of the manipulated
    variable rather than of chance.
    """

    import enum
    import typing

    from pydantic import BaseModel

    from department.agents import (
        AnesthesiaAgent,
        DirectorAgent,
        NeurologyAgent,
        PhysiatristAgent,
        RadiologyAgent,
        ResidentAgent,
        ViceDirectorAgent,
    )
    from services.llm_client import FakeModelClient

    def enum_values(annotation) -> set[str]:
        found: set[str] = set()
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            # Role names are constrained to the people in the room, which the
            # director's prompt states as a list; the enum itself is wider.
            if annotation is not AgentRole:
                found.update(str(member.value) for member in annotation)
            return found
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            # A nested object's keys and values are part of the contract too.
            for name, nested in annotation.model_fields.items():
                found.add(name)
                found |= enum_values(nested.annotation)
            return found
        for argument in typing.get_args(annotation):
            found |= enum_values(argument)
        return found

    missing: list[str] = []
    for agent_type in (
        ResidentAgent,
        RadiologyAgent,
        NeurologyAgent,
        AnesthesiaAgent,
        PhysiatristAgent,
        ViceDirectorAgent,
        DirectorAgent,
    ):
        agent = agent_type(FakeModelClient({}))
        prompt = agent.system_prompt
        for name, field in agent.output_model.model_fields.items():
            if name not in prompt:
                missing.append(f"{agent_type.__name__}: field {name}")
            for value in enum_values(field.annotation):
                if value not in prompt:
                    missing.append(f"{agent_type.__name__}: {name} value {value!r}")

    assert missing == []


def test_the_director_is_told_who_is_in_the_room():
    """``addressed_roles`` advertises ten roles and accepts six."""

    from department.agents import DirectorAgent
    from services.llm_client import FakeModelClient

    prompt = DirectorAgent(FakeModelClient({})).system_prompt
    for role in (
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.VICE_DIRECTOR,
    ):
        assert f"``{role.value}``" in prompt
    # Roles the validator rejects. "director" is not checked this way because the
    # word names the reader; the prompt rules that out in words instead.
    for absent in (
        AgentRole.MODERATOR,
        AgentRole.SPINE_SURGEON,
        AgentRole.PERIOPERATIVE_PHYSICIAN,
    ):
        assert absent.value not in prompt
    assert "name yourself" in prompt


@pytest.mark.asyncio
async def test_the_schema_sent_to_a_text_model_is_actually_json():
    """The sentence says "this JSON Schema"; it has to be one.

    A Python object interpolated into the prompt writes single quotes and
    ``False``/``None``, which no JSON parser accepts. Six of the seven roles go
    down this path, in a mode where the schema is the only description of the
    contract they get.
    """

    import json

    from config import ProviderConfig
    from department.schemas import DirectorRuling
    from services.llm_client import DeepSeekTextClient

    client = DeepSeekTextClient(
        ProviderConfig(
            api_key="unused-in-this-test",
            model="fake",
            base_url="https://example.invalid",
            timeout_seconds=1,
            max_retries=0,
        )
    )
    sent: dict = {}

    class _Capture:
        async def create(self, **kwargs):
            sent.update(kwargs)
            raise RuntimeError("stop before the network")

    client._client.chat.completions = _Capture()
    with pytest.raises(Exception):
        await client.generate_structured(
            system_prompt="s",
            user_prompt="u",
            output_model=DirectorRuling,
        )

    body = sent["messages"][1]["content"]
    _, _, schema_text = body.partition("JSON Schema:\n")
    assert json.loads(schema_text)["properties"]["decision"]


@pytest.mark.asyncio
async def test_the_films_are_read_in_batches_and_every_reading_is_kept():
    """The radiologist is the only role that sees the images.

    A consolidation that drops or softens what a batch reported is otherwise
    indistinguishable from one that never saw it, and the films cost the most
    of anything in the run to read again.
    """

    from department.agents import RadiologyAgent
    from department.schemas import ImagingFindings, LevelStatus
    from schemas import CaseImage, Modality
    from services.llm_client import FakeModelClient
    from tests.department_factories import imaging_findings

    first = imaging_findings(levels=["C3-4"], public_message="Batch one.")
    second = imaging_findings(
        levels=[],
        level_status=LevelStatus.NONDIAGNOSTIC,
        public_message="Batch two was unreadable.",
    )
    merged = imaging_findings(levels=["C3-4"], public_message="Merged.")
    client = FakeModelClient({ImagingFindings: [first, second, merged]})
    agent = RadiologyAgent(client, batch_max_images=1)

    result = await agent.run(
        case_summary_md="Summary.",
        images=[
            CaseImage(image_id="a", modality=Modality.MRI, path_or_url="/tmp/a.png"),
            CaseImage(image_id="b", modality=Modality.MRI, path_or_url="/tmp/b.png"),
        ],
    )

    assert result.output.public_message == "Merged."
    assert [item.public_message for item in result.intermediates] == [
        "Batch one.",
        "Batch two was unreadable.",
    ]


def test_a_red_flag_has_to_say_what_it_changes():
    """A list of equal-weight doubts can only be answered with another test.

    Across four real cases the neurologist produced five or six ungraded flags
    every time, and six rulings out of nine ordered imaging against them.
    """

    import pytest as _pytest

    from department.schemas import (
        FlagRelevance,
        LocalizationMatch,
        NeurologyAssessment,
        RedFlag,
    )

    with _pytest.raises(ValueError):
        RedFlag(finding="Absent ankle jerks.", relevance=FlagRelevance.COEXISTING)

    graded = NeurologyAssessment(
        localization="Mid-cervical cord.",
        match_level=LocalizationMatch.MEDIUM,
        red_flags=[
            RedFlag(
                finding="Perineal sensory loss.",
                relevance=FlagRelevance.CHANGES_PLAN,
                rationale="Not explained by cervical compression; image the conus.",
            ),
            RedFlag(
                finding="Absent ankle jerks.",
                relevance=FlagRelevance.COEXISTING,
                rationale="A length-dependent neuropathy beside the myelopathy.",
            ),
        ],
        public_message="Two flags, graded.",
    )

    assert [flag.relevance for flag in graded.red_flags] == [
        FlagRelevance.CHANGES_PLAN,
        FlagRelevance.COEXISTING,
    ]


@pytest.mark.asyncio
async def test_only_the_negotiation_turns_read_the_new_persuasion_standard():
    """Protocol dept-v2 changes what can move the advocate, and nothing else.

    Its opening is frozen in the baseline all three arms share, so the opening
    prompt must stay exactly what produced it; only the turns taken after it
    read the added instruction.
    """

    from department.agents import PhysiatristAgent
    from department.agents.physiatrist import NEGOTIATION_PROMPT_FILENAME
    from department.schemas import RehabAssessment
    from services.llm_client import FakeModelClient
    from tests.department_factories import (
        imaging_findings,
        neurology_assessment,
        rehab_assessment,
        resident_report,
    )

    client = FakeModelClient({RehabAssessment: rehab_assessment()})
    agent = PhysiatristAgent(client)
    opening_file = (
        Path(__file__).resolve().parent.parent / "department" / "prompts" / "physiatrist.md"
    ).read_text(encoding="utf-8").strip()

    await agent.run(
        resident_report=resident_report(),
        imaging=imaging_findings(),
        neurology=neurology_assessment(),
    )
    await agent.respond(prior=rehab_assessment(), chat_log=[], round_number=1)

    opening, response = (call["system_prompt"] for call in client.calls)
    assert opening == opening_file
    assert response.startswith(opening_file)
    assert response.endswith(agent.load_department_prompt(NEGOTIATION_PROMPT_FILENAME))
    assert "already in the record" in response
    assert "already in the record" not in opening


def test_the_director_is_never_asked_to_name_its_own_conduct_as_a_fault():
    """A question that names a fault gets denied.

    Under dept-v1 the director was asked whether it had transferred liability to
    the family. It said no in all nine clean rulings, including one that
    arranged for the family to decide in the same breath. It now reports what it
    asked for, and the observer judges what that amounts to.
    """

    from department.agents import DirectorAgent
    from services.llm_client import FakeModelClient

    prompt = DirectorAgent(FakeModelClient({})).system_prompt.lower()

    # "Core responsibilities" is every prompt's section heading and is not the
    # loaded use; the phrases below are what dept-v1 asked and dept-v2 removed.
    for loaded in (
        "liabil",
        "责任",
        "blame",
        "moved the decision onto",
        "responsibility for its consequences",
    ):
        assert loaded not in prompt, loaded
