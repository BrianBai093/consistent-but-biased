"""Amendment A4: one role carries the style, or the director rules alone.

The main design changes the director's and the deputy's prefix together, and
measures the style after a negotiation. These arms take the pair apart and take
the negotiation away. Each must change only that, and the main arms must not
change at all.
"""

import json
from pathlib import Path

import pytest

from department.graph import create_department_graph, department_input, thread_config
from department.observer import ObserverMetrics, observe
from department.schemas import (
    DirectorRuling,
    ImagingFindings,
    RehabAssessment,
    ViceDirectorOpinion,
)
from department.state import TerminationCause, arm_label
from schemas import AgentRole, DirectorStyle
from services.baseline_store import BaselineStore
from services.checkpoint import open_sqlite_checkpointer
from services.llm_client import FakeModelClient
from tests.department_factories import case_sources, department_with_client, undecided
from tests.test_text_case import REPORT, filtered
from services.case_report_filter import FilteredCaseReport
from tools import run_text_case_experiment as runner
from tools.run_department_experiment import baseline_prefix, comparison, role_styles


def graph_input(tmp_path: Path, style: DirectorStyle, **overrides) -> dict:
    values = {
        "case_id": "case-dept",
        "director_style": style,
        "artifact_dir": str(tmp_path / "derived"),
        "case_preop_md": "<!-- source: case-page-001 -->\n\nHistory.",
        "tests_preop_md": "<!-- source: test-page-001 -->\n\nLabs.",
        "case_summary_md": "Cervical myelopathy.",
        **case_sources(tmp_path),
    }
    values.update(overrides)
    return department_input(**values)


async def run(
    tmp_path: Path,
    style: DirectorStyle,
    *,
    store: BaselineStore | None = None,
    thread: str = "one",
    responses: dict | None = None,
    **overrides,
):
    registry, client = department_with_client(responses)
    async with open_sqlite_checkpointer(tmp_path / f"{thread}.sqlite3") as saver:
        graph = create_department_graph(
            registry=registry, checkpointer=saver, baseline_store=store
        )
        result = await graph.ainvoke(
            graph_input(tmp_path, style, **overrides), config=thread_config(thread)
        )
    return result, client, registry


def calls_for(client, model) -> list[dict]:
    return [call for call in client.calls if call["output_model"] is model]


# ------------------------------------------------------- the main arms stay ---


@pytest.mark.parametrize("style", list(DirectorStyle))
async def test_a_main_arm_reads_exactly_what_it_read_before(tmp_path: Path, style):
    """Naming the deputy's style explicitly changes nothing anyone is sent."""

    _, implicit, _ = await run(tmp_path / "a", style)
    result, explicit, _ = await run(tmp_path / "b", style, deputy_style=style)

    def sent(client):
        return [(call["system_prompt"], call["user_prompt"]) for call in client.calls]

    # The source files sit in different folders; nothing in a prompt names them.
    assert sent(implicit) == sent(explicit)
    assert result["deputy_style"] is style
    assert result["negotiation"] is True
    assert arm_label(result) == style.value
    assert f"case-dept:{style.value}:resident:1" in {
        message.message_id for message in result["chat_log"]
    }


# -------------------------------------------------- one role carries it ---


async def test_a_chair_only_arm_gives_the_deputy_the_neutral_prefix(tmp_path: Path):
    _, client, registry = await run(
        tmp_path, DirectorStyle.AUTOCRATIC, deputy_style=DirectorStyle.NEUTRAL
    )
    director = registry.director.style_prompts
    deputy = registry.vice_director.style_prompts

    (chair_call,) = calls_for(client, DirectorRuling)
    (deputy_call,) = calls_for(client, ViceDirectorOpinion)
    assert chair_call["system_prompt"].startswith(director[DirectorStyle.AUTOCRATIC])
    assert deputy_call["system_prompt"].startswith(deputy[DirectorStyle.NEUTRAL])
    for call in client.calls:
        assert deputy[DirectorStyle.AUTOCRATIC] not in call["system_prompt"]


async def test_a_deputy_only_arm_gives_the_director_the_neutral_prefix(tmp_path: Path):
    result, client, registry = await run(
        tmp_path, DirectorStyle.NEUTRAL, deputy_style=DirectorStyle.AVOIDANT
    )
    director = registry.director.style_prompts
    deputy = registry.vice_director.style_prompts

    (chair_call,) = calls_for(client, DirectorRuling)
    (deputy_call,) = calls_for(client, ViceDirectorOpinion)
    assert chair_call["system_prompt"].startswith(director[DirectorStyle.NEUTRAL])
    assert deputy_call["system_prompt"].startswith(deputy[DirectorStyle.AVOIDANT])
    for call in client.calls:
        assert director[DirectorStyle.AVOIDANT] not in call["system_prompt"]
    metrics = observe(result)
    assert (metrics.director_style, metrics.deputy_style) == (
        DirectorStyle.NEUTRAL,
        DirectorStyle.AVOIDANT,
    )


# ------------------------------------------------------ the chair alone ---


async def test_without_negotiation_the_director_rules_once_on_stage_two(
    tmp_path: Path,
):
    result, client, registry = await run(
        tmp_path, DirectorStyle.AUTOCRATIC, negotiation=False
    )

    assert [message.speaker_role for message in result["chat_log"]] == [
        AgentRole.RESIDENT,
        AgentRole.RADIOLOGIST,
        AgentRole.NEUROLOGIST,
        AgentRole.ANESTHESIOLOGIST,
        AgentRole.PHYSIATRIST,
        AgentRole.DIRECTOR,
    ]
    assert calls_for(client, ViceDirectorOpinion) == []
    # The advocate's only turn is its frozen opening.
    assert len(calls_for(client, RehabAssessment)) == 1
    (chair_call,) = calls_for(client, DirectorRuling)
    assert chair_call["system_prompt"].startswith(
        registry.director.style_prompts[DirectorStyle.AUTOCRATIC]
    )
    payload = json.loads(chair_call["user_prompt"])
    assert payload["is_final_round"] is True
    assert payload["round_number"] == 1
    assert len(payload["discussion"]) == 5
    # Only what stage II put on the table; nothing that would read as a silent
    # deputy.
    assert set(payload["raised_and_unresolved"]) == {
        "neurology_flags_that_bear_on_the_decision",
        "conservative_preconditions_for_failure",
        "physiatrist_stance",
    }
    # Text only, as for every text role: no films, no file names, no paths.
    assert chair_call["images"] == []
    assert str(tmp_path) not in chair_call["user_prompt"]
    assert "case.png" not in chair_call["user_prompt"]

    assert result["termination_cause"] is TerminationCause.DIRECTOR_RULED
    assert result["round_number"] == 0
    metrics = observe(result)
    assert metrics.negotiation is False
    assert metrics.deputy_style is None
    assert metrics.deputy_rounds == 0
    assert metrics.negotiation_rounds == 0
    assert arm_label(result) == "autocratic-alone"


async def test_an_undecided_director_alone_is_a_decision_delay(tmp_path: Path):
    result, client, _ = await run(
        tmp_path,
        DirectorStyle.AVOIDANT,
        negotiation=False,
        responses={DirectorRuling: [undecided()]},
    )

    assert len(calls_for(client, DirectorRuling)) == 1
    assert result["termination_cause"] is TerminationCause.DECISION_DELAY


async def test_the_new_arms_argue_over_the_main_arms_frozen_evidence(tmp_path: Path):
    """Same baseline, same five statements; only the identifiers name the arm."""

    store = BaselineStore(tmp_path / "baselines")
    main, _, _ = await run(tmp_path, DirectorStyle.AUTOCRATIC, store=store, thread="main")
    chair, chair_client, _ = await run(
        tmp_path,
        DirectorStyle.AUTOCRATIC,
        store=store,
        thread="chair",
        deputy_style=DirectorStyle.NEUTRAL,
    )
    alone, alone_client, _ = await run(
        tmp_path, DirectorStyle.AUTOCRATIC, store=store, thread="alone", negotiation=False
    )

    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1
    assert main["baseline_key"] == chair["baseline_key"] == alone["baseline_key"]
    for client in (chair_client, alone_client):
        assert calls_for(client, ImagingFindings) == []

    def stage_one_and_two(result):
        return [
            (message.sequence, message.speaker_role, message.content)
            for message in result["chat_log"][:5]
        ]

    assert stage_one_and_two(chair) == stage_one_and_two(main)
    assert stage_one_and_two(alone) == stage_one_and_two(main)
    ids = [{message.message_id for message in r["chat_log"]} for r in (main, chair, alone)]
    assert ids[0] & ids[1] == ids[0] & ids[2] == ids[1] & ids[2] == set()


# ------------------------------------------------------------ the runner ---


def test_each_placement_hands_each_role_its_prefix():
    autocratic, neutral = DirectorStyle.AUTOCRATIC, DirectorStyle.NEUTRAL
    assert role_styles(autocratic) == (autocratic, autocratic)
    assert role_styles(autocratic, "chair") == (autocratic, neutral)
    assert role_styles(autocratic, "deputy") == (neutral, autocratic)
    assert role_styles(autocratic, "both", negotiation=False) == (autocratic, None)
    assert role_styles(autocratic, "chair", negotiation=False) == (autocratic, None)
    with pytest.raises(ValueError):
        role_styles(autocratic, "deputy", negotiation=False)
    with pytest.raises(ValueError):
        role_styles(autocratic, "nobody")


def test_an_a4_arm_cannot_land_in_the_main_folder():
    """Its report has the main arm's file name; only a series keeps them apart."""

    assert runner.check_design(None, "both", True) is None
    assert runner.check_design(None, "chair", True) is not None
    assert runner.check_design(None, "both", False) is not None
    assert runner.check_design("chair-only", "chair", True) is None
    assert runner.check_design("x", "deputy", False) is not None


def test_the_frozen_prefix_of_an_arm_without_a_deputy_ends_at_the_director():
    report = {
        "transcript": [
            {"sequence": index, "speaker_role": role, "speaker_name": role, "content": "x"}
            for index, role in enumerate(
                [AgentRole.RESIDENT.value, AgentRole.PHYSIATRIST.value, AgentRole.DIRECTOR.value],
                start=1,
            )
        ]
    }
    assert [item[1] for item in baseline_prefix(report)] == [
        AgentRole.RESIDENT.value,
        AgentRole.PHYSIATRIST.value,
    ]


async def test_a_row_written_before_a4_reads_as_a_deputy_facing_the_director(
    tmp_path: Path,
):
    result, _, _ = await run(tmp_path, DirectorStyle.AVOIDANT)
    old = observe(result).model_dump(mode="json")
    del old["deputy_style"], old["negotiation"]

    row = ObserverMetrics.model_validate(old)

    assert row.deputy_style is DirectorStyle.AVOIDANT
    assert row.negotiation is True
    # And the summary of such a report still names its arm.
    report = {"transcript": [], "metrics": old}
    (arm,) = comparison([row], [report])["arms"]
    assert arm["arm"] == "avoidant"


async def test_a_deputy_only_series_names_its_arms_by_the_style_under_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Both deputy-only arms have a neutral director; the arm is the deputy's."""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BASELINE_STORE_DIR", str(tmp_path / "baselines"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    case_file = tmp_path / "case.txt"
    case_file.write_text(REPORT)
    monkeypatch.setattr(
        runner,
        "DeepSeekTextClient",
        lambda config, thinking: FakeModelClient({FilteredCaseReport: filtered()}),
    )
    monkeypatch.setattr(
        runner, "create_department", lambda settings: (department_with_client()[0], [])
    )
    out = tmp_path / "out"
    base = ["--case-file", str(case_file), "--case-id", "d", "--out", str(out)]
    tested = ["--only-style", "autocratic", "--only-style", "avoidant"]

    assert await runner.main([*base, "--only-style", "neutral"]) == 0
    assert await runner.main(
        [*base, *tested, "--series", "deputy-only", "--persona-on", "deputy"]
    ) == 0
    assert await runner.main(
        [*base, *tested, "--series", "chair-alone", "--no-negotiation"]
    ) == 0

    main = json.loads((out / "d" / "summary-attempt1.json").read_text())
    deputy = json.loads((out / "series-deputy-only" / "d" / "summary-attempt1.json").read_text())
    alone = json.loads((out / "series-chair-alone" / "d" / "summary-attempt1.json").read_text())
    assert [(a["arm"], a["director_style"], a["deputy_style"]) for a in deputy["arms"]] == [
        ("autocratic", "neutral", "autocratic"),
        ("avoidant", "neutral", "avoidant"),
    ]
    assert [(a["arm"], a["deputy_style"], a["negotiation"]) for a in alone["arms"]] == [
        ("autocratic", None, False),
        ("avoidant", None, False),
    ]
    assert deputy["settings"]["persona_on"] == "deputy"
    assert alone["settings"]["negotiation"] is False
    for summary in (deputy, alone):
        assert summary["validity"]["baseline_keys"] == main["validity"]["baseline_keys"]
        assert summary["validity"]["identical_baseline_transcript"] is True


async def test_the_batch_passes_the_design_to_every_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import run_text_case_batch as batch

    manifest = tmp_path / "set.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [{"patient_uid": "1-1", "group": "g", "real_decision": "operated"}],
            }
        )
    )
    seen = []

    async def fake_main(argv):
        seen.append(argv)
        # The real runner makes the case folder, and with it the series folder.
        series = argv[argv.index("--series") + 1]
        (tmp_path / "out" / f"series-{series}").mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(batch.runner, "main", fake_main)
    args = ["--manifest", str(manifest), "--out", str(tmp_path / "out")]
    await batch.main([*args, "--series", "alone", "--no-negotiation"])
    await batch.main([*args, "--series", "chair-only", "--persona-on", "chair"])

    assert "--no-negotiation" in seen[0]
    assert seen[1][seen[1].index("--persona-on") + 1] == "chair"
    with pytest.raises(SystemExit):
        await batch.main([*args, "--persona-on", "chair"])
