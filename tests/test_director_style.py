import sqlite3
from pathlib import Path

import pytest

from api_schemas import CaseCreateMetadata, RunStatus
from graph import create_opening_graph, opening_input, thread_config
from schemas import DirectorStyle, RunPhase
from services.baseline_store import BaselineStore
from services.checkpoint import open_sqlite_checkpointer
from services.run_store import RunStore
from state import CHECKPOINT_SCHEMA_ALLOWLIST, create_initial_state
from tests.dynamic_factories import registry_with_client
from tests.test_baseline_store import case_with_local_sources


def test_the_arm_is_a_property_of_the_run_not_of_the_case():
    """The same case is run once per style, so style must not sit on the case."""

    from schemas import CaseInput

    assert "director_style" not in CaseInput.model_fields
    assert "director_style" in CaseCreateMetadata.model_fields


def test_the_arm_survives_a_checkpoint_round_trip():
    assert ("schemas", "DirectorStyle") in CHECKPOINT_SCHEMA_ALLOWLIST


def test_initial_state_records_the_arm(tmp_path: Path):
    case = case_with_local_sources(tmp_path)

    state = create_initial_state(case, director_style=DirectorStyle.AUTOCRATIC)

    assert state["director_style"] is DirectorStyle.AUTOCRATIC


def test_a_run_without_an_arm_is_simply_unattributed(tmp_path: Path):
    case = case_with_local_sources(tmp_path)

    assert create_initial_state(case)["director_style"] is None


@pytest.mark.asyncio
async def test_the_arm_reaches_graph_state_and_survives_resume(tmp_path: Path):
    case = case_with_local_sources(tmp_path)
    registry, _ = registry_with_client(moderator_responses=[])
    config = thread_config("thread-arm")

    async with open_sqlite_checkpointer(tmp_path / "arm.sqlite3") as saver:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            interrupt_after=["prepare_documents"],
        )
        interrupted = await graph.ainvoke(
            opening_input(case, director_style=DirectorStyle.AVOIDANT),
            config=config,
        )
        assert interrupted["director_style"] is DirectorStyle.AVOIDANT

    async with open_sqlite_checkpointer(tmp_path / "arm.sqlite3") as saver:
        graph = create_opening_graph(agent_registry=registry, checkpointer=saver)
        resumed = await graph.ainvoke(None, config=config)

    assert resumed["director_style"] is DirectorStyle.AVOIDANT
    assert resumed["phase"] is RunPhase.INDEPENDENT_SPECIALIST_ROUND


@pytest.mark.asyncio
async def test_every_arm_of_one_case_shares_a_single_baseline(tmp_path: Path):
    """The manipulated variable must not leak into the frozen evidence."""

    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    registry, client = registry_with_client(moderator_responses=[])
    results = {}

    for index, style in enumerate(DirectorStyle):
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_opening_graph(
                agent_registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            results[style] = await graph.ainvoke(
                opening_input(case, director_style=style),
                config=thread_config(f"thread-arm-{index}"),
            )
        if index == 0:
            calls_after_first_arm = len(client.calls)

    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1
    assert len(client.calls) == calls_after_first_arm
    first = results[DirectorStyle.DEMOCRATIC]
    for style, result in results.items():
        assert result["director_style"] is style
        assert result["case_summary_md"] == first["case_summary_md"]
        assert result["radiology_report"] == first["radiology_report"]


@pytest.mark.asyncio
async def test_the_arm_is_persisted_and_published_with_run_status(tmp_path: Path):
    store = await RunStore.open(tmp_path / "runs.sqlite3")
    try:
        created = await store.create_run(
            "thread-arm",
            "case-arm",
            director_style=DirectorStyle.DEMOCRATIC,
        )
        reloaded = await store.get_run("thread-arm")
        unattributed = await store.create_run("thread-plain", "case-arm")
    finally:
        await store.close()

    assert created.director_style is DirectorStyle.DEMOCRATIC
    assert reloaded is not None
    assert reloaded.director_style is DirectorStyle.DEMOCRATIC
    assert reloaded.status is RunStatus.QUEUED
    assert unattributed.director_style is None


@pytest.mark.asyncio
async def test_an_existing_database_gains_the_column_without_losing_runs(
    tmp_path: Path,
):
    """CREATE TABLE IF NOT EXISTS leaves an older table untouched."""

    database = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(database)
    legacy.executescript(
        """
        CREATE TABLE api_runs (
            thread_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            status TEXT NOT NULL,
            last_message_sequence INTEGER NOT NULL DEFAULT 0,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO api_runs VALUES (
            'legacy-thread', 'legacy-case', 'COMPLETED', 7, NULL, NULL,
            '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    legacy.commit()
    legacy.close()

    store = await RunStore.open(database)
    try:
        preserved = await store.get_run("legacy-thread")
    finally:
        await store.close()

    assert preserved is not None
    assert preserved.last_message_sequence == 7
    assert preserved.director_style is None


@pytest.mark.asyncio
async def test_reopening_a_migrated_database_is_idempotent(tmp_path: Path):
    database = tmp_path / "runs.sqlite3"
    for _ in range(3):
        store = await RunStore.open(database)
        await store.close()

    connection = sqlite3.connect(database)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_runs)")]
    finally:
        connection.close()

    assert columns.count("director_style") == 1
