from pathlib import Path

import pytest

from schemas import CaseImage, DocumentPage, Modality, RunPhase
from services.baseline_store import BaselineStore
from services.checkpoint import open_sqlite_checkpointer
from graph import create_opening_graph, opening_input, thread_config
from tests.dynamic_factories import registry_with_client
from tests.factories import radiology_report


def case_with_local_sources(tmp_path: Path, *, case_id: str = "case-baseline"):
    from schemas import CaseInput

    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, payload in (
        ("case-page-001", b"case bytes"),
        ("test-page-001", b"test bytes"),
        ("mri-1", b"image bytes"),
    ):
        path = source / f"{name}.bin"
        path.write_bytes(payload)
        written[name] = path
    return CaseInput(
        case_id=case_id,
        case_pages=[
            DocumentPage(
                page_id="case-page-001",
                page_number=1,
                path_or_url=str(written["case-page-001"]),
                original_filename="case.png",
            )
        ],
        test_pages=[
            DocumentPage(
                page_id="test-page-001",
                page_number=1,
                path_or_url=str(written["test-page-001"]),
                original_filename="tests.png",
            )
        ],
        images=[
            CaseImage(
                image_id="mri-1",
                modality=Modality.MRI,
                path_or_url=str(written["mri-1"]),
            )
        ],
        artifact_dir=str(tmp_path / "derived"),
    )


async def store_baseline(store: BaselineStore, case, **overrides):
    payload = {
        "case_pages": case.case_pages,
        "test_pages": case.test_pages,
        "images": case.images,
        "case_id": case.case_id,
        "case_preop_md": "<!-- source: case-page-001 -->\n\nHistory.",
        "tests_preop_md": "<!-- source: test-page-001 -->\n\nLabs.",
        "case_summary_md": "Summary.",
        "radiology_preop_md": "Radiology.",
        "preparation_warnings": ["[OCR可疑] case-page-001: 第 3 行"],
        "radiology_report": radiology_report(),
    }
    payload.update(overrides)
    return await store.store(**payload)


async def test_a_stored_baseline_is_returned_for_identical_sources(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)

    key = await store_baseline(store, case)
    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )

    assert key is not None
    assert loaded is not None
    assert loaded.baseline_key == key
    assert loaded.case_summary_md == "Summary."
    assert loaded.preparation_warnings == ["[OCR可疑] case-page-001: 第 3 行"]
    assert loaded.radiology_report == radiology_report()


async def test_changed_source_bytes_miss(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    await store_baseline(store, case)

    Path(case.case_pages[0].path_or_url).write_bytes(b"different bytes")
    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )

    assert loaded is None


async def test_a_different_case_id_misses(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    await store_baseline(store, case)

    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id="some-other-case",
    )

    assert loaded is None


async def test_renamed_identifiers_miss_because_the_report_cites_them(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    await store_baseline(store, case)
    renamed = case.images[0].model_copy(update={"image_id": "mri-renamed"})

    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=[renamed],
        case_id=case.case_id,
    )

    assert loaded is None


async def test_two_cases_over_the_same_files_do_not_share_one_slot(tmp_path: Path):
    """A miss must never land on a record that another case is using.

    The same screened files can be prepared twice under different case labels.
    If the key covered only the bytes, the second case would miss on the first
    case's record -- and then overwrite it, destroying preparation that was paid
    for and sending the first case silently back to the model.
    """

    store = BaselineStore(tmp_path / "baselines")
    first = case_with_local_sources(tmp_path, case_id="case-one")
    second = first.model_copy(update={"case_id": "case-two"})
    await store_baseline(store, first)
    await store_baseline(store, second, case_summary_md="A different summary.")

    assert len(list((tmp_path / "baselines").glob("*.json"))) == 2
    kept = await store.load(
        case_pages=first.case_pages,
        test_pages=first.test_pages,
        images=first.images,
        case_id=first.case_id,
    )
    assert kept is not None
    assert kept.case_summary_md == "Summary."


async def test_repreparing_a_case_keeps_its_frozen_department_stages(
    tmp_path: Path,
):
    """Stage I-II was paid for once; re-running v5 preparation must not drop it."""

    from tests.department_factories import department_payload

    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    await store_baseline(store, case)
    await store.store_department(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md="History.",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=["[OCR可疑] case-page-001: 第 3 行"],
        payload=department_payload(),
    )

    await store_baseline(store, case, case_summary_md="Re-prepared.")

    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    assert loaded is not None
    assert loaded.case_summary_md == "Re-prepared."
    assert loaded.format_version == "v2"
    assert loaded.department is not None


async def test_a_record_this_build_cannot_read_still_keeps_its_frozen_stages(
    tmp_path: Path,
):
    """An unreadable record is not an empty slot.

    A field added by a later build makes the record fail validation. If that
    counted as "nothing stored here", re-running preparation would overwrite
    stage I-II that was paid for once and is meant to outlive every arm.
    """

    import json

    from tests.department_factories import department_payload

    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    await store_baseline(store, case)
    await store.store_department(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md="History.",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=["[OCR可疑] case-page-001: 第 3 行"],
        payload=department_payload(),
    )
    path = next((tmp_path / "baselines").glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["a_field_from_a_later_build"] = True
    path.write_text(json.dumps(record), encoding="utf-8")

    await store_baseline(store, case, case_summary_md="Re-prepared.")

    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    assert loaded is not None
    assert loaded.case_summary_md == "Re-prepared."
    assert loaded.department is not None


async def test_preparation_is_frozen_before_any_discussion_happens(tmp_path: Path):
    """Transcription is the expensive half and has nothing to do with the arms."""

    from tests.department_factories import department_payload

    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    warnings = ["[OCR可疑] case-page-001: 第 3 行"]
    await store.store_preparation(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md="History.",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=warnings,
    )

    loaded = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    assert loaded is not None
    assert loaded.case_summary_md == "Summary."
    assert loaded.preparation_warnings == warnings
    assert loaded.department is None

    # A later department run attaches to the same record rather than replacing it.
    await store.store_department(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
        case_preop_md="History.",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=warnings,
        payload=department_payload(),
    )
    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1
    again = await store.load(
        case_pages=case.case_pages,
        test_pages=case.test_pages,
        images=case.images,
        case_id=case.case_id,
    )
    assert again is not None and again.department is not None
    assert again.preparation_warnings == warnings


async def test_a_disabled_store_never_reads_or_writes(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines", enabled=False)
    case = case_with_local_sources(tmp_path)

    assert await store_baseline(store, case) is None
    assert not (tmp_path / "baselines").exists()


async def test_remote_sources_have_no_stable_fingerprint(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    remote = case.case_pages[0].model_copy(
        update={"path_or_url": "https://example.test/case-page-001.png"}
    )

    assert await store_baseline(store, case, case_pages=[remote]) is None


@pytest.mark.asyncio
async def test_a_second_run_reuses_the_baseline_without_calling_a_model(
    tmp_path: Path,
):
    """Three arms of one case must argue over byte-identical evidence."""

    store = BaselineStore(tmp_path / "baselines")
    case = case_with_local_sources(tmp_path)
    registry, client = registry_with_client(moderator_responses=[])

    async with open_sqlite_checkpointer(tmp_path / "first.sqlite3") as saver:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        first = await graph.ainvoke(
            opening_input(case), config=thread_config("arm-one")
        )

    calls_after_first = len(client.calls)
    ocr_calls_after_first = len(registry.case_document.engine.calls)
    assert calls_after_first > 0

    async with open_sqlite_checkpointer(tmp_path / "second.sqlite3") as saver:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        second = await graph.ainvoke(
            opening_input(case), config=thread_config("arm-two")
        )

    assert len(client.calls) == calls_after_first
    assert len(registry.case_document.engine.calls) == ocr_calls_after_first
    for field in (
        "case_preop_md",
        "tests_preop_md",
        "case_summary_md",
        "radiology_preop_md",
    ):
        assert second[field] == first[field]
    assert second["radiology_report"] == first["radiology_report"]
    assert second["phase"] is RunPhase.INDEPENDENT_SPECIALIST_ROUND
    assert second["run_metadata"].radiology_cache_hit is False


@pytest.mark.asyncio
async def test_the_reused_run_still_writes_its_own_artifacts(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines")
    first_case = case_with_local_sources(tmp_path)
    registry, _ = registry_with_client(moderator_responses=[])

    async with open_sqlite_checkpointer(tmp_path / "first.sqlite3") as saver:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        await graph.ainvoke(opening_input(first_case), config=thread_config("arm-one"))

    second_case = first_case.model_copy(
        update={"artifact_dir": str(tmp_path / "derived-two")}
    )
    async with open_sqlite_checkpointer(tmp_path / "second.sqlite3") as saver:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            baseline_store=store,
        )
        result = await graph.ainvoke(
            opening_input(second_case), config=thread_config("arm-two")
        )

    derived = Path(second_case.artifact_dir)
    for filename, field in (
        ("case_preop.md", "case_preop_md"),
        ("tests_preop.md", "tests_preop_md"),
        ("case_summary.md", "case_summary_md"),
        ("radiology_preop.md", "radiology_preop_md"),
    ):
        assert (derived / filename).read_text(encoding="utf-8") == result[field]
