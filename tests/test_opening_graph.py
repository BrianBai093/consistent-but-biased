import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from graph import create_opening_graph, opening_input, thread_config
from schemas import (
    AgentRole,
    CaseDocumentMarkdown,
    CaseImage,
    DocumentPage,
    Modality,
    RunPhase,
    TestDocumentMarkdown as LabDocumentMarkdown,
)
from services.checkpoint import open_sqlite_checkpointer
from tests.dynamic_factories import registry_with_client
from tests.factories import case_input as make_case_input, radiology_report


def case_input():
    return make_case_input(
        case_id="case-opening",
        images=[
            CaseImage(
                image_id="mri-1",
                modality=Modality.MRI,
                path_or_url="https://example.test/mri-1.jpg",
                body_region="lumbar spine",
                view_or_plane="sagittal",
                sequence="T2",
            ),
            CaseImage(
                image_id="mri-2",
                modality=Modality.MRI,
                path_or_url="https://example.test/mri-2.jpg",
                body_region="lumbar spine",
                view_or_plane="axial",
                sequence="T2",
                level="L4-L5",
            ),
        ],
    )


def registry_and_client():
    return registry_with_client(moderator_responses=[])


def with_document_pages(case_count: int, test_count: int):
    case = case_input()
    return case.model_copy(
        update={
            "case_pages": [
                DocumentPage(
                    page_id=f"case-page-{index:03d}",
                    page_number=index,
                    path_or_url=f"https://example.test/case-{index}.jpg",
                    original_filename=f"private-case-{index}.jpg",
                )
                for index in range(1, case_count + 1)
            ],
            "test_pages": [
                DocumentPage(
                    page_id=f"test-page-{index:03d}",
                    page_number=index,
                    path_or_url=f"https://example.test/test-{index}.jpg",
                    original_filename=f"private-test-{index}.jpg",
                )
                for index in range(1, test_count + 1)
            ],
        }
    )


@pytest.mark.asyncio
async def test_fixed_opening_order_and_state(tmp_path: Path) -> None:
    registry, client = registry_and_client()
    async with open_sqlite_checkpointer(tmp_path / "opening.sqlite3") as checkpointer:
        graph = create_opening_graph(agent_registry=registry, checkpointer=checkpointer)
        result = await graph.ainvoke(
            opening_input(case_input()),
            config=thread_config("thread-opening"),
        )

        assert result["phase"] is RunPhase.INDEPENDENT_SPECIALIST_ROUND
        assert result["radiology_ready"] is True
        assert result["radiology_report"] == radiology_report()
        assert [message.speaker_role for message in result["chat_log"]] == [
            AgentRole.MODERATOR,
            AgentRole.RADIOLOGIST,
        ]
        assert [message.sequence for message in result["chat_log"]] == [1, 2]
        assert result["speaker_counts"][AgentRole.RADIOLOGIST.value] == 1
        assert all(
            count == 0
            for role, count in result["speaker_counts"].items()
            if role != AgentRole.RADIOLOGIST.value
        )
        assert result["run_metadata"].specialist_message_count == 1
        # Case-record transcription is deterministic OCR, not a model call.
        assert len(result["run_metadata"].model_calls) == 3
        assert len(client.calls) == 3
        assert client.calls[-1]["images"] == case_input().images


@pytest.mark.asyncio
async def test_graph_transcribes_all_case_pages_before_test_pages(tmp_path: Path) -> None:
    registry, client = registry_and_client()
    case = with_document_pages(6, 10).model_copy(
        update={"artifact_dir": str(tmp_path / "artifacts")}
    )
    async with open_sqlite_checkpointer(tmp_path / "sequential-documents.sqlite3") as saver:
        graph = create_opening_graph(agent_registry=registry, checkpointer=saver)
        result = await graph.ainvoke(
            opening_input(case),
            config=thread_config("thread-sequential-documents"),
        )

    # Only test reports still go through a model; case records are transcribed
    # by deterministic OCR and therefore make no model call at all.
    document_calls = [
        call
        for call in client.calls
        if call["output_model"] in {CaseDocumentMarkdown, LabDocumentMarkdown}
    ]
    assert [call["output_model"] for call in document_calls] == [LabDocumentMarkdown] * 10
    assert all(len(call["images"]) == 1 for call in document_calls)
    assert json.loads(document_calls[0]["user_prompt"])["previous_transcription_md"] is None
    assert registry.case_document.engine.calls == [
        page.path_or_url for page in case.case_pages
    ]
    assert result["case_preop_md"].count("<!-- source: case-page-") == 6
    assert result["tests_preop_md"].count("<!-- source: test-page-") == 10
    assert "test-page-" not in result["case_preop_md"]
    assert "case-page-" not in result["tests_preop_md"]
    artifact_dir = Path(case.artifact_dir)
    assert (artifact_dir / "case_preop.md").read_text(encoding="utf-8") == result[
        "case_preop_md"
    ]
    assert (artifact_dir / "tests_preop.md").read_text(encoding="utf-8") == result[
        "tests_preop_md"
    ]


@pytest.mark.asyncio
async def test_case_document_checkpoint_prevents_replay_on_resume(tmp_path: Path) -> None:
    database = tmp_path / "document-resume.sqlite3"
    config = thread_config("thread-document-resume")
    registry, client = registry_and_client()
    case = with_document_pages(2, 2).model_copy(
        update={"artifact_dir": str(tmp_path / "resume-artifacts")}
    )

    async with open_sqlite_checkpointer(database) as saver:
        interrupted_graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=saver,
            interrupt_after=["prepare_documents"],
        )
        interrupted = await interrupted_graph.ainvoke(opening_input(case), config=config)
        assert interrupted["phase"] is RunPhase.DOCUMENT_PREPARATION
        assert interrupted["case_preop_md"]
        assert interrupted["tests_preop_md"]
        assert len(client.calls) == 2

    async with open_sqlite_checkpointer(database) as saver:
        resumed_graph = create_opening_graph(agent_registry=registry, checkpointer=saver)
        resumed = await resumed_graph.ainvoke(None, config=config)

    # The OCR engine and the test-report model are each asked for their two
    # pages exactly once: resuming must not replay a completed preparation node.
    assert len(registry.case_document.engine.calls) == 2
    test_calls = [
        call for call in client.calls if call["output_model"] is LabDocumentMarkdown
    ]
    assert len(test_calls) == 2
    assert resumed["phase"] is RunPhase.INDEPENDENT_SPECIALIST_ROUND


@pytest.mark.asyncio
async def test_sqlite_resume_does_not_replay_completed_events(tmp_path: Path) -> None:
    database = tmp_path / "resume.sqlite3"
    config = thread_config("thread-resume")
    registry, client = registry_and_client()

    async with open_sqlite_checkpointer(database) as checkpointer:
        interrupted_graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            interrupt_after=["moderator_opening"],
        )
        interrupted = await interrupted_graph.ainvoke(opening_input(case_input()), config=config)
        assert [message.speaker_role for message in interrupted["chat_log"]] == [
            AgentRole.MODERATOR
        ]
        assert len(client.calls) == 2

    async with open_sqlite_checkpointer(database) as checkpointer:
        resumed_graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
        )
        resumed = await resumed_graph.ainvoke(None, config=config)
        assert resumed["phase"] is RunPhase.INDEPENDENT_SPECIALIST_ROUND
        assert [message.sequence for message in resumed["chat_log"]] == [1, 2]
        assert len(client.calls) == 3

        completed_again = await resumed_graph.ainvoke(None, config=config)
        assert [message.sequence for message in completed_again["chat_log"]] == [1, 2]
        assert len(client.calls) == 3

        snapshot = await resumed_graph.aget_state(config)
        assert snapshot.values["radiology_report"] == radiology_report()


@pytest.mark.asyncio
async def test_normalization_rejects_invalid_input(tmp_path: Path) -> None:
    registry, _ = registry_and_client()
    async with open_sqlite_checkpointer(tmp_path / "invalid.sqlite3") as checkpointer:
        graph = create_opening_graph(agent_registry=registry, checkpointer=checkpointer)
        with pytest.raises(ValidationError, match="modality"):
            await graph.ainvoke(
                {
                    "case_id": "case-invalid",
                    "case_pages": [
                        {
                            "page_id": "case-page-001",
                            "page_number": 1,
                            "path_or_url": "case.jpg",
                            "original_filename": "case.jpg",
                        }
                    ],
                    "test_pages": [
                        {
                            "page_id": "test-page-001",
                            "page_number": 1,
                            "path_or_url": "tests.jpg",
                            "original_filename": "tests.jpg",
                        }
                    ],
                    "images": [
                        {
                            "image_id": "bad-1",
                            "modality": "PET",
                            "path_or_url": "bad.jpg",
                            "body_region": "lumbar spine",
                            "view_or_plane": "unknown",
                        }
                    ],
                    "artifact_dir": str(tmp_path / "artifacts"),
                },
                config=thread_config("thread-invalid"),
            )


def test_thread_id_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="thread_id"):
        thread_config("   ")
