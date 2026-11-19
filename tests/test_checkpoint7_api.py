import asyncio
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import graph as graph_module
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
)
from agents import (
    TestDocumentAgent as LabDocumentAgent,
)
from app import create_app
from config import Settings
from exceptions import (
    ModelRateLimitError,
    ModelTimeoutError,
    OcrError,
    StructuredOutputError,
)
from schemas import (
    AgentRole,
    AnesthesiologistOutput,
    CaseDocumentMarkdown,
    CaseSummaryMarkdown,
    FinalPlan,
    FinalPlanReview,
    ModeratorDecision,
    RadiologyReport,
)
from schemas import (
    TestDocumentMarkdown as LabDocumentOutput,
)
from services.llm_client import FakeModelClient
from services.run_manager import _external_input_update_node
from tests.dynamic_factories import registry_with_client
from services.ocr_engine import FakeOcrEngine
from tests.factories import all_fake_responses, fake_ocr_engine
from tests.test_dynamic_graph import STANDARD_ROUTE


def image_bytes(format_name: str) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 4), "white").save(stream, format=format_name)
    return stream.getvalue()


PNG = image_bytes("PNG")
JPEG = image_bytes("JPEG")


def api_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "checkpoint_db_path": tmp_path / "checkpoints.sqlite3",
        "run_store_db_path": tmp_path / "runs.sqlite3",
        "upload_dir": tmp_path / "uploads",
        "radiology_cache_dir": tmp_path / "radiology-cache",
        "baseline_store_dir": tmp_path / "baselines",
        "sse_heartbeat_seconds": 0.02,
        "max_upload_bytes": 1024,
        "max_images_per_case": 5,
        "max_additional_input_chars": 200,
        "openai_api_key": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def case_metadata(*, image_count: int = 2) -> dict:
    images = [
        {
            "image_id": "mri-1",
            "modality": "MRI",
            "body_region": "lumbar spine",
            "view_or_plane": "sagittal",
            "sequence": "T2",
            "level": "L4-L5",
            "description": None,
        },
        {
            "image_id": "ct-1",
            "modality": "CT",
            "body_region": "lumbar spine",
            "view_or_plane": "axial",
            "sequence": None,
            "level": "L4-L5",
            "description": None,
        },
    ]
    return {
        "case_id": "case-api-001",
        "confirmed_preoperative_only": True,
        "images": images[:image_count],
    }


def multipart_case(*, image_count: int = 2):
    files = [
        ("case_files", ("case-01.png", PNG, "image/png")),
        ("test_files", ("tests-01.jpg", JPEG, "image/jpeg")),
        ("image_files", ("mri.png", PNG, "image/png")),
        ("image_files", ("ct.jpg", JPEG, "image/jpeg")),
    ]
    return {
        "metadata": json.dumps(case_metadata(image_count=image_count))
    }, files[:2] + files[2 : 2 + image_count]


def package_files(image_files: list[tuple[str, bytes, str]]) -> list[tuple]:
    return [
        ("case_files", ("case-01.png", PNG, "image/png")),
        ("test_files", ("tests-01.jpg", JPEG, "image/jpeg")),
        *[("image_files", item) for item in image_files],
    ]


async def wait_for_status(
    client: AsyncClient,
    thread_id: str,
    expected: set[str],
    *,
    timeout: float = 3,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/runs/{thread_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in expected:
            return body
        await asyncio.sleep(0.01)
    pytest.fail(f"Run {thread_id} did not reach one of {sorted(expected)}")


async def open_client(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


def fake_registry():
    registry, _ = registry_with_client(moderator_responses=STANDARD_ROUTE)
    return registry


@pytest.mark.parametrize(
    ("role", "expected_node"),
    [
        ("radiologist", "radiologist_dynamic"),
        ("spine_surgeon", "spine_surgeon"),
        ("anesthesiologist", "anesthesiologist"),
        ("neurologist", "neurologist"),
        ("perioperative_physician", "perioperative_physician"),
    ],
)
def test_addendum_checkpoint_anchor_uses_real_specialist_node(
    role: str,
    expected_node: str,
) -> None:
    parsed_role = next(item for item in AgentRole if item.value == role)

    assert _external_input_update_node(
        {
            "phase": "dynamic_discussion",
            "chat_log": [SimpleNamespace(speaker_role=parsed_role)],
        }
    ) == expected_node


def test_addendum_anchor_does_not_repeat_completed_independent_round() -> None:
    initial_counts = {
        role.value: 1
        for role in (
            AgentRole.SPINE_SURGEON,
            AgentRole.NEUROLOGIST,
            AgentRole.ANESTHESIOLOGIST,
            AgentRole.PERIOPERATIVE_PHYSICIAN,
        )
    }

    assert _external_input_update_node(
        {"phase": "independent_specialist_round", "speaker_counts": {}}
    ) == "moderator_review"
    assert _external_input_update_node(
        {"phase": "awaiting_dynamic_routing", "speaker_counts": initial_counts}
    ) == "independent_specialist_round"
    assert _external_input_update_node(
        {"phase": "awaiting_dynamic_routing", "speaker_counts": {}}
    ) == "moderator_review"


@pytest.mark.asyncio
async def test_case_submission_completes_and_replays_sse(tmp_path: Path) -> None:
    registry, fake_client = registry_with_client(moderator_responses=STANDARD_ROUTE)
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry,
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case()
            submitted = await client.post("/v1/cases", data=data, files=files)
            assert submitted.status_code == 202, submitted.text
            thread_id = submitted.json()["thread_id"]

            completed = await wait_for_status(client, thread_id, {"COMPLETED"})
            assert completed["case_id"] == "case-api-001"
            assert completed["phase"] == "completed"
            assert completed["is_complete"] is True
            assert completed["final_plan"] is not None
            assert completed["final_plan"]["definitive_surgery_status"] == "established"
            assert completed["final_plan"]["conditional_surgical_options"] == []
            assert {
                item["approach"]
                for item in completed["final_plan"]["approach_comparison"]["assessments"]
            } == {"anterior", "posterior", "combined"}
            assert completed["run_metadata"] is not None
            sequences = [message["sequence"] for message in completed["new_messages"]]
            assert sequences == list(range(1, len(sequences) + 1))
            image_calls = [
                call
                for call in fake_client.calls
                if call["output_model"] is RadiologyReport
            ]
            assert len(image_calls) == 1
            assert image_calls[0]["output_model"] is RadiologyReport
            assert [image.modality.value for image in image_calls[0]["images"]] == [
                "MRI",
                "CT",
            ]
            assert completed["prepared_documents"]["case_preop_md"]
            assert completed["prepared_documents"]["tests_preop_md"]
            assert completed["prepared_documents"]["case_summary_md"]
            assert completed["prepared_documents"]["radiology_preop_md"]
            assert completed["radiology_progress"] == {
                "stage": "completed",
                "completed_images": 2,
                "total_images": 2,
                "batch_number": 1,
                "total_batches": 1,
                "current_image_start": None,
                "current_image_end": None,
            }

            incremental = await client.get(
                f"/v1/runs/{thread_id}",
                params={"after_sequence": sequences[-2]},
            )
            assert [item["sequence"] for item in incremental.json()["new_messages"]] == [
                sequences[-1]
            ]

            result = await client.get(f"/v1/runs/{thread_id}/result")
            assert result.status_code == 200
            assert result.json()["final_plan"] == completed["final_plan"]

            stream = await client.get(f"/v1/runs/{thread_id}/events")
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            event_ids = [
                int(line.removeprefix("id: "))
                for line in stream.text.splitlines()
                if line.startswith("id: ")
            ]
            assert event_ids == sorted(event_ids)
            assert "event: chat.message" in stream.text
            assert "event: radiology.progress" in stream.text
            assert "event: run.completed" in stream.text
            persisted_events = (
                await app.state.run_manager.run_store.list_events(thread_id)
            )
            chat_events = [
                event for event in persisted_events if event.event_type == "chat.message"
            ]
            assert [event.data["sequence"] for event in chat_events] == sequences
            assert all(event.data["content"] for event in chat_events)
            progress_events = [
                event
                for event in persisted_events
                if event.event_type == "radiology.progress"
            ]
            assert [event.data["stage"] for event in progress_events] == [
                "reviewing",
                "completed",
            ]
            stored_progress = (
                await app.state.run_manager.run_store.get_radiology_progress(
                    thread_id
                )
            )
            assert stored_progress is not None
            assert stored_progress.stage == "completed"
            assert stored_progress.completed_images == 2

            replay = await client.get(
                f"/v1/runs/{thread_id}/events",
                headers={"Last-Event-ID": str(event_ids[-2])},
            )
            replay_ids = [
                int(line.removeprefix("id: "))
                for line in replay.text.splitlines()
                if line.startswith("id: ")
            ]
            assert replay_ids == [event_ids[-1]]


@pytest.mark.asyncio
async def test_health_openapi_and_missing_provider_are_safe(tmp_path: Path) -> None:
    app = create_app(settings=api_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            assert (await client.get("/health")).status_code == 200
            openapi = await client.get("/openapi.json")
            assert openapi.status_code == 200
            paths = openapi.json()["paths"]
            assert "/v1/cases" in paths
            assert "/v1/runs/{thread_id}/events" in paths
            assert all("websocket" not in json.dumps(value).lower() for value in paths.values())

            data, files = multipart_case(image_count=1)
            response = await client.post("/v1/cases", data=data, files=files)
            assert response.status_code == 503
            assert response.json() == {
                "code": "PROVIDER_NOT_CONFIGURED",
                "message": "Required model provider configuration is unavailable.",
                "thread_id": None,
            }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "files", "status_code", "error_code"),
    [
        (
            case_metadata(image_count=2),
            package_files([("image.png", PNG, "image/png")]),
            422,
            "IMAGE_METADATA_MISMATCH",
        ),
        (
            case_metadata(image_count=1),
            package_files([("image.gif", PNG, "image/png")]),
            415,
            "INVALID_IMAGE_EXTENSION",
        ),
        (
            case_metadata(image_count=1),
            package_files([("image.png", PNG, "image/jpeg")]),
            415,
            "INVALID_IMAGE_CONTENT_TYPE",
        ),
        (
            case_metadata(image_count=1),
            package_files([("image.png", b"not-png", "image/png")]),
            415,
            "INVALID_IMAGE_CONTENT",
        ),
        (
            case_metadata(image_count=1),
            package_files(
                [("image.png", b"\x89PNG\r\n\x1a\ncorrupt", "image/png")]
            ),
            415,
            "INVALID_IMAGE_CONTENT",
        ),
        (
            case_metadata(image_count=1),
            package_files(
                [("image.png", b"\x89PNG\r\n\x1a\n" + b"x" * 2000, "image/png")]
            ),
            413,
            "IMAGE_TOO_LARGE",
        ),
        (
            {
                **case_metadata(image_count=1),
                "images": [
                    {
                        **case_metadata(image_count=1)["images"][0],
                        "modality": "DICOM",
                    }
                ],
            },
            package_files([("image.png", PNG, "image/png")]),
            422,
            "INVALID_CASE_METADATA",
        ),
        (
            {
                **case_metadata(image_count=1),
                "images": [
                    {
                        **case_metadata(image_count=1)["images"][0],
                        "image_id": f"mri-{index}",
                    }
                        for index in range(4)
                    ],
                },
            package_files(
                [(f"image-{index}.png", PNG, "image/png") for index in range(4)]
            ),
            413,
            "TOO_MANY_FILES",
        ),
    ],
)
async def test_invalid_uploads_are_rejected_and_staging_is_cleaned(
    tmp_path: Path,
    metadata: dict,
    files: list,
    status_code: int,
    error_code: str,
) -> None:
    settings = api_settings(tmp_path)
    app = create_app(settings=settings, registry_factory=fake_registry)
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            response = await client.post(
                "/v1/cases",
                data={"metadata": json.dumps(metadata)},
                files=files,
            )
            assert response.status_code == status_code, response.text
            assert response.json()["code"] == error_code
            assert not list(settings.upload_dir.glob(".staging-*"))


class GateFakeModelClient(FakeModelClient):
    def __init__(self) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        super().__init__(responses)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked = False

    async def generate_structured(self, **kwargs):
        if not self._blocked:
            self._blocked = True
            self.started.set()
            await self.release.wait()
        return await super().generate_structured(**kwargs)


class RadiologyGateFakeModelClient(FakeModelClient):
    def __init__(self) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        super().__init__(responses)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked = False

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is RadiologyReport and not self._blocked:
            self._blocked = True
            self.started.set()
            await self.release.wait()
        return await super().generate_structured(**kwargs)


def registry_for_client(
    client: FakeModelClient,
    *,
    ocr_engine=None,
) -> AgentRegistry:
    return AgentRegistry(
        case_document=CaseDocumentTranscriber(ocr_engine or fake_ocr_engine()),
        test_document=LabDocumentAgent(client),
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


@pytest.mark.asyncio
async def test_radiology_progress_is_visible_while_model_request_is_running(
    tmp_path: Path,
) -> None:
    gate = RadiologyGateFakeModelClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(gate),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=2)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await asyncio.wait_for(gate.started.wait(), timeout=1)

            running = (await client.get(f"/v1/runs/{thread_id}")).json()
            assert running["status"] == "RUNNING"
            assert running["radiology_progress"] == {
                "stage": "reviewing",
                "completed_images": 0,
                "total_images": 2,
                "batch_number": 1,
                "total_batches": 1,
                "current_image_start": 1,
                "current_image_end": 2,
            }
            events = await app.state.run_manager.run_store.list_events(thread_id)
            assert any(event.event_type == "radiology.progress" for event in events)

            gate.release.set()
            await wait_for_status(client, thread_id, {"COMPLETED"})


@pytest.mark.asyncio
async def test_pause_addendum_resume_and_cancel_at_node_boundary(tmp_path: Path) -> None:
    gate = GateFakeModelClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(gate),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await asyncio.wait_for(gate.started.wait(), timeout=1)

            pause = await client.post(f"/v1/runs/{thread_id}/pause")
            assert pause.status_code == 202
            assert pause.json()["status"] == "PAUSE_REQUESTED"
            gate.release.set()
            paused = await wait_for_status(client, thread_id, {"PAUSED"})
            paused_sequences = [item["sequence"] for item in paused["new_messages"]]

            addendum = await client.post(
                f"/v1/runs/{thread_id}/input",
                json={"content": "New hemoglobin is 128 g/L."},
            )
            assert addendum.status_code == 202
            values = await app.state.run_manager._checkpoint_values(thread_id)
            assert "New hemoglobin is 128 g/L." in values["case_preop_md"]
            assert all(
                "New hemoglobin is 128 g/L." not in message.content
                for message in values["chat_log"]
            )

            resumed = await client.post(f"/v1/runs/{thread_id}/resume")
            assert resumed.status_code == 202
            completed = await wait_for_status(client, thread_id, {"COMPLETED"})
            completed_sequences = [item["sequence"] for item in completed["new_messages"]]
            assert completed_sequences[: len(paused_sequences)] == paused_sequences
            assert len(completed_sequences) == len(set(completed_sequences))
            assert any(
                "New hemoglobin is 128 g/L." in call["user_prompt"]
                for call in gate.calls
            )
            assert sum(
                call["output_model"] is RadiologyReport for call in gate.calls
            ) == 1

            invalid_resume = await client.post(f"/v1/runs/{thread_id}/resume")
            assert invalid_resume.status_code == 409
            assert invalid_resume.json()["code"] == "INVALID_RUN_TRANSITION"
            late_input = await client.post(
                f"/v1/runs/{thread_id}/input",
                json={"content": "Too late"},
            )
            assert late_input.status_code == 409
            event_types = [
                event.event_type
                for event in await app.state.run_manager.run_store.list_events(thread_id)
            ]
            assert {
                "run.pause_requested",
                "run.paused",
                "input.accepted",
                "run.resumed",
                "run.completed",
            }.issubset(event_types)

    cancel_gate = GateFakeModelClient()
    cancel_settings = api_settings(tmp_path / "cancel")
    cancel_app = create_app(
        settings=cancel_settings,
        registry_factory=lambda: registry_for_client(cancel_gate),
    )
    async with cancel_app.router.lifespan_context(cancel_app):
        async with await open_client(cancel_app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await asyncio.wait_for(cancel_gate.started.wait(), timeout=1)
            requested = await client.post(f"/v1/runs/{thread_id}/cancel")
            assert requested.status_code == 202
            assert requested.json()["status"] == "CANCEL_REQUESTED"
            cancel_gate.release.set()
            await wait_for_status(client, thread_id, {"CANCELLED"})
            rejected = await client.post(f"/v1/runs/{thread_id}/resume")
            assert rejected.status_code == 409
            cancel_events = await cancel_app.state.run_manager.run_store.list_events(
                thread_id
            )
            cancel_event_types = [event.event_type for event in cancel_events]
            assert "run.cancel_requested" in cancel_event_types
            assert cancel_event_types[-1] == "run.cancelled"
            assert cancel_event_types.index("run.cancel_requested") < (
                cancel_event_types.index("run.cancelled")
            )


@pytest.mark.asyncio
async def test_restart_marks_run_interrupted_and_concurrent_resume_is_singleton(
    tmp_path: Path,
) -> None:
    settings = api_settings(tmp_path)
    first_gate = GateFakeModelClient()
    first_app = create_app(
        settings=settings,
        registry_factory=lambda: registry_for_client(first_gate),
    )
    async with first_app.router.lifespan_context(first_app):
        async with await open_client(first_app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await asyncio.wait_for(first_gate.started.wait(), timeout=1)
            running = await wait_for_status(client, thread_id, {"RUNNING"})
            checkpointed_sequences = [
                message["sequence"] for message in running["new_messages"]
            ]

    second_gate = GateFakeModelClient()
    second_app = create_app(
        settings=settings,
        registry_factory=lambda: registry_for_client(second_gate),
    )
    async with second_app.router.lifespan_context(second_app):
        async with await open_client(second_app) as client:
            interrupted = await client.get(f"/v1/runs/{thread_id}")
            assert interrupted.status_code == 200
            assert interrupted.json()["status"] == "INTERRUPTED"

            first, second = await asyncio.gather(
                client.post(f"/v1/runs/{thread_id}/resume"),
                client.post(f"/v1/runs/{thread_id}/resume"),
            )
            assert first.status_code == second.status_code == 202
            await asyncio.wait_for(second_gate.started.wait(), timeout=1)
            assert len(second_app.state.run_manager._tasks) == 1
            second_gate.release.set()
            completed = await wait_for_status(client, thread_id, {"COMPLETED"})
            sequences = [message["sequence"] for message in completed["new_messages"]]
            assert sequences[: len(checkpointed_sequences)] == checkpointed_sequences
            assert len(sequences) == len(set(sequences))

            events = await second_app.state.run_manager.run_store.list_events(thread_id)
            assert sum(event.event_type == "run.resumed" for event in events) == 1
            assert sum(event.event_type == "run.interrupted" for event in events) >= 1


class ExplodingFakeModelClient(FakeModelClient):
    async def generate_structured(self, **kwargs):
        raise RuntimeError("provider exploded with secret api-key-value")


class TimeoutFakeModelClient(FakeModelClient):
    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        raise ModelTimeoutError("provider-specific timeout detail")


class AnesthesiaTimeoutClient(FakeModelClient):
    def __init__(self) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        super().__init__(responses, max_retries=3)

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is AnesthesiologistOutput:
            self.calls.append(kwargs)
            raise ModelTimeoutError(
                "private timeout api-key-value H:\\patient\\scan.jpg"
            )
        return await super().generate_structured(**kwargs)


class FinalReviewTimeoutClient(FakeModelClient):
    def __init__(self) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        super().__init__(responses, max_retries=1)

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is FinalPlanReview:
            self.calls.append(kwargs)
            raise ModelTimeoutError(
                "private review timeout api-key-value H:\\patient\\scan.jpg"
            )
        return await super().generate_structured(**kwargs)


class RecoverableFinalSynthesisClient(FakeModelClient):
    def __init__(self) -> None:
        responses = all_fake_responses()
        responses[ModeratorDecision] = STANDARD_ROUTE
        super().__init__(responses, max_retries=1)
        self.invalid_final_attempts = 2

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is FinalPlan and self.invalid_final_attempts:
            self.calls.append(kwargs)
            self.invalid_final_attempts -= 1
            raise StructuredOutputError("invalid final plan")
        return await super().generate_structured(**kwargs)


@pytest.mark.asyncio
async def test_critical_final_review_failure_closes_run_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    failing_client = FinalReviewTimeoutClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(failing_client),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(client, thread_id, {"FAILED"})

            assert failed["final_plan"] is None
            assert failed["prepared_documents"]["case_preop_md"]
            assert failed["prepared_documents"]["tests_preop_md"]
            assert failed["prepared_documents"]["case_summary_md"]
            assert failed["prepared_documents"]["radiology_preop_md"]
            assert failed["error_code"] == "final_plan_review_MODEL_TIMEOUT"
            failure = next(
                item
                for item in failed["run_metadata"]["model_failures"]
                if item["stage"] == "final_plan_review"
            )
            assert failure["outcome"] == "failed"
            assert failure["attempt_count"] == 2
            assert failure["attempt_categories"] == ["MODEL_TIMEOUT"] * 2
            serialized = json.dumps(failed)
            assert "api-key-value" not in serialized
            assert "H:\\\\patient" not in serialized


@pytest.mark.asyncio
async def test_failed_finalization_resumes_from_checkpoint_without_replaying_work(
    tmp_path: Path,
) -> None:
    recovering_client = RecoverableFinalSynthesisClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(recovering_client),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(client, thread_id, {"FAILED"})

            assert failed["error_code"] == (
                "final_plan_generation_MODEL_INVALID_OUTPUT"
            )
            calls_before_resume = list(recovering_client.calls)
            messages_before_resume = list(failed["new_messages"])

            resumed = await client.post(f"/v1/runs/{thread_id}/resume")
            assert resumed.status_code == 202, resumed.text
            completed = await wait_for_status(
                client,
                thread_id,
                {"COMPLETED"},
                timeout=5,
            )

            assert completed["final_plan"] is not None
            assert completed["error_code"] is None
            assert completed["new_messages"][:-1] == messages_before_resume
            assert completed["new_messages"][-1]["speaker_role"] == "moderator"
            assert sum(
                call["output_model"] is CaseDocumentMarkdown
                for call in recovering_client.calls
            ) == sum(
                call["output_model"] is CaseDocumentMarkdown
                for call in calls_before_resume
            )
            assert sum(
                call["output_model"] is RadiologyReport
                for call in recovering_client.calls
            ) == sum(
                call["output_model"] is RadiologyReport
                for call in calls_before_resume
            )
            assert sum(
                call["output_model"] is FinalPlan
                for call in recovering_client.calls
            ) == 3
            assert any(
                failure["stage"] == "final_plan_generation"
                for failure in completed["run_metadata"]["model_failures"]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_version", ["v1", "v2", "v3"])
async def test_unfinished_legacy_checkpoint_is_readable_but_cannot_resume_or_mutate(
    tmp_path: Path,
    legacy_version: str,
) -> None:
    gate = GateFakeModelClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(gate),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await asyncio.wait_for(gate.started.wait(), timeout=1)
            assert (await client.post(f"/v1/runs/{thread_id}/pause")).status_code == 202
            gate.release.set()
            await wait_for_status(client, thread_id, {"PAUSED"})

            await app.state.run_manager._graph.aupdate_state(
                graph_module.thread_config(thread_id),
                {"workflow_version": legacy_version},
                as_node="prepare_documents",
            )

            readable = await client.get(f"/v1/runs/{thread_id}")
            assert readable.status_code == 200
            assert readable.json()["status"] == "PAUSED"

            resume = await client.post(f"/v1/runs/{thread_id}/resume")
            assert resume.status_code == 409
            assert resume.json()["code"] == "WORKFLOW_VERSION_INCOMPATIBLE"
            addendum = await client.post(
                f"/v1/runs/{thread_id}/input",
                json={"content": "Do not apply to a legacy checkpoint."},
            )
            assert addendum.status_code == 409
            assert addendum.json()["code"] == "WORKFLOW_VERSION_INCOMPATIBLE"


@pytest.mark.asyncio
async def test_specialist_failure_is_persisted_to_snapshot_sqlite_and_sse(
    tmp_path: Path,
) -> None:
    failing_client = AnesthesiaTimeoutClient()
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(failing_client),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            completed = await wait_for_status(client, thread_id, {"COMPLETED"})

            snapshot_failure = next(
                item
                for item in completed["run_metadata"]["model_failures"]
                if item["agent_role"] == "anesthesiologist"
            )
            assert snapshot_failure["category"] == "MODEL_TIMEOUT"
            assert snapshot_failure["attempt_count"] == 4
            assert snapshot_failure["attempt_categories"] == ["MODEL_TIMEOUT"] * 4

            stored = await app.state.run_manager.run_store.list_model_failures(
                thread_id
            )
            stored_failure = next(
                item
                for item in stored
                if item.agent_role.value == "anesthesiologist"
            )
            assert stored_failure.model_dump(mode="json") == snapshot_failure

            events = await app.state.run_manager.run_store.list_events(thread_id)
            failure_events = [
                event
                for event in events
                if event.event_type == "model.failure"
                and event.data["agent_role"] == "anesthesiologist"
            ]
            assert len(failure_events) == 1
            assert failure_events[0].data["category"] == "MODEL_TIMEOUT"

            duplicate = await app.state.run_manager.run_store.record_model_failure(
                thread_id,
                stored_failure,
            )
            assert duplicate is None
            events_after_duplicate = (
                await app.state.run_manager.run_store.list_events(thread_id)
            )
            assert sum(
                event.event_type == "model.failure"
                and event.data["agent_role"] == "anesthesiologist"
                for event in events_after_duplicate
            ) == 1

            stream = await client.get(f"/v1/runs/{thread_id}/events")
            assert "event: model.failure" in stream.text
            serialized = json.dumps(
                {
                    "snapshot": snapshot_failure,
                    "event": failure_events[0].data,
                }
            )
            assert "api-key-value" not in serialized
            assert "patient" not in serialized
            assert "scan.jpg" not in serialized


@pytest.mark.asyncio
async def test_model_timeout_has_a_stable_actionable_run_error(tmp_path: Path) -> None:
    timeout_client = TimeoutFakeModelClient(all_fake_responses())
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(timeout_client),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(client, thread_id, {"FAILED"})

            # Case records no longer reach a model, so only the test-report
            # stage can time out here.
            assert failed["error_code"] == "TEST_DOCUMENT_TRANSCRIPTION_MODEL_TIMEOUT"
            assert failed["error_message"] == (
                "Test-report transcription failed for test-page-001 because the "
                "model request timed out. The run did not enter MDT discussion."
            )
            assert "provider-specific" not in json.dumps(failed)
            assert len(timeout_client.calls) == 1
            assert timeout_client.calls[0]["output_model"] is LabDocumentOutput
            stored = await app.state.run_manager.run_store.get_run(thread_id)
            assert stored is not None
            assert stored.error_code == failed["error_code"]
            events = await app.state.run_manager.run_store.list_events(thread_id)
            failed_event = next(event for event in events if event.event_type == "run.failed")
            assert failed_event.data["error_code"] == failed["error_code"]
            stream = await client.get(f"/v1/runs/{thread_id}/events")
            assert failed["error_code"] in stream.text


@pytest.mark.asyncio
async def test_case_ocr_failure_has_a_stable_actionable_run_error(tmp_path: Path) -> None:
    """A dead OCR backend must fail the run closed, exactly like a dead provider."""

    failing_ocr = FakeOcrEngine({}, default=OcrError("ocr backend unavailable"))
    client = FakeModelClient(all_fake_responses())
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(client, ocr_engine=failing_ocr),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as api_client:
            data, files = multipart_case(image_count=1)
            submitted = await api_client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(api_client, thread_id, {"FAILED"})

            assert failed["error_code"] == (
                "CASE_DOCUMENT_TRANSCRIPTION_MODEL_INVOCATION_FAILED"
            )
            assert "case-page-001" in failed["error_message"]
            assert "ocr backend unavailable" not in json.dumps(failed)
            snapshot = await api_client.get(f"/v1/runs/{thread_id}")
            assert snapshot.json()["final_plan"] is None


@pytest.mark.asyncio
async def test_the_experiment_arm_is_visible_in_snapshots_and_events(
    tmp_path: Path,
) -> None:
    """A run has to be attributable to its arm from the moment it starts."""

    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(FakeModelClient(all_fake_responses())),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            metadata = json.loads(data["metadata"])
            metadata["director_style"] = "autocratic"
            data["metadata"] = json.dumps(metadata)

            submitted = await client.post("/v1/cases", data=data, files=files)
            assert submitted.status_code == 202
            thread_id = submitted.json()["thread_id"]
            assert submitted.json()["director_style"] == "autocratic"

            snapshot = await client.get(f"/v1/runs/{thread_id}")
            assert snapshot.json()["director_style"] == "autocratic"

            events = await app.state.run_manager.run_store.list_events(thread_id)
            started = next(
                event for event in events if event.event_type in {"run.started", "run.queued"}
            )
            assert started.data["director_style"] == "autocratic"


@pytest.mark.asyncio
async def test_a_run_submitted_without_an_arm_stays_unattributed(
    tmp_path: Path,
) -> None:
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(FakeModelClient(all_fake_responses())),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)

            assert submitted.json()["director_style"] is None


class RateLimitedLabClient(FakeModelClient):
    def __init__(self, responses, case_written: asyncio.Event) -> None:
        super().__init__(responses)
        self.case_written = case_written

    async def generate_structured(self, **kwargs):
        if kwargs["output_model"] is LabDocumentOutput:
            await self.case_written.wait()
            self.calls.append(kwargs)
            raise ModelRateLimitError("provider detail")
        return await super().generate_structured(**kwargs)


@pytest.mark.asyncio
async def test_test_transcription_failure_is_named_and_blocks_later_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case_written = asyncio.Event()
    original_write_markdown = graph_module.write_markdown

    async def tracked_write_markdown(directory, filename, content):
        result = await original_write_markdown(directory, filename, content)
        if filename == "case_preop.md":
            case_written.set()
        return result

    monkeypatch.setattr(graph_module, "write_markdown", tracked_write_markdown)
    client_with_failure = RateLimitedLabClient(all_fake_responses(), case_written)
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=lambda: registry_for_client(client_with_failure),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(client, thread_id, {"FAILED"})

            assert failed["error_code"] == (
                "TEST_DOCUMENT_TRANSCRIPTION_MODEL_RATE_LIMITED"
            )
            output_models = [call["output_model"] for call in client_with_failure.calls]
            assert CaseSummaryMarkdown not in output_models
            assert RadiologyReport not in output_models
            assert failed["prepared_documents"]["case_preop_md"]
            assert not failed["prepared_documents"]["tests_preop_md"]
            assert "test-page-001" in failed["error_message"]
            case_markdown = tmp_path / "uploads" / thread_id / "derived" / "case_preop.md"
            assert case_markdown.is_file()


@pytest.mark.asyncio
async def test_unexpected_runtime_error_is_persisted_and_sanitized(tmp_path: Path) -> None:
    responses = all_fake_responses()
    responses[ModeratorDecision] = STANDARD_ROUTE
    exploding = ExplodingFakeModelClient(responses)
    app = create_app(
        settings=api_settings(
            tmp_path,
            openai_api_key="api-key-value",
        ),
        registry_factory=lambda: registry_for_client(exploding),
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            failed = await wait_for_status(client, thread_id, {"FAILED"})
            serialized = json.dumps(failed)
            assert failed["error_code"] == "RUN_EXECUTION_FAILED"
            assert failed["error_message"] == (
                "The run failed due to an unexpected execution error."
            )
            assert "api-key-value" not in serialized
            assert "provider exploded" not in serialized

            stream = await client.get(f"/v1/runs/{thread_id}/events")
            assert "event: run.failed" in stream.text
            assert "api-key-value" not in stream.text
            assert "provider exploded" not in stream.text

            result = await client.get(f"/v1/runs/{thread_id}/result")
            assert result.status_code == 409
            assert result.json()["code"] == "RUN_NOT_COMPLETE"


@pytest.mark.asyncio
async def test_unknown_run_and_invalid_sse_cursor_use_stable_errors(tmp_path: Path) -> None:
    app = create_app(
        settings=api_settings(tmp_path),
        registry_factory=fake_registry,
    )
    async with app.router.lifespan_context(app):
        async with await open_client(app) as client:
            missing = await client.get("/v1/runs/not-found")
            assert missing.status_code == 404
            assert missing.json()["code"] == "RUN_NOT_FOUND"

            data, files = multipart_case(image_count=1)
            submitted = await client.post("/v1/cases", data=data, files=files)
            thread_id = submitted.json()["thread_id"]
            await wait_for_status(client, thread_id, {"COMPLETED"})
            bad_cursor = await client.get(
                f"/v1/runs/{thread_id}/events",
                headers={"Last-Event-ID": "invalid"},
            )
            assert bad_cursor.status_code == 422
            assert bad_cursor.json()["code"] == "INVALID_LAST_EVENT_ID"
            negative_cursor = await client.get(
                f"/v1/runs/{thread_id}/events",
                headers={"Last-Event-ID": "-1"},
            )
            assert negative_cursor.status_code == 422
            assert negative_cursor.json()["code"] == "INVALID_LAST_EVENT_ID"
