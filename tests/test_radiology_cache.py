from pathlib import Path

import pytest

from graph import create_opening_graph, opening_input, thread_config
from schemas import (
    CaseImage,
    CaseInput,
    CompressionDistribution,
    DocumentPage,
    DynamicStability,
    FixationImagingSupport,
    Modality,
    OperativePlanningAssessment,
    PosteriorCordDriftFeasibility,
    SagittalAlignment,
)
from services.checkpoint import open_sqlite_checkpointer
from services.radiology_cache import RadiologyCache, migrate_pre_v4_report
from tests.dynamic_factories import registry_with_client
from tests.factories import radiology_report


def _local_case(
    tmp_path: Path,
    *,
    case_id: str,
    image_id: str,
    artifact_name: str,
) -> CaseInput:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    case_page = source / "case.png"
    test_page = source / "test.png"
    image = source / "image.jpg"
    case_page.write_bytes(b"same case page bytes")
    test_page.write_bytes(b"same test page bytes")
    image.write_bytes(b"same preoperative image bytes")
    return CaseInput(
        case_id=case_id,
        case_pages=[
            DocumentPage(
                page_id="case-page-001",
                page_number=1,
                path_or_url=str(case_page),
                original_filename="private-case.png",
            )
        ],
        test_pages=[
            DocumentPage(
                page_id="test-page-001",
                page_number=1,
                path_or_url=str(test_page),
                original_filename="private-test.png",
            )
        ],
        images=[
            CaseImage(
                image_id=image_id,
                modality=Modality.MRI,
                path_or_url=str(image),
                original_filename="private-image.jpg",
                body_region="lumbar spine",
                view_or_plane="sagittal",
                sequence="T2",
            )
        ],
        artifact_dir=str(tmp_path / artifact_name),
    )


@pytest.mark.asyncio
async def test_cache_round_trip_remaps_ids_and_omits_paths(tmp_path: Path) -> None:
    cache = RadiologyCache(tmp_path / "cache")
    original = _local_case(
        tmp_path,
        case_id="case-original",
        image_id="mri-1",
        artifact_name="original-derived",
    )
    key = await cache.store(
        case_pages=original.case_pages,
        test_pages=original.test_pages,
        images=original.images,
        report=radiology_report(),
    )
    assert key is not None

    repeated = _local_case(
        tmp_path,
        case_id="case-repeated",
        image_id="new-image-id",
        artifact_name="repeated-derived",
    )
    loaded = await cache.load(
        case_pages=repeated.case_pages,
        test_pages=repeated.test_pages,
        images=repeated.images,
    )

    assert loaded is not None
    assert loaded.mri_findings[0].image_ids == ["new-image-id"]
    assert loaded.operative_planning.evidence_ids == ["new-image-id"]
    serialized = (tmp_path / "cache" / f"{key}.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "path_or_url" not in serialized
    assert "original_filename" not in serialized
    assert "mri-1" not in serialized
    assert "case-original" not in serialized


@pytest.mark.asyncio
async def test_cache_misses_when_sanitized_metadata_changes(tmp_path: Path) -> None:
    cache = RadiologyCache(tmp_path / "cache")
    original = _local_case(
        tmp_path,
        case_id="case-original",
        image_id="mri-1",
        artifact_name="original-derived",
    )
    await cache.store(
        case_pages=original.case_pages,
        test_pages=original.test_pages,
        images=original.images,
        report=radiology_report(),
    )
    changed_images = [
        original.images[0].model_copy(update={"body_region": "cervical spine"})
    ]

    loaded = await cache.load(
        case_pages=original.case_pages,
        test_pages=original.test_pages,
        images=changed_images,
    )

    assert loaded is None


@pytest.mark.asyncio
async def test_opening_graph_skips_radiologist_model_on_exact_cache_hit(
    tmp_path: Path,
) -> None:
    cache = RadiologyCache(tmp_path / "cache")
    registry, client = registry_with_client(moderator_responses=[])
    first = _local_case(
        tmp_path,
        case_id="case-first",
        image_id="mri-1",
        artifact_name="first-derived",
    )
    second = _local_case(
        tmp_path,
        case_id="case-second",
        image_id="remapped-mri-id",
        artifact_name="second-derived",
    )
    async with open_sqlite_checkpointer(tmp_path / "cache-hit.sqlite3") as checkpointer:
        graph = create_opening_graph(
            agent_registry=registry,
            checkpointer=checkpointer,
            radiology_cache=cache,
        )
        first_result = await graph.ainvoke(
            opening_input(first),
            config=thread_config("thread-cache-first"),
        )
        second_result = await graph.ainvoke(
            opening_input(second),
            config=thread_config("thread-cache-second"),
        )

    radiology_calls = [
        call for call in client.calls if call["output_model"].__name__ == "RadiologyReport"
    ]
    assert len(radiology_calls) == 1
    assert first_result["run_metadata"].radiology_cache_hit is False
    assert second_result["run_metadata"].radiology_cache_hit is True
    assert second_result["radiology_report"].mri_findings[0].image_ids == [
        "remapped-mri-id"
    ]
    # Test-report transcription and case summarization remain model calls; the
    # case record is deterministic OCR and the radiology call was a cache hit.
    assert len(second_result["run_metadata"].model_calls) == 2


def test_pre_v4_migration_is_conservative_and_drops_addenda() -> None:
    old = radiology_report().model_copy(
        update={
            "alignment": "Cervical lordosis is straightened. Dynamic alignment is unknown.",
            "neural_compression": "Multilevel ventral cord compression is present.",
            "operative_planning": OperativePlanningAssessment(),
        }
    )

    migrated = migrate_pre_v4_report(old)

    assert migrated.operative_planning.sagittal_alignment is (
        SagittalAlignment.STRAIGHTENED
    )
    assert migrated.operative_planning.dynamic_stability is DynamicStability.NOT_ASSESSED
    assert migrated.operative_planning.compression_distribution is (
        CompressionDistribution.VENTRAL
    )
    assert migrated.operative_planning.posterior_cord_drift_feasibility is (
        PosteriorCordDriftFeasibility.INDETERMINATE
    )
    assert migrated.operative_planning.fixation_imaging_support is (
        FixationImagingSupport.NOT_ASSESSED
    )
    assert migrated.operative_planning.evidence_ids == ["mri-1"]
    assert migrated.addenda == []
