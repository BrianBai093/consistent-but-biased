from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from api_schemas import ImageUploadMetadata
from schemas import Modality
from services.upload_store import UploadStore, UploadValidationError


def png_bytes(width: int, height: int, *, bands: int = 0) -> bytes:
    """A white image, optionally with evenly spaced black text-like bands."""

    image = Image.new("L", (width, height), color=255)
    if bands:
        gap = height // (bands + 1)
        for index in range(bands):
            top = gap * (index + 1) - 10
            for y in range(top, top + 20):
                for x in range(10, width - 10):
                    image.putpixel((x, y), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def upload(name: str, payload: bytes) -> UploadFile:
    return UploadFile(
        file=BytesIO(payload),
        filename=name,
        headers=Headers({"content-type": "image/png"}),
    )


def store(tmp_path: Path, **kwargs) -> UploadStore:
    return UploadStore(
        tmp_path / "uploads",
        max_upload_bytes=20 * 1024 * 1024,
        max_images_per_case=50,
        **kwargs,
    )


async def save(upload_store: UploadStore, *, case_payloads: list[bytes]):
    return await upload_store.save_case_package(
        thread_id="thread-slicing",
        case_id="case-slicing",
        image_metadata=[ImageUploadMetadata(image_id="mri-1", modality=Modality.MRI)],
        case_files=[
            upload(f"case-{index:02d}.png", payload)
            for index, payload in enumerate(case_payloads, start=1)
        ],
        test_files=[upload("tests-01.png", png_bytes(400, 300))],
        image_files=[upload("mri.png", png_bytes(400, 300))],
    )


async def test_a_short_upload_stays_a_single_page(tmp_path: Path):
    case = await save(store(tmp_path), case_payloads=[png_bytes(400, 300)])

    assert [page.page_id for page in case.case_pages] == ["case-page-001"]
    assert Path(case.case_pages[0].path_or_url).is_file()


async def test_a_tall_upload_becomes_one_page_per_slice(tmp_path: Path):
    tall = png_bytes(400, 3000, bands=20)

    case = await save(store(tmp_path, page_slice_max_height=800), case_payloads=[tall])

    assert len(case.case_pages) > 1
    assert [page.page_id for page in case.case_pages] == [
        f"case-page-{index:03d}" for index in range(1, len(case.case_pages) + 1)
    ]
    assert all(Path(page.path_or_url).is_file() for page in case.case_pages)


async def test_page_numbers_stay_contiguous_across_several_uploads(tmp_path: Path):
    """``CaseInput`` requires page numbers to run 1..N with no gaps."""

    case = await save(
        store(tmp_path, page_slice_max_height=800),
        case_payloads=[
            png_bytes(400, 3000, bands=20),
            png_bytes(400, 300),
            png_bytes(400, 2000, bands=12),
        ],
    )

    assert [page.page_number for page in case.case_pages] == list(
        range(1, len(case.case_pages) + 1)
    )
    assert len(case.case_pages) > 3


async def test_a_sliced_upload_leaves_no_duplicate_of_the_whole_screenshot(
    tmp_path: Path,
):
    """The un-sliced original would otherwise duplicate every line it contains."""

    case = await save(
        store(tmp_path, page_slice_max_height=800),
        case_payloads=[png_bytes(400, 3000, bands=20)],
    )

    stored = sorted(Path(case.case_pages[0].path_or_url).parent.iterdir())
    assert len(stored) == len(case.case_pages)


async def test_original_filenames_are_preserved_for_every_slice(tmp_path: Path):
    case = await save(
        store(tmp_path, page_slice_max_height=800),
        case_payloads=[png_bytes(400, 3000, bands=20)],
    )

    assert {page.original_filename for page in case.case_pages} == {"case-01.png"}


async def test_an_analysis_only_directory_is_reserved_beside_the_sources(
    tmp_path: Path,
):
    """Per-case labels live outside the agent-visible tree by construction."""

    case = await save(store(tmp_path), case_payloads=[png_bytes(400, 300)])

    run_directory = Path(case.artifact_dir).parent
    assert (run_directory / "analysis_only").is_dir()
    assert sorted(path.name for path in run_directory.iterdir()) == [
        "analysis_only",
        "derived",
        "source",
    ]


async def test_an_explosive_expansion_is_rejected(tmp_path: Path):
    upload_store = store(tmp_path, page_slice_max_height=200, max_pages_per_document=5)

    with pytest.raises(UploadValidationError) as caught:
        await save(upload_store, case_payloads=[png_bytes(400, 4000, bands=30)])

    assert caught.value.code == "TOO_MANY_PAGES_AFTER_SLICING"
