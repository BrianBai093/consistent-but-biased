"""Validated, path-safe storage for preoperative case packages."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from api_schemas import ImageUploadMetadata
from exceptions import UploadValidationError
from services.image_loader import validate_image_bytes
from services.page_slicer import (
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MIN_BLANK_RUN,
    slice_image_file,
)
from schemas import CaseImage, CaseInput, DocumentPage


IMAGE_TYPES = {
    ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
    ".jpg": ("image/jpeg", b"\xff\xd8\xff"),
    ".jpeg": ("image/jpeg", b"\xff\xd8\xff"),
}


class UploadStore:
    def __init__(
        self,
        root: Path | str,
        *,
        max_upload_bytes: int,
        max_images_per_case: int,
        page_slice_max_height: int = DEFAULT_MAX_HEIGHT,
        page_slice_min_blank_run: int = DEFAULT_MIN_BLANK_RUN,
        max_pages_per_document: int = 200,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_images_per_case = max_images_per_case
        self.page_slice_max_height = page_slice_max_height
        self.page_slice_min_blank_run = page_slice_min_blank_run
        self.max_pages_per_document = max_pages_per_document
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_case_package(
        self,
        *,
        thread_id: str,
        case_id: str,
        image_metadata: list[ImageUploadMetadata],
        case_files: list[UploadFile],
        test_files: list[UploadFile],
        image_files: list[UploadFile],
    ) -> CaseInput:
        if not case_files or not test_files or not image_files:
            raise UploadValidationError(
                "MISSING_CASE_ASSETS",
                "Case pages, test pages, and preoperative images are all required.",
            )
        if len(image_metadata) != len(image_files):
            raise UploadValidationError(
                "IMAGE_METADATA_MISMATCH",
                "The number of image metadata entries must match the number of image files.",
            )
        total_files = len(case_files) + len(test_files) + len(image_files)
        if total_files > self.max_images_per_case:
            raise UploadValidationError(
                "TOO_MANY_FILES",
                "The case package contains more files than the configured limit.",
                status_code=413,
            )

        staging = self.root / f".staging-{thread_id}-{uuid4().hex}"
        final_directory = self.root / thread_id
        source = staging / "source"
        case_directory = source / "case_pages"
        test_directory = source / "test_pages"
        image_directory = source / "preop_images"
        artifact_directory = staging / "derived"
        # Reserved for per-case labels that must never reach an agent: whether
        # the surgical indication is clear, what operation was actually chosen,
        # and emergency vs elective admission. Those are analysis-layer only.
        analysis_directory = staging / "analysis_only"
        for directory in (
            case_directory,
            test_directory,
            image_directory,
            artifact_directory,
            analysis_directory,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        try:
            case_pages = await self._save_pages(case_files, case_directory, "case-page")
            test_pages = await self._save_pages(test_files, test_directory, "test-page")
            images = await self._save_images(
                image_files,
                image_metadata,
                image_directory,
            )
            if final_directory.exists():
                raise UploadValidationError(
                    "UPLOAD_DIRECTORY_CONFLICT",
                    "The run upload directory already exists.",
                    status_code=409,
                )
            await asyncio.to_thread(staging.replace, final_directory)
            return CaseInput(
                case_id=case_id,
                case_pages=[
                    page.model_copy(
                        update={"path_or_url": _final_path(page.path_or_url, staging, final_directory)}
                    )
                    for page in case_pages
                ],
                test_pages=[
                    page.model_copy(
                        update={"path_or_url": _final_path(page.path_or_url, staging, final_directory)}
                    )
                    for page in test_pages
                ],
                images=[
                    image.model_copy(
                        update={"path_or_url": _final_path(image.path_or_url, staging, final_directory)}
                    )
                    for image in images
                ],
                artifact_dir=str(final_directory / "derived"),
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, staging, True)
            raise

    async def _save_pages(
        self,
        files: list[UploadFile],
        directory: Path,
        identifier_prefix: str,
    ) -> list[DocumentPage]:
        """Store uploads and expand tall stitched screenshots into real pages.

        A phone screenshot of a laboratory report is often one very tall image
        holding several independent reports. Keeping it as a single page would
        make every citation of it ambiguous, so it becomes one page per slice
        and page numbers run continuously across the whole document.
        """

        pages: list[DocumentPage] = []
        for upload_index, upload in enumerate(files, start=1):
            original = await self._save_upload(upload, directory, upload_index)
            slices = await asyncio.to_thread(
                slice_image_file,
                original,
                directory,
                f"{upload_index:03d}-{uuid4().hex}",
                max_height=self.page_slice_max_height,
                min_blank_run=self.page_slice_min_blank_run,
            )
            if len(pages) + len(slices) > self.max_pages_per_document:
                raise UploadValidationError(
                    "TOO_MANY_PAGES_AFTER_SLICING",
                    "The uploaded screenshots expand to more pages than the limit.",
                    status_code=413,
                )
            if len(slices) > 1:
                # The original is superseded by its slices; keeping it would
                # duplicate every line of the document.
                await asyncio.to_thread(original.unlink)
                sources = slices
            else:
                await asyncio.to_thread(slices[0].unlink)
                sources = [original]
            for source in sources:
                page_number = len(pages) + 1
                pages.append(
                    DocumentPage(
                        page_id=f"{identifier_prefix}-{page_number:03d}",
                        page_number=page_number,
                        path_or_url=str(source),
                        original_filename=upload.filename or f"page-{upload_index:03d}",
                    )
                )
        return pages

    async def _save_images(
        self,
        files: list[UploadFile],
        metadata: list[ImageUploadMetadata],
        directory: Path,
    ) -> list[CaseImage]:
        images: list[CaseImage] = []
        for index, (item, upload) in enumerate(zip(metadata, files, strict=True), start=1):
            destination = await self._save_upload(upload, directory, index)
            images.append(
                CaseImage(
                    image_id=item.image_id,
                    modality=item.modality,
                    path_or_url=str(destination),
                    original_filename=upload.filename or f"image-{index:03d}",
                    body_region=item.body_region,
                    view_or_plane=item.view_or_plane,
                    sequence=item.sequence,
                    level=item.level,
                    description=item.description,
                )
            )
        return images

    async def _save_upload(
        self,
        upload: UploadFile,
        directory: Path,
        index: int,
    ) -> Path:
        suffix = Path(upload.filename or "").suffix.lower()
        expected = IMAGE_TYPES.get(suffix)
        if expected is None:
            raise UploadValidationError(
                "INVALID_IMAGE_EXTENSION",
                "Only PNG, JPG, and JPEG files are supported.",
                status_code=415,
            )
        expected_content_type, magic = expected
        if upload.content_type != expected_content_type:
            raise UploadValidationError(
                "INVALID_IMAGE_CONTENT_TYPE",
                "The uploaded image content type does not match its extension.",
                status_code=415,
            )
        content = await _read_limited(upload, self.max_upload_bytes)
        if not content.startswith(magic):
            raise UploadValidationError(
                "INVALID_IMAGE_CONTENT",
                "The uploaded file is not a valid PNG or JPEG image.",
                status_code=415,
            )
        try:
            await asyncio.to_thread(validate_image_bytes, content)
        except (OSError, ValueError) as exc:
            raise UploadValidationError(
                "INVALID_IMAGE_CONTENT",
                "The uploaded file is not a decodable PNG or JPEG image.",
                status_code=415,
            ) from exc
        normalized_suffix = ".png" if suffix == ".png" else ".jpg"
        destination = directory / f"{index:03d}-{uuid4().hex}{normalized_suffix}"
        await asyncio.to_thread(destination.write_bytes, content)
        return destination


def _final_path(path: str, staging: Path, final_directory: Path) -> str:
    return str(final_directory / Path(path).relative_to(staging))


async def _read_limited(upload: UploadFile, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise UploadValidationError(
                "IMAGE_TOO_LARGE",
                "An uploaded image exceeds the configured size limit.",
                status_code=413,
            )
        chunks.append(chunk)
    if total == 0:
        raise UploadValidationError(
            "EMPTY_IMAGE",
            "Uploaded image files must not be empty.",
        )
    return b"".join(chunks)
