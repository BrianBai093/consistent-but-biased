"""Content addressing for one case's manually screened source material.

Two things need to answer the same question -- "is this the identical set of
source files?" -- so they share one fingerprint: the radiology cache, which
skips an image-model call on an exact repeat, and the baseline store, which lets
the three discussion arms of an experiment read one frozen preparation instead
of regenerating it.

A fingerprint covers file bytes plus sanitized image metadata. It never contains
a local path, an original filename, a case ID, or a thread ID.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from pydantic import Field

from schemas import CaseImage, DocumentPage, StrictModel

SLOT_PREFIX = "cached-image-"


class CachedImageDescriptor(StrictModel):
    """One image reduced to its content hash and sanitized metadata.

    ``slot_id`` is positional rather than the run's own image ID, so a record
    stays valid for a later run that names the same images differently.
    """

    slot_id: str = Field(pattern=r"^cached-image-\d{3}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    modality: str
    body_region: str | None = None
    view_or_plane: str | None = None
    sequence: str | None = None
    level: str | None = None
    description: str | None = None


class SourceFingerprint(StrictModel):
    case_page_sha256: list[str] = Field(min_length=1)
    test_page_sha256: list[str] = Field(min_length=1)
    images: list[CachedImageDescriptor] = Field(min_length=1)

    def key(self, format_version: str) -> str:
        """A stable hex digest, salted by the caller's own record format.

        Including the format version means a record written under an older
        layout can never be mistaken for a current one.
        """

        payload = {
            "format_version": format_version,
            **self.model_dump(mode="json"),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


async def fingerprint_inputs(
    case_pages: Sequence[DocumentPage],
    test_pages: Sequence[DocumentPage],
    images: Sequence[CaseImage],
) -> SourceFingerprint | None:
    """Hash every source file, or return None when one cannot be read.

    A remote or data URL has no stable local bytes, so it yields no fingerprint
    at all rather than a weaker one.
    """

    if not case_pages or not test_pages or not images:
        return None
    paths = [
        *(item.path_or_url for item in case_pages),
        *(item.path_or_url for item in test_pages),
        *(item.path_or_url for item in images),
    ]
    digests = await asyncio.gather(*(sha256_local_file(path) for path in paths))
    if any(digest is None for digest in digests):
        return None
    digest_values = [str(digest) for digest in digests]
    case_end = len(case_pages)
    test_end = case_end + len(test_pages)
    image_digests = digest_values[test_end:]
    descriptors = [
        CachedImageDescriptor(
            slot_id=f"{SLOT_PREFIX}{index:03d}",
            content_sha256=digest,
            modality=image.modality.value,
            body_region=image.body_region,
            view_or_plane=image.view_or_plane,
            sequence=image.sequence,
            level=image.level,
            description=image.description,
        )
        for index, (image, digest) in enumerate(
            zip(images, image_digests, strict=True),
            start=1,
        )
    ]
    return SourceFingerprint(
        case_page_sha256=digest_values[:case_end],
        test_page_sha256=digest_values[case_end:test_end],
        images=descriptors,
    )


async def sha256_local_file(path_or_url: str) -> str | None:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return None
    path = Path(path_or_url)
    try:
        return await asyncio.to_thread(_sha256_file, path)
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_record_atomically(root: Path, key: str, payload: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{key}.{uuid4().hex}.tmp"
    destination = root / f"{key}.json"
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
