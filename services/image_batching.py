"""Bounded partitioning of case images for multimodal review.

A single request carrying every image of a case is both slow and liable to hit
provider limits, so images are reviewed in ordered, bounded batches. Order is
preserved because a reader cites images by position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from schemas import CaseImage

DEFAULT_BATCH_MAX_IMAGES = 4
DEFAULT_BATCH_MAX_BYTES = 12 * 1024 * 1024


def local_image_size(image: CaseImage) -> int | None:
    """Size on disk, or None for a remote image whose size is unknown."""

    path_or_url = image.path_or_url
    if path_or_url.startswith(("http://", "https://", "data:")):
        return None
    try:
        return Path(path_or_url).stat().st_size
    except OSError:
        return None


def partition_images(
    images: Sequence[CaseImage],
    *,
    max_images: int = DEFAULT_BATCH_MAX_IMAGES,
    max_bytes: int = DEFAULT_BATCH_MAX_BYTES,
) -> list[list[CaseImage]]:
    """Split images into ordered batches bounded by count and by total bytes.

    An image whose size cannot be determined still counts toward the image
    limit, so an unreadable path can never produce an unbounded batch.
    """

    batches: list[list[CaseImage]] = []
    current: list[CaseImage] = []
    current_bytes = 0
    for image in images:
        image_bytes = local_image_size(image)
        exceeds_count = len(current) >= max_images
        exceeds_bytes = bool(
            current
            and image_bytes is not None
            and current_bytes + image_bytes > max_bytes
        )
        if exceeds_count or exceeds_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(image)
        if image_bytes is not None:
            current_bytes += image_bytes
    if current:
        batches.append(current)
    return batches
