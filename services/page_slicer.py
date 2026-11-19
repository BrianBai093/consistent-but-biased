"""Deterministic slicing of tall stitched screenshots into page-sized images.

A phone screenshot of a laboratory report is frequently one very tall image that
actually contains several independent report pages. Sending it as a single image
wastes resolution and makes evidence citation meaningless, so it is split before
it ever becomes a ``DocumentPage``.

Everything here is deterministic: the same bytes always produce the same cuts.
There is no model, no randomness, and no pixel enhancement -- slices are exact
crops of the source image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_MAX_HEIGHT = 2400
DEFAULT_MIN_BLANK_RUN = 8
DEFAULT_INK_THRESHOLD = 200
DEFAULT_BLANK_TOLERANCE = 1.0


@dataclass(frozen=True, slots=True)
class SliceBounds:
    """A half-open vertical range ``[top, bottom)`` of the source image."""

    top: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class BlankRun:
    top: int
    bottom: int

    @property
    def midpoint(self) -> int:
        return (self.top + self.bottom) // 2


def row_ink(
    image: Image.Image,
    *,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
) -> list[float]:
    """Return the per-row count of dark pixels.

    The BOX filter averages every column of a row into a single pixel, which is
    exactly a horizontal projection profile. This matches a NumPy per-row sum
    row for row while keeping Pillow as the only image dependency.
    """

    width, height = image.size
    mask = image.convert("L").point(lambda value: 255 if value < ink_threshold else 0)
    column = mask.resize((1, height), Image.BOX)
    scale = width / 255
    return [column.getpixel((0, y)) * scale for y in range(height)]


def find_blank_runs(
    ink: list[float],
    *,
    min_blank_run: int = DEFAULT_MIN_BLANK_RUN,
    blank_tolerance: float = DEFAULT_BLANK_TOLERANCE,
) -> list[BlankRun]:
    """Return the vertical bands that carry no text and are safe to cut inside.

    "Blank" is measured against the image's own quietest row rather than against
    absolute white. Several of these screenshots frame a white page inside grey
    margins, so every single row contains dark pixels and an absolute threshold
    would find no blank band at all.
    """

    if not ink:
        return []
    quietest = min(ink)
    limit = quietest + blank_tolerance
    runs: list[BlankRun] = []
    start: int | None = None
    for y, value in enumerate(ink):
        if value <= limit:
            if start is None:
                start = y
            continue
        if start is not None:
            if y - start >= min_blank_run:
                runs.append(BlankRun(start, y))
            start = None
    if start is not None and len(ink) - start >= min_blank_run:
        runs.append(BlankRun(start, len(ink)))
    return runs


def plan_slices(
    image: Image.Image,
    *,
    max_height: int = DEFAULT_MAX_HEIGHT,
    min_blank_run: int = DEFAULT_MIN_BLANK_RUN,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
    blank_tolerance: float = DEFAULT_BLANK_TOLERANCE,
) -> list[SliceBounds]:
    """Plan contiguous slices that never cut through a line of text.

    The returned ranges always tile the image exactly: no gaps, no overlap. An
    image that already fits within ``max_height`` yields a single full-height
    slice, so callers can treat slicing as unconditional.
    """

    if max_height < 1:
        raise ValueError("max_height must be positive")
    height = image.size[1]
    if height <= max_height:
        return [SliceBounds(0, height)]

    # Cutting through a line of text is the one outcome worth avoiding at any
    # cost, so a wide blank band is merely preferred, not required: a single
    # blank row is still a safe cut, and a hard cut is the last resort for an
    # image with no blank row at all.
    narrow_runs = find_blank_runs(
        row_ink(image, ink_threshold=ink_threshold),
        min_blank_run=1,
        blank_tolerance=blank_tolerance,
    )
    wide_runs = [run for run in narrow_runs if run.bottom - run.top >= min_blank_run]

    cuts: list[int] = []
    position = 0
    while height - position > max_height:
        upper = position + max_height

        def in_window(runs: list[BlankRun], *, start: int = position, end: int = upper):
            return [run for run in runs if start < run.midpoint <= end]

        candidates = in_window(wide_runs) or in_window(narrow_runs)
        position = candidates[-1].midpoint if candidates else upper
        cuts.append(position)
    boundaries = [0, *cuts, height]
    return [
        SliceBounds(top, bottom)
        for top, bottom in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def slice_image_file(
    source: Path,
    destination_dir: Path,
    stem: str,
    *,
    max_height: int = DEFAULT_MAX_HEIGHT,
    min_blank_run: int = DEFAULT_MIN_BLANK_RUN,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
    blank_tolerance: float = DEFAULT_BLANK_TOLERANCE,
) -> list[Path]:
    """Write the planned slices as lossless PNG files and return them in order.

    An image that needs no slicing still produces one file, so the caller never
    has to branch on whether slicing happened. PNG avoids the generational JPEG
    loss that would otherwise accumulate on every derived crop.
    """

    with Image.open(source) as image:
        image.load()
        bounds = plan_slices(
            image,
            max_height=max_height,
            min_blank_run=min_blank_run,
            ink_threshold=ink_threshold,
            blank_tolerance=blank_tolerance,
        )
        width = image.size[0]
        destination_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for index, slice_bounds in enumerate(bounds, start=1):
            destination = destination_dir / f"{stem}-s{index:02d}.png"
            crop = image.crop((0, slice_bounds.top, width, slice_bounds.bottom))
            crop.save(destination, format="PNG")
            written.append(destination)
        return written
