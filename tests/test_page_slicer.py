from pathlib import Path

import pytest
from PIL import Image

from services.page_slicer import (
    SliceBounds,
    find_blank_runs,
    plan_slices,
    row_ink,
    slice_image_file,
)


def stitched_report(
    *,
    blocks: int,
    block_height: int,
    gap_height: int,
    width: int = 200,
) -> Image.Image:
    """A white image with evenly spaced black text bands separated by blank gaps."""

    height = blocks * block_height + (blocks + 1) * gap_height
    image = Image.new("L", (width, height), color=255)
    for index in range(blocks):
        top = gap_height + index * (block_height + gap_height)
        for y in range(top, top + block_height):
            for x in range(10, width - 10):
                image.putpixel((x, y), 0)
    return image


def text_rows(image: Image.Image) -> set[int]:
    return {y for y, value in enumerate(row_ink(image)) if value > 1.0}


def test_short_image_is_returned_as_a_single_full_height_slice():
    image = stitched_report(blocks=2, block_height=40, gap_height=20)

    bounds = plan_slices(image, max_height=image.size[1] + 1)

    assert bounds == [SliceBounds(0, image.size[1])]


def test_slices_tile_the_image_without_gaps_or_overlap():
    image = stitched_report(blocks=40, block_height=40, gap_height=30)

    bounds = plan_slices(image, max_height=300)

    assert bounds[0].top == 0
    assert bounds[-1].bottom == image.size[1]
    for earlier, later in zip(bounds[:-1], bounds[1:], strict=True):
        assert earlier.bottom == later.top
    assert sum(item.height for item in bounds) == image.size[1]


def test_no_cut_lands_inside_a_line_of_text():
    image = stitched_report(blocks=40, block_height=40, gap_height=30)
    inked = text_rows(image)

    bounds = plan_slices(image, max_height=300)

    interior_cuts = [item.top for item in bounds[1:]]
    assert interior_cuts
    assert not [cut for cut in interior_cuts if cut in inked]


def test_slice_height_never_exceeds_the_configured_maximum():
    image = stitched_report(blocks=40, block_height=40, gap_height=30)

    bounds = plan_slices(image, max_height=300)

    assert all(item.height <= 300 for item in bounds)


def test_a_uniformly_inked_page_still_terminates_and_tiles_exactly():
    image = Image.new("L", (200, 900), color=0)

    bounds = plan_slices(image, max_height=300)

    assert bounds[0].top == 0
    assert bounds[-1].bottom == 900
    assert all(item.height <= 300 for item in bounds)
    assert sum(item.height for item in bounds) == 900


def test_a_white_page_inside_grey_margins_still_finds_blank_bands():
    """Several real screenshots frame a white page inside grey margins.

    Every row then contains dark pixels, so an absolute ink threshold finds no
    blank band at all and the planner is forced to cut through text.
    """

    width, page_left, page_right = 300, 80, 220
    image = stitched_report(blocks=40, block_height=40, gap_height=30, width=width)
    for y in range(image.size[1]):
        for x in list(range(page_left)) + list(range(page_right, width)):
            image.putpixel((x, y), 133)
    ink = row_ink(image)
    inked = {y for y, value in enumerate(ink) if value > min(ink) + 1.0}

    bounds = plan_slices(image, max_height=300)

    interior_cuts = [item.top for item in bounds[1:]]
    assert interior_cuts
    assert not [cut for cut in interior_cuts if cut in inked]


def test_planning_is_deterministic_for_identical_input():
    image = stitched_report(blocks=40, block_height=40, gap_height=30)

    first = plan_slices(image, max_height=300)
    second = plan_slices(image, max_height=300)

    assert first == second


def test_blank_runs_shorter_than_the_minimum_are_not_cut_candidates():
    ink = [0.0] * 4 + [9.0] * 10 + [0.0] * 12

    runs = find_blank_runs(ink, min_blank_run=8)

    assert [(run.top, run.bottom) for run in runs] == [(14, 26)]


def test_written_slices_are_ordered_lossless_crops_of_the_source(tmp_path: Path):
    image = stitched_report(blocks=40, block_height=40, gap_height=30)
    source = tmp_path / "source.png"
    image.save(source)

    written = slice_image_file(source, tmp_path / "out", "case-page-001", max_height=300)

    assert [path.name for path in written] == [
        f"case-page-001-s{index:02d}.png" for index in range(1, len(written) + 1)
    ]
    bounds = plan_slices(image, max_height=300)
    for path, slice_bounds in zip(written, bounds, strict=True):
        with Image.open(path) as crop:
            crop.load()
            expected = image.crop((0, slice_bounds.top, image.size[0], slice_bounds.bottom))
            assert crop.size == expected.size
            assert crop.convert("L").tobytes() == expected.tobytes()


def test_writing_is_deterministic_across_repeated_runs(tmp_path: Path):
    image = stitched_report(blocks=40, block_height=40, gap_height=30)
    source = tmp_path / "source.png"
    image.save(source)

    first = slice_image_file(source, tmp_path / "a", "case-page-001", max_height=300)
    second = slice_image_file(source, tmp_path / "b", "case-page-001", max_height=300)

    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_an_image_needing_no_slicing_still_produces_one_file(tmp_path: Path):
    image = stitched_report(blocks=2, block_height=40, gap_height=20)
    source = tmp_path / "source.png"
    image.save(source)

    written = slice_image_file(source, tmp_path / "out", "test-page-007", max_height=10_000)

    assert [path.name for path in written] == ["test-page-007-s01.png"]


def test_a_non_positive_maximum_height_is_rejected():
    image = stitched_report(blocks=2, block_height=40, gap_height=20)

    with pytest.raises(ValueError):
        plan_slices(image, max_height=0)
