"""The runner's own decisions, the ones made before any money is spent."""

from pathlib import Path

import pytest

from tools.run_department_experiment import (
    AmbiguousFolder,
    baseline_prefix,
    collect_sources,
    _category,
    _modality,
)
from schemas import AgentRole, Modality


def build_case(root: Path, folders: dict[str, list[str]]) -> Path:
    case = root / "case"
    for folder, names in folders.items():
        directory = case / folder
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return case


def test_a_folder_that_fits_two_categories_stops_the_run():
    """Guessing here would change the evidence all three arms share.

    "术前检查" reads as both preoperative imaging and an examination. Taking the
    first match would send the sheets to the radiologist as film and drop them
    from the test pages, and the log would show only a count.
    """

    with pytest.raises(AmbiguousFolder):
        _category("术前检查")
    with pytest.raises(AmbiguousFolder):
        _category("随访资料")


def test_postoperative_material_is_excluded_whatever_else_the_name_says():
    """It states which operation was performed: that is the answer, not an input."""

    assert _category("术后") == "excluded"
    assert _category("术后复查检验") == "excluded"


def test_the_two_spellings_of_the_test_folder_both_work():
    assert _category("入院常规检验及检查") == "test_pages"
    assert _category("入院常规检查及检验") == "test_pages"
    assert _category("基础资料") == "case_pages"
    assert _category("术前") == "images"


def test_sources_are_sorted_by_the_folder_each_file_sits_in(tmp_path: Path):
    case = build_case(
        tmp_path,
        {
            "基础资料": ["入院记录1.png", "病例1.png"],
            "入院常规检验及检查": ["1.png"],
            "术前": ["MRI术前.png"],
            "术后": ["MRI术后.png"],
        },
    )

    sources = collect_sources(case)

    assert len(sources.case_pages) == 2
    assert len(sources.test_pages) == 1
    assert len(sources.images) == 1
    assert len(sources.excluded) == 1
    assert sources.routing["术后"] == "excluded"


def test_an_image_without_a_modality_in_its_name_is_refused(tmp_path: Path):
    """A film filed under the wrong modality is evidence nobody can audit."""

    with pytest.raises(ValueError):
        _modality(Path("术前/scan1.jpg"))
    assert _modality(Path("术前/MRI术前20250206 (1).JPG")) is Modality.MRI
    assert _modality(Path("术前/X-RAY术前20250205.JPG")) is Modality.XRAY


def test_the_baseline_prefix_ends_where_the_style_begins():
    """Only the frozen part is required to be identical between arms."""

    report = {
        "transcript": [
            {
                "sequence": index,
                "speaker_role": role,
                "speaker_name": role,
                "content": "x",
            }
            for index, role in enumerate(
                [
                    AgentRole.RESIDENT.value,
                    AgentRole.RADIOLOGIST.value,
                    AgentRole.VICE_DIRECTOR.value,
                    AgentRole.DIRECTOR.value,
                ],
                start=1,
            )
        ]
    }

    prefix = baseline_prefix(report)

    assert [item[1] for item in prefix] == [
        AgentRole.RESIDENT.value,
        AgentRole.RADIOLOGIST.value,
    ]


def test_the_operative_note_is_never_read_whatever_folder_it_is_in(tmp_path: Path):
    """A rule set by the study's clinician: 手术.jpg is never an input.

    It is excluded before ingestion, upstream of the leak screen, so no flag
    that relaxes the screen can bring it back. The discharge summary states the
    same operation and goes with it.
    """

    case = build_case(
        tmp_path,
        {
            "基础资料": ["入院.jpg", "手术.jpg", "出院.jpg", "病案.jpg"],
            "入院常规检验及检查": ["1.png"],
            "术前": ["MRI术前.png"],
        },
    )

    sources = collect_sources(case)

    assert sorted(path.name for path in sources.case_pages) == ["入院.jpg", "病案.jpg"]
    assert sorted(path.name for path in sources.excluded) == ["出院.jpg", "手术.jpg"]


def test_a_repeated_arm_does_not_overwrite_the_first():
    """Two runs of one cell are the evidence of whether the cell is stable."""

    from schemas import DirectorStyle
    from tools.run_department_experiment import report_name

    assert report_name("arm", 2, DirectorStyle.AVOIDANT) == "arm-avoidant-attempt2.json"
    assert report_name("arm", 3, DirectorStyle.AVOIDANT) != report_name(
        "arm", 2, DirectorStyle.AVOIDANT
    )
    assert report_name("summary", 2) == "summary-attempt2.json"
