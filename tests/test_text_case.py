"""A published, text-only case: what changes, and what must not.

The hospital path is untouched by all of this -- its screen and its baseline
keys are what existing signatures were given over -- so several tests here
check the Chinese defaults as much as the English additions.
"""

import json
from pathlib import Path

import pytest

from agents.case_report_filter import MAX_FILTER_ATTEMPTS, CaseReportFilterAgent
from department.graph import create_department_graph, department_input, thread_config
from department.schemas import ImagingFindings
from schemas import DirectorStyle, DocumentPage
from services.baseline_store import BaselineStore, TextCaseBaselineStore
from services.checkpoint import open_sqlite_checkpointer
from services.llm_client import FakeModelClient
from exceptions import StructuredOutputError
from services.case_report_filter import (
    FilteredCaseReport,
    FilterViolation,
    check_deletion_only,
)
from services.record_screen import screen_markdown
from tests.department_factories import department_with_client
from tools import run_text_case_experiment as runner

KEPT = (
    "A 62-year-old man presented with gait disturbance for 6 months (Fig. 1). "
    "His temperature was 36.5 C. Hoffmann sign was positive bilaterally and the JOA "
    "score was 14. MRI showed severe stenosis at C4-5 with T2 signal change."
)


def filtered(text: str = KEPT) -> FilteredCaseReport:
    return FilteredCaseReport.model_validate(
        {
            "preoperative_text": text,
            "removals": [{"category": "decision", "reason": "the operation"}],
        }
    )


REPORT = (
    "A 62-year-old man presented with gait disturbance for 6 months (Fig. 1). "
    "His temperature was 36.5 C. Hoffmann sign was positive bilaterally and the JOA "
    "score was 14. MRI showed severe stenosis at C4-5 with T2 signal change. "
    "He underwent C3-7 laminoplasty. At the 1-year follow-up he walked unaided.\n"
)


# ------------------------------------------------------------------ screen ---


def test_the_english_screen_finds_the_decision_and_the_follow_up():
    categories = {item.category for item in screen_markdown(REPORT, language="en")}
    assert {"decision-already-taken", "postoperative-course", "implant-or-procedure"} <= (
        categories
    )


def test_an_adverb_alone_can_state_the_treatment():
    assert screen_markdown(
        "The patient was surgically treated after a couple of months.", language="en"
    )


def test_a_decision_not_to_operate_is_also_the_answer():
    text = "She was not offered surgery and opted for conservative management."
    assert screen_markdown(text, language="en")


def test_findings_alone_do_not_trip_the_english_screen():
    findings = (
        "MRI showed severe stenosis at C4-5 with T2 signal change. Hoffmann sign "
        "was positive. He had numbness of both hands for two years."
    )
    assert screen_markdown(findings, language="en") == []


def test_the_chinese_screen_did_not_gain_the_english_words():
    """New hits there would block re-runs of records already signed."""

    assert screen_markdown("He underwent laminoplasty at follow-up.") == []


# ------------------------------------------------------------------ filter ---

MIXED = (
    "A 63-year-old man with a 1-year history of difficulty walking (Nurick grade 3, "
    "mJOA score of 15) presented to the clinic in preparation for a C3-C6 laminectomy. "
    "There was no weakness. MRI showed mild stenosis at C5-6. He underwent surgery."
)
MIXED_KEPT = (
    "A 63-year-old man with a 1-year history of difficulty walking (Nurick grade 3, "
    "mJOA score of 15). There was no weakness. MRI showed mild stenosis at C5-6."
)


def test_a_clause_can_be_cut_and_the_finding_beside_it_kept():
    """The case sentence deletion could not handle: presentation and plan in one."""

    passages = check_deletion_only(MIXED, MIXED_KEPT)
    assert [item.text for item in passages] == [
        "presented to the clinic in preparation for a C3-C6 laminectomy",
        "He underwent surgery.",
    ]


@pytest.mark.parametrize(
    "edited",
    [
        MIXED_KEPT.replace("no weakness", "weakness"),
        MIXED_KEPT.replace("mild ", ""),
    ],
)
def test_a_word_or_two_cut_from_a_kept_sentence_is_refused(edited: str):
    """Dropping "no" or "mild" changes a finding instead of removing one."""

    with pytest.raises(FilterViolation, match="inside a sentence"):
        check_deletion_only(MIXED, edited)


def test_a_paraphrase_is_refused():
    with pytest.raises(FilterViolation, match="not in the record"):
        check_deletion_only(MIXED, MIXED_KEPT.replace("difficulty walking", "gait trouble"))


def test_a_bare_connective_may_go():
    source = "He had neck pain and he underwent fusion. Reflexes were brisk."
    check_deletion_only(source, "He had neck pain. Reflexes were brisk.")


async def test_a_refused_answer_is_asked_for_again_with_the_reason():
    client = FakeModelClient(
        {FilteredCaseReport: [filtered(KEPT.replace("severe ", "")), filtered()]}
    )
    result, passages = await CaseReportFilterAgent(client).run(record=REPORT)

    assert result.output.preoperative_text == KEPT
    assert len(client.calls) == 2
    second = json.loads(client.calls[1]["user_prompt"])
    assert "severe" in second["your_previous_answer_was_refused_because"]
    assert "laminoplasty" in " ".join(item.text for item in passages)


async def test_a_filter_refused_every_time_fails_the_case():
    bad = filtered(KEPT + " She recovered well.")
    client = FakeModelClient({FilteredCaseReport: [bad] * MAX_FILTER_ATTEMPTS})
    with pytest.raises(StructuredOutputError):
        await CaseReportFilterAgent(client).run(record=REPORT)


# ------------------------------------------------------------------ sources ---


def test_only_the_patient_summary_is_read_never_the_title(tmp_path: Path):
    csv_path = tmp_path / "pmc.csv"
    csv_path.write_text(
        "patient_id,patient_uid,PMID,file_path,title,patient,age,gender,"
        "relevant_articles,similar_patients\n"
        '0,123-1,9,x.xml,Surgical treatment for myelopathy,"Line one.\\nLine two.",'
        "[],M,{},{}\n",
        encoding="utf-8",
    )
    text, pmid = runner.read_pmc_patient(csv_path, "123-1")
    assert text == "Line one.\nLine two.\n"
    assert pmid == "9"
    with pytest.raises(KeyError):
        runner.read_pmc_patient(csv_path, "missing")


def _page(path: Path) -> DocumentPage:
    return DocumentPage(
        page_id="case-page-001",
        page_number=1,
        path_or_url=str(path),
        original_filename="case-page-001.txt",
    )


async def test_a_text_case_is_keyed_by_its_bytes_not_its_path(tmp_path: Path):
    store = TextCaseBaselineStore(tmp_path / "baselines")
    first, second, other = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"
    first.write_text(REPORT)
    second.write_text(REPORT)
    other.write_text(REPORT + "More.")

    key = await store._key([_page(first)], [], [], "pmc")
    assert key is not None
    assert key == await store._key([_page(second)], [], [], "pmc")
    assert key != await store._key([_page(other)], [], [], "pmc")
    assert key != await store._key([_page(first)], [], [], "another-label")


async def test_the_image_store_still_refuses_a_case_without_films(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text(REPORT)
    assert await BaselineStore(tmp_path / "b")._key([_page(path)], [], [], "pmc") is None


# -------------------------------------------------------------------- graph ---


def _text_input(tmp_path: Path, style: DirectorStyle) -> dict:
    path = tmp_path / "source" / "case-page-001.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(REPORT)
    return department_input(
        case_id="pmc-dept",
        director_style=style,
        artifact_dir=str(tmp_path / "derived"),
        case_pages=[_page(path)],
        test_pages=[],
        images=[],
        case_preop_md=runner.record_markdown("A 62-year-old man. MRI showed stenosis."),
        tests_preop_md=runner.NO_TEST_SHEET_MD,
        case_summary_md="Cervical myelopathy.",
    )


async def test_without_films_the_radiologist_reads_the_record_and_no_one_gets_images(
    tmp_path: Path,
):
    store = TextCaseBaselineStore(tmp_path / "baselines")
    calls = []
    for index, style in enumerate(DirectorStyle):
        registry, client = department_with_client()
        async with open_sqlite_checkpointer(tmp_path / f"arm-{index}.sqlite3") as saver:
            graph = create_department_graph(
                registry=registry,
                checkpointer=saver,
                baseline_store=store,
            )
            result = await graph.ainvoke(
                _text_input(tmp_path, style),
                config=thread_config(f"text-arm-{index}"),
            )
        assert result["baseline_key"] is not None
        calls.append(client.calls)

    first_arm = calls[0]
    radiology = [call for call in first_arm if call["output_model"] is ImagingFindings]
    assert len(radiology) == 1
    payload = json.loads(radiology[0]["user_prompt"])
    assert set(payload) == {"task", "case_preop_md"}
    assert "MRI showed stenosis" in payload["case_preop_md"]
    assert registry.radiologist.record_only_prompt in radiology[0]["system_prompt"]
    assert all(call["images"] == [] for arm in calls for call in arm)
    # Stage I-II was paid for once and frozen; the later arms only negotiate.
    assert not any(call["output_model"] is ImagingFindings for call in calls[1])
    assert len(list((tmp_path / "baselines").glob("*.json"))) == 1


# ------------------------------------------------------------- whole runner ---


async def test_the_runner_filters_once_freezes_it_and_runs_three_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BASELINE_STORE_DIR", str(tmp_path / "baselines"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    case_file = tmp_path / "case.txt"
    case_file.write_text(REPORT)
    filter_client = FakeModelClient({FilteredCaseReport: filtered()})
    monkeypatch.setattr(runner, "DeepSeekTextClient", lambda config, thinking: filter_client)
    clients = []

    def fake_department(settings):
        registry, client = department_with_client()
        clients.append(client)
        return registry, []

    monkeypatch.setattr(runner, "create_department", fake_department)
    out = tmp_path / "out"
    base = ["--case-file", str(case_file), "--case-id", "pmc-test", "--out", str(out)]

    assert await runner.main(base) == 0
    assert await runner.main([*base, "--attempt", "2"]) == 0

    # Filtered once; the second run read the frozen filter and the frozen baseline.
    assert len(filter_client.calls) == 1
    case_preop = (out / "pmc-test" / "case_preop.md").read_text()
    assert "laminoplasty" not in case_preop and "follow-up" not in case_preop
    assert "Hoffmann sign was positive" in case_preop
    summary = json.loads((out / "pmc-test" / "summary-attempt1.json").read_text())
    assert summary["filter"]["approved_by"] == "automatic"
    assert summary["filter"]["removed_passages"] == 1
    assert summary["imaging_source"] == "record_text"
    assert summary["validity"]["one_baseline"] is True
    assert summary["validity"]["identical_baseline_transcript"] is True
    assert "case_report_filter" in summary["prompt_fingerprints"]
    removed = json.loads((out / "pmc-test" / "analysis_only" / "removed_passages.json").read_text())
    assert "laminoplasty" in removed[0]["text"]
    assert all(call["images"] == [] for client in clients for call in client.calls)


async def test_a_refiltered_case_negotiates_on_new_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The old transcript of a finished thread must not collide with the new one."""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BASELINE_STORE_DIR", str(tmp_path / "baselines"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    case_file = tmp_path / "case.txt"
    case_file.write_text(REPORT)
    answers = iter([filtered(), filtered(KEPT.split(" MRI showed")[0])])
    monkeypatch.setattr(
        runner,
        "DeepSeekTextClient",
        lambda config, thinking: FakeModelClient({FilteredCaseReport: next(answers)}),
    )
    monkeypatch.setattr(
        runner, "create_department", lambda settings: (department_with_client()[0], [])
    )
    out = tmp_path / "out"
    base = ["--case-file", str(case_file), "--case-id", "refilter", "--out", str(out)]

    assert await runner.main(base) == 0
    assert await runner.main([*base, "--redo-filter", "--attempt", "1"]) == 0
    summary = json.loads((out / "refilter" / "summary-attempt1.json").read_text())
    assert summary["failed_arms"] == []


async def test_a_frozen_preparation_is_reused_even_though_the_record_ends_in_a_newline(
    tmp_path: Path,
):
    path = tmp_path / "case.txt"
    path.write_text(REPORT)
    case = runner.TextCase(
        case_id="reuse",
        case_pages=[_page(path)],
        test_pages=[],
        images=[],
        artifact_dir=str(tmp_path),
    )
    store = TextCaseBaselineStore(tmp_path / "baselines")
    await store.store_preparation(
        case_pages=case.case_pages,
        test_pages=[],
        images=[],
        case_id="reuse",
        case_preop_md=runner.record_markdown(REPORT),
        tests_preop_md=runner.NO_TEST_SHEET_MD,
        case_summary_md="Frozen summary.",
        preparation_warnings=[],
        redaction_sha="a" * 64,
    )
    prepared = await runner.prepare_text(
        case,
        store=store,
        case_preop_md=runner.record_markdown(REPORT),
        redaction_sha="a" * 64,
    )
    assert prepared.reused and prepared.case_summary_md == "Frozen summary."


# -------------------------------------------------------------------- batch ---


async def test_the_batch_passes_only_the_uid_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import run_text_case_batch as batch

    manifest = tmp_path / "set.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {"patient_uid": "1-1", "group": "g", "real_decision": "not_operated"},
                    {"patient_uid": "2-1", "group": "g", "real_decision": "operated"},
                ],
            }
        )
    )
    out = tmp_path / "out"
    done = out / "pmc-1-1"
    done.mkdir(parents=True)
    (done / "summary-attempt1.json").write_text(
        json.dumps(
            {
                "arms": [],
                "validity": {"one_baseline": True, "identical_baseline_transcript": True},
                "filter": {"attempts": 1, "removed_passages": 2, "unreviewed_screen_hits": []},
                "failed_arms": [],
            }
        )
    )
    for style in DirectorStyle:
        (done / f"arm-{style.value}-attempt1.json").write_text("{}")
    seen = []

    async def fake_main(argv):
        seen.append(argv)
        return 0

    monkeypatch.setattr(batch.runner, "main", fake_main)
    code = await batch.main(["--manifest", str(manifest), "--out", str(out)])

    # The finished case was not run again; the other was, with nothing but its uid.
    assert len(seen) == 1
    assert seen[0][seen[0].index("--pmc-uid") + 1] == "2-1"
    assert not any("operated" in item for item in seen[0])
    assert [seen[0][i + 1] for i, item in enumerate(seen[0]) if item == "--only-style"] == [
        style.value for style in DirectorStyle
    ]
    report = json.loads((out / "batch-tiny-attempt1.json").read_text())
    assert [row["result"] is not None for row in report["cases"]] == [True, False]
    assert code == 1


async def test_a_text_case_makes_no_summary_call(tmp_path: Path):
    """No role reads a summary when there are no films; none is paid for."""

    path = tmp_path / "case.txt"
    path.write_text(REPORT)
    case = runner.TextCase(
        case_id="nosummary",
        case_pages=[_page(path)],
        test_pages=[],
        images=[],
        artifact_dir=str(tmp_path),
    )
    prepared = await runner.prepare_text(
        case,
        store=TextCaseBaselineStore(tmp_path / "baselines"),
        case_preop_md=runner.record_markdown(REPORT),
        redaction_sha="b" * 64,
    )
    assert prepared.model_calls == 0
    assert prepared.case_summary_md == runner.NO_SUMMARY_MD


async def test_an_arm_added_later_joins_the_summary_instead_of_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The control arm was added after the three styles had run."""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BASELINE_STORE_DIR", str(tmp_path / "baselines"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    case_file = tmp_path / "case.txt"
    case_file.write_text(REPORT)
    monkeypatch.setattr(
        runner,
        "DeepSeekTextClient",
        lambda config, thinking: FakeModelClient({FilteredCaseReport: filtered()}),
    )
    monkeypatch.setattr(
        runner, "create_department", lambda settings: (department_with_client()[0], [])
    )
    out = tmp_path / "out"
    base = ["--case-file", str(case_file), "--case-id", "later", "--out", str(out)]
    styles = [style for style in DirectorStyle if style is not DirectorStyle.NEUTRAL]

    assert await runner.main(
        [*base, *[arg for style in styles for arg in ("--only-style", style.value)]]
    ) == 0
    assert await runner.main([*base, "--only-style", "neutral"]) == 0

    summary = json.loads((out / "later" / "summary-attempt1.json").read_text())
    assert [arm["director_style"] for arm in summary["arms"]] == [
        style.value for style in DirectorStyle
    ]
    # The control arm read the same frozen evidence as the three that ran first.
    assert summary["validity"]["one_baseline"] is True
    assert summary["validity"]["identical_baseline_transcript"] is True


async def test_a_series_runs_on_its_own_threads_over_the_same_frozen_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """M5/M6 change who decides or how they are told, never what they read."""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BASELINE_STORE_DIR", str(tmp_path / "baselines"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    case_file = tmp_path / "case.txt"
    case_file.write_text(REPORT)
    filter_client = FakeModelClient({FilteredCaseReport: filtered()})
    monkeypatch.setattr(runner, "DeepSeekTextClient", lambda config, thinking: filter_client)
    clients = []

    def fake_department(settings):
        registry, client = department_with_client()
        clients.append(client)
        return registry, []

    monkeypatch.setattr(runner, "create_department", fake_department)
    out = tmp_path / "out"
    base = ["--case-file", str(case_file), "--case-id", "s", "--out", str(out)]

    assert await runner.main(base) == 0
    assert await runner.main([*base, "--series", "wording-b"]) == 0

    main = json.loads((out / "s" / "summary-attempt1.json").read_text())
    series = json.loads((out / "series-wording-b" / "s" / "summary-attempt1.json").read_text())
    assert series["series"] == "wording-b"
    assert series["validity"]["baseline_keys"] == main["validity"]["baseline_keys"]
    # The filter and stage I-II were paid for once, by the main run.
    assert len(filter_client.calls) == 1
    assert not any(call["output_model"] is ImagingFindings for call in clients[1].calls)


async def test_the_batch_stops_when_every_arm_of_a_case_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An exhausted balance fails every arm; the remaining cases must not be tried."""

    from tools import run_text_case_batch as batch

    manifest = tmp_path / "set.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {"patient_uid": uid, "group": "g", "real_decision": "operated"}
                    for uid in ("1-1", "2-1", "3-1")
                ],
            }
        )
    )
    out = tmp_path / "out"
    seen = []

    async def refused(argv):
        case_id = argv[argv.index("--case-id") + 1]
        seen.append(case_id)
        folder = out / case_id
        folder.mkdir(parents=True, exist_ok=True)
        for style in DirectorStyle:
            (folder / f"arm-{style.value}-attempt1-failed.json").write_text("{}")
        return 1

    monkeypatch.setattr(batch.runner, "main", refused)
    assert await batch.main(["--manifest", str(manifest), "--out", str(out)]) == 1
    assert seen == ["pmc-1-1"]
