"""A model says what to remove; the program removes it; a person signs the diff."""

from pathlib import Path

import pytest

from agents.record_redactor import RecordRedactorAgent
from services.baseline_store import BaselineStore
from services.llm_client import FakeModelClient
from services.record_redaction import (
    Deletion,
    RedactionCategory,
    RedactionPlan,
    RedactionRecord,
    apply,
    removed_sentences,
    render_review,
    segment,
    source_sha256,
    units,
)
from tests.department_factories import case_sources, department_payload

RECORD = (
    "<!-- source: case-page-001 -->\n\n"
    "主诉：颈痛伴双上肢麻木20天。查体：会阴部皮肤感觉减退，\n"
    "小指展肌肌力IV级。诊断：脊髓型颈椎病。\n\n"
    "<!-- source: case-page-002 -->\n\n"
    "术前讨论：手术方案为后路椎板切除减压。患者及家属手术意愿强烈。\n\n"
    "<!-- source: case-page-003 -->\n\n"
    "术后神经功能恢复良好。同意预约明日出院。\n"
)


def unit_number(fragment: str) -> int:
    return next(unit.number for unit in units(segment(RECORD)) if fragment in unit.text)


def plan(*pairs: tuple[str, RedactionCategory]) -> RedactionPlan:
    return RedactionPlan(
        deletions=[
            Deletion(unit=unit_number(fragment), category=category, reason="states it")
            for fragment, category in pairs
        ]
    )


def record_for(redaction_plan: RedactionPlan, **overrides) -> RedactionRecord:
    values = {
        "source_sha256": source_sha256(RECORD),
        "unit_count": len(units(segment(RECORD))),
        "deletions": list(redaction_plan.deletions),
    }
    values.update(overrides)
    return RedactionRecord(**values)


# ------------------------------------------------------------- segmentation ---


def test_numbering_the_record_loses_nothing():
    segments = segment(RECORD)

    assert "".join(item.text for item in segments) == RECORD
    assert apply(segments, []) == RECORD


def test_a_unit_is_a_sentence_even_when_ocr_broke_the_line():
    """OCR line breaks are page layout; a sentence often runs over two lines."""

    spanning = next(unit for unit in units(segment(RECORD)) if "会阴部" in unit.text)

    assert "\n" in spanning.text
    assert spanning.text.endswith("小指展肌肌力IV级。")


def test_a_page_without_sentence_punctuation_is_still_divided():
    """A form or a table would otherwise be one unit, and deleting it takes everything."""

    table = "<!-- source: case-page-001 -->\n\n" + "".join(
        f"项目{index}  结果{index}\n" for index in range(40)
    )

    assert len(units(segment(table))) > 1


# ------------------------------------------------------------------ applying ---


def test_the_cleaned_record_is_the_transcription_minus_the_removed_sentences():
    """The model returns numbers, so the program's output can only be a subsequence."""

    segments = segment(RECORD)
    removal = {unit_number("手术方案"), unit_number("手术意愿")}

    cleaned = apply(segments, removal)

    remaining = iter(RECORD)
    assert all(character in remaining for character in cleaned)
    assert "手术方案" not in cleaned
    assert "会阴部皮肤感觉减退" in cleaned


def test_a_page_with_nothing_left_is_dropped_with_its_heading():
    segments = segment(RECORD)
    whole_page = {unit.number for unit in units(segments) if unit.page_id == "case-page-003"}

    cleaned = apply(segments, whole_page)

    assert "case-page-003" not in cleaned
    assert "case-page-001" in cleaned and "case-page-002" in cleaned


# ----------------------------------------------------------------- signature ---


def test_the_signature_names_exactly_what_is_removed_from_exactly_what():
    base = record_for(plan(("手术方案", RedactionCategory.DECISION)))

    more = base.model_copy(update={"reviewer_removed": [unit_number("手术意愿")]})
    other_source = base.model_copy(update={"source_sha256": "0" * 64})

    assert base.signature() == record_for(plan(("手术方案", RedactionCategory.DECISION))).signature()
    assert more.signature() != base.signature()
    assert other_source.signature() != base.signature()


def test_a_signature_stops_counting_the_moment_the_deletions_change():
    base = record_for(plan(("手术方案", RedactionCategory.DECISION)))
    signed = base.model_copy(update={"approved_sha": base.signature()})

    corrected = signed.model_copy(update={"reviewer_restored": [unit_number("手术方案")]})

    assert signed.approved is True
    assert corrected.approved is False


def test_a_reviewer_can_take_back_evidence_and_add_what_was_missed():
    record = record_for(
        plan(
            ("手术方案", RedactionCategory.DECISION),
            ("会阴部", RedactionCategory.DECISION),
        ),
        reviewer_removed=[unit_number("手术意愿")],
        reviewer_restored=[unit_number("会阴部")],
    )

    assert record.removals() == {unit_number("手术方案"), unit_number("手术意愿")}


# -------------------------------------------------------- the analysis layer ---


def test_what_was_removed_is_kept_for_the_analysis_not_thrown_away():
    record = record_for(
        plan(("术后神经功能", RedactionCategory.POSTOPERATIVE)),
        reviewer_removed=[unit_number("明日出院")],
    )

    rows = removed_sentences(segment(RECORD), record)

    assert [row["category"] for row in rows] == ["postoperative", "reviewer"]
    assert rows[0]["text"].startswith("术后神经功能恢复良好")
    assert rows[0]["page_id"] == "case-page-003"


# -------------------------------------------------------------- the review ---


def test_the_review_puts_a_removed_finding_in_front_of_the_reviewer():
    """Removing evidence makes the indication look cleaner; that is the worse error."""

    record = record_for(
        plan(
            ("会阴部", RedactionCategory.DECISION),
            ("手术方案", RedactionCategory.DECISION),
        )
    )

    review = render_review(
        case_id="case-1",
        segments=segment(RECORD),
        record=record,
        screen_findings=[],
    )

    finding_line = next(line for line in review.splitlines() if "会阴部" in line)
    plan_line = next(line for line in review.splitlines() if "手术方案" in line)
    assert "⚠" in finding_line
    assert "⚠" not in plan_line
    assert record.signature() in review
    assert "Keyword screen on the cleaned record: clean" in review


# ------------------------------------------------------------------- the agent ---


async def test_the_redactor_returns_numbers_and_is_held_to_the_record():
    """A number past the end is refused and retried with a reason, not applied."""

    segments = segment(RECORD)
    count = len(units(segments))
    beyond = RedactionPlan(
        deletions=[Deletion(unit=count + 5, category=RedactionCategory.DECISION, reason="x")]
    )
    valid = plan(("手术方案", RedactionCategory.DECISION))
    client = FakeModelClient({RedactionPlan: [beyond, valid]}, max_retries=1)

    result = await RecordRedactorAgent(client).run(segments=segments)

    assert result.output == valid
    assert len(client.calls) == 2
    assert "less than or equal to" in client.calls[1]["user_prompt"]
    # The sentences travel with their numbers; nothing else is asked of it.
    assert '"n": 1' in client.calls[0]["user_prompt"]


# -------------------------------------------------------------- the whole flow ---


def complete_plan() -> RedactionPlan:
    """Everything in RECORD that states the answer, so the screen is left with nothing."""

    return plan(
        ("手术方案", RedactionCategory.DECISION),
        ("手术意愿", RedactionCategory.DECISION),
        ("术后神经功能", RedactionCategory.POSTOPERATIVE),
        ("明日出院", RedactionCategory.POSTOPERATIVE),
    )


def case_for(tmp_path: Path):
    from schemas import CaseInput

    return CaseInput(case_id="case-1", artifact_dir=str(tmp_path / "derived"), **case_sources(tmp_path))


def planner(redaction_plan: RedactionPlan, calls: list):
    async def plan_for(segments):
        calls.append(len(units(segments)))
        return redaction_plan, 1

    return plan_for


async def test_the_first_run_plans_and_waits_for_a_signature(tmp_path: Path):
    from tools.run_department_experiment import redact

    store = BaselineStore(tmp_path / "baselines")
    calls: list = []
    out = tmp_path / "out"
    out.mkdir()

    redaction = await redact(
        raw_md=RECORD,
        case=case_for(tmp_path),
        store=store,
        plan_for=planner(plan(("手术方案", RedactionCategory.DECISION)), calls),
        out=out,
    )

    assert calls == [len(units(segment(RECORD)))]
    assert redaction.record.approved is False
    assert "手术方案" not in (out / "case_preop.md").read_text(encoding="utf-8")
    # The unredacted record exists only in the analysis layer.
    assert "手术方案" in (out / "analysis_only" / "case_preop.raw.md").read_text(encoding="utf-8")
    assert (out / "redaction-review.md").exists()


async def test_a_signed_plan_is_read_back_and_never_asked_for_again(tmp_path: Path):
    """A second plan would differ, and the signature would describe neither."""

    from tools.run_department_experiment import redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()
    case = case_for(tmp_path)
    calls: list = []
    first = await redact(
        raw_md=RECORD,
        case=case,
        store=store,
        plan_for=planner(complete_plan(), calls),
        out=out,
    )

    signed = await redact(
        raw_md=RECORD,
        case=case,
        store=store,
        plan_for=planner(RedactionPlan(deletions=[]), calls),
        out=out,
        approve=first.record.signature(),
    )
    later = await redact(
        raw_md=RECORD,
        case=case,
        store=store,
        plan_for=planner(RedactionPlan(deletions=[]), calls),
        out=out,
    )

    assert len(calls) == 1
    assert signed.record.approved is True
    assert later.record.approved is True
    assert later.clean_md == first.clean_md


async def test_a_signature_for_a_different_set_of_deletions_is_refused(tmp_path: Path):
    from tools.run_department_experiment import SignatureMismatch, redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()
    case = case_for(tmp_path)
    first = await redact(
        raw_md=RECORD,
        case=case,
        store=store,
        plan_for=planner(plan(("手术方案", RedactionCategory.DECISION)), []),
        out=out,
    )

    with pytest.raises(SignatureMismatch):
        await redact(
            raw_md=RECORD,
            case=case,
            store=store,
            plan_for=planner(RedactionPlan(deletions=[]), []),
            out=out,
            also_remove=[unit_number("手术意愿")],
            approve=first.record.signature(),
        )


async def test_what_the_redaction_left_behind_is_found_by_the_screen(tmp_path: Path):
    """The diff shows what was removed; it cannot show what was missed."""

    from tools.run_department_experiment import redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()

    redaction = await redact(
        raw_md=RECORD,
        case=case_for(tmp_path),
        store=store,
        plan_for=planner(plan(("手术方案", RedactionCategory.DECISION)), []),
        out=out,
    )

    assert any("手术意愿" in line for line in redaction.findings)


async def test_a_new_redaction_retires_the_stage_one_and_two_built_on_the_old_one(
    tmp_path: Path,
):
    """The five roles described a record the arms would no longer read."""

    store = BaselineStore(tmp_path / "baselines")
    case = case_for(tmp_path)
    identity = {
        "case_pages": case.case_pages,
        "test_pages": case.test_pages,
        "images": case.images,
        "case_id": case.case_id,
    }
    await store.store_preparation(
        **identity,
        case_preop_md="cleaned A",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=[],
        redaction_sha="a" * 64,
    )
    await store.store_department(
        **identity,
        case_preop_md="cleaned A",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=[],
        payload=department_payload(),
    )

    await store.store_preparation(
        **identity,
        case_preop_md="cleaned B",
        tests_preop_md="Labs.",
        case_summary_md="Summary.",
        preparation_warnings=[],
        redaction_sha="b" * 64,
    )

    loaded = await store.load(**identity)
    assert loaded is not None
    assert loaded.redaction_sha == "b" * 64
    assert loaded.department is None


PAST_HISTORY = (
    "<!-- source: case-page-001 -->\n\n"
    "既往史：阑尾炎术后十年。查体：小指展肌肌力IV级。\n\n"
    "<!-- source: case-page-002 -->\n\n"
    "术前讨论：手术方案为后路椎板切除减压。\n"
)


async def test_signing_over_what_the_screen_still_flags_needs_a_judgement(tmp_path: Path):
    """The screen is noisy on purpose; a person decides what its hits are.

    "阑尾炎术后" is past surgical history, not this operation. Without a way to
    say so, the only exit was to switch the screen off entirely -- including the
    later screens of the test pages and the summary.
    """

    from tools.run_department_experiment import UnacceptedScreenFindings, redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()
    case = case_for(tmp_path)
    conference = next(
        unit.number for unit in units(segment(PAST_HISTORY)) if "手术方案" in unit.text
    )
    only_the_plan = RedactionPlan(
        deletions=[Deletion(unit=conference, category=RedactionCategory.DECISION, reason="plan")]
    )
    first = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
    )
    assert first.unaccepted  # 术后 in the past history

    with pytest.raises(UnacceptedScreenFindings):
        await redact(
            raw_md=PAST_HISTORY, case=case, store=store,
            plan_for=planner(only_the_plan, []), out=out,
            approve=first.record.signature(),
        )

    signed = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
        approve=first.record.signature(), accept_screen_hits=True,
    )
    assert signed.record.approved is True
    assert signed.unaccepted == []
    assert signed.record.accepted_screen_findings == first.findings
    review = (out / "redaction-review.md").read_text(encoding="utf-8")
    assert "阑尾炎术后十年" in review  # the flagged sentence is shown in its words


async def test_a_correction_withdraws_the_judgement_on_the_old_record(tmp_path: Path):
    from tools.run_department_experiment import redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()
    case = case_for(tmp_path)
    conference = next(
        unit.number for unit in units(segment(PAST_HISTORY)) if "手术方案" in unit.text
    )
    history = next(
        unit.number for unit in units(segment(PAST_HISTORY)) if "阑尾炎" in unit.text
    )
    only_the_plan = RedactionPlan(
        deletions=[Deletion(unit=conference, category=RedactionCategory.DECISION, reason="plan")]
    )
    first = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
    )
    await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
        approve=first.record.signature(), accept_screen_hits=True,
    )

    corrected = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
        also_remove=[history],
    )

    assert corrected.record.approved is False
    assert corrected.record.accepted_screen_findings == []


async def test_a_judgement_on_the_screen_survives_being_read_back(tmp_path: Path):
    """The next run reads the record from disk, not from the signing run's memory.

    The stored record strips surrounding whitespace and the screen's report
    indents its lines, so a finding accepted at signing stopped matching once
    read back. The signing run itself went through; every later run of that
    case was refused on hits a person had already judged.
    """

    from tools.run_department_experiment import redact

    store = BaselineStore(tmp_path / "baselines")
    out = tmp_path / "out"
    out.mkdir()
    case = case_for(tmp_path)
    conference = next(
        unit.number for unit in units(segment(PAST_HISTORY)) if "手术方案" in unit.text
    )
    only_the_plan = RedactionPlan(
        deletions=[Deletion(unit=conference, category=RedactionCategory.DECISION, reason="plan")]
    )
    first = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
    )
    await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
        approve=first.record.signature(), accept_screen_hits=True,
    )

    later = await redact(
        raw_md=PAST_HISTORY, case=case, store=store,
        plan_for=planner(only_the_plan, []), out=out,
    )

    assert later.record.approved is True
    assert later.findings  # the screen still sees the past history ...
    assert later.unaccepted == []  # ... and the reviewer's ruling still holds
