"""A record written after the fact already contains the answer."""

from services.record_screen import screen_markdown, summarise


def test_the_operative_note_and_the_discharge_diagnosis_are_caught():
    markdown = "<!-- source: case-page-004 -->\n\n手术记录：…。出院诊断：…。"

    findings = screen_markdown(markdown)

    assert {"operative-record", "discharge"} <= {item.category for item in findings}
    assert all(item.page_id == "case-page-004" for item in findings)


def test_the_preoperative_conference_is_caught_although_it_is_preoperative():
    """The most dangerous page is dated before the operation.

    A filter that removed "postoperative" material would keep the preoperative
    conference, and that is where the attending states the plan that was
    actually carried out.
    """

    markdown = (
        "<!-- source: case-page-003 -->\n\n"
        "术前讨论…手术方案：后路椎板切除减压、侧块螺钉植骨内固定术。"
    )

    categories = {item.category for item in screen_markdown(markdown)}

    assert "decision-already-taken" in categories
    assert "implant-or-procedure" in categories


def test_a_preoperative_record_passes():
    markdown = (
        "<!-- source: case-page-001 -->\n\n"
        "主诉：双手麻木一年。查体：小指展肌左/右=IV/IV级，Hoffmann征阴性。"
        "诊疗计划：入院常规及术前常规检查，颈椎三维CT/MRI。"
    )

    assert screen_markdown(markdown) == []


def test_the_report_names_the_document_not_the_folder():
    """What gets deleted is a file; a page number is not something to act on.

    The document's own name is written. The folder holding it is the patient's
    name and never appears.
    """

    markdown = "<!-- source: case-page-010 -->\n\n手术记录：…"
    findings = screen_markdown(markdown)

    text = summarise(findings, filenames={"case-page-010": "手术.jpg"})

    assert "手术.jpg" in text
    assert "Documents to remove or redact: 手术.jpg" in text


def test_an_untranscribed_document_is_still_screened():
    """Text with no page markers is screened as one section rather than skipped."""

    findings = screen_markdown("出院诊断：颈椎病术后")

    assert findings
    assert findings[0].page_id == "(whole document)"


def test_a_decision_stated_in_ordinary_words_is_caught():
    """Records do not always use the heading; they often just say it.

    Every line here appeared in, or reads like, the real records, and every one
    of them passed the first version of this screen.
    """

    for text in (
        "患者及家属手术意愿强烈，请示上级医生指导下一步治疗方案。",
        "建议行颈椎后路减压手术治疗。",
        "经科室讨论，同意手术。",
        "拟于明日在全麻下行颈椎手术。",
    ):
        assert screen_markdown(text), text


def test_a_postoperative_note_that_never_says_postoperative_is_caught():
    """Wound, dressing and drains date a note more reliably than the word does."""

    text = "2025-02-08 患者一般情况可，切口敷料干燥，四肢肌力较前改善。"

    categories = {item.category for item in screen_markdown(text)}

    assert categories == {"postoperative-course"}


def test_an_ordinary_preoperative_plan_is_not_flagged():
    """A screen that fires on every plan line is a screen that gets ignored.

    "拟行" is ordinary in a preoperative plan and only leaks when what is
    planned is the operation.
    """

    for text in (
        "诊疗计划：1.Ⅲ级护理，普食。2.入院常规及术前常规检查。3.拟行颈椎MRI检查。",
        "主诉：双手麻木一年。查体：Hoffmann征阴性。计划完善术前检查。",
        "患者双上肢肌力IV级，双下肢肌张力略高，膝腱反射亢进。",
    ):
        assert screen_markdown(text) == [], text


def test_a_postoperative_ward_round_is_caught():
    """The note that got through the first real clean run.

    It never writes 术后第 or 出院诊断, only 术后 and 出院 in passing, and it was
    291 characters long. It told the director the patient had already had the
    operation, and the director ruled that no further surgery was warranted.
    """

    text = (
        "日常查房记录 2025-03-31,08:35 患者一般状态良好。"
        "术后神经功能恢复良好，双上肢疼痛症状明显改善。"
        "患者及家属要求出院，请示上级医师后，同意预约明日出院。"
    )

    assert screen_markdown(text)
