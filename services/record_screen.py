"""Refuse to run on a record that already contains the answer.

The experiment asks a department to decide whether to operate. A case record
written after the fact contains the operative note, the discharge diagnosis and
-- most dangerously -- the real preoperative discussion, in which the attending
states the plan that was actually carried out. Any of those reaching an agent
turns the simulation into a reading exercise.

Curating this by hand does not work. In a four-case batch the obvious files
(``手术.jpg``, ``出院.jpg``) had been noticed, and the contamination was still
present on thirteen of nineteen pages of one case, because the progress notes
run from admission to discharge and carry the preoperative conference inside
them. So the screen runs over the transcribed text, where the words actually
are, rather than over filenames.

It is deliberately noisy. A screen that stops a clean case costs a minute of
someone's attention; a screen that passes a contaminated one costs the study.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

#: Patterns that reveal what was decided or done. Each is a label and a regular
#: expression; a plain phrase is one. The label is what the report prints.
#:
#: Some of these have to be patterns rather than phrases. "拟行" is ordinary in
#: a preoperative plan ("拟行颈椎MRI"), and only becomes a leak when what is
#: planned is the operation, so the operation has to be in the pattern. A screen
#: that fires on every case's plan line is a screen that gets ignored.
DECISION_REVEALING: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "operative-record",
        (
            "手术记录",
            "手术经过",
            "术中所见",
            "麻醉记录单",
            "手术安全核查",
            "术者",
            "手术日期",
            "麻醉方式",
        ),
    ),
    (
        "discharge",
        ("出院诊断", "出院记录", "出院情况", "出院小结", "治愈出院", "出院医嘱"),
    ),
    (
        # The front sheet of a Chinese medical record has fields for exactly
        # what this study is asking the department to decide.
        "front-sheet",
        (
            "手术及操作名称",
            "手术操作名称",
            "主要手术",
            "手术级别",
            "切口愈合",
            "离院方式",
            "住院天数",
            "手术编码",
            "治疗结果",
        ),
    ),
    (
        "decision-already-taken",
        (
            "术前讨论",
            "术前小结",
            "手术方案",
            "手术指征明确",
            "手术意愿",
            "同意手术",
            "建议[^。；\n]{0,12}手术",
            "拟[行于在][^。；\n]{0,20}(?:手术|术|减压|融合|成形)",
            "全麻下行",
            "手术知情同意",
            "手术同意书",
        ),
    ),
    (
        "postoperative-course",
        (
            # Bare, after a compound-only version let a postoperative ward round
            # through: "术后神经功能恢复良好…患者及家属要求出院". A preoperative
            # record does discuss 术后 risks, so this over-flags -- which now
            # costs one dropped page rather than the study.
            "术后",
            "出院",
            "术后第",
            "术后复查",
            "术后处理",
            "术后恢复",
            "术后病程",
            # A postoperative note need never use the word "术后".
            "拆线",
            "切口",
            "敷料",
            "引流管",
            "负压引流",
            "换药",
            "伤口",
        ),
    ),
    (
        "implant-or-procedure",
        (
            "侧块螺钉",
            "椎弓根螺钉",
            "钛网",
            "钛板",
            "植骨融合",
            "内固定术",
            "椎板切除",
            "单开门",
            "椎管扩大成形",
            "颈前路减压",
            "椎间融合器",
            "cage",
        ),
    ),
)

#: The same screen for a published English case report. Such a report is not a
#: hospital record: it has no operative-note headings or front sheet, and states
#: the answer in narrative instead -- "underwent C3-7 laminoplasty", "was not
#: offered surgical intervention", "at the 1-year follow-up". It also states the
#: decision when the decision was *not* to operate ("opted for conservative
#: management", "refused surgery"), which for this study is just as much the
#: answer. Kept apart from the Chinese list, so that adding a word here cannot
#: start flagging the records that were already reviewed and signed.
ENGLISH_DECISION_REVEALING: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "decision-already-taken",
        (
            r"underwent\b",
            r"was (?:then )?(?:performed|planned|scheduled|recommended|advised|offered)",
            r"(?:recommend|advis|offer|plann|schedul|decid)\w*[^.\n]{0,40}"
            r"(?:surg|operat|decompress|fusion|laminoplast|laminectom|discectom)",
            r"(?:surg|operat)\w*[^.\n]{0,30}(?:was|were) (?:planned|recommended|indicated|performed)",
            r"(?:declin|refus|opted|elect|chose)\w*[^.\n]{0,40}"
            r"(?:surg|operat|conservative|non-?operative)",
            r"conservative(?:ly)? (?:treat|manag|therap)\w*",
            r"non-?(?:operative|surgical) (?:treat|manag)\w*",
            r"informed consent",
            r"treated (?:with|by)\b",
            r"(?:surgically|operatively|conservatively|medically) (?:treated|managed)",
            r"treated (?:surgically|operatively|conservatively)",
        ),
    ),
    (
        "intraoperative",
        (
            r"intra-?operative",
            r"\bincision\b",
            r"general an(?:a)?esthesia",
            r"prone position",
            r"supine position",
            r"blood loss",
            r"operating (?:room|theat)",
        ),
    ),
    (
        "postoperative-course",
        (
            r"post-?operative",
            r"post-?surg",
            r"after (?:the )?(?:surgery|operation|procedure|decompression)",
            r"follow-?up",
            r"discharged",
            r"\bdischarge\b",
            r"recover(?:ed|y)",
            r"at \d+ (?:days?|weeks?|months?|years?)",
        ),
    ),
    (
        "implant-or-procedure",
        (
            r"laminoplast\w*",
            r"laminectom\w*",
            r"laminotom\w*",
            r"discectom\w*",
            r"corpectom\w*",
            r"\bACDF\b",
            r"\bACCF\b",
            r"\bfusion\b",
            r"arthroplast\w*",
            r"\bscrews?\b",
            r"\bcage\b",
            r"\bplate\b",
            r"instrumentation",
            r"foraminotom\w*",
            r"open-door",
        ),
    ),
)

_PATTERN_SETS = {"zh": DECISION_REVEALING, "en": ENGLISH_DECISION_REVEALING}

_SOURCE_MARKER = re.compile(r"<!--\s*source:\s*(\S+?)\s*-->")


@lru_cache(maxsize=256)
def _matcher(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LeakFinding:
    page_id: str
    category: str
    phrase: str
    occurrences: int


def screen_markdown(markdown: str, *, language: str = "zh") -> list[LeakFinding]:
    """Report every decision-revealing phrase, page by page.

    The text is split on the ``<!-- source: ... -->`` markers the transcribers
    write, so a finding names the page a human has to look at. ``language``
    picks the phrase list: ``zh`` for hospital records, ``en`` for a published
    English case report.
    """

    patterns = _PATTERN_SETS[language]
    findings: list[LeakFinding] = []
    parts = _SOURCE_MARKER.split(markdown)
    sections: Iterable[tuple[str, str]]
    if len(parts) > 1:
        sections = zip(parts[1::2], parts[2::2])
    else:
        sections = [("(whole document)", markdown)]
    for page_id, body in sections:
        for category, phrases in patterns:
            for phrase in phrases:
                count = len(_matcher(phrase).findall(body))
                if count:
                    findings.append(
                        LeakFinding(
                            page_id=page_id,
                            category=category,
                            phrase=phrase,
                            occurrences=count,
                        )
                    )
    return findings


def summarise(
    findings: Sequence[LeakFinding],
    *,
    filenames: dict[str, str] | None = None,
) -> str:
    """A report that names what to remove, without quoting the record.

    ``filenames`` maps a page id to the document it came from, because a page
    number is not something anyone can act on: what gets deleted is a file.
    The document's own name is written -- never the folder holding it, which is
    the patient's.
    """

    if not findings:
        return "No decision-revealing content found."
    by_page: dict[str, list[LeakFinding]] = {}
    for finding in findings:
        by_page.setdefault(finding.page_id, []).append(finding)
    lines = [
        f"{len(findings)} decision-revealing phrases on {len(by_page)} page(s). "
        "These state what was decided or done, so an agent that reads them is "
        "not deciding anything.",
    ]
    for page_id, items in by_page.items():
        detail = ", ".join(
            f"{item.phrase}x{item.occurrences}" for item in sorted(
                items, key=lambda entry: entry.phrase
            )
        )
        categories = sorted({item.category for item in items})
        source = (filenames or {}).get(page_id)
        where = f"{page_id} ({source})" if source else page_id
        lines.append(f"  {where}: [{', '.join(categories)}] {detail}")
    files = sorted({(filenames or {}).get(page.page_id, "") for page in findings} - {""})
    if files:
        lines.append("")
        lines.append("Documents to remove or redact: " + ", ".join(files))
    return "\n".join(lines)
