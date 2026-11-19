"""Remove what a case record says about the decision, and prove that is all.

A case record written after the fact states the answer the department is being
asked to reach: the preoperative conference, the plan, the operation, the
postoperative days. Keyword screening alone both over- and under-blocks, so the
record goes through three steps that each check a different thing:

1. **A model decides what to remove -- and nothing else.** It is shown the record
   as numbered sentences and returns the numbers to delete, each with a category
   and a reason. It never returns text. So it cannot paraphrase, correct an OCR
   error, merge two sentences, or summarise, and the cleaned record is by
   construction a subsequence of the transcription, character for character.
   That preserves the reason the case record is transcribed by deterministic OCR
   in the first place.
2. **This module applies the deletions.** The diff between the transcription and
   the cleaned record is therefore exactly the deleted sentences.
3. **A person reviews that diff and signs it.** The signature is a hash of the
   transcription and the set of deletions; change either, and the signature no
   longer matches. Removing too much is the failure that matters most here --
   deleting an examination finding makes the surgical indication look cleaner
   than it was, which would manufacture the result the study is measuring -- and
   only a reviewer reading the diff can see it. Removing too little is not
   visible in the diff at all, which is why the keyword screen still runs over
   the cleaned record.

The removed sentences are not discarded. They are the analysis-only layer of the
case: evidence of what was decided and done, kept for the analysis and never
shown to an agent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Sequence

from pydantic import Field

from schemas import StrictModel

#: A unit ends where a sentence does. OCR line breaks are page layout, not
#: meaning -- a sentence routinely runs across two lines -- so they stay inside
#: the unit rather than splitting it.
_SENTENCE_END = re.compile(r"(?<=[。；！？])")
_LINE_END = re.compile(r"(?<=\n)")
_MARKER = re.compile(r"(<!--\s*source:\s*\S+?\s*-->)")
_MARKER_ID = re.compile(r"source:\s*(\S+?)\s*-->")
#: A page without sentence punctuation -- a form, a table -- would otherwise be
#: one unit, and deleting it would take everything with it.
MAX_UNIT_CHARS = 200


class RedactionCategory(str, Enum):
    """Why a sentence may not reach the department."""

    #: Whether to operate, which operation, which levels, which approach, which
    #: implants, consent to operate.
    DECISION = "decision"
    #: What happened in theatre.
    INTRAOPERATIVE = "intraoperative"
    #: Anything after the operation, including discharge.
    POSTOPERATIVE = "postoperative"
    #: The real department's own opinion on treatment -- including "the surgical
    #: indication is clear", which is the answer in one phrase.
    TREATMENT_OPINION = "treatment_opinion"


class Deletion(StrictModel):
    unit: int = Field(ge=1)
    category: RedactionCategory
    reason: str = Field(min_length=1)


class RedactionPlan(StrictModel):
    """What the model returns: numbers to delete, never text."""

    deletions: list[Deletion]


class RedactionRecord(StrictModel):
    """A plan, the reviewer's corrections to it, and whether it was signed.

    Frozen per case, so every arm -- and every later run -- reads the same
    cleaned record, and a signature given once stays valid until either the
    transcription or the set of deletions changes.
    """

    source_sha256: str = Field(min_length=64, max_length=64)
    unit_count: int = Field(ge=0)
    deletions: list[Deletion]
    #: Added by the reviewer: something the model left in that states the answer.
    reviewer_removed: list[int] = Field(default_factory=list)
    #: Taken back by the reviewer: something the model removed that is evidence.
    reviewer_restored: list[int] = Field(default_factory=list)
    approved_sha: str | None = None
    approved_at: datetime | None = None
    #: Keyword-screen hits on the cleaned record that the reviewer read and
    #: judged not to be leaks -- "阑尾炎术后" is past history, not this operation.
    #: The screen is meant to be noisy; a person is the one who decides.
    accepted_screen_findings: list[str] = Field(default_factory=list)

    def unaccepted(self, findings: Sequence[str]) -> list[str]:
        """Findings the reviewer has not ruled on, compared as text, not layout.

        The record strips surrounding whitespace when it is stored (as every
        ``StrictModel`` does) and the screen's report indents its lines. Compared
        raw, a finding accepted at signing stopped matching the moment the
        record was read back from disk, and every later run of that case was
        refused on hits a person had already judged.
        """

        accepted = {item.strip() for item in self.accepted_screen_findings}
        return [
            item
            for item in findings
            if item.strip() and item.strip() not in accepted
        ]

    def removals(self) -> set[int]:
        removed = {item.unit for item in self.deletions} | set(self.reviewer_removed)
        return removed - set(self.reviewer_restored)

    def signature(self) -> str:
        """Names exactly what will be deleted from exactly which transcription."""

        body = self.source_sha256 + ":" + ",".join(str(n) for n in sorted(self.removals()))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @property
    def approved(self) -> bool:
        return self.approved_sha is not None and self.approved_sha == self.signature()


@dataclass(frozen=True, slots=True)
class Segment:
    """A piece of the transcription.

    ``number`` is set for a sentence that may be deleted; markers and the
    whitespace between sentences have none and are always kept, so that the
    page structure survives any set of deletions.
    """

    text: str
    page_id: str
    number: int | None


class UnknownUnitError(ValueError):
    """The plan names a sentence that does not exist."""


def source_sha256(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def segment(markdown: str) -> list[Segment]:
    """Split a transcription into numbered sentences, losing nothing.

    Joining every segment's text reproduces the input exactly.
    """

    segments: list[Segment] = []
    number = 0
    page_id = "(before the first page)"
    for piece in _MARKER.split(markdown):
        if not piece:
            continue
        if _MARKER.fullmatch(piece):
            page_id = _MARKER_ID.search(piece).group(1)
            segments.append(Segment(piece, page_id, None))
            continue
        for sentence in _SENTENCE_END.split(piece):
            parts = (
                _LINE_END.split(sentence)
                if len(sentence) > MAX_UNIT_CHARS
                else [sentence]
            )
            for part in parts:
                if not part:
                    continue
                stripped = part.lstrip()
                lead = part[: len(part) - len(stripped)]
                if lead:
                    segments.append(Segment(lead, page_id, None))
                if stripped:
                    number += 1
                    segments.append(Segment(stripped, page_id, number))
    return segments


def units(segments: Sequence[Segment]) -> list[Segment]:
    return [item for item in segments if item.number is not None]


def check_plan(plan: RedactionPlan, unit_count: int) -> None:
    """A deletion that names no sentence is a plan that cannot be applied."""

    unknown = sorted({item.unit for item in plan.deletions if item.unit > unit_count})
    if unknown:
        raise UnknownUnitError(
            f"the plan names sentences that do not exist: {unknown} "
            f"(the record has {unit_count})"
        )


def apply(segments: Sequence[Segment], removals: Iterable[int]) -> str:
    """The transcription with the given sentences removed.

    A page left with no sentences is dropped entirely rather than kept as an
    empty heading.
    """

    removed = set(removals)
    pages: list[tuple[str, list[Segment]]] = []
    for item in segments:
        if not pages or (item.number is None and _MARKER.fullmatch(item.text)):
            pages.append((item.page_id, []))
        pages[-1][1].append(item)

    kept: list[str] = []
    for _, page in pages:
        page_units = [item for item in page if item.number is not None]
        if page_units and all(item.number in removed for item in page_units):
            continue
        kept.extend(
            item.text
            for item in page
            if item.number is None or item.number not in removed
        )
    return "".join(kept)


def removed_sentences(
    segments: Sequence[Segment],
    record: RedactionRecord,
) -> list[dict]:
    """The analysis-only layer: what was decided and done, never shown to an agent."""

    by_number = {item.number: item for item in units(segments)}
    plan = {item.unit: item for item in record.deletions}
    rows = []
    for number in sorted(record.removals()):
        unit = by_number[number]
        deletion = plan.get(number)
        rows.append(
            {
                "unit": number,
                "page_id": unit.page_id,
                "text": unit.text,
                "category": deletion.category.value if deletion else "reviewer",
                "reason": deletion.reason if deletion else "removed by the reviewer",
            }
        )
    return rows


#: Vocabulary of a preoperative finding. A removed sentence carrying any of it is
#: the one a reviewer most needs to look at: removing evidence makes the
#: indication look cleaner than it was.
FINDING_VOCABULARY = (
    "查体",
    "肌力",
    "反射",
    "感觉",
    "征",
    "阳性",
    "阴性",
    "压痛",
    "叩击痛",
    "麻木",
    "疼痛",
    "萎缩",
    "步态",
    "踩棉",
    "MRI",
    "CT",
    "狭窄",
    "突出",
    "信号",
    "主诉",
    "现病史",
    "既往",
)


def render_review(
    *,
    case_id: str,
    segments: Sequence[Segment],
    record: RedactionRecord,
    screen_findings: Sequence[str],
    flagged_sentences: Sequence[tuple[int, str]] = (),
    filenames: dict[str, str] | None = None,
    context: int = 1,
) -> str:
    """The diff a reviewer signs, with each removal shown in its context."""

    all_units = units(segments)
    removed = record.removals()
    plan = {item.unit: item for item in record.deletions}
    counts: dict[str, int] = {}
    for number in removed:
        deletion = plan.get(number)
        key = deletion.category.value if deletion else "reviewer"
        counts[key] = counts.get(key, 0) + 1

    lines = [
        f"# Redaction review: {case_id}",
        "",
        f"Signature to approve: `{record.signature()}`",
        "",
        f"Removed {len(removed)} of {len(all_units)} sentences: "
        + (", ".join(f"{key} {value}" for key, value in sorted(counts.items())) or "none"),
    ]
    if record.reviewer_removed or record.reviewer_restored:
        lines.append(
            f"Reviewer corrections: +{len(record.reviewer_removed)} removed, "
            f"{len(record.reviewer_restored)} restored"
        )
    lines.append(
        "Keyword screen on the cleaned record: "
        + ("clean" if not screen_findings else f"{len(screen_findings)} hit(s)")
    )
    lines.extend(f"  - {finding}" for finding in screen_findings)
    if flagged_sentences:
        lines.extend(
            [
                "",
                "Kept sentences the screen flags. Remove the ones that state this "
                "operation or its aftermath; if every one is something else -- past "
                "surgical history, a risk explained at consent -- sign with "
                "--accept-screen-hits and the judgement is recorded with the signature:",
            ]
        )
        lines.extend(
            f"  [{number}] {' '.join(text.split())}" for number, text in flagged_sentences
        )
    lines.extend(
        [
            "",
            "Check both directions:",
            "- Nothing marked `-` may be a preoperative finding -- history, symptom, "
            "examination, test or imaging result. ⚠ marks a removal that uses the "
            "vocabulary of one; read those first. Removing evidence makes the "
            "indication look cleaner than it was.",
            "- Nothing left unmarked may say whether, what or where to operate, or "
            "anything from after the decision.",
            "",
            "Correct with --also-remove N,N and --restore N,N; approve with "
            "--approve-redaction <signature>.",
        ]
    )

    position = {item.number: index for index, item in enumerate(all_units)}
    shown: set[int] = set()
    for number in sorted(removed):
        index = position[number]
        start, end = max(0, index - context), min(len(all_units), index + context + 1)
        shown.update(all_units[i].number for i in range(start, end))

    current_page = None
    previous = None
    for unit in all_units:
        if unit.number not in shown:
            continue
        if unit.page_id != current_page:
            current_page = unit.page_id
            source = (filenames or {}).get(unit.page_id)
            lines.extend(["", f"## {unit.page_id}" + (f" ({source})" if source else "")])
            previous = None
        if previous is not None and unit.number != previous + 1:
            lines.append("      …")
        text = " ".join(unit.text.split())
        if unit.number in removed:
            deletion = plan.get(unit.number)
            label = (
                f"〔{deletion.category.value}〕{deletion.reason}"
                if deletion
                else "〔reviewer〕"
            )
            flag = " ⚠" if any(word in unit.text for word in FINDING_VOCABULARY) else ""
            lines.append(f"  - [{unit.number}]{flag} {text}   {label}")
        else:
            lines.append(f"    [{unit.number}] {text}")
        previous = unit.number
    return "\n".join(lines) + "\n"


def flagged_sentences(
    segments: Sequence[Segment],
    removals: Iterable[int],
    matches,
) -> list[tuple[int, str]]:
    """Kept sentences that ``matches`` flags, so a hit can be judged in its words."""

    removed = set(removals)
    return [
        (item.number, item.text)
        for item in units(segments)
        if item.number not in removed and matches(item.text)
    ]


def parse_numbers(raw: str | None) -> list[int]:
    """``"3,17, 18"`` -> ``[3, 17, 18]``; the CLI's way of naming sentences."""

    if not raw:
        return []
    return sorted({int(item) for item in raw.replace(" ", "").split(",") if item})
