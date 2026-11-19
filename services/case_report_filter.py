"""Check that a model's filtered case report only deleted, and show what it deleted.

A published case report runs from presentation through treatment to follow-up,
often inside one sentence: "A 63-year-old man with ... (mJOA 15) presented in
preparation for a C3-6 laminectomy." Deleting whole sentences either keeps the
answer or loses the presentation with it. So for these cases a model returns
the preoperative text itself, cut wherever it needs to be cut.

A model that returns text can also rewrite it, so the text is checked here,
without a model, against the source:

1. **Deletion only.** The output's words must appear in the source in the same
   order. Nothing paraphrased, corrected, merged or invented gets through. A
   sentence-ending mark may be added where a clause was cut.
2. **No word-level surgery.** A deletion of one or two words inside a sentence
   that is otherwise kept is refused, unless the words are a bare connective.
   That is how "no weakness" becomes "weakness" and "mild stenosis" becomes
   "stenosis" -- a deletion that changes a finding rather than removing one.

What was deleted is reconstructed from the two texts, never taken from the
model's own account, and kept as the analysis-only layer of the case.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from pydantic import Field

from schemas import StrictModel
from services.record_redaction import RedactionCategory

_TOKEN = re.compile(r"\w+(?:[-'.]\w+)*|[^\w\s]")
#: Marks the filter may add where it cut a clause short.
ADDABLE = frozenset({".", ";", ","})
SENTENCE_END = frozenset({".", "!", "?"})
#: Words whose removal on their own joins two kept clauses rather than
#: changing a finding.
CONNECTIVES = frozenset(
    {"and", "or", "but", "which", "who", "then", "with", "while", "after", "as", "the"}
)
#: The most a deletion inside a kept sentence may be and still be refused.
SHORT_RUN = 2


class FilterRemoval(StrictModel):
    category: RedactionCategory
    reason: str = Field(min_length=1)


class FilteredCaseReport(StrictModel):
    """What the filter returns: the kept text and its own account of the cuts."""

    preoperative_text: str = Field(min_length=1)
    removals: list[FilterRemoval]


class RemovedPassageRecord(StrictModel):
    text: str
    before: str
    after: str


class FilterRecord(StrictModel):
    """One case's accepted filter, frozen so every later run reads the same text."""

    source_sha256: str = Field(min_length=64, max_length=64)
    preoperative_text: str = Field(min_length=1)
    removals: list[FilterRemoval]
    #: Reconstructed from the two texts, not taken from the model's account.
    removed_passages: list[RemovedPassageRecord]
    provider: str
    model: str
    attempts: int = Field(ge=1)
    created_at: datetime


class FilterViolation(ValueError):
    """The filtered text is not a deletion-only edit of the source."""


@dataclass(frozen=True, slots=True)
class RemovedPassage:
    text: str
    #: Kept words immediately before and after, so a reader sees where it was.
    before: str
    after: str


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _sentence_ids(source: Sequence[str]) -> list[int]:
    ids, current = [], 0
    for token in source:
        ids.append(current)
        if token in SENTENCE_END:
            current += 1
    return ids


def check_deletion_only(source_text: str, filtered_text: str) -> list[RemovedPassage]:
    """The passages removed from ``source_text``, or ``FilterViolation``."""

    source = tokens(source_text)
    kept = tokens(filtered_text)
    if not kept:
        raise FilterViolation("the filtered text is empty")
    matcher = difflib.SequenceMatcher(a=source, b=kept, autojunk=False)
    kept_source = [False] * len(source)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            for index in range(a0, a1):
                kept_source[index] = True
        elif tag in ("insert", "replace"):
            added = [token for token in kept[b0:b1] if token not in ADDABLE]
            if added:
                raise FilterViolation(
                    "the filtered text contains words that are not in the record, in "
                    f"this order: {' '.join(added[:12])!r}. Copy the record's words "
                    "exactly; only delete."
                )
    sentence = _sentence_ids(source)
    passages: list[RemovedPassage] = []
    index = 0
    while index < len(source):
        if kept_source[index]:
            index += 1
            continue
        start = index
        while index < len(source) and not kept_source[index]:
            index += 1
        run = source[start:index]
        words = [token for token in run if re.match(r"\w", token)]
        inside = (
            start > 0
            and index < len(source)
            and kept_source[start - 1]
            and kept_source[index]
            and sentence[start - 1] == sentence[index]
        )
        if (
            inside
            and 0 < len(words) <= SHORT_RUN
            and not all(word.lower() in CONNECTIVES for word in words)
        ):
            raise FilterViolation(
                f"you deleted {' '.join(run)!r} from inside a sentence you otherwise "
                "kept. Deleting a word or two changes a finding rather than removing "
                "one. Keep those words, or remove the whole clause they belong to."
            )
        if words:
            passages.append(
                RemovedPassage(
                    text=_join(run),
                    before=_join(source[max(0, start - 6) : start]),
                    after=_join(source[index : index + 6]),
                )
            )
    return passages


def _join(run: Sequence[str]) -> str:
    text = " ".join(run)
    return re.sub(r" ([.,;:!?)\]])", r"\1", re.sub(r"([(\[]) ", r"\1", text))


def render_filter_review(
    *,
    case_id: str,
    passages: Sequence[RemovedPassage],
    removals: Sequence[dict],
    screen_findings: Sequence[str],
) -> str:
    """What was cut, in context, for anyone auditing the case later."""

    lines = [
        f"# Filter review: {case_id}",
        "",
        "Approved automatically: no person has read this. The kept text was checked "
        "to be a deletion-only edit of the record.",
        "",
        f"Removed {len(passages)} passage(s).",
        "Keyword screen on the kept text: "
        + ("clean" if not screen_findings else f"{len(screen_findings)} hit(s)"),
    ]
    lines.extend(f"  - {finding}" for finding in screen_findings)
    lines.extend(["", "## The filter's own account", ""])
    lines.extend(f"- 〔{item['category']}〕{item['reason']}" for item in removals)
    lines.extend(["", "## Removed passages, in context", ""])
    for number, passage in enumerate(passages, start=1):
        lines.append(f"{number}. …{passage.before} **[−] {passage.text}** {passage.after}…")
    return "\n".join(lines) + "\n"
