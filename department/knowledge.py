"""The seam for per-role knowledge retrieval.

Every role in the protocol declares its own knowledge base, and each agent is
meant to consult its own sources autonomously. That retrieval is not built yet.

What exists here is the shape it will take, so that adding it later is a change
inside one agent rather than a restructuring. Nothing in this package hard-codes
"retrieval is disabled": an agent simply has no source attached, which is a
local, visible fact rather than a global switch to hunt for.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from schemas import StrictModel
from pydantic import Field


class KnowledgeExcerpt(StrictModel):
    """One retrieved passage, carrying the citation a role must reproduce."""

    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class KnowledgeSource(Protocol):
    """A role's own library. Implementations arrive with retrieval."""

    name: str

    async def lookup(self, query: str, *, limit: int = 5) -> Sequence[KnowledgeExcerpt]: ...
