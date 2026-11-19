"""Disabled medical retrieval boundary reserved for a later checkpoint."""

from typing import Protocol

from exceptions import RetrievalDisabledError
from schemas import EvidenceRecord


class MedicalRetriever(Protocol):
    async def retrieve(self, query: str, *, limit: int = 5) -> list[EvidenceRecord]: ...


class DisabledMedicalRetriever:
    async def retrieve(self, query: str, *, limit: int = 5) -> list[EvidenceRecord]:
        raise RetrievalDisabledError("Medical retrieval is disabled in this version.")
