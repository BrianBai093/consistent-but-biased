from pathlib import Path

import pytest

from exceptions import RetrievalDisabledError
from tools.medical_retriever import DisabledMedicalRetriever


@pytest.mark.asyncio
async def test_disabled_retriever_always_returns_controlled_error() -> None:
    with pytest.raises(RetrievalDisabledError, match="disabled"):
        await DisabledMedicalRetriever().retrieve("spine surgery guideline")


def test_guideline_directory_remains_empty() -> None:
    guideline_directory = Path(__file__).resolve().parent.parent / "data" / "guidelines"
    # git does not keep empty folders, so a fresh clone has none
    assert not guideline_directory.exists() or list(guideline_directory.iterdir()) == []

