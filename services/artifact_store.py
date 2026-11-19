"""Atomic persistence for Markdown artifacts produced during intake."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4


async def write_markdown(artifact_dir: str, filename: str, content: str) -> None:
    directory = Path(artifact_dir)
    destination = directory / filename
    temporary = directory / f".{filename}.{uuid4().hex}.tmp"

    def _write() -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)

    await asyncio.to_thread(_write)


async def read_markdown(artifact_dir: str, filename: str) -> str:
    """Read a prepared artifact when a failed parallel branch was not checkpointed."""

    path = Path(artifact_dir) / filename

    def _read() -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    for attempt in range(3):
        try:
            return await asyncio.to_thread(_read)
        except PermissionError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.01)
    return ""
