"""Async SQLite checkpoint lifecycle management."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from state import create_checkpoint_serializer


@asynccontextmanager
async def open_sqlite_checkpointer(path: Path | str) -> AsyncIterator[AsyncSqliteSaver]:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(str(database_path))
    saver = AsyncSqliteSaver(connection, serde=create_checkpoint_serializer())
    await saver.setup()
    try:
        yield saver
    finally:
        await connection.close()

