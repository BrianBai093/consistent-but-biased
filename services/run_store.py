"""Persistent run catalog, SSE event log, and case addendum store."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
import aiosqlite

from api_schemas import RunRecord, RunStatus, StoredEvent
from schemas import DirectorStyle, ModelFailureRecord, RadiologyProgress


ACTIVE_STATUSES = (
    RunStatus.QUEUED,
    RunStatus.RUNNING,
    RunStatus.PAUSE_REQUESTED,
    RunStatus.CANCEL_REQUESTED,
)
NONTERMINAL_STATUSES = (
    RunStatus.QUEUED,
    RunStatus.RUNNING,
    RunStatus.PAUSE_REQUESTED,
    RunStatus.PAUSED,
    RunStatus.CANCEL_REQUESTED,
)


class RunStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._write_lock = asyncio.Lock()
        self._event_condition = asyncio.Condition()

    @classmethod
    async def open(cls, path: Path | str) -> "RunStore":
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(database_path))
        connection.row_factory = aiosqlite.Row
        store = cls(connection)
        await store.setup()
        return store

    async def setup(self) -> None:
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA busy_timeout=5000")
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_runs (
                thread_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                status TEXT NOT NULL,
                last_message_sequence INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_api_events_thread_event
                ON api_events(thread_id, event_id);
            CREATE TABLE IF NOT EXISTS model_failures (
                thread_id TEXT NOT NULL,
                failure_id TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                stage TEXT NOT NULL,
                outcome TEXT NOT NULL,
                category TEXT NOT NULL,
                attempt_categories_json TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                safe_message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(thread_id, failure_id)
            );
            CREATE INDEX IF NOT EXISTS idx_model_failures_thread
                ON model_failures(thread_id, created_at);
            CREATE TABLE IF NOT EXISTS radiology_progress (
                thread_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                completed_images INTEGER NOT NULL,
                total_images INTEGER NOT NULL,
                batch_number INTEGER,
                total_batches INTEGER NOT NULL,
                current_image_start INTEGER,
                current_image_end INTEGER,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_addenda (
                addendum_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_case_addenda_thread
                ON case_addenda(thread_id, created_at);
            """
        )
        await self._add_missing_columns()
        await self._connection.commit()

    async def _add_missing_columns(self) -> None:
        """Bring an existing database up to the current api_runs shape.

        ``CREATE TABLE IF NOT EXISTS`` leaves an older table untouched, so a
        column added after a database was first created has to be applied here.
        """

        cursor = await self._connection.execute("PRAGMA table_info(api_runs)")
        try:
            existing = {row[1] for row in await cursor.fetchall()}
        finally:
            await cursor.close()
        if "director_style" not in existing:
            await self._connection.execute(
                "ALTER TABLE api_runs ADD COLUMN director_style TEXT"
            )

    async def close(self) -> None:
        await self._connection.close()

    async def create_run(
        self,
        thread_id: str,
        case_id: str,
        *,
        director_style: DirectorStyle | None = None,
    ) -> RunRecord:
        now = datetime.now(UTC)
        record = RunRecord(
            thread_id=thread_id,
            case_id=case_id,
            status=RunStatus.QUEUED,
            director_style=director_style,
            created_at=now,
            updated_at=now,
        )
        async with self._write_lock:
            await self._connection.execute(
                """
                INSERT INTO api_runs(
                    thread_id, case_id, status, last_message_sequence,
                    error_code, error_message, created_at, updated_at,
                    director_style
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    record.thread_id,
                    record.case_id,
                    record.status.value,
                    record.last_message_sequence,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.director_style.value if record.director_style else None,
                ),
            )
            await self._connection.commit()
        return record

    async def get_run(self, thread_id: str) -> RunRecord | None:
        cursor = await self._connection.execute(
            "SELECT * FROM api_runs WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _run_from_row(row) if row else None

    async def update_status(
        self,
        thread_id: str,
        status: RunStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> RunRecord | None:
        now = datetime.now(UTC).isoformat()
        async with self._write_lock:
            await self._connection.execute(
                """
                UPDATE api_runs
                SET status = ?, error_code = ?, error_message = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (status.value, error_code, error_message, now, thread_id),
            )
            await self._connection.commit()
        return await self.get_run(thread_id)

    async def update_last_message_sequence(
        self,
        thread_id: str,
        sequence: int,
    ) -> None:
        async with self._write_lock:
            await self._connection.execute(
                """
                UPDATE api_runs
                SET last_message_sequence = MAX(last_message_sequence, ?), updated_at = ?
                WHERE thread_id = ?
                """,
                (sequence, datetime.now(UTC).isoformat(), thread_id),
            )
            await self._connection.commit()

    async def append_event(
        self,
        thread_id: str,
        event_type: str,
        data: dict,
    ) -> StoredEvent:
        created_at = datetime.now(UTC)
        serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            cursor = await self._connection.execute(
                """
                INSERT INTO api_events(thread_id, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, event_type, serialized, created_at.isoformat()),
            )
            await self._connection.commit()
            event_id = int(cursor.lastrowid)
            await cursor.close()
        event = StoredEvent(
            event_id=event_id,
            thread_id=thread_id,
            event_type=event_type,
            data=data,
            created_at=created_at,
        )
        async with self._event_condition:
            self._event_condition.notify_all()
        return event

    async def record_model_failure(
        self,
        thread_id: str,
        failure: ModelFailureRecord,
    ) -> StoredEvent | None:
        """Persist one safe failure record and its SSE event atomically."""

        created_at = datetime.now(UTC)
        data = {
            "thread_id": thread_id,
            **failure.model_dump(mode="json"),
        }
        serialized_event = json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        serialized_categories = json.dumps(
            failure.attempt_categories,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        async with self._write_lock:
            cursor = await self._connection.execute(
                """
                INSERT OR IGNORE INTO model_failures(
                    thread_id, failure_id, agent_role, stage, outcome, category,
                    attempt_categories_json, attempt_count, max_attempts,
                    safe_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    failure.failure_id,
                    failure.agent_role.value,
                    failure.stage,
                    failure.outcome,
                    failure.category,
                    serialized_categories,
                    failure.attempt_count,
                    failure.max_attempts,
                    failure.safe_message,
                    created_at.isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            await cursor.close()
            if not inserted:
                await self._connection.commit()
                return None
            event_cursor = await self._connection.execute(
                """
                INSERT INTO api_events(thread_id, event_type, data_json, created_at)
                VALUES (?, 'model.failure', ?, ?)
                """,
                (thread_id, serialized_event, created_at.isoformat()),
            )
            event_id = int(event_cursor.lastrowid)
            await event_cursor.close()
            await self._connection.commit()
        event = StoredEvent(
            event_id=event_id,
            thread_id=thread_id,
            event_type="model.failure",
            data=data,
            created_at=created_at,
        )
        async with self._event_condition:
            self._event_condition.notify_all()
        return event

    async def list_model_failures(
        self,
        thread_id: str,
    ) -> list[ModelFailureRecord]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM model_failures
            WHERE thread_id = ?
            ORDER BY created_at ASC, failure_id ASC
            """,
            (thread_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ModelFailureRecord(
                failure_id=row["failure_id"],
                agent_role=row["agent_role"],
                stage=row["stage"],
                outcome=row["outcome"],
                category=row["category"],
                attempt_categories=json.loads(row["attempt_categories_json"]),
                attempt_count=row["attempt_count"],
                max_attempts=row["max_attempts"],
                safe_message=row["safe_message"],
            )
            for row in rows
        ]

    async def record_radiology_progress(
        self,
        thread_id: str,
        progress: RadiologyProgress,
    ) -> StoredEvent:
        """Persist the latest safe image-review progress and emit it over SSE."""

        created_at = datetime.now(UTC)
        data = {"thread_id": thread_id, **progress.model_dump(mode="json")}
        serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            await self._connection.execute(
                """
                INSERT INTO radiology_progress(
                    thread_id, stage, completed_images, total_images,
                    batch_number, total_batches, current_image_start,
                    current_image_end, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    stage = excluded.stage,
                    completed_images = excluded.completed_images,
                    total_images = excluded.total_images,
                    batch_number = excluded.batch_number,
                    total_batches = excluded.total_batches,
                    current_image_start = excluded.current_image_start,
                    current_image_end = excluded.current_image_end,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    progress.stage,
                    progress.completed_images,
                    progress.total_images,
                    progress.batch_number,
                    progress.total_batches,
                    progress.current_image_start,
                    progress.current_image_end,
                    created_at.isoformat(),
                ),
            )
            cursor = await self._connection.execute(
                """
                INSERT INTO api_events(thread_id, event_type, data_json, created_at)
                VALUES (?, 'radiology.progress', ?, ?)
                """,
                (thread_id, serialized, created_at.isoformat()),
            )
            event_id = int(cursor.lastrowid)
            await cursor.close()
            await self._connection.commit()
        event = StoredEvent(
            event_id=event_id,
            thread_id=thread_id,
            event_type="radiology.progress",
            data=data,
            created_at=created_at,
        )
        async with self._event_condition:
            self._event_condition.notify_all()
        return event

    async def get_radiology_progress(
        self,
        thread_id: str,
    ) -> RadiologyProgress | None:
        cursor = await self._connection.execute(
            "SELECT * FROM radiology_progress WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return RadiologyProgress(
            stage=row["stage"],
            completed_images=row["completed_images"],
            total_images=row["total_images"],
            batch_number=row["batch_number"],
            total_batches=row["total_batches"],
            current_image_start=row["current_image_start"],
            current_image_end=row["current_image_end"],
        )

    async def list_events(
        self,
        thread_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 200,
    ) -> list[StoredEvent]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM api_events
            WHERE thread_id = ? AND event_id > ?
            ORDER BY event_id ASC
            LIMIT ?
            """,
            (thread_id, after_event_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_event_from_row(row) for row in rows]

    async def wait_for_events(
        self,
        thread_id: str,
        *,
        after_event_id: int,
        timeout: float,
    ) -> list[StoredEvent]:
        events = await self.list_events(thread_id, after_event_id=after_event_id)
        if events:
            return events
        try:
            async with self._event_condition:
                await asyncio.wait_for(self._event_condition.wait(), timeout=timeout)
        except TimeoutError:
            return []
        return await self.list_events(thread_id, after_event_id=after_event_id)

    async def add_addendum(
        self,
        *,
        addendum_id: str,
        thread_id: str,
        content: str,
        created_at: datetime,
    ) -> None:
        async with self._write_lock:
            await self._connection.execute(
                """
                INSERT INTO case_addenda(addendum_id, thread_id, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (addendum_id, thread_id, content, created_at.isoformat()),
            )
            await self._connection.commit()

    async def mark_active_runs_interrupted(self) -> list[RunRecord]:
        return await self._mark_statuses_interrupted(ACTIVE_STATUSES)

    async def mark_nonterminal_runs_interrupted(self) -> list[RunRecord]:
        return await self._mark_statuses_interrupted(NONTERMINAL_STATUSES)

    async def _mark_statuses_interrupted(
        self,
        statuses: tuple[RunStatus, ...],
    ) -> list[RunRecord]:
        placeholders = ",".join("?" for _ in statuses)
        cursor = await self._connection.execute(
            f"SELECT * FROM api_runs WHERE status IN ({placeholders})",
            tuple(status.value for status in statuses),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        records = [_run_from_row(row) for row in rows]
        if not records:
            return []
        now = datetime.now(UTC).isoformat()
        async with self._write_lock:
            await self._connection.execute(
                f"""
                UPDATE api_runs
                SET status = ?, updated_at = ?
                WHERE status IN ({placeholders})
                """,
                (
                    RunStatus.INTERRUPTED.value,
                    now,
                    *(status.value for status in statuses),
                ),
            )
            await self._connection.commit()
        return [
            record.model_copy(
                update={"status": RunStatus.INTERRUPTED, "updated_at": datetime.fromisoformat(now)}
            )
            for record in records
        ]


def _run_from_row(row: aiosqlite.Row) -> RunRecord:
    return RunRecord(
        thread_id=row["thread_id"],
        case_id=row["case_id"],
        status=RunStatus(row["status"]),
        director_style=(
            DirectorStyle(row["director_style"]) if row["director_style"] else None
        ),
        last_message_sequence=row["last_message_sequence"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _event_from_row(row: aiosqlite.Row) -> StoredEvent:
    return StoredEvent(
        event_id=row["event_id"],
        thread_id=row["thread_id"],
        event_type=row["event_type"],
        data=json.loads(row["data_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
