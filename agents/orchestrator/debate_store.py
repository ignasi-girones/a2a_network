"""Persistent storage for debates and their SSE event streams.

Backed by SQLite (via :mod:`aiosqlite`). The orchestrator writes here on
every progress event so the frontend can:

  - Recover the running debate after F5 (replay all events ⇒ rebuild UI).
  - Browse past debates from a sidebar.

Schema (see :data:`CREATE_SQL`):

  debates(id, prompt, status, created_at, updated_at, verdict, error)
      One row per debate. Status moves running ⇒ completed | failed.
      A partial UNIQUE index enforces "at most one running debate" at the
      database level on top of the asyncio.Lock — defence in depth.

  events(id, debate_id, seq, stage, message, data, created_at)
      One row per ProgressCallback.on_progress call. ``seq`` is monotonic
      per debate_id, assigned under :attr:`_lock`. ``data`` is JSON-serialized.

In-process pub/sub (``_subscribers`` dict keyed by debate_id) lets the
SSE follow-stream emit live events as :meth:`record_event` writes them.
The DB is the source of truth; the queues are pure fan-out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)


# ── Schema ─────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS debates (
    id          TEXT PRIMARY KEY,
    prompt      TEXT NOT NULL,
    status      TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    verdict     TEXT,
    error       TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS one_running
    ON debates(status) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id   TEXT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    stage       TEXT NOT NULL,
    message     TEXT,
    data        TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(debate_id, seq)
);

CREATE INDEX IF NOT EXISTS events_by_debate ON events(debate_id, seq);

CREATE TABLE IF NOT EXISTS attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id    TEXT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    mime_type    TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    extracted_text TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS attachments_by_debate ON attachments(debate_id);
"""


class DebateAlreadyActiveError(RuntimeError):
    """Raised when create_debate is called and another debate is still running."""


# ── DTOs ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DebateRow:
    id: str
    prompt: str
    status: str
    created_at: str
    updated_at: str
    verdict: str | None
    error: str | None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "DebateRow":
        return cls(
            id=row["id"],
            prompt=row["prompt"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            verdict=row["verdict"],
            error=row["error"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verdict": self.verdict,
            "error": self.error,
        }


@dataclass(frozen=True)
class EventRow:
    seq: int
    stage: str
    message: str | None
    data: dict[str, Any] | None
    created_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "EventRow":
        raw = row["data"]
        return cls(
            seq=row["seq"],
            stage=row["stage"],
            message=row["message"],
            data=json.loads(raw) if raw else None,
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AttachmentRow:
    id: int
    debate_id: str
    filename: str
    mime_type: str
    size_bytes: int
    storage_path: str
    extracted_text: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "AttachmentRow":
        return cls(
            id=row["id"],
            debate_id=row["debate_id"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            storage_path=row["storage_path"],
            extracted_text=row["extracted_text"],
            created_at=row["created_at"],
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "debate_id": self.debate_id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }
        if include_text:
            d["extracted_text"] = self.extracted_text
        return d


# ── Store ───────────────────────────────────────────────────────────────────


class DebateStore:
    """Async SQLite-backed store for debates + their event streams.

    One instance per orchestrator process. The connection is shared
    (SQLite serializes writes anyway) and protected by ``_lock`` to keep
    the ``seq`` numbers monotonic per debate without races.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        # debate_id -> set of asyncio.Queue listeners receiving live events
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    async def initialize(self) -> None:
        """Open the connection, ensure schema, enable foreign keys."""
        # Make sure the parent directory exists (mounted volume case).
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.executescript(CREATE_SQL)
        await self._db.commit()
        logger.info("DebateStore initialized at %s", self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DebateStore not initialized — call initialize() first")
        return self._db

    # ── Debate lifecycle ────────────────────────────────────────────────

    async def create_debate(self, prompt: str) -> str:
        """Insert a new running debate. Raises if another is already running."""
        debate_id = str(uuid.uuid4())
        async with self._lock:
            try:
                await self.db.execute(
                    "INSERT INTO debates (id, prompt, status) VALUES (?, ?, 'running')",
                    (debate_id, prompt),
                )
                await self.db.commit()
            except aiosqlite.IntegrityError as e:
                # Two possible causes:
                #   - UUID collision on PK (vanishingly rare)
                #   - The partial unique index on status='running' fired
                # SQLite's error mentions "debates.status" for the latter
                # and "debates.id" for the former.
                msg = str(e)
                if "status" in msg:
                    raise DebateAlreadyActiveError(
                        "another debate is already running"
                    ) from e
                raise
        logger.info("Created debate %s", debate_id)
        return debate_id

    async def mark_completed(self, debate_id: str, verdict: str) -> None:
        """Move debate to ``completed`` and emit a synthetic verdict event.

        The synthetic event lets the frontend's reducer surface the verdict
        identically whether replaying or following live — no special path
        for "the run is done".
        """
        async with self._lock:
            await self.db.execute(
                "UPDATE debates SET status='completed', verdict=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (verdict, debate_id),
            )
            seq = await self._next_seq_locked(debate_id)
            await self._insert_event_locked(
                debate_id, seq, "verdict", "Veredicto", {"text": verdict}
            )
            await self.db.commit()
        await self._publish(
            debate_id, _event_payload(seq, "verdict", "Veredicto", {"text": verdict})
        )
        logger.info("Marked debate %s completed", debate_id)

    async def mark_failed(self, debate_id: str, error: str) -> None:
        """Move debate to ``failed`` and emit a synthetic failed event."""
        async with self._lock:
            await self.db.execute(
                "UPDATE debates SET status='failed', error=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (error, debate_id),
            )
            seq = await self._next_seq_locked(debate_id)
            await self._insert_event_locked(
                debate_id, seq, "failed", "Debate fallido", {"error": error}
            )
            await self.db.commit()
        await self._publish(
            debate_id, _event_payload(seq, "failed", "Debate fallido", {"error": error})
        )
        logger.info("Marked debate %s failed: %s", debate_id, error)

    async def cleanup_orphans(self, error: str = "orchestrator restart") -> int:
        """At process startup, mark any leftover ``running`` rows as failed.

        Returns the number of orphans cleaned up. Without this an
        orchestrator crash would leave the DB blocking new debates forever
        (the partial unique index on status='running' would still be hit).
        """
        async with self._lock:
            cursor = await self.db.execute(
                "SELECT id FROM debates WHERE status='running'"
            )
            orphans = [row["id"] for row in await cursor.fetchall()]
            if orphans:
                await self.db.execute(
                    "UPDATE debates SET status='failed', error=?, "
                    "updated_at=CURRENT_TIMESTAMP WHERE status='running'",
                    (error,),
                )
                await self.db.commit()
        if orphans:
            logger.warning(
                "Cleaned up %d orphaned running debate(s): %s",
                len(orphans),
                orphans,
            )
        return len(orphans)

    # ── Events ──────────────────────────────────────────────────────────

    async def record_event(
        self,
        debate_id: str,
        stage: str,
        message: str | None,
        data: dict[str, Any] | None,
    ) -> int:
        """Persist one progress event, returning its monotonic ``seq``.

        Also publishes the event to any in-process /stream listeners.
        """
        async with self._lock:
            seq = await self._next_seq_locked(debate_id)
            await self._insert_event_locked(debate_id, seq, stage, message, data)
            await self.db.execute(
                "UPDATE debates SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (debate_id,),
            )
            await self.db.commit()
        await self._publish(debate_id, _event_payload(seq, stage, message, data))
        return seq

    async def _next_seq_locked(self, debate_id: str) -> int:
        """Compute next seq under the held lock — caller MUST hold _lock."""
        cursor = await self.db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM events WHERE debate_id=?",
            (debate_id,),
        )
        row = await cursor.fetchone()
        return int(row["next"])

    async def _insert_event_locked(
        self,
        debate_id: str,
        seq: int,
        stage: str,
        message: str | None,
        data: dict[str, Any] | None,
    ) -> None:
        """Insert one event row. Caller MUST hold _lock and commit later."""
        await self.db.execute(
            "INSERT INTO events (debate_id, seq, stage, message, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                debate_id,
                seq,
                stage,
                message,
                json.dumps(data, ensure_ascii=False) if data is not None else None,
            ),
        )

    # ── Queries ─────────────────────────────────────────────────────────

    async def get_debate(self, debate_id: str) -> DebateRow | None:
        cursor = await self.db.execute(
            "SELECT * FROM debates WHERE id=?", (debate_id,)
        )
        row = await cursor.fetchone()
        return DebateRow.from_row(row) if row else None

    async def get_active(self) -> DebateRow | None:
        cursor = await self.db.execute(
            "SELECT * FROM debates WHERE status='running' LIMIT 1"
        )
        row = await cursor.fetchone()
        return DebateRow.from_row(row) if row else None

    async def list_debates(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[DebateRow]:
        # CURRENT_TIMESTAMP has second resolution — debates created in the
        # same second tie. Break ties with rowid (insertion order) so the
        # newest is always first.
        cursor = await self.db.execute(
            "SELECT * FROM debates ORDER BY created_at DESC, rowid DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [DebateRow.from_row(r) for r in await cursor.fetchall()]

    async def get_events(
        self, debate_id: str, *, since_seq: int = -1
    ) -> list[EventRow]:
        """Return events with seq strictly greater than ``since_seq``.

        ``since_seq=-1`` (the default) yields every event. The frontend
        passes ``since=0`` for a full replay; live followers pass the seq
        of the last event they already have to skip duplicates.
        """
        cursor = await self.db.execute(
            "SELECT seq, stage, message, data, created_at FROM events "
            "WHERE debate_id=? AND seq > ? ORDER BY seq ASC",
            (debate_id, since_seq),
        )
        return [EventRow.from_row(r) for r in await cursor.fetchall()]

    # ── Attachments ─────────────────────────────────────────────────────

    async def add_attachment(
        self,
        debate_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        storage_path: str,
        extracted_text: str | None,
    ) -> int:
        """Insert an attachment row and return its id."""
        cursor = await self.db.execute(
            "INSERT INTO attachments "
            "(debate_id, filename, mime_type, size_bytes, storage_path, extracted_text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (debate_id, filename, mime_type, size_bytes, storage_path, extracted_text),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_attachments(self, debate_id: str) -> list[AttachmentRow]:
        cursor = await self.db.execute(
            "SELECT * FROM attachments WHERE debate_id=? ORDER BY id",
            (debate_id,),
        )
        return [AttachmentRow.from_row(r) for r in await cursor.fetchall()]

    # ── Pub/Sub ─────────────────────────────────────────────────────────

    async def _publish(self, debate_id: str, payload: dict[str, Any]) -> None:
        """Fan event payload out to any live /stream listeners."""
        queues = list(self._subscribers.get(debate_id, ()))
        for q in queues:
            # put_nowait so a slow client can never stall record_event.
            # If the queue is full something is wrong with the client; drop.
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning(
                    "subscriber queue full for debate %s; dropping event seq=%s",
                    debate_id,
                    payload.get("seq"),
                )

    @asynccontextmanager
    async def subscribe(
        self, debate_id: str, *, maxsize: int = 256
    ) -> AsyncIterator[asyncio.Queue]:
        """Yield a queue receiving live events for ``debate_id``.

        Use as ``async with store.subscribe(id) as q: ...``. The queue is
        unregistered on exit even if the consumer raises, so leaks don't
        accumulate when clients disconnect mid-stream.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.setdefault(debate_id, set()).add(q)
        try:
            yield q
        finally:
            subs = self._subscribers.get(debate_id)
            if subs is not None:
                subs.discard(q)
                if not subs:
                    self._subscribers.pop(debate_id, None)


# ── Helpers ────────────────────────────────────────────────────────────────


def _event_payload(
    seq: int,
    stage: str,
    message: str | None,
    data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the dict shape /stream and /events return."""
    return {
        "seq": seq,
        "stage": stage,
        "message": message,
        "data": data,
    }
