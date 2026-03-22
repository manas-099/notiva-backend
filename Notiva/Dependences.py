"""
dependencies.py — Shared app-level state + FastAPI dependency providers
────────────────────────────────────────────────────────────────────────
Holds:
  - The asyncpg pool (one per process, created at startup)
  - The active note_taker registry (meeting_id → AsyncNoteTaker)
  - The SSE connection manager (meeting_id → list of queues)

FastAPI routes import get_pool(), get_registry(), get_sse_manager()
as Depends() — they never touch AppState directly.
"""

import asyncio
from typing import AsyncGenerator

import asyncpg
import structlog

from .Note_taker import AsyncNoteTaker

log = structlog.get_logger(__name__)


# ── SSE Manager ───────────────────────────────────────────────────────────────

class SSEManager:
    """
    Manages per-meeting Server-Sent Event queues.
    Each connected browser tab gets its own asyncio.Queue.
    When _push_to_ui fires, it puts an event into every queue for that meeting.
    """

    def __init__(self) -> None:
        # meeting_id → list of asyncio.Queue
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._log = log.bind(component="SSEManager")

    def subscribe(self, meeting_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queues.setdefault(meeting_id, []).append(q)
        self._log.info("sse.subscribed", meeting_id=meeting_id,
                       subscribers=len(self._queues[meeting_id]))
        return q

    def unsubscribe(self, meeting_id: str, q: asyncio.Queue) -> None:
        if meeting_id in self._queues:
            try:
                self._queues[meeting_id].remove(q)
            except ValueError:
                pass
            if not self._queues[meeting_id]:
                del self._queues[meeting_id]
        self._log.info("sse.unsubscribed", meeting_id=meeting_id)

    async def push(self, meeting_id: str, event: dict) -> None:
        queues = self._queues.get(meeting_id, [])
        if not queues:
            return
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._log.warning("sse.queue_full", meeting_id=meeting_id)
        self._log.debug("sse.pushed", meeting_id=meeting_id, subscribers=len(queues))

    async def push_done(self, meeting_id: str) -> None:
        """Signal stream end to all subscribers."""
        await self.push(meeting_id, {"type": "done"})


# ── Note-taker registry ───────────────────────────────────────────────────────

class NoteRegistry:
    """
    Tracks all live AsyncNoteTaker instances by meeting_id.
    Routes end-meeting calls and SSE pushes to the right instance.
    """

    def __init__(self) -> None:
        self._registry: dict[str, AsyncNoteTaker] = {}
        self._log = log.bind(component="NoteRegistry")

    def register(self, note_taker: AsyncNoteTaker) -> None:
        self._registry[note_taker.meeting_id] = note_taker
        self._log.info("registry.registered", meeting_id=note_taker.meeting_id,
                       total_active=len(self._registry))

    def get(self, meeting_id: str) -> AsyncNoteTaker | None:
        return self._registry.get(meeting_id)

    def remove(self, meeting_id: str) -> None:
        self._registry.pop(meeting_id, None)
        self._log.info("registry.removed", meeting_id=meeting_id,
                       total_active=len(self._registry))

    def active_meetings(self) -> list[str]:
        return list(self._registry.keys())


# ── App-level state (singleton) ───────────────────────────────────────────────

class AppState:
    pool        : asyncpg.Pool | None = None
    sse_manager : SSEManager          = SSEManager()
    registry    : NoteRegistry        = NoteRegistry()


_state = AppState()


# ── Dependency providers ──────────────────────────────────────────────────────

async def get_pool() -> asyncpg.Pool:
    if _state.pool is None:
        raise RuntimeError("DB pool not initialised — startup failed")
    return _state.pool


def get_sse_manager() -> SSEManager:
    return _state.sse_manager


def get_registry() -> NoteRegistry:
    return _state.registry


def get_app_state() -> AppState:
    return _state