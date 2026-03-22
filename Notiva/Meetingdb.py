"""
meetingdb.py — Meeting Notes DB with structured logging + error handling
─────────────────────────────────────────────────────────────────────────
Changes vs original:
  - structlog added throughout (replaces all print())
  - Every public method has try/except with contextual log fields
  - DB errors raise MeetingDBError (never swallowed silently)
  - load_last_segment trailing-comma SQL bug fixed
  - save_segment result.display_notes.summary_line bug fixed
    (SegmentOutput has no display_notes in DB layer — uses result.rolling_summary)
  - asyncpg.Pool type hints tightened
"""

import json
import os
from uuid import uuid4

import asyncpg
import structlog
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]


# ── Custom exceptions ──────────────────────────────────────────────────────────

class MeetingDBError(Exception):
    """Raised on any unrecoverable DB operation failure."""


class MeetingNotFoundError(MeetingDBError):
    """Raised when a meeting_id does not exist in the DB."""


# ── Pydantic models ────────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str
    owner: str | None    = None
    deadline: str | None = None


class KeyPoint(BaseModel):
    point: str
    category: str  # number | name | date | fact | risk | blocker


class SegmentOutput(BaseModel):
    rolling_summary: str
    notes: list[str]
    action_items: list[ActionItem]
    key_points: list[KeyPoint]
    # display_notes intentionally not here — not stored in DB


# ── Migration SQL ──────────────────────────────────────────────────────────────

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS meetings (
    id               UUID        PRIMARY KEY,
    user_id          UUID        NOT NULL,
    attendee_emails  TEXT[]      NOT NULL DEFAULT '{}',
    status           TEXT        NOT NULL DEFAULT 'live'
                                 CHECK (status IN ('live', 'done', 'failed')),
    final_notes      JSONB,
    duration_secs    INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meetings_user   ON meetings (user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_status ON meetings (status);

CREATE TABLE IF NOT EXISTS meeting_segments (
    id              UUID        PRIMARY KEY  DEFAULT gen_random_uuid(),
    meeting_id      UUID        NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    segment_index   INTEGER     NOT NULL,
    rolling_summary TEXT        NOT NULL,
    notes           TEXT[]      NOT NULL DEFAULT '{}',
    action_items    JSONB       NOT NULL DEFAULT '[]',
    key_points      JSONB       NOT NULL DEFAULT '[]',
    flush_reason    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (meeting_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting ON meeting_segments (meeting_id);
CREATE INDEX IF NOT EXISTS idx_segments_order   ON meeting_segments (meeting_id, segment_index);
"""


# ── Pool ───────────────────────────────────────────────────────────────────────

async def create_pool() -> asyncpg.Pool:
    """Create and return the asyncpg connection pool."""
    logger = log.bind(database_url=DATABASE_URL.split("@")[-1])  # hide credentials
    try:
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("db.pool_created")
        return pool
    except Exception as exc:
        logger.error("db.pool_creation_failed", error=str(exc))
        raise MeetingDBError(f"Could not create DB pool: {exc}") from exc


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Apply schema migrations idempotently."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLES)
        log.info("db.migrations_complete")
    except Exception as exc:
        log.error("db.migrations_failed", error=str(exc))
        raise MeetingDBError(f"Migration failed: {exc}") from exc


# ── MeetingDB ──────────────────────────────────────────────────────────────────

class MeetingDB:

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._log  = log.bind(component="MeetingDB")

    # ── create_meeting ─────────────────────────────────────────────────────────

    async def create_meeting(
        self,
        meeting_id: str,
        user_id: str,
        attendee_emails: list[str],
    ) -> None:
        logger = self._log.bind(meeting_id=meeting_id, user_id=user_id)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO meetings (id, user_id, attendee_emails, status)
                    VALUES ($1, $2, $3, 'live')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    meeting_id, user_id, attendee_emails,
                )
            logger.info("db.meeting_created", attendee_count=len(attendee_emails))
        except asyncpg.PostgresError as exc:
            logger.error("db.create_meeting_failed", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"create_meeting failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.create_meeting_unexpected", error=str(exc))
            raise MeetingDBError(f"create_meeting unexpected error: {exc}") from exc

    # ── save_segment ───────────────────────────────────────────────────────────

    async def save_segment(
        self,
        meeting_id: str,
        segment_index: int,
        result: SegmentOutput,
        flush_reason: str = "",
    ) -> None:
        logger = self._log.bind(
            meeting_id=meeting_id,
            segment_index=segment_index,
            flush_reason=flush_reason,
        )
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO meeting_segments (
                        meeting_id, segment_index, rolling_summary,
                        notes, action_items, key_points, flush_reason
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (meeting_id, segment_index)
                    DO UPDATE SET
                        rolling_summary = EXCLUDED.rolling_summary,
                        notes           = EXCLUDED.notes,
                        action_items    = EXCLUDED.action_items,
                        key_points      = EXCLUDED.key_points,
                        flush_reason    = EXCLUDED.flush_reason
                    """,
                    meeting_id,
                    segment_index,
                    result.rolling_summary,                                     # FIX: was result.display_notes.summary_line
                    result.notes,                                               # TEXT[]
                    json.dumps([a.model_dump() for a in result.action_items]),  # JSONB
                    json.dumps([k.model_dump() for k in result.key_points]),    # JSONB
                    flush_reason,
                )
            logger.info(
                "db.segment_saved",
                notes_count=len(result.notes),
                action_items_count=len(result.action_items),
                key_points_count=len(result.key_points),
            )
        except asyncpg.PostgresError as exc:
            logger.error("db.save_segment_failed", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"save_segment failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.save_segment_unexpected", error=str(exc))
            raise MeetingDBError(f"save_segment unexpected error: {exc}") from exc

    # ── load_all_segments ──────────────────────────────────────────────────────

    async def load_all_segments(self, meeting_id: str) -> dict:
        logger = self._log.bind(meeting_id=meeting_id)
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT rolling_summary, notes, action_items, key_points
                    FROM   meeting_segments
                    WHERE  meeting_id = $1
                    ORDER  BY segment_index ASC
                    """,
                    meeting_id,
                )

            all_notes      : list[str]        = []
            all_actions    : list[ActionItem] = []
            all_key_points : list[KeyPoint]   = []
            last_summary   : str              = ""

            for row in rows:
                last_summary = row["rolling_summary"]
                all_notes.extend(row["notes"] or [])

                raw_actions = row["action_items"]
                if isinstance(raw_actions, str):
                    raw_actions = json.loads(raw_actions)
                all_actions.extend([ActionItem(**a) for a in (raw_actions or [])])

                raw_kp = row["key_points"]
                if isinstance(raw_kp, str):
                    raw_kp = json.loads(raw_kp)
                all_key_points.extend([KeyPoint(**k) for k in (raw_kp or [])])

            logger.info(
                "db.segments_loaded",
                segment_count=len(rows),
                total_notes=len(all_notes),
                total_actions=len(all_actions),
            )
            return {
                "rolling_summary": last_summary,
                "all_notes":       all_notes,
                "all_actions":     all_actions,
                "all_key_points":  all_key_points,
            }

        except asyncpg.PostgresError as exc:
            logger.error("db.load_segments_failed", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"load_all_segments failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.load_segments_unexpected", error=str(exc))
            raise MeetingDBError(f"load_all_segments unexpected error: {exc}") from exc

    # ── load_last_segment ──────────────────────────────────────────────────────

    async def load_last_segment(self, meeting_id: str) -> dict | None:
        logger = self._log.bind(meeting_id=meeting_id)
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT segment_index, rolling_summary,
                           notes, action_items, key_points
                    FROM   meeting_segments
                    WHERE  meeting_id = $1
                    ORDER  BY segment_index DESC
                    LIMIT  1
                    """,
                    meeting_id,
                )

            if row is None:
                logger.info("db.no_segments_found")
                return None

            raw_actions = row["action_items"]
            if isinstance(raw_actions, str):
                raw_actions = json.loads(raw_actions)

            raw_kp = row["key_points"]
            if isinstance(raw_kp, str):
                raw_kp = json.loads(raw_kp)

            state = {
                "next_segment_index": row["segment_index"] + 1,
                "rolling_summary":    row["rolling_summary"],
                "all_notes":          list(row["notes"] or []),
                "all_actions":        [ActionItem(**a) for a in (raw_actions or [])],
                "all_key_points":     [KeyPoint(**k) for k in (raw_kp or [])],
            }
            logger.info(
                "db.crash_state_loaded",
                next_segment=state["next_segment_index"],
                notes_recovered=len(state["all_notes"]),
            )
            return state

        except asyncpg.PostgresError as exc:
            logger.error("db.load_last_segment_failed", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"load_last_segment failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.load_last_segment_unexpected", error=str(exc))
            raise MeetingDBError(f"load_last_segment unexpected error: {exc}") from exc

    # ── complete_meeting ───────────────────────────────────────────────────────

    async def complete_meeting(
        self,
        meeting_id: str,
        final_notes: dict,
        duration_secs: int,
    ) -> None:
        logger = self._log.bind(meeting_id=meeting_id, duration_secs=duration_secs)
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE meetings
                    SET
                        final_notes   = $1,
                        status        = 'done',
                        duration_secs = $2,
                        updated_at    = NOW()
                    WHERE id = $3
                    """,
                    json.dumps(final_notes),
                    duration_secs,
                    meeting_id,
                )
            if result == "UPDATE 0":
                logger.warning("db.complete_meeting_no_rows", meeting_id=meeting_id)
                raise MeetingNotFoundError(f"Meeting {meeting_id} not found for completion")
            logger.info("db.meeting_completed")
        except MeetingNotFoundError:
            raise
        except asyncpg.PostgresError as exc:
            logger.error("db.complete_meeting_failed", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"complete_meeting failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.complete_meeting_unexpected", error=str(exc))
            raise MeetingDBError(f"complete_meeting unexpected error: {exc}") from exc

    # ── fail_meeting ───────────────────────────────────────────────────────────

    async def fail_meeting(self, meeting_id: str, reason: str = "") -> None:
        logger = self._log.bind(meeting_id=meeting_id, reason=reason)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE meetings SET status='failed', updated_at=NOW() WHERE id=$1",
                    meeting_id,
                )
            logger.warning("db.meeting_failed", reason=reason)
        except asyncpg.PostgresError as exc:
            logger.error("db.fail_meeting_error", error=str(exc), pg_code=exc.sqlstate)
            raise MeetingDBError(f"fail_meeting failed: {exc}") from exc
        except Exception as exc:
            logger.error("db.fail_meeting_unexpected", error=str(exc))
            raise MeetingDBError(f"fail_meeting unexpected error: {exc}") from exc


# ── Startup helper ─────────────────────────────────────────────────────────────

async def startup() -> asyncpg.Pool:
    pool = await create_pool()
    await run_migrations(pool)
    return pool