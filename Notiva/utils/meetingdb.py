"""
Meeting Notes DB — segment storage
────────────────────────────────────
One meeting → many segments
Each segment stores: rolling_summary, notes, action_items,
                     key_points
display_notes is NOT stored — only used for live UI display

Schema:
  meetings         — one row per meeting
  meeting_segments — one row per LLM flush
"""

import json
import os
from uuid import uuid4

import asyncpg
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


# ── Models matching LLM output ─────────────────────────────────────────────────

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
 
    # display_notes intentionally not here — not stored


# ── Migration ──────────────────────────────────────────────────────────────────

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
    decisions       TEXT[]      NOT NULL DEFAULT '{}',
    key_points      JSONB       NOT NULL DEFAULT '[]',
    topics_covered  TEXT[]      NOT NULL DEFAULT '{}',

    flush_reason    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (meeting_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting ON meeting_segments (meeting_id);
CREATE INDEX IF NOT EXISTS idx_segments_order   ON meeting_segments (meeting_id, segment_index);

"""


# ── Pool ───────────────────────────────────────────────────────────────────────

async def create_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )

async def run_migrations(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES)
    print("Migrations complete")


# ── DB class ───────────────────────────────────────────────────────────────────

class MeetingDB:

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    # ── Start meeting ─────────────────────────────────────────────────────────

    async def create_meeting(
        self,
        meeting_id: str,
        user_id: str,
        attendee_emails: list[str],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meetings (id, user_id, attendee_emails, status)
                VALUES ($1, $2, $3, 'live')
                ON CONFLICT (id) DO NOTHING
                """,
                meeting_id, user_id, attendee_emails,
            )

    # ── Save segment after each flush ─────────────────────────────────────────

    async def save_segment(
        self,
        meeting_id: str,
        segment_index: int,
        result: SegmentOutput,
        flush_reason: str = "",
    ) -> None:
        """
        Call after every buffer trigger fires and LLM returns a SegmentOutput.
        Uses ON CONFLICT so retries are safe — just overwrites the same row.
        display_notes is intentionally not saved here.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meeting_segments (
                    meeting_id,
                    segment_index,
                    rolling_summary,
                    notes,
                    action_items,
                   
                    key_points,
                 
                    flush_reason
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
                result.display_notes.summary_line,
                result.notes,                                          # TEXT[]
                json.dumps([a.model_dump() for a in result.action_items]),  # JSONB
                                                  # TEXT[]
                json.dumps([k.model_dump() for k in result.key_points]),    # JSONB
                                         
                flush_reason,
            )

    # ── Load all segments → assemble for final LLM call ───────────────────────

    async def load_all_segments(self, meeting_id: str) -> dict:
        """
        Call at end_meeting().
        Reads every segment in order, merges into flat lists.
        Returns exactly what llm.call_final() expects.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT rolling_summary, notes, action_items,
                        key_points
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

        return {
            "rolling_summary": last_summary,
            "all_notes":       all_notes,
            "all_actions":     all_actions,
           
            "all_key_points":  all_key_points,
        
        }

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def load_last_segment(self, meeting_id: str) -> dict | None:
        """
        Call on reconnect after crash.
        Returns last saved state so NoteTaker resumes from correct index.
        Returns None if no segments saved yet.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT segment_index, rolling_summary,
                       notes, action_items,
                       key_points
                FROM   meeting_segments
                WHERE  meeting_id = $1
                ORDER  BY segment_index DESC
                LIMIT  1
                """,
                meeting_id
            )

        if row is None:
            return None

        raw_actions = row["action_items"]
        if isinstance(raw_actions, str):
            raw_actions = json.loads(raw_actions)

        raw_kp = row["key_points"]
        if isinstance(raw_kp, str):
            raw_kp = json.loads(raw_kp)

        return {
            "next_segment_index": row["segment_index"] + 1,
            "rolling_summary":    row["rolling_summary"],
            "all_notes":          list(row["notes"] or []),
            "all_actions":        [ActionItem(**a) for a in (raw_actions or [])],
           
            "all_key_points":     [KeyPoint(**k) for k in (raw_kp or [])],
           
        }

    # ── Complete meeting ───────────────────────────────────────────────────────

    async def complete_meeting(
        self,
        meeting_id: str,
        final_notes: dict,
        duration_secs: int,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
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

    async def fail_meeting(self, meeting_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE meetings SET status='failed', updated_at=NOW() WHERE id=$1",
                meeting_id,
            )


# ── How this plugs into MeetingNoteTaker ──────────────────────────────────────
#
# async def _flush(self, reason):
#     chunk = " ".join(self._buffer)
#     self._buffer.clear()
#
#     result = await run_in_executor(self._llm.call_segment, chunk, self._prev_summary)
#
#     # update RAM
#     self._prev_summary = result.rolling_summary
#     self._all_notes.extend(result.notes)
#     self._all_actions.extend(result.action_items)
#     self._all_decisions.extend(result.decisions)
#     self._all_key_points.extend(result.key_points)
#     self._all_topics.extend(result.topics_covered)
#
#     # push display_notes to live UI (websocket / SSE) — NOT saved to DB
#     await self._push_to_ui(result.display_notes)
#
#     # save to DB in background — does not block STT stream
#     asyncio.create_task(
#         self._db.save_segment(
#             self.meeting_id,
#             self._seg_index,
#             result,
#             flush_reason=reason,
#         )
#     )
#     self._seg_index += 1
#
#
# async def end_meeting(self):
#     accumulated = await self._db.load_all_segments(self.meeting_id)
#
#     final = await run_in_executor(
#         self._llm.call_final,
#         accumulated["rolling_summary"],
#         accumulated["all_notes"],
#         accumulated["all_actions"],
#         accumulated["all_decisions"],
#         accumulated["all_key_points"],
#         accumulated["all_topics"],
#     )
#
#     await self._db.complete_meeting(
#         self.meeting_id,
#         final.model_dump(),
#         duration_secs,
#     )
#     return final
#
# ─────────────────────────────────────────────────────────────────────────────


async def startup() -> asyncpg.Pool:
    pool = await create_pool()
    await run_migrations(pool)
    return pool