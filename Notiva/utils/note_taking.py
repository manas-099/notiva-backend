"""
AsyncNoteTaker — connected to MeetingDB
────────────────────────────────────────
Tracks: notes, action_items, key_points only.
Decisions and topics_covered removed entirely.

File structure:
  meeting_db.py   ← DB layer
  meeting_llm.py  ← LLM layer
  note_taker.py   ← this file
"""

import asyncio
import time

import asyncpg
from dotenv import load_dotenv
load_dotenv()
from .meetingdb import MeetingDB, ActionItem, KeyPoint
from .meeting_llm import MeetingLLM, SegmentOutput

# ── Config ────────────────────────────────────────────────────────────────────

WORD_COUNT_LIMIT   = 200
TIME_CAP_SECONDS   = 90
SILENCE_SECONDS    = 4
POLL_INTERVAL      = 1
MIN_WORDS_TO_FLUSH = 8


# ── Note Taker ────────────────────────────────────────────────────────────────

class AsyncNoteTaker:

    def __init__(
        self,
        meeting_id: str,
        user_id: str,
        attendee_emails: list[str],
        db: MeetingDB,
        llm: MeetingLLM,
    ):
        self.meeting_id      = meeting_id
        self.user_id         = user_id
        self.attendee_emails = attendee_emails

        self._db  = db
        self._llm = llm

        # buffer
        self._buffer        = []
        self._last_text_at  = time.monotonic()
        self._last_flush_at = time.monotonic()
        self._meeting_start = time.monotonic()

        # memory — only prev_summary passed to LLM each call
        self._prev_summary = ""

        # accumulated — notes, actions, key_points only
        self._all_notes      : list[str]        = []
        self._all_actions    : list[ActionItem] = []
        self._all_key_points : list[KeyPoint]   = []

        self._seg_index  = 0
        self._flush_lock = asyncio.Lock()
        self._running    = True

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        meeting_id: str,
        user_id: str,
        attendee_emails: list[str],
        db: MeetingDB,
        llm: MeetingLLM,
    ) -> "AsyncNoteTaker":
        """Always use this instead of __init__ — creates DB row first."""
        instance = cls(meeting_id, user_id, attendee_emails, db, llm)
        await db.create_meeting(meeting_id, user_id, attendee_emails)
        return instance

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def resume_if_crashed(self) -> bool:
        """
        Call after create() if server may have crashed mid-meeting.
        Reloads last saved state from DB so nothing is lost.
        Returns True if recovered, False if fresh start.
        """
        state = await self._db.load_last_segment(self.meeting_id)
        if state is None:
            return False

        self._seg_index      = state["next_segment_index"]
        self._prev_summary   = state["rolling_summary"]
        self._all_notes      = state["all_notes"]
        self._all_actions    = state["all_actions"]
        self._all_key_points = state["all_key_points"]

        print(f"Resumed from segment {self._seg_index - 1}")
        return True

    # ── Public API ────────────────────────────────────────────────────────────

    async def feed(self, text: str):
        """Call from Sarvam STT response handler with each transcript chunk."""
        text = text.strip()
        if not text:
            return
        self._buffer.append(text)
        self._last_text_at = time.monotonic()

        if self._word_count() >= WORD_COUNT_LIMIT:
            await self._flush("WORD_LIMIT")

    async def on_speech_end(self):
        """Call when Sarvam sends response.type == 'speech_end'."""
        await self._flush("VAD_SPEECH_END")

    async def run_trigger_loop(self):
        """Run as: asyncio.create_task(note_taker.run_trigger_loop())"""
        while self._running:
            await asyncio.sleep(POLL_INTERVAL)
            now = time.monotonic()

            if (now - self._last_text_at >= SILENCE_SECONDS
                    and self._word_count() >= MIN_WORDS_TO_FLUSH):
                await self._flush("SILENCE_FALLBACK")
                continue

            if (now - self._last_flush_at >= TIME_CAP_SECONDS
                    and self._word_count() >= MIN_WORDS_TO_FLUSH):
                await self._flush("TIME_CAP")

    async def end_meeting(self) -> dict:
        """
        1. Flush remaining buffer
        2. Load all segments from DB
        3. Final LLM call with notes, actions, key_points
        4. Save final_notes to meetings table
        5. Return final dict
        """
        self._running = False

        if self._word_count() >= 1:
            await self._flush("MEETING_END")

        duration_secs = int(time.monotonic() - self._meeting_start)
        print("\nGenerating final notes...")

        accumulated = await self._db.load_all_segments(self.meeting_id)

        final = await asyncio.get_event_loop().run_in_executor(
            None,
            self._llm.call_final,
            accumulated["rolling_summary"],
            accumulated["all_notes"],
            accumulated["all_actions"],
            accumulated["all_key_points"],
        )

        await self._db.complete_meeting(
            self.meeting_id,
            final.model_dump(),
            duration_secs,
        )

        print(f"Done — {self._seg_index} segments, {duration_secs}s")
        return final.model_dump()

    # ── Internal flush ────────────────────────────────────────────────────────

    async def _flush(self, reason: str):
        async with self._flush_lock:
            if self._word_count() < MIN_WORDS_TO_FLUSH:
                return

            chunk = " ".join(self._buffer)
            self._buffer.clear()
            self._last_flush_at = time.monotonic()

        current_index = self._seg_index
        self._seg_index += 1

        print(f"\n[{reason}] Segment {current_index + 1} → LLM ({chunk.count(' ') + 1} words)")

        result: SegmentOutput = await asyncio.get_event_loop().run_in_executor(
            None,
            self._llm.call_segment,
            chunk,
            self._prev_summary,
        )

        # update RAM — notes, actions, key_points only
        self._prev_summary = result.rolling_summary
        self._all_notes.extend(result.notes)
        self._all_actions.extend(result.action_items)
        self._all_key_points.extend(result.key_points)

        # push display_notes to live UI — NOT saved to DB
        self._push_to_ui(current_index, result)

        # save segment to DB in background
        asyncio.create_task(
            self._db.save_segment(
                self.meeting_id,
                current_index,
                result,
                flush_reason=reason,
            )
        )

    def _push_to_ui(self, index: int, result: SegmentOutput):
        """
        Replace this body with your WebSocket / SSE push.
        display_notes lives here only — never written to DB.
        """
        print(f"\nSummary  : {result.rolling_summary}")
        print(f"Notes    : {result.notes}")
        if hasattr(result, "display_notes") and result.display_notes:
            dn = result.display_notes
            print(f"--------------dsp-----------------")
            print(f"Headline : {dn.headline}")
            print(f"Summary  : {dn.summary_line}")
            for b in dn.notes:
                print(f"  · {b}")
            for h in dn.highlights:
                print(f"  !! {h}")

    def _word_count(self) -> int:
        return sum(len(x.split()) for x in self._buffer)


# ── One-liner factory ─────────────────────────────────────────────────────────

async def create_note_taker(
    meeting_id: str,
    user_id: str,
    attendee_emails: list[str],
    pool: asyncpg.Pool,
) -> AsyncNoteTaker:
    db  = MeetingDB(pool)
    llm = MeetingLLM()
    return await AsyncNoteTaker.create(
        meeting_id, user_id, attendee_emails, db, llm
    )


# # ── Usage ─────────────────────────────────────────────────────────────────────

# from meeting_db import create_pool, run_migrations
# from note_taker import create_note_taker

# async def run_meeting():
#       pool       = await create_pool()
#       await run_migrations(pool)

#       note_taker = await create_note_taker(
#           meeting_id      = str(uuid4()),
#           user_id         = "user-123",
#           attendee_emails = ["alice@co.com"],
#           pool            = pool,
#       )
#       asyncio.create_task(note_taker.run_trigger_loop())

#       async with sarvam_client.speech_to_text_streaming.connect(...) as ws:
#           while True:
#               response = await ws.recv()
#               if response.type == "data":
#                   await note_taker.feed(response.data.transcript)
#               elif response.type == "speech_end":
#                   await note_taker.on_speech_end()

#       final = await note_taker.end_meeting()

# ─────────────────────────────────────────────────────────────────────────────