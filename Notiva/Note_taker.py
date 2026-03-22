"""
note_taker.py — AsyncNoteTaker with structured logging + error handling
────────────────────────────────────────────────────────────────────────
Changes vs original:
  - structlog added throughout — every flush, feed, end_meeting logged
  - NoteTakerError raised on critical failures (DB save, LLM call)
  - _flush never crashes the STT stream — LLM/DB errors logged + segment skipped
  - run_trigger_loop catches and logs all exceptions (never kills the loop)
  - end_meeting marks meeting as failed in DB if LLM/DB error occurs
  - _push_to_ui is a clean no-op placeholder with a log statement
  - All timing tracked: flush latency, meeting duration
"""

import asyncio
import time

import asyncpg
import structlog
from dotenv import load_dotenv

load_dotenv()

from Meetingdb import MeetingDB, ActionItem, KeyPoint, MeetingDBError
from Meeting_llm import MeetingLLM, SegmentOutput, LLMError
from dashboard import print_segment, print_info, print_status

log = structlog.get_logger(__name__)


# ── Custom exceptions ──────────────────────────────────────────────────────────

class NoteTakerError(Exception):
    """Raised when note taker cannot recover from a critical failure."""


# ── Config ────────────────────────────────────────────────────────────────────

WORD_COUNT_LIMIT   = 200
TIME_CAP_SECONDS   = 90
SILENCE_SECONDS    = 4
POLL_INTERVAL      = 1
MIN_WORDS_TO_FLUSH = 8


# ── AsyncNoteTaker ─────────────────────────────────────────────────────────────

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
        self._log = log.bind(meeting_id=meeting_id, user_id=user_id)

        # buffer
        self._buffer        : list[str] = []
        self._last_text_at  : float     = time.monotonic()
        self._last_flush_at : float     = time.monotonic()
        self._meeting_start : float     = time.monotonic()

        # LLM memory — only prev_summary passed forward
        self._prev_summary : str = ""

        # accumulated state
        self._all_notes      : list[str]        = []
        self._all_actions    : list[ActionItem] = []
        self._all_key_points : list[KeyPoint]   = []

        self._seg_index     : int            = 0
        self._flush_lock    : asyncio.Lock   = asyncio.Lock()
        self._running       : bool           = True

        # stats
        self._total_words_fed   : int = 0
        self._successful_flushes: int = 0
        self._failed_flushes    : int = 0

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
        try:
            await db.create_meeting(meeting_id, user_id, attendee_emails)
            instance._log.info(
                "note_taker.created",
                attendee_count=len(attendee_emails),
            )
        except MeetingDBError as exc:
            instance._log.error("note_taker.create_failed", error=str(exc))
            raise NoteTakerError(f"Could not create meeting in DB: {exc}") from exc
        return instance

    # ── Crash recovery ────────────────────────────────────────────────────────

    async def resume_if_crashed(self) -> bool:
        """
        Call after create() if server may have crashed mid-meeting.
        Returns True if state was recovered, False if fresh start.
        """
        try:
            state = await self._db.load_last_segment(self.meeting_id)
        except MeetingDBError as exc:
            self._log.error("note_taker.resume_failed", error=str(exc))
            return False

        if state is None:
            self._log.info("note_taker.fresh_start")
            return False

        self._seg_index      = state["next_segment_index"]
        self._prev_summary   = state["rolling_summary"]
        self._all_notes      = state["all_notes"]
        self._all_actions    = state["all_actions"]
        self._all_key_points = state["all_key_points"]

        self._log.info(
            "note_taker.resumed",
            from_segment=self._seg_index - 1,
            notes_recovered=len(self._all_notes),
            actions_recovered=len(self._all_actions),
        )
        return True

    # ── Public API ────────────────────────────────────────────────────────────

    async def feed(self, text: str) -> None:
        """Call from Sarvam STT handler with each transcript chunk."""
        text = text.strip()
        if not text:
            return

        self._buffer.append(text)
        self._last_text_at = time.monotonic()
        word_count = len(text.split())
        self._total_words_fed += word_count

        self._log.debug(
            "note_taker.fed",
            words=word_count,
            buffer_words=self._word_count(),
            total_words=self._total_words_fed,
        )

        if self._word_count() >= WORD_COUNT_LIMIT:
            self._log.info("note_taker.word_limit_trigger", buffer_words=self._word_count())
            await self._flush("WORD_LIMIT")

    async def on_speech_end(self) -> None:
        """Call when Sarvam sends response.type == 'speech_end'."""
        self._log.debug("note_taker.speech_end_received")
        await self._flush("VAD_SPEECH_END")

    async def run_trigger_loop(self) -> None:
        """Run as: asyncio.create_task(note_taker.run_trigger_loop())"""
        self._log.info("note_taker.trigger_loop_started")
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                now = time.monotonic()

                silence_elapsed = now - self._last_text_at
                time_cap_elapsed = now - self._last_flush_at
                buf_words = self._word_count()

                if silence_elapsed >= SILENCE_SECONDS and buf_words >= MIN_WORDS_TO_FLUSH:
                    self._log.info(
                        "note_taker.silence_trigger",
                        silence_s=round(silence_elapsed, 1),
                        buffer_words=buf_words,
                    )
                    await self._flush("SILENCE_FALLBACK")
                    continue

                if time_cap_elapsed >= TIME_CAP_SECONDS and buf_words >= MIN_WORDS_TO_FLUSH:
                    self._log.info(
                        "note_taker.time_cap_trigger",
                        elapsed_s=round(time_cap_elapsed, 1),
                        buffer_words=buf_words,
                    )
                    await self._flush("TIME_CAP")

            except asyncio.CancelledError:
                self._log.info("note_taker.trigger_loop_cancelled")
                break
            except Exception as exc:
                # Never kill the loop on unexpected errors — log and continue
                self._log.error(
                    "note_taker.trigger_loop_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        self._log.info("note_taker.trigger_loop_stopped")

    async def end_meeting(self) -> dict:
        """
        1. Stop trigger loop
        2. Flush remaining buffer
        3. Load all segments from DB
        4. Final LLM call
        5. Save final_notes to meetings table
        6. Return final dict
        On failure: marks meeting as failed in DB and raises NoteTakerError.
        """
        self._running = False
        self._log.info(
            "note_taker.ending_meeting",
            total_segments=self._seg_index,
            total_words_fed=self._total_words_fed,
            successful_flushes=self._successful_flushes,
            failed_flushes=self._failed_flushes,
        )

        # Flush any remaining buffer
        if self._word_count() >= 1:
            await self._flush("MEETING_END")

        duration_secs = int(time.monotonic() - self._meeting_start)
        self._log.info("note_taker.generating_final_notes", duration_secs=duration_secs)

        try:
            accumulated = await self._db.load_all_segments(self.meeting_id)
        except MeetingDBError as exc:
            self._log.error("note_taker.end_load_segments_failed", error=str(exc))
            await self._safe_fail_meeting("load_segments_failed")
            raise NoteTakerError(f"end_meeting: could not load segments: {exc}") from exc

        try:
            t0 = time.monotonic()
            final = await asyncio.get_event_loop().run_in_executor(
                None,
                self._llm.call_final,
                accumulated["rolling_summary"],
                accumulated["all_notes"],
                accumulated["all_actions"],
                accumulated["all_key_points"],
            )
            self._log.info(
                "note_taker.final_llm_done",
                elapsed_s=round(time.monotonic() - t0, 2),
                title=final.title,
            )
        except LLMError as exc:
            self._log.error("note_taker.final_llm_failed", error=str(exc))
            await self._safe_fail_meeting("final_llm_failed")
            raise NoteTakerError(f"end_meeting: LLM final call failed: {exc}") from exc

        try:
            await self._db.complete_meeting(
                self.meeting_id,
                final.model_dump(),
                duration_secs,
            )
            self._log.info(
                "note_taker.meeting_complete",
                duration_secs=duration_secs,
                segment_count=self._seg_index,
            )
        except MeetingDBError as exc:
            self._log.error("note_taker.complete_meeting_db_failed", error=str(exc))
            # Don't fail the whole meeting — we have the final notes in memory
            self._log.warning("note_taker.returning_notes_despite_db_failure")

        return final.model_dump()

    # ── Internal flush ────────────────────────────────────────────────────────

    async def _flush(self, reason: str) -> None:
        async with self._flush_lock:
            if self._word_count() < MIN_WORDS_TO_FLUSH:
                self._log.debug(
                    "note_taker.flush_skipped",
                    reason=reason,
                    buffer_words=self._word_count(),
                    min_words=MIN_WORDS_TO_FLUSH,
                )
                return

            chunk = " ".join(self._buffer)
            self._buffer.clear()
            self._last_flush_at = time.monotonic()

        current_index = self._seg_index
        self._seg_index += 1
        chunk_words = chunk.count(" ") + 1

        flush_log = self._log.bind(
            segment=current_index,
            flush_reason=reason,
            chunk_words=chunk_words,
        )
        flush_log.info("note_taker.flush_start")

        # ── LLM call ──────────────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            result: SegmentOutput = await asyncio.get_event_loop().run_in_executor(
                None,
                self._llm.call_segment,
                chunk,
                self._prev_summary,
            )
        except LLMError as exc:
            self._failed_flushes += 1
            flush_log.error(
                "note_taker.flush_llm_failed",
                error=str(exc),
                elapsed_s=round(time.monotonic() - t0, 2),
            )
            # Do not crash the STT stream — skip this segment
            return
        except Exception as exc:
            self._failed_flushes += 1
            flush_log.error(
                "note_taker.flush_unexpected_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return

        elapsed = round(time.monotonic() - t0, 2)
        flush_log.info(
            "note_taker.flush_llm_done",
            elapsed_s=elapsed,
            notes_returned=len(result.notes),
            actions_returned=len(result.action_items),
        )

        # ── Update in-memory state ─────────────────────────────────────────────
        self._prev_summary = result.rolling_summary
        self._all_notes.extend(result.notes)
        self._all_actions.extend(result.action_items)
        self._all_key_points.extend(result.key_points)
        self._successful_flushes += 1

        # ── Push to UI (non-blocking) ──────────────────────────────────────────
        self._push_to_ui(current_index, result)

        # ── Save to DB in background (non-blocking, failure logged) ───────────
        asyncio.create_task(
            self._safe_save_segment(current_index, result, reason)
        )

    async def _safe_save_segment(
        self,
        segment_index: int,
        result: SegmentOutput,
        flush_reason: str,
    ) -> None:
        """Saves segment to DB. Logs error but does NOT propagate — STT must keep running."""
        try:
            await self._db.save_segment(
                self.meeting_id,
                segment_index,
                result,
                flush_reason=flush_reason,
            )
        except MeetingDBError as exc:
            self._log.error(
                "note_taker.segment_save_failed",
                segment=segment_index,
                error=str(exc),
            )
        except Exception as exc:
            self._log.error(
                "note_taker.segment_save_unexpected",
                segment=segment_index,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _safe_fail_meeting(self, reason: str) -> None:
        """Mark meeting as failed in DB. Logs but never raises."""
        try:
            await self._db.fail_meeting(self.meeting_id, reason=reason)
        except Exception as exc:
            self._log.error("note_taker.fail_meeting_error", error=str(exc))

    def _push_to_ui(self, index: int, result: SegmentOutput) -> None:
        """
        Print segment to terminal dashboard + forward to SSE when FastAPI is running.
        display_notes lives here only — never written to DB.
        """
        self._log.info(
            "note_taker.ui_push",
            segment=index,
            headline=result.display_notes.headline if result.display_notes else "",
        )

        # ── Terminal dashboard output ──────────────────────────────────────────
        if result.display_notes:
            print_segment(index, result.display_notes)

        # ── SSE push (active when running under FastAPI) ───────────────────────
        # Patched by routes/meetings.py start_meeting() at runtime:
        # asyncio.create_task(sse_manager.push(meeting_id, result.display_notes.model_dump()))

    def _word_count(self) -> int:
        return sum(len(x.split()) for x in self._buffer)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "meeting_id":        self.meeting_id,
            "segments_flushed":  self._seg_index,
            "successful_flushes": self._successful_flushes,
            "failed_flushes":    self._failed_flushes,
            "total_words_fed":   self._total_words_fed,
            "notes_accumulated": len(self._all_notes),
            "actions_found":     len(self._all_actions),
            "key_points_found":  len(self._all_key_points),
            "duration_secs":     int(time.monotonic() - self._meeting_start),
        }


# ── Factory function ───────────────────────────────────────────────────────────

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