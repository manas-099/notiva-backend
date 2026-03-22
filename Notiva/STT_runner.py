"""
stt.py — Sarvam STT + AsyncNoteTaker  (CLI / standalone runner)
────────────────────────────────────────────────────────────────
Use this file to run the note-taker from the command line WITHOUT
the FastAPI server (e.g. local testing with a real microphone).

When deployed on Railway the FastAPI server (main.py) owns the process
and this file is NOT imported — the WebSocket endpoint in
routes/meetings.py handles audio instead.

Logging rule:
  configure_logging() is called ONCE here (the entry point).
  meetingdb, meeting_llm, note_taker never call configure_logging() —
  they only do `log = structlog.get_logger(__name__)`.
"""

import asyncio
import base64
import sys
from uuid import uuid4

import os
import structlog
from dotenv import load_dotenv

load_dotenv()

# ── Logging: configured HERE and nowhere else ─────────────────────────────────
from .logging_config import configure_logging
configure_logging()          # reads LOG_LEVEL + LOG_FORMAT from env
# ─────────────────────────────────────────────────────────────────────────────

import sounddevice as sd
from sarvamai import AsyncSarvamAI

from .Note_taker import create_note_taker, NoteTakerError
from .Meetingdb import create_pool, run_migrations, MeetingDBError
from .dashboard import (
    print_meeting_start,
    print_final_notes,
    print_error,
    print_info,
    print_status,
)

log = structlog.get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

SARVAM_API_KEY  = os.environ.get("SARVAM_API_KEY", "")
USER_ID         = os.environ.get("MEETING_USER_ID", str(uuid4()))
ATTENDEE_EMAILS = os.environ.get("ATTENDEE_EMAILS", "alice@co.com").split(",")
SAMPLE_RATE     = 16_000

if not SARVAM_API_KEY:
    log.warning("stt.missing_sarvam_api_key", hint="Set SARVAM_API_KEY in .env")

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)


# ── STT stream ────────────────────────────────────────────────────────────────

async def stream_audio(note_taker) -> None:
    log.info(
        "stt.stream_start",
        sample_rate=SAMPLE_RATE,
        user_id=USER_ID,
    )

    loop = asyncio.get_running_loop()
    chunk_errors = 0

    try:
        async with client.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="codemix",
            language_code="hi-IN",
            sample_rate=SAMPLE_RATE,
            input_audio_codec="pcm_s16le",
            high_vad_sensitivity=True,
            vad_signals=True,
        ) as ws:
            log.info("stt.websocket_connected")

            def callback(indata, frames, time_info, status):
                nonlocal chunk_errors
                if status:
                    log.warning("stt.audio_callback_status", status=str(status))
                try:
                    audio_bytes  = indata.copy().tobytes()
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        ws.transcribe(audio=audio_base64),
                    )
                except Exception as exc:
                    chunk_errors += 1
                    log.error(
                        "stt.chunk_send_failed",
                        error=str(exc),
                        total_chunk_errors=chunk_errors,
                    )

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=callback,
            ):
                log.info("stt.mic_open")
                while True:
                    try:
                        response = await ws.recv()

                        if response.type == "data":
                            text = response.data.transcript
                            if text.strip():
                                log.debug("stt.transcript_chunk", text=text[:80])
                                await note_taker.feed(text)

                        elif response.type == "speech_end":
                            log.debug("stt.speech_end")
                            await note_taker.on_speech_end()

                        elif response.type == "speech_start":
                            log.debug("stt.speech_start")

                        elif response.type == "error":
                            log.error("stt.server_error", response=str(response))

                        else:
                            log.debug("stt.unknown_event", type=response.type)

                    except asyncio.CancelledError:
                        log.info("stt.stream_cancelled")
                        break
                    except KeyboardInterrupt:
                        log.info("stt.keyboard_interrupt")
                        break
                    except Exception as exc:
                        log.error(
                            "stt.recv_error",
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )
                        # Continue loop — transient errors should not kill the stream
                        await asyncio.sleep(0.1)

    except Exception as exc:
        log.error(
            "stt.stream_fatal",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise

    log.info("stt.stream_ended", chunk_errors=chunk_errors)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    meeting_id = str(uuid4())
    log.info("app.startup", meeting_id=meeting_id, user_id=USER_ID, attendees=ATTENDEE_EMAILS)

    # ── 1. DB setup ───────────────────────────────────────────────────────────
    try:
        pool = await create_pool()
        await run_migrations(pool)
    except MeetingDBError as exc:
        log.error("app.db_setup_failed", error=str(exc))
        print_error(f"Database setup failed: {exc}")
        sys.exit(1)

    # ── 2. Create note taker ──────────────────────────────────────────────────
    try:
        note_taker = await create_note_taker(
            meeting_id      = meeting_id,
            user_id         = USER_ID,
            attendee_emails = ATTENDEE_EMAILS,
            pool            = pool,
        )
    except NoteTakerError as exc:
        log.error("app.note_taker_create_failed", error=str(exc))
        print_error(f"Could not create note taker: {exc}")
        await pool.close()
        sys.exit(1)

    # Show dashboard banner
    print_meeting_start(meeting_id, USER_ID, ATTENDEE_EMAILS)

    # ── 3. Start trigger loop ─────────────────────────────────────────────────
    trigger_task = asyncio.create_task(note_taker.run_trigger_loop())

    # ── 4. Stream audio — always call end_meeting on exit ─────────────────────
    final = None
    try:
        await stream_audio(note_taker)
    except KeyboardInterrupt:
        log.info("app.interrupted_by_user")
        print_info("Stopped by user — generating final notes...")
    except Exception as exc:
        log.error("app.stream_error", error=str(exc), error_type=type(exc).__name__)
        print_error(f"Stream error: {exc}")
    finally:
        trigger_task.cancel()
        try:
            await trigger_task
        except asyncio.CancelledError:
            pass

        print_info("Generating final notes…")
        try:
            final = await note_taker.end_meeting()
        except NoteTakerError as exc:
            log.error("app.end_meeting_failed", error=str(exc))
            print_error(f"End meeting failed: {exc}")
        except Exception as exc:
            log.error("app.end_meeting_unexpected", error=str(exc))
            print_error(f"Unexpected error: {exc}")

    # ── 5. Print final notes via dashboard ────────────────────────────────────
    if final:
        log.info(
            "app.final_notes",
            title=final.get("title", ""),
            executive_summary=final.get("executive_summary", "")[:120],
            action_item_count=len(final.get("action_items", [])),
            key_point_count=len(final.get("key_points", [])),
        )
        print_final_notes(final)
        log.info("app.session_stats", **note_taker.get_stats())
    else:
        log.warning("app.no_final_notes_produced")
        print_error("No notes were produced for this meeting.")

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    try:
        await pool.close()
        log.info("app.pool_closed")
    except Exception as exc:
        log.error("app.pool_close_failed", error=str(exc))

    log.info("app.shutdown_complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass