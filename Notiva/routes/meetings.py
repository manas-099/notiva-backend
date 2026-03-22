"""
routes/meetings.py — v3: two WebSocket connections per meeting
───────────────────────────────────────────────────────────────
Each meeting now receives TWO WebSocket connections:
  /meetings/{id}/audio  — tab audio (remote speakers) from offscreen.js
  /meetings/{id}/audio  — mic audio (your voice) from content.js

Both connect to the SAME endpoint. The server accepts multiple connections
per meeting_id, mixes their PCM streams into one buffer, and feeds Sarvam.

Mixing strategy: both streams write into a shared asyncio.Queue per meeting.
A single Sarvam STT session reads from that queue.
"""

import asyncio
import base64
import json
import os
import time
from uuid import uuid4

import asyncpg
import structlog
from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    WebSocket, WebSocketDisconnect, status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from ..Dependences import NoteRegistry, SSEManager, get_pool, get_registry, get_sse_manager
from ..Meetingdb import MeetingDBError
from ..Note_taker import NoteTakerError, create_note_taker

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/meetings", tags=["meetings"])
SAMPLE_RATE = 16_000

# ── Per-meeting audio mixer ────────────────────────────────────────────────────
# Holds the shared queue and Sarvam session for each live meeting
_audio_sessions: dict[str, asyncio.Queue] = {}   # meeting_id → Queue of bytes


# ── Schemas ────────────────────────────────────────────────────────────────────

class StartMeetingRequest(BaseModel):
    user_id:         str
    attendee_emails: list[str]

    @field_validator("attendee_emails")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("attendee_emails must not be empty")
        return v

class StartMeetingResponse(BaseModel):
    meeting_id: str
    status:     str
    ws_url:     str
    sse_url:    str

class EndMeetingResponse(BaseModel):
    meeting_id:    str
    status:        str
    title:         str
    duration_secs: int


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(pool=Depends(get_pool), registry: NoteRegistry=Depends(get_registry)):
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as exc:
        log.error("health.db_failed", error=str(exc)); db_ok = False

    key = os.environ.get("SARVAM_API_KEY", "")
    return {
        "status":          "ok" if db_ok else "degraded",
        "db":              "ok" if db_ok else "error",
        "active_meetings": len(registry.active_meetings()),
        "sarvam_key_set":  bool(key),
        "sarvam_key_hint": key[:8] + "..." if key else "NOT SET",
    }


# ── POST /meetings/start ───────────────────────────────────────────────────────

@router.post("/start", response_model=StartMeetingResponse, status_code=201)
async def start_meeting(
    body:        StartMeetingRequest,
    request:     Request,
    pool=        Depends(get_pool),
    registry:    NoteRegistry = Depends(get_registry),
    sse_manager: SSEManager   = Depends(get_sse_manager),
):
    meeting_id = str(uuid4())
    log.info("api.start_meeting", meeting_id=meeting_id, user_id=body.user_id,
             attendee_count=len(body.attendee_emails))

    try:
        note_taker = await create_note_taker(
            meeting_id=meeting_id, user_id=body.user_id,
            attendee_emails=body.attendee_emails, pool=pool,
        )
    except NoteTakerError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Wire SSE push
    original_push = note_taker._push_to_ui
    def patched_push(index, result):
        original_push(index, result)
        if result.display_notes:
            asyncio.create_task(sse_manager.push(meeting_id, {
                "type": "segment", "segment": index,
                "data": result.display_notes.model_dump(),
            }))
    note_taker._push_to_ui = patched_push

    registry.register(note_taker)
    asyncio.create_task(note_taker.run_trigger_loop())

    # Create shared audio queue for this meeting
    # Both WebSocket connections (tab + mic) will write raw PCM bytes here
    _audio_sessions[meeting_id] = asyncio.Queue(maxsize=500)

    # Start the Sarvam STT consumer in the background
    asyncio.create_task(_sarvam_consumer(meeting_id, note_taker))

    base_url = str(request.base_url).rstrip("/")
    ws_base  = base_url.replace("https://", "wss://").replace("http://", "ws://")

    log.info("api.start_meeting_ok", meeting_id=meeting_id)
    return StartMeetingResponse(
        meeting_id=meeting_id, status="live",
        ws_url  = f"{ws_base}/meetings/{meeting_id}/audio",
        sse_url = f"{base_url}/meetings/{meeting_id}/stream",
    )


# ── Sarvam STT consumer ────────────────────────────────────────────────────────
# Reads mixed PCM from the queue and streams to Sarvam

async def _sarvam_consumer(meeting_id: str, note_taker) -> None:
    ws_log = log.bind(meeting_id=meeting_id, component="sarvam_consumer")
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        ws_log.error("sarvam.no_api_key"); return

    queue = _audio_sessions.get(meeting_id)
    if not queue:
        ws_log.error("sarvam.no_queue"); return

    try:
        from sarvamai import AsyncSarvamAI
        client = AsyncSarvamAI(api_subscription_key=key)

        async with client.speech_to_text_streaming.connect(
            model                = "saaras:v3",   
            mode                 = "codemix",
            language_code        = "en-IN",
            sample_rate          = SAMPLE_RATE,
            input_audio_codec    = "pcm_s16le",
            high_vad_sensitivity = True,
            vad_signals          = True,
        ) as stt_ws:
            ws_log.info("sarvam.connected")
            transcript_count = 0
            chunk_count = 0

            async def send_audio():
                nonlocal chunk_count
                while True:
                    try:
                        raw = await asyncio.wait_for(queue.get(), timeout=60.0)
                        if raw is None:  # poison pill — stop signal
                            ws_log.info("sarvam.stop_signal_received")
                            break
                        chunk_count += 1
                        audio_b64 = base64.b64encode(raw).decode("utf-8")
                        await stt_ws.transcribe(audio=audio_b64)
                        if chunk_count % 100 == 0:
                            ws_log.info("sarvam.chunks_sent", count=chunk_count)
                    except asyncio.TimeoutError:
                        ws_log.debug("sarvam.queue_timeout_keepalive")
                    except asyncio.CancelledError:
                        break

            async def recv_transcript():
                nonlocal transcript_count
                while True:
                    try:
                        response = await stt_ws.recv()
                        ws_log.info("sarvam.response", type=response.type)

                        if response.type == "data":
                            text = getattr(response.data, "transcript", "") or ""
                            ws_log.info("sarvam.transcript", text=text[:100], has_text=bool(text.strip()))
                            if text.strip():
                                transcript_count += 1
                                await note_taker.feed(text)
                                ws_log.info("sarvam.fed", count=transcript_count,
                                            buffer_words=note_taker._word_count())

                        elif response.type == "speech_end":
                            ws_log.info("sarvam.speech_end")
                            await note_taker.on_speech_end()

                        elif response.type == "speech_start":
                            ws_log.info("sarvam.speech_start")

                        elif response.type == "error":
                            ws_log.error("sarvam.error_response", detail=str(response))

                    except asyncio.CancelledError:
                        break
                    except Exception as exc:
                        ws_log.error("sarvam.recv_error", error=str(exc))
                        await asyncio.sleep(0.1)

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_transcript())
            await send_task
            recv_task.cancel()
            try: await recv_task
            except asyncio.CancelledError: pass

    except Exception as exc:
        ws_log.error("sarvam.fatal", error=str(exc), error_type=type(exc).__name__)
    finally:
        _audio_sessions.pop(meeting_id, None)
        ws_log.info("sarvam.session_ended")


# ── WS /meetings/{id}/audio ────────────────────────────────────────────────────
# Both offscreen.js (tab) and content.js (mic) connect here
# Each sends raw binary Int16 PCM — we put it straight into the queue

@router.websocket("/{meeting_id}/audio")
async def audio_websocket(
    websocket:  WebSocket,
    meeting_id: str,
    registry:   NoteRegistry = Depends(get_registry),
):
    ws_log = log.bind(meeting_id=meeting_id)

    note_taker = registry.get(meeting_id)
    if not note_taker:
        ws_log.warning("ws.meeting_not_found")
        await websocket.close(code=4004, reason="Meeting not found"); return

    queue = _audio_sessions.get(meeting_id)
    if not queue:
        ws_log.warning("ws.no_audio_queue")
        await websocket.close(code=4005, reason="Audio session not ready"); return

    await websocket.accept()
    ws_log.info("ws.client_connected")

    chunk_count    = 0
    bytes_received = 0

    try:
        while True:
            raw = await websocket.receive_bytes()
            bytes_received += len(raw)
            chunk_count    += 1

            if chunk_count == 1:
                ws_log.info("ws.first_chunk", bytes=len(raw))

            # Put raw PCM bytes into the shared queue for Sarvam consumer
            try:
                queue.put_nowait(raw)
            except asyncio.QueueFull:
                ws_log.warning("ws.queue_full_dropping_chunk")

            if chunk_count % 100 == 0:
                ws_log.info("ws.audio_progress", chunks=chunk_count,
                            kb=round(bytes_received / 1024, 1))

    except WebSocketDisconnect:
        ws_log.info("ws.client_disconnected", chunks=chunk_count,
                    kb=round(bytes_received / 1024, 1))
    except Exception as exc:
        ws_log.error("ws.error", error=str(exc))
    finally:
        try: await websocket.close()
        except: pass


# ── POST /meetings/{id}/end ────────────────────────────────────────────────────

@router.post("/{meeting_id}/end", response_model=EndMeetingResponse)
async def end_meeting(
    meeting_id:  str,
    pool=        Depends(get_pool),
    registry:    NoteRegistry = Depends(get_registry),
    sse_manager: SSEManager   = Depends(get_sse_manager),
):
    note_taker = registry.get(meeting_id)
    if not note_taker:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Send poison pill to stop Sarvam consumer
    q = _audio_sessions.get(meeting_id)
    if q:
        try: q.put_nowait(None)
        except: pass

    try:
        final = await note_taker.end_meeting()
    except NoteTakerError as exc:
        registry.remove(meeting_id)
        await sse_manager.push_done(meeting_id)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        registry.remove(meeting_id)
        await sse_manager.push_done(meeting_id)
        raise HTTPException(status_code=500, detail="Unexpected error")

    registry.remove(meeting_id)
    await sse_manager.push_done(meeting_id)
    asyncio.create_task(_send_final_email(note_taker.attendee_emails, meeting_id, final))

    log.info("api.end_meeting_ok", meeting_id=meeting_id, title=final.get("title",""))
    return EndMeetingResponse(
        meeting_id=meeting_id, status="done",
        title=final.get("title",""),
        duration_secs=note_taker.get_stats()["duration_secs"],
    )


# ── GET /meetings/{id}/stream (SSE) ───────────────────────────────────────────

@router.get("/{meeting_id}/stream")
async def stream_notes(meeting_id: str, sse_manager: SSEManager=Depends(get_sse_manager)):
    log.info("sse.client_connected", meeting_id=meeting_id)
    q = sse_manager.subscribe(meeting_id)

    async def gen():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"; continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done": break
        except asyncio.CancelledError:
            log.info("sse.disconnected", meeting_id=meeting_id)
        finally:
            sse_manager.unsubscribe(meeting_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── GET /meetings/{id}/stats ───────────────────────────────────────────────────

@router.get("/{meeting_id}/stats")
async def get_stats(meeting_id: str, registry: NoteRegistry=Depends(get_registry)):
    nt = registry.get(meeting_id)
    if not nt: raise HTTPException(status_code=404, detail="Meeting not found")
    return nt.get_stats()


# ── GET /meetings/{id}/notes ───────────────────────────────────────────────────

@router.get("/{meeting_id}/notes")
async def get_notes(meeting_id: str, pool=Depends(get_pool)):
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT final_notes, status, duration_secs FROM meetings WHERE id=$1",
                meeting_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="DB error")
    if not row: raise HTTPException(status_code=404, detail="Meeting not found")
    if row["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Meeting still {row['status']}")
    notes = row["final_notes"]
    if isinstance(notes, str): notes = json.loads(notes)
    return {"meeting_id": meeting_id, "status": row["status"],
            "duration_secs": row["duration_secs"], "notes": notes}


# ── Email ──────────────────────────────────────────────────────────────────────

async def _send_final_email(emails, meeting_id, final):
    key = os.environ.get("SENDGRID_API_KEY",""); from_e = os.environ.get("FROM_EMAIL","")
    if not key or not from_e or not emails: return
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail
        actions = "\n".join(f"  • {a.get('task')} — {a.get('owner') or 'unassigned'}"
                            for a in final.get("action_items",[])) or "  None"
        body = (f"Meeting Notes: {final.get('title','')}\n\n"
                f"Summary:\n{final.get('executive_summary','')}\n\n"
                f"Action Items:\n{actions}")
        sg = sendgrid.SendGridAPIClient(api_key=key)
        for email in emails:
            msg = Mail(from_email=from_e, to_emails=email,
                       subject=f"Meeting Notes: {final.get('title','')}",
                       plain_text_content=body)
            sg.client.mail.send.post(request_body=msg.get())
            log.info("email.sent", to=email)
    except ImportError:
        log.warning("email.sendgrid_not_installed")
    except Exception as exc:
        log.error("email.failed", error=str(exc))