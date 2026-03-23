import asyncio
import base64
import json
import os
from uuid import uuid4

import structlog
from fastapi import (
    APIRouter, Depends, HTTPException, Request,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from Dependences import NoteRegistry, SSEManager, get_pool, get_registry, get_sse_manager
from Meetingdb import MeetingDBError
from Note_taker import NoteTakerError, create_note_taker

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/meetings", tags=["meetings"])

SAMPLE_RATE = 16_000

# ── Per-meeting storage ────────────────────────────────────────────
_audio_sessions: dict[str, asyncio.Queue] = {}
_meeting_keys: dict[str, dict] = {}   # ✅ NEW


# ── Schemas ───────────────────────────────────────────────────────

class StartMeetingRequest(BaseModel):
    user_id: str
    attendee_emails: list[str]
    openrouter_api_key: str | None = None
    sarvam_api_key: str | None = None

    @field_validator("attendee_emails")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("attendee_emails must not be empty")
        return v


class StartMeetingResponse(BaseModel):
    meeting_id: str
    status: str
    ws_url: str
    sse_url: str


class EndMeetingResponse(BaseModel):
    meeting_id: str
    status: str
    title: str
    duration_secs: int


# ── START MEETING ─────────────────────────────────────────────────

@router.post("/start", response_model=StartMeetingResponse, status_code=201)
async def start_meeting(
    body: StartMeetingRequest,
    request: Request,
    pool=Depends(get_pool),
    registry: NoteRegistry = Depends(get_registry),
    sse_manager: SSEManager = Depends(get_sse_manager),
):
    meeting_id = str(uuid4())

    # ✅ Resolve keys (UI > env fallback)
    openrouter_key = body.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    sarvam_key     = body.sarvam_api_key     or os.environ.get("SARVAM_API_KEY", "")

    if not openrouter_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key required")

    # ✅ Store per meeting (IMPORTANT)
    _meeting_keys[meeting_id] = {
        "openrouter": openrouter_key,
        "sarvam": sarvam_key,
    }

    log.info(
        "api.start_meeting",
        meeting_id=meeting_id,
        has_openrouter_key=True,
        has_sarvam_key=bool(sarvam_key),
    )

    try:
        note_taker = await create_note_taker(
            meeting_id=meeting_id,
            user_id=body.user_id,
            attendee_emails=body.attendee_emails,
            pool=pool,
            openrouter_api_key=openrouter_key,
        )
    except NoteTakerError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # SSE hook
    original_push = note_taker._push_to_ui

    def patched_push(index, result):
        original_push(index, result)
        if result.display_notes:
            asyncio.create_task(sse_manager.push(meeting_id, {
                "type": "segment",
                "segment": index,
                "data": result.display_notes.model_dump(),
            }))

    note_taker._push_to_ui = patched_push

    registry.register(note_taker)
    asyncio.create_task(note_taker.run_trigger_loop())

    _audio_sessions[meeting_id] = asyncio.Queue(maxsize=500)

    # ✅ Pass meeting_id (NOT env)
    asyncio.create_task(_sarvam_consumer(meeting_id, note_taker))

    base_url = str(request.base_url).rstrip("/")
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")

    return StartMeetingResponse(
        meeting_id=meeting_id,
        status="live",
        ws_url=f"{ws_base}/meetings/{meeting_id}/audio",
        sse_url=f"{base_url}/meetings/{meeting_id}/stream",
    )


# ── SARVAM CONSUMER ──────────────────────────────────────────────

async def _sarvam_consumer(meeting_id: str, note_taker):
    keys = _meeting_keys.get(meeting_id, {})
    sarvam_key = keys.get("sarvam")
    queue = _audio_sessions.get(meeting_id)

    if not sarvam_key or not queue:
        log.info("sarvam.disabled", meeting_id=meeting_id)
        return

    from sarvamai import AsyncSarvamAI

    client = AsyncSarvamAI(api_subscription_key=sarvam_key)

    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="codemix",
        language_code="en-IN",
        sample_rate=SAMPLE_RATE,
        input_audio_codec="pcm_s16le",
    ) as stt_ws:

        async def send_audio():
            while True:
                raw = await queue.get()
                if raw is None:
                    break
                audio_b64 = base64.b64encode(raw).decode()
                await stt_ws.transcribe(audio=audio_b64)

        async def recv():
            while True:
                response = await stt_ws.recv()
                if response.type == "data":
                    text = getattr(response.data, "transcript", "") or ""
                    if text.strip():
                        await note_taker.feed(text)
                elif response.type == "speech_end":
                    await note_taker.on_speech_end()

        await asyncio.gather(send_audio(), recv())


# ── WEBSOCKET AUDIO ──────────────────────────────────────────────

@router.websocket("/{meeting_id}/audio")
async def audio_websocket(
    websocket: WebSocket,
    meeting_id: str,
    registry: NoteRegistry = Depends(get_registry),
):
    note_taker = registry.get(meeting_id)
    queue = _audio_sessions.get(meeting_id)

    if not note_taker or not queue:
        await websocket.close()
        return

    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_bytes()
            try:
                queue.put_nowait(raw)
            except asyncio.QueueFull:
                pass
    except WebSocketDisconnect:
        pass


# ── END MEETING ──────────────────────────────────────────────────

@router.post("/{meeting_id}/end", response_model=EndMeetingResponse)
async def end_meeting(
    meeting_id: str,
    pool=Depends(get_pool),
    registry: NoteRegistry = Depends(get_registry),
    sse_manager: SSEManager = Depends(get_sse_manager),
):
    note_taker = registry.get(meeting_id)
    if not note_taker:
        raise HTTPException(status_code=404, detail="Meeting not found")

    q = _audio_sessions.get(meeting_id)
    if q:
        q.put_nowait(None)

    final = await note_taker.end_meeting()

    registry.remove(meeting_id)
    _meeting_keys.pop(meeting_id, None)   # ✅ CLEANUP
    await sse_manager.push_done(meeting_id)

    return EndMeetingResponse(
        meeting_id=meeting_id,
        status="done",
        title=final.get("title", ""),
        duration_secs=note_taker.get_stats()["duration_secs"],
    )


# ── SSE STREAM ───────────────────────────────────────────────────

@router.get("/{meeting_id}/stream")
async def stream_notes(meeting_id: str, sse_manager: SSEManager = Depends(get_sse_manager)):
    q = sse_manager.subscribe(meeting_id)

    async def gen():
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")

@router.get("/health")
async def health(
    registry: NoteRegistry = Depends(get_registry),
):
    return {
        "status": "ok",
        "active_meetings": len(registry.active_meetings()),
    }