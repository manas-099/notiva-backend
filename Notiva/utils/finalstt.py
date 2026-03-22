"""
stt.py — Sarvam STT wired into AsyncNoteTaker
───────────────────────────────────────────────
Fixes applied vs your original:
  1. speech_end was nested inside `if response.type == "data"` — never fired. Fixed.
  2. note_taker now created via create_note_taker() with DB + LLM injected.
  3. end_meeting() called after stream closes.
"""

import asyncio
import base64
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
from sarvamai import AsyncSarvamAI


from .note_taking import create_note_taker
from .meetingdb import create_pool, run_migrations
from uuid import uuid4
import asyncio            

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_API_KEY = "sk_c372d77m_9GtuKyS4y24M0dk1defklQIP"

USER_ID         = str(uuid4())
ATTENDEE_EMAILS = ["alice@co.com"]
SAMPLE_RATE     = 16000

client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)


# ── STT stream ────────────────────────────────────────────────────────────────

async def stream_audio(note_taker):
    print("🎤 Listening... (Ctrl+C to stop)")

    loop = asyncio.get_running_loop()

    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="codemix",
        language_code="hi-IN",
        sample_rate=SAMPLE_RATE,
        input_audio_codec="pcm_s16le",
        high_vad_sensitivity=True,
        vad_signals=True,           # ← enables speech_end events
    ) as ws:

        def callback(indata, frames, time, status):
            audio_bytes  = indata.copy().tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            loop.call_soon_threadsafe(
                asyncio.create_task,
                ws.transcribe(audio=audio_base64)
            )

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while True:
                try:
                    response = await ws.recv()

                    # ── transcript chunk ──────────────────────────────────────
                    if response.type == "data":
                        text = response.data.transcript
                        if text.strip():
                            print(f"  STT: {text}")
                            await note_taker.feed(text)

                    # ── VAD: real speech boundary — best flush trigger ────────
                    # BUG FIX: this was nested inside `if response.type == "data"`
                    # in your original code — it could never fire there.
                    elif response.type == "speech_end":
                        print("  [speech end]")
                        await note_taker.on_speech_end()

                    elif response.type == "speech_start":
                        print("  [speech start]")

                except asyncio.CancelledError:
                    print("\nStream closed")
                    break
                except KeyboardInterrupt:
                    break


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # 1. set up DB pool
    pool = await create_pool()
    await run_migrations(pool)

    # 2. create note taker — DB row created, LLM injected
    note_taker = await create_note_taker(
        meeting_id      = str(uuid4()),
        user_id         = USER_ID,
        attendee_emails = ATTENDEE_EMAILS,
        pool            = pool,
    )

    # 3. start background trigger loop (silence + time cap)
    asyncio.create_task(note_taker.run_trigger_loop())

    # 4. run STT stream — feeds text into note_taker
    try:
        await stream_audio(note_taker)
    except KeyboardInterrupt:
        pass

    # 5. meeting ended — final LLM call + save to DB
    print("\n🏁 Meeting ended — generating final notes...")
    final = await note_taker.end_meeting()

    print("\n" + "═" * 50)
    print("FINAL NOTES")
    print("═" * 50)
    print(f"Title   : {final.get('title', '')}")
    print(f"Summary : {final.get('executive_summary', '')}")
    print("Actions :")
    for a in final.get("action_items", []):
        print(f"  • {a.get('task')} — {a.get('owner') or 'unassigned'}")
    print("Key points :")
    for k in final.get("key_points", []):
        print(f"  [{k.get('category')}] {k.get('point')}")
    print("═" * 50)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())