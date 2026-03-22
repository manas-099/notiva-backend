# ── Usage ─────────────────────────────────────────────────────────────────────

from .meetingdb import create_pool, run_migrations
from .note_taking import create_note_taker
from uuid import uuid4
import asyncio
async def run_meeting():
      pool       = await create_pool()
      await run_migrations(pool)

      note_taker = await create_note_taker(
          meeting_id      = str(uuid4()),
          user_id         = "user-123",
          attendee_emails = ["alice@co.com"],
          pool            = pool,
      )
      asyncio.create_task(note_taker.run_trigger_loop())

      async with sarvam_client.speech_to_text_streaming.connect(...) as ws:
          while True:
              response = await ws.recv()
              if response.type == "data":
                  await note_taker.feed(response.data.transcript)
              elif response.type == "speech_end":
                  await note_taker.on_speech_end()

      final = await note_taker.end_meeting()

─────────────────────────────────────────────────────────────────────────────