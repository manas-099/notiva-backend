"""
Production-grade AsyncNoteTaker
---------------------------------
Triggers (in priority order):
  1. VAD speech_end  — Sarvam tells you speech stopped (best signal, use this)
  2. Word count      — buffer hit 200+ words, flush now regardless
  3. Time cap        — 90s elapsed, guarantees progress on long monologues
  4. Silence poll    — fallback if VAD signals aren't enabled (4s no new text)

No audio files saved. No polling threads. Pure asyncio.
"""

import asyncio
import time

# ── Config ────────────────────────────────────────────────────────────────────

WORD_COUNT_LIMIT    = 200   # flush if buffer grows this large
TIME_CAP_SECONDS    = 90    # flush at least every N seconds
SILENCE_SECONDS     = 4     # fallback: flush after N seconds of no new text
POLL_INTERVAL       = 1     # how often the trigger loop checks (seconds)
MIN_WORDS_TO_FLUSH  = 8     # don't flush tiny fragments (filler words, noise)


# ── Note Taker ────────────────────────────────────────────────────────────────

class AsyncNoteTaker:
    """
    Feed text with .feed()
    Signal VAD boundary with .on_speech_end()
    Call .end_meeting() when the call ends.
    """

    def __init__(self, llm_callback):
        self.llm_callback   = llm_callback   # async fn(chunk: str, prev: str) -> str
        self._buffer        = []
        self._last_text_at  = time.monotonic()
        self._last_flush_at = time.monotonic()
        self._prev_summary  = ""
        self._prev_conversation = ""
        self._seg_index     = 0
        self._flush_lock    = asyncio.Lock()  # prevents double-flush races
        self._running       = True

    # ── Public API ────────────────────────────────────────────────────────────

    async def feed(self, text: str):
        """Call this from your STT response handler."""
        text = text.strip()
        if not text:
            return
        self._buffer.append(text)
        self._last_text_at = time.monotonic()

        # Trigger 2: word count overflow
        if self._word_count() >= WORD_COUNT_LIMIT:
            await self._flush("WORD_LIMIT")

    async def on_speech_end(self):
        """
        Call this when Sarvam sends response.type == 'speech_end'.
        This is the highest-quality trigger — it means a real pause happened.
        """
        await self._flush("VAD_SPEECH_END")

    async def end_meeting(self) -> str:
        """Flush remainder and return final consolidated notes."""
        self._running = False
        await self._flush("MEETING_END")
        return await self._final_notes()

    async def run_trigger_loop(self):
        """
        Run as a background task: asyncio.create_task(note_taker.run_trigger_loop())
        Handles time-cap and silence fallback triggers.
        """
        while self._running:
            await asyncio.sleep(POLL_INTERVAL)
            now = time.monotonic()

            # Trigger 4: silence fallback (only fires if VAD signals aren't working)
            silence = now - self._last_text_at
            if silence >= SILENCE_SECONDS and self._word_count() >= MIN_WORDS_TO_FLUSH:
                await self._flush("SILENCE_FALLBACK")
                continue

            # Trigger 3: time cap (long monologue guard)
            elapsed = now - self._last_flush_at
            if elapsed >= TIME_CAP_SECONDS and self._word_count() >= MIN_WORDS_TO_FLUSH:
                await self._flush("TIME_CAP")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _flush(self, reason: str):
        # Lock prevents two triggers firing at the same millisecond
        async with self._flush_lock:
            if self._word_count() < MIN_WORDS_TO_FLUSH:
                return   # don't waste an LLM call on "okay", "yeah", "hmm"

            chunk = " ".join(self._buffer)
            self._buffer.clear()
            self._last_flush_at = time.monotonic()

        self._seg_index += 1
        print(f"\n🚀 [{reason}] Segment {self._seg_index} → LLM  ({chunk.count(' ')+1} words)")

        result = await self.llm_callback(chunk, self._prev_summary)
        self._prev_summary = result['summary']
        # self._prev_conversation= result

        print(f"📝 summary:\n{result['summary']}")
        print("\n")
        
        print(f"📝 Notes:\n{result}")
        print("_______________________________")
        # print(f"📝 Full :\n{result}")

    # async def _final_notes(self) -> str:
    #     if not self._prev_summary:
    #         return "No content captured."
    #     consolidation_prompt = f"Previous notes: {self._prev_summary}"
    #     return await self.llm_callback(consolidation_prompt, "")
    async def final_notes(self)->str:
        if not self._prev_summary:
            return "No content captured."
        prompt="""
you are a note taker
draft the notes based the context provided 
context or past conversatation :{self._prev_summary}
notes: write proper notes from the given context
         """
        response=await self.llm_callback(prompt, "")
        return response.content

    

    def _word_count(self) -> int:
        return sum(len(x.split()) for x in self._buffer)






