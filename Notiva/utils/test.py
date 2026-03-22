import asyncio
import time


# ─── FAKE LLM (for testing) ─────────────────────
# Replace this later with Anthropic / OpenAI

async def llm_call(chunk, prev):
    await asyncio.sleep(1)  # simulate API delay

    return f"""
Summary: {chunk[:80]}...
Action Items: None
"""



# ─── NOTE TAKER CLASS ───────────────────────────

class AsyncNoteTaker:
    def __init__(self, llm_callback):
        self.buffer = []
        self.last_chunk_time = time.time()
        self.last_flush_time = time.time()
        self.llm_callback = llm_callback
        self.previous_summary = ""

    async def feed(self, text):
        print("🎤 STT:", text)
        self.buffer.append(text)
        self.last_chunk_time = time.time()

    async def trigger_loop(self):
        while True:
            await asyncio.sleep(1)

            now = time.time()

            # 🔥 Silence trigger
            if now - self.last_chunk_time > 4 and self._word_count() > 10:
                await self.flush("SILENCE")

            # 🔥 Time trigger
            elif now - self.last_flush_time > 15 and self._word_count() > 10:
                await self.flush("TIME")

    async def flush(self, reason):
        if not self.buffer:
            return

        chunk = " ".join(self.buffer)
        self.buffer = []
        self.last_flush_time = time.time()

        print(f"\n🚀 LLM CALL ({reason})")
        print("📦 CHUNK:", chunk)

        output = await self.llm_callback(chunk, self.previous_summary)
        self.previous_summary = output

        print("📝 NOTES:", output) 

    def _word_count(self):
        return sum(len(x.split()) for x in self.buffer)



# # ─── MAIN ───────────────────────────────────────

# async def main():
#     note_taker = AsyncNoteTaker(llm_call)

#     # start trigger loop
#     asyncio.create_task(note_taker.trigger_loop())

#     # start fake STT
#     await fake_stt_stream(note_taker)

# # run
# asyncio.run(main())