"""
meeting_llm.py
───────────────
Wraps your existing model.py (OpenRouter / LangChain) and produces
SegmentOutput / FinalOutput Pydantic models that note_taker.py expects.

Removed from prompts:
  - decisions
  - topics_covered
  - bullets (was commented out anyway)
  - comments inside JSON templates (# lines — invalid JSON)
"""

import asyncio
import json
from pydantic import BaseModel

from langchain_openrouter import ChatOpenRouter
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENROUTER_API_KEY"] = "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"

model = ChatOpenRouter(
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.8,
)


# ── Pydantic output models ─────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str
    owner: str | None    = None
    deadline: str | None = None

class KeyPoint(BaseModel):
    point: str
    category: str  # number | name | date | fact | risk | blocker

class DisplayNotes(BaseModel):
    headline: str
    summary_line: str
    notes: list[str]
    highlights: list[str]

class SegmentOutput(BaseModel):
    rolling_summary: str           # memory — passed to next call only
    notes: list[str]               # raw bullets accumulated across meeting
    action_items: list[ActionItem]
    key_points: list[KeyPoint]
    display_notes: DisplayNotes    # UI only — never stored in DB

class FinalOutput(BaseModel):
    title: str
    executive_summary: str
    notes: list[str]
    action_items: list[ActionItem]
    key_points: list[KeyPoint]
    next_steps: list[str]
    display_notes: DisplayNotes


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a professional meeting note taker.
Rules:
- Output valid JSON only. No markdown, no explanation, no fences.
- rolling_summary: always covers the ENTIRE meeting so far, not just this segment.
- notes: concise bullet points. Start each with a verb or noun. No filler.
- action_items: only extract if someone is clearly assigned a task.
- key_points: flag numbers, names, dates, risks, blockers, important facts.
- key_point category must be one of: number | name | date | fact | risk | blocker
- display_notes.highlights: max 3 items, prefix BLOCKER: / NUMBER: / RISK: / DECISION: only."""

_SEGMENT_TEMPLATE = """{context}

New transcript segment:
{chunk}

Respond with this exact JSON:
{{
  "rolling_summary": "2-3 sentence summary of the ENTIRE meeting so far including this segment",
  "notes": [
    "bullet point from this segment"
  ],
  "action_items": [
    {{"task": "description", "owner": "name or null", "deadline": "timeframe or null"}}
  ],
  "key_points": [
    {{"point": "the important detail", "category": "number|name|date|fact|risk|blocker"}}
  ],
  "display_notes": {{
    "headline": "one sharp sentence about what this segment was mainly about",
    "summary_line": "1-2 sentences expanding the headline with key context",
    "notes": [
      "bullet point from this segment"
    ],
    "highlights": [
      "BLOCKER: example or NUMBER: example — max 3, skip if nothing critical"
    ]
  }}
}}

UPDATE (not repeat): update rolling_summary without repeating previous points.
Only include information explicitly present in the transcript.
Do NOT assume or add external knowledge."""

_FINAL_TEMPLATE = """Meeting summary:
{summary}

All notes collected:
{notes}

All action items:
{actions}

Key points tracked:
{key_points}

Produce final meeting notes. Deduplicate everything. Respond with this exact JSON:
{{
  "title": "short meeting title max 8 words",
  "executive_summary": "3-5 professional sentences covering the full meeting",
  "notes": ["clean deduplicated bullet points for the full meeting"],
  "action_items": [
    {{"task": "...", "owner": "... or null", "deadline": "... or null"}}
  ],
  "key_points": [
    {{"point": "...", "category": "number|name|date|fact|risk|blocker"}}
  ],
  "next_steps": ["forward-looking items"],
  "display_notes": {{
    "headline": "one sentence capturing the entire meeting outcome",
    "summary_line": "2-3 sentences: what was discussed and what happens next",
    "notes": ["clean deduplicated bullet points for the full meeting"],
    "highlights": [
      "DECISION: ...",
      "BLOCKER: ...",
      "DATE: ...",
      "NUMBER: ..."
    ]
  }}
}}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    cleaned = raw.strip().strip("```").replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback — return minimal valid structure
        return {
            "rolling_summary": cleaned,
            "notes": [],
            "action_items": [],
            "key_points": [],
            "display_notes": {
                "headline": "Segment processed",
                "summary_line": cleaned[:200],
                "notes": [],
                "highlights": [],
            },
        }

def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "None"


# ── MeetingLLM class ───────────────────────────────────────────────────────────

class MeetingLLM:
    """
    Drop-in replacement for the old MeetingLLM.
    Uses your OpenRouter model from model.py under the hood.
    note_taker.py imports this and calls call_segment() / call_final().
    """

    # ── Segment call ──────────────────────────────────────────────────────────

    def call_segment(self, chunk: str, prev_summary: str) -> SegmentOutput:
        """
        Sync — called via run_in_executor from note_taker._flush().
        """
        context = (
            f"Meeting so far:\n{prev_summary}"
            if prev_summary
            else "This is the first segment of the meeting."
        )
        prompt = _SEGMENT_TEMPLATE.format(context=context, chunk=chunk.strip())
        full_prompt = f"{_SYSTEM}\n\n{prompt}"

        response = model.invoke(full_prompt)
        data = _parse_json(response.content)

        return SegmentOutput(
            rolling_summary = data.get("rolling_summary", ""),
            notes           = data.get("notes", []),
            action_items    = [ActionItem(**a) for a in data.get("action_items", [])],
            key_points      = [KeyPoint(**k)   for k in data.get("key_points", [])],
            display_notes   = DisplayNotes(**data.get("display_notes", {
                "headline": "", "summary_line": "", "notes": [], "highlights": []
            })),
        )

    # ── Final call ────────────────────────────────────────────────────────────

    def call_final(
        self,
        rolling_summary: str,
        all_notes: list[str],
        all_actions: list[ActionItem],
        all_key_points: list[KeyPoint],
    ) -> FinalOutput:
        """
        Sync — called via run_in_executor from note_taker.end_meeting().
        """
        prompt = _FINAL_TEMPLATE.format(
            summary    = rolling_summary or "No summary available.",
            notes      = _bullets(all_notes),
            actions    = "\n".join(
                f"- {a.task} | owner: {a.owner or 'unassigned'} | deadline: {a.deadline or 'none'}"
                for a in all_actions
            ) or "None",
            key_points = "\n".join(
                f"- [{k.category}] {k.point}" for k in all_key_points
            ) or "None",
        )
        full_prompt = f"{_SYSTEM}\n\n{prompt}"

        response = model.invoke(full_prompt)
        data = _parse_json(response.content)

        return FinalOutput(
            title             = data.get("title", "Meeting notes"),
            executive_summary = data.get("executive_summary", ""),
            notes             = data.get("notes", []),
            action_items      = [ActionItem(**a) for a in data.get("action_items", [])],
            key_points        = [KeyPoint(**k)   for k in data.get("key_points", [])],
            next_steps        = data.get("next_steps", []),
            display_notes     = DisplayNotes(**data.get("display_notes", {
                "headline": "", "summary_line": "", "notes": [], "highlights": []
            })),
        )


# ── Async wrapper (kept for backwards compat with your old model.py) ───────────

async def llm_call(chunk: str, prev_summary: str) -> dict:
    """
    Your original async llm_call from model.py — still works if anything imports it.
    Returns raw dict (not Pydantic) — only use this if you need the old interface.
    """
    context = (
        f"Meeting so far:\n{prev_summary}"
        if prev_summary
        else "This is the first segment."
    )
    prompt = f"{_SYSTEM}\n\n{_SEGMENT_TEMPLATE.format(context=context, chunk=chunk)}"

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.invoke, prompt)
    return _parse_json(response.content)