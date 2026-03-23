"""
meeting_llm.py — LLM layer with structured logging + error handling

"""

import asyncio
import json
import time
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from langchain_openrouter import ChatOpenRouter
import os

log = structlog.get_logger(__name__)



# from dotenv import load_dotenv
# load_dotenv()
# print("KEY:", os.environ.get("OPENROUTER_API_KEY", "NOT FOUND"))

MODEL_NAME   = "arcee-ai/trinity-large-preview:free"
MAX_RETRIES  = 2
RETRY_DELAY  = 2.0


def _get_model(api_key: str = "") -> ChatOpenRouter:
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    os.environ["OPENROUTER_API_KEY"] = key
    return ChatOpenRouter(
        model=MODEL_NAME,
        temperature=0.8,
    )


# ── Custom exceptions ──────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


# ── Pydantic output models ─────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str
    owner: str | None    = None
    deadline: str | None = None


class KeyPoint(BaseModel):
    point: str
    category: str


class DisplayNotes(BaseModel):
    headline: str
    summary_line: str
    notes: list[str]
    highlights: list[str]


class SegmentOutput(BaseModel):
    rolling_summary: str
    notes: list[str]
    action_items: list[ActionItem]
    key_points: list[KeyPoint]
    display_notes: DisplayNotes


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

def _parse_json(raw: str, context: str = "") -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning(
            "llm.json_parse_failed",
            context=context,
            error=str(exc),
            raw_preview=raw[:200],
        )
        return {
            "rolling_summary": cleaned[:500],
            "notes": [],
            "action_items": [],
            "key_points": [],
            "display_notes": {
                "headline": "Segment processed (parse error)",
                "summary_line": cleaned[:200],
                "notes": [],
                "highlights": [],
            },
        }


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "None"


def _invoke_with_retry(prompt: str, call_label: str, api_key: str = "") -> str:
    model = _get_model(api_key)
    logger = log.bind(call_label=call_label, model=MODEL_NAME)
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            t0 = time.monotonic()
            response = model.invoke(prompt)
            elapsed = round(time.monotonic() - t0, 2)
            content = response.content
            logger.info(
                "llm.call_success",
                attempt=attempt,
                elapsed_s=elapsed,
                response_len=len(content),
            )
            return content

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "llm.call_failed",
                attempt=attempt,
                max_attempts=MAX_RETRIES + 1,
                error=str(exc),
            )
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    logger.error("llm.all_retries_exhausted", error=str(last_exc))
    raise LLMError(f"LLM call '{call_label}' failed after {MAX_RETRIES + 1} attempts: {last_exc}") from last_exc


# ── MeetingLLM ─────────────────────────────────────────────────────────────────

class MeetingLLM:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def call_segment(self, chunk: str, prev_summary: str) -> SegmentOutput:
        logger = log.bind(
            method="call_segment",
            chunk_words=len(chunk.split()),
            has_prev_summary=bool(prev_summary),
        )
        logger.info("llm.segment_call_start")

        context = (
            f"Meeting so far:\n{prev_summary}"
            if prev_summary
            else "This is the first segment of the meeting."
        )
        prompt = f"{_SYSTEM}\n\n{_SEGMENT_TEMPLATE.format(context=context, chunk=chunk.strip())}"

        raw = _invoke_with_retry(prompt, call_label="segment", api_key=self.api_key)
        data = _parse_json(raw, context="call_segment")

        try:
            result = SegmentOutput(
                rolling_summary = data.get("rolling_summary", ""),
                notes           = data.get("notes", []),
                action_items    = [ActionItem(**a) for a in data.get("action_items", [])],
                key_points      = [KeyPoint(**k)   for k in data.get("key_points", [])],
                display_notes   = DisplayNotes(**data.get("display_notes", {
                    "headline": "", "summary_line": "", "notes": [], "highlights": []
                })),
            )
            logger.info(
                "llm.segment_parsed",
                notes_count=len(result.notes),
                action_items_count=len(result.action_items),
                key_points_count=len(result.key_points),
            )
            return result

        except (ValidationError, TypeError, KeyError) as exc:
            logger.error("llm.segment_schema_error", error=str(exc), data_keys=list(data.keys()))
            raise LLMParseError(f"SegmentOutput validation failed: {exc}") from exc

    def call_final(
        self,
        rolling_summary: str,
        all_notes: list[str],
        all_actions: list[ActionItem],
        all_key_points: list[KeyPoint],
    ) -> FinalOutput:
        logger = log.bind(
            method="call_final",
            total_notes=len(all_notes),
            total_actions=len(all_actions),
            total_key_points=len(all_key_points),
        )
        logger.info("llm.final_call_start")

        prompt = f"{_SYSTEM}\n\n{_FINAL_TEMPLATE.format(
            summary    = rolling_summary or 'No summary available.',
            notes      = _bullets(all_notes),
            actions    = '\n'.join(
                f'- {a.task} | owner: {a.owner or "unassigned"} | deadline: {a.deadline or "none"}'
                for a in all_actions
            ) or 'None',
            key_points = '\n'.join(
                f'- [{k.category}] {k.point}' for k in all_key_points
            ) or 'None',
        )}"

        raw = _invoke_with_retry(prompt, call_label="final", api_key=self.api_key)
        data = _parse_json(raw, context="call_final")

        try:
            result = FinalOutput(
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
            logger.info(
                "llm.final_parsed",
                title=result.title,
                notes_count=len(result.notes),
                next_steps_count=len(result.next_steps),
            )
            return result

        except (ValidationError, TypeError, KeyError) as exc:
            logger.error("llm.final_schema_error", error=str(exc), data_keys=list(data.keys()))
            raise LLMParseError(f"FinalOutput validation failed: {exc}") from exc


# ── Async wrapper ──────────────────────────────────────────────────────────────

async def llm_call(chunk: str, prev_summary: str, api_key: str = "") -> dict:
    context = (
        f"Meeting so far:\n{prev_summary}"
        if prev_summary
        else "This is the first segment."
    )
    prompt = f"{_SYSTEM}\n\n{_SEGMENT_TEMPLATE.format(context=context, chunk=chunk)}"

    loop = asyncio.get_running_loop()
    try:
        model = _get_model(api_key)
        response = await loop.run_in_executor(None, model.invoke, prompt)
        return _parse_json(response.content, context="llm_call_legacy")
    except Exception as exc:
        log.error("llm.legacy_call_failed", error=str(exc))
        raise LLMError(f"llm_call failed: {exc}") from exc