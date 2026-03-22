from langchain_openrouter import ChatOpenRouter
import os
import asyncio
import json
os.environ['OPENROUTER_API_KEY']='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'

model = ChatOpenRouter(
    model="arcee-ai/trinity-large-preview:free",
    temperature=0.8,
)

# Example usage
# response = model.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")
# print(response.content)

# async def llm_call(chunk: str, prev_summary: str):
#     prompt = f"""
# You are taking notes for an ongoing meeting.

# What happened so far (your context):
# {prev_summary or "This is the first segment."}

# New transcript segment:
# {chunk}

# Respond in JSON:
# {{
#   "summary": "Updated 2-3 sentence summary of the ENTIRE meeting so far",
#   "action_items": ["item1", "item2"],
#   "decisions": ["decision1"]
# }}
# """
#     response = model.invoke(prompt)
#     return response.content
SYSTEM = """You are a professional meeting note taker.
Rules:
- Output valid JSON only. No markdown, no explanation, no fences.
- notes: concise bullet points. Start each with a verb or noun. No filler.
- summary: always covers the ENTIRE meeting so far, not just this segment.
- action_items: only extract if someone is clearly assigned a task.
- decisions: only log things explicitly agreed or confirmed. Not suggestions.
- key_points: flag numbers, names, dates, risks, blockers, important facts.
- key_point category must be one of: number | name | date | fact | risk | blocker
- topics_covered: short labels for subjects discussed in this segment only."""

_SEGMENT_TEMPLATE = """

 
Respond with this exact JSON:
{{
  "summary": "2-3 sentence summary of the ENTIRE meeting so far including this segment",
#   "notes": [
#     "bullet point from this segment"
#   ],
 "action_items": [
    {{"task": "description", "owner": "name or null", "deadline": "timeframe or null"}}
  ],


  "key_points": [
    {{"point": "the important detail", "category": "number|name|date|fact|risk|blocker"}}
  ],
  "topics_covered": ["topic label"],
  "display_notes": {{
    "headline": "one sharp sentence about what this segment was mainly about",
    "summary_line": "1-2 sentences expanding the headline with key context",
    "notes": [
     "bullet point from this segment"
             ],
    # "bullets": [
    #   "Specific note bullet with names and numbers where relevant"
    # ],
    "highlights": [
      "BLOCKER: example",
      "NUMBER: example"
    ]
}}"""
_FINAL_TEMPLATE = """Meeting summary:
{summary}
 
All notes collected:
{notes}
 
All action items:
{actions}

 
Key points tracked:
{key_points}
 
Topics covered:
{topics}
 
Produce final meeting notes. Deduplicate everything. Respond with this exact JSON:
{{
  "title": "short meeting title max 8 words",
  "executive_summary": "3-5 professional sentences covering the full meeting",
  
  "action_items": [
    {{"task": "...", "owner": "... or null", "deadline": "... or null"}}
  ],
  "decisions": ["deduplicated confirmed decisions"],
  "key_points": [
    {{"point": "...", "category": "number|name|date|fact|risk|blocker"}}
  ],
  "next_steps": ["forward-looking items"],
  "topics_covered": ["deduplicated topic labels"],
  "display_notes": {{
    "headline": "one sentence capturing the entire meeting outcome",
    "summary_line": "2-3 sentences: what was discussed, decided, and what happens next",
    "notes": ["clean deduplicated bullet points for the full meeting"],
    "highlights": [
      "DECISION: ...",
      "BLOCKER: ...",
      "DATE: ...",
      "NUMBER: ..."
    ]
  }}
}}"""
# Async LLM call
async def llm_call(chunk: str, prev_summary: str,):
    prompt = f"""
{SYSTEM}

What happened so far (your context):
{prev_summary or "This is the first segment."}



New transcript segment:
{chunk}
{_SEGMENT_TEMPLATE}
UPDATE (not repeat)
Update the summary without repeating previous points.
Only include information explicitly present in transcript.
Do NOT assume or add external knowledge.

"""
    # Note: invoke() is sync, so wrap in loop.run_in_executor to keep async
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.invoke, prompt)
        # Clean JSON (remove backticks if any)
    content = response.content.strip().strip("```").replace("json", "").strip()

    # Parse JSON safely
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if parsing fails
        data = {
            "summary": content,
         
        }
    return data
    # return response.content

