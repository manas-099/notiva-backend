"""
dashboard.py — Live terminal dashboard for Meeting Notes AI


"""

import time
from datetime import timedelta

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.columns import Columns
from rich.rule    import Rule
from rich.live    import Live
from rich.layout  import Layout
from rich.align   import Align
from rich import box

console = Console()


# ── Pill colours for highlight categories ─────────────────────────────────────

PILL_STYLES = {
    "blocker":  ("bold white on red",          "🚫 BLOCKER"),
    "risk":     ("bold black on yellow",        "⚠  RISK"),
    "decision": ("bold white on blue",          "✓  DECISION"),
    "number":   ("bold black on green",         "#  NUMBER"),
    "date":     ("bold white on dark_orange",   "📅 DATE"),
    "name":     ("bold white on purple",        "👤 NAME"),
    "fact":     ("bold white on cyan",          "ℹ  FACT"),
}

def _pill(highlight: str) -> Text:
    lower = highlight.lower()
    for key, (style, prefix) in PILL_STYLES.items():
        if lower.startswith(key):
            content = highlight[len(key):].lstrip(": ").strip()
            t = Text()
            t.append(f" {prefix}: ", style=style)
            t.append(content + " ", style="default")
            return t
    t = Text()
    t.append(f" {highlight} ", style="bold black on white")
    return t


# ── Segment card ───────────────────────────────────────────────────────────────

def print_segment(index: int, display_notes) -> None:
    """
    Print one segment card to the terminal.
    Called from note_taker._push_to_ui() every time LLM returns a segment.
    """
    from rich.markup import escape

    # Header line
    console.print()
    console.rule(
        f"[bold cyan]  Segment {index + 1}  [/bold cyan]",
        style="cyan",
    )

    # Headline
    console.print(
        f"\n  [bold white]{escape(display_notes.headline)}[/bold white]"
    )

    # Summary line
    if display_notes.summary_line:
        console.print(
            f"  [dim]{escape(display_notes.summary_line)}[/dim]\n"
        )

    # Bullet notes
    if display_notes.notes:
        for note in display_notes.notes:
            console.print(f"  [green]·[/green] {escape(note)}")
        console.print()

    # Highlight pills — pills use Text() directly, already safe
    if display_notes.highlights:
        pills = []
        for h in display_notes.highlights:
            if h.strip():
                pills.append(_pill(h))
        if pills:
            row = Text("  ")
            for p in pills:
                row.append_text(p)
                row.append("  ")
            console.print(row)
            console.print()


# ── Status line ────────────────────────────────────────────────────────────────

def print_status(state: str, meeting_id: str, segments: int, start_time: float) -> None:
    """Print a one-line status update (called on each state change)."""
    elapsed = int(time.monotonic() - start_time)
    dur     = str(timedelta(seconds=elapsed))

    state_colours = {
        "live":    "[bold green]● LIVE[/bold green]",
        "ending":  "[bold yellow]⏳ ENDING[/bold yellow]",
        "done":    "[bold green]✓ DONE[/bold green]",
        "error":   "[bold red]✕ ERROR[/bold red]",
    }
    state_label = state_colours.get(state, f"[dim]{state}[/dim]")

    console.print(
        f"  {state_label}  "
        f"[dim]meeting:[/dim] [cyan]{meeting_id[:8]}…[/cyan]  "
        f"[dim]duration:[/dim] [white]{dur}[/white]  "
        f"[dim]segments:[/dim] [white]{segments}[/white]"
    )


# ── Meeting start banner ────────────────────────────────────────────────────────

def print_meeting_start(meeting_id: str, user_id: str, attendees: list[str]) -> None:
    console.print()
    console.print(Panel(
        f"[bold green]Meeting started[/bold green]\n\n"
        f"[dim]Meeting ID :[/dim] [cyan]{meeting_id}[/cyan]\n"
        f"[dim]User       :[/dim] [white]{user_id[:16]}…[/white]\n"
        f"[dim]Attendees  :[/dim] [white]{', '.join(attendees)}[/white]\n\n"
        f"[dim]Listening… speak into your microphone[/dim]",
        title="[bold]Meeting Notes AI[/bold]",
        border_style="green",
        padding=(1, 4),
    ))
    console.print()


# ── Final notes ────────────────────────────────────────────────────────────────

def print_final_notes(final: dict) -> None:
    """
    Print the complete final notes panel.
    Called once at end of meeting after LLM final call.
    """
    console.print()
    console.rule("[bold green]  Final Meeting Notes  [/bold green]", style="green")
    console.print()

    # Title + summary — escape LLM text so Rich doesn't parse [] as markup
    from rich.markup import escape
    console.print(Panel(
        f"[bold white]{escape(final.get('executive_summary', ''))}[/bold white]",
        title=f"[bold cyan]{escape(final.get('title', 'Meeting Notes'))}[/bold cyan]",
        border_style="cyan",
        padding=(1, 3),
    ))

    # Notes
    notes = final.get("notes", [])
    if notes:
        console.print("\n[bold]Notes[/bold]")
        for n in notes:
            console.print(f"  [green]·[/green] {escape(n)}")

    # Action items
    actions = final.get("action_items", [])
    if actions:
        console.print("\n[bold]Action items[/bold]")
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        tbl.add_column("Task",     style="white",   no_wrap=False)
        tbl.add_column("Owner",    style="cyan",    no_wrap=True)
        tbl.add_column("Deadline", style="yellow",  no_wrap=True)
        for a in actions:
            tbl.add_row(
                a.get("task", ""),
                a.get("owner") or "—",
                a.get("deadline") or "—",
            )
        console.print(tbl)

    # Key points
    key_points = final.get("key_points", [])
    if key_points:
        console.print("\n[bold]Key points[/bold]")
        for k in key_points:
            cat   = escape(k.get("category", "fact").upper())
            point = escape(k.get("point", ""))
            console.print(f"  [dim][[/dim][cyan]{cat}[/cyan][dim]][/dim] {point}")

    # Next steps
    next_steps = final.get("next_steps", [])
    if next_steps:
        console.print("\n[bold]Next steps[/bold]")
        for s in next_steps:
            console.print(f"  [blue]→[/blue] {escape(s)}")

    console.print()
    console.rule(style="green")
    console.print()


# ── Error ─────────────────────────────────────────────────────────────────────

def print_error(message: str) -> None:
    console.print(f"\n  [bold red]✕[/bold red] {message}\n")


# ── Info ─────────────────────────────────────────────────────────────────────

def print_info(message: str) -> None:
    console.print(f"  [dim]→[/dim] {message}")