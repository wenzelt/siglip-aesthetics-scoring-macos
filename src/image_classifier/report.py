from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from image_classifier.classifier import RATING_BANDS, Timings

_BAR_WIDTH = 30
_STAR_COLORS = {5: "green", 4: "green", 3: "yellow", 2: "orange3", 1: "red"}


def star_display(rating: int) -> str:
    """Return a unicode star string for the given 1–5 rating."""
    return "★" * rating + "☆" * (5 - rating)


def format_duration(seconds: float) -> str:
    """Format a duration like '850 ms', '42.0 s', '3m 05s' or '2h 14m'."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _bar(count: int, largest: int) -> str:
    """A bar scaled to the largest bucket; any non-empty bucket gets a visible bar."""
    if count <= 0 or largest <= 0:
        return ""
    return "█" * max(1, round(count / largest * _BAR_WIDTH))


@dataclass
class RunSummary:
    """Everything the end-of-run report needs, as plain data."""

    scored: int
    skipped: int
    errors: int
    rating_counts: Mapping[int, int]
    elapsed_s: float = 0.0
    dry_run: bool = False
    # Most recent first, as (file name, error message).
    failures: Sequence[tuple[str, str]] = field(default_factory=tuple)
    failures_total: int = 0


def print_profile_summary(
    console: Console, all_timings: Sequence[Timings], elapsed_s: float = 0.0
) -> None:
    """Print a table of mean and max milliseconds per phase."""
    if not all_timings:
        return
    n = len(all_timings)
    phases = [
        ("Load", [t.load_ms for t in all_timings]),
        ("Preprocess", [t.preprocess_ms for t in all_timings]),
        ("Infer", [t.infer_ms for t in all_timings]),
        ("Upsert", [t.upsert_ms for t in all_timings]),
        ("Exiftool", [t.exiftool_ms for t in all_timings]),
        ("Xattr", [t.xattr_ms for t in all_timings]),
        ("Total", [t.total_ms for t in all_timings]),
    ]

    table = Table(title=f"Timing Profile ({n} images)", box=None, padding=(0, 2))
    table.add_column("Phase", style="cyan")
    table.add_column("Mean (ms)", justify="right", style="magenta")
    table.add_column("Max (ms)", justify="right", style="magenta")

    for name, values in phases:
        mean = sum(values) / n
        mx = max(values)
        if name == "Total":
            table.add_section()
            table.add_row(
                f"[bold]{name}[/bold]",
                f"[bold]{mean:.1f}[/bold]",
                f"[bold]{mx:.1f}[/bold]",
            )
        else:
            table.add_row(name, f"{mean:.1f}", f"{mx:.1f}")

    console.print()
    console.print(Panel(table, expand=False, border_style="dim", title="⏱️ Timings"))
    if elapsed_s > 0:
        console.print(
            f"[dim]Per-image phase times overlap when running several workers; "
            f"wall-clock throughput was {n / elapsed_s:.1f} images/s.[/dim]"
        )


def print_summary(
    console: Console, summary: RunSummary, log_path: Path | None = None
) -> None:
    """Print the run summary, any failures and the rating distribution."""
    table = Table(box=None, padding=(0, 2))
    table.add_row(
        "🖼️  Would score" if summary.dry_run else "🖼️  Scored",
        f"[bold]{summary.scored:>4}[/bold] images",
    )
    table.add_row(
        "⏭️  Skipped", f"[dim]{summary.skipped:>4}[/dim] (already in database)"
    )
    if summary.errors:
        where = f" (logged to {log_path})" if log_path else ""
        table.add_row("❌  Errors", f"[red]{summary.errors:>4}[/red]{where}")
    else:
        table.add_row("✅  Errors", f"[green]{summary.errors:>4}[/green]")
    if summary.scored and summary.elapsed_s > 0:
        rate = summary.scored / summary.elapsed_s
        table.add_row(
            "⏱️  Time", f"{format_duration(summary.elapsed_s)} ({rate:.1f} images/s)"
        )

    title = (
        "📊 Summary (dry run: nothing was written)" if summary.dry_run else "📊 Summary"
    )
    console.print()
    console.print(Panel(table, expand=False, title=title, border_style="blue"))

    if summary.errors and summary.failures:
        error_list = Table(box=None, header_style="bold red")
        error_list.add_column("File")
        error_list.add_column("Error")
        for name, error in summary.failures:
            error_list.add_row(name, f"[red]{error}[/red]")
        hidden = summary.failures_total - len(summary.failures)
        console.print(
            Panel(
                error_list,
                title="Recent Failures",
                subtitle=f"and {hidden} more" if hidden > 0 else None,
                border_style="red",
                expand=False,
            )
        )

    _print_distribution(console, summary)


def _print_distribution(console: Console, summary: RunSummary) -> None:
    counts = {
        stars: summary.rating_counts.get(stars, 0) for _, stars, _ in RATING_BANDS
    }
    total = sum(counts.values())
    if not total:
        return
    largest = max(counts.values())

    dist = Table(title="Rating Distribution", box=None, padding=(0, 1))
    dist.add_column("Rating", justify="left")
    dist.add_column("Score Range", justify="right", style="dim")
    dist.add_column("Count", justify="right")
    dist.add_column("Share", justify="right", style="dim")
    dist.add_column("Bar", justify="left")

    for _, stars, label in RATING_BANDS:
        count = counts[stars]
        color = _STAR_COLORS[stars]
        dist.add_row(
            star_display(stars),
            f"({label})",
            f"{count:,}",
            f"{count / total:.0%}",
            f"[{color}]{_bar(count, largest)}[/]",
        )

    console.print()
    title = "📈 Distribution (this run)" if summary.dry_run else "📈 Distribution"
    console.print(Panel(dist, expand=False, title=title, border_style="cyan"))
