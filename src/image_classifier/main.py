from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

from image_classifier.classifier import (
    Timings,
    get_device,
    load_model,
)
from image_classifier.database import (
    DB_PATH,
    all_scores,
    all_failures,
    is_processed,
    make_connection,
)
from image_classifier.metadata import check_exiftool
from image_classifier.logging_utils import LOG_PATH, setup_logger, log_error
from image_classifier.scanner import scan_images
from image_classifier.processor import ImageProcessor

console = Console()


def star_display(rating: int) -> str:
    """Return a unicode star string for the given 1–5 rating."""
    return "★" * rating + "☆" * (5 - rating)


def print_profile_summary(all_timings: list[Timings]) -> None:
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
            table.add_row(f"[bold]{name}[/bold]", f"[bold]{mean:.1f}[/bold]", f"[bold]{mx:.1f}[/bold]")
        else:
            table.add_row(name, f"{mean:.1f}", f"{mx:.1f}")

    console.print()
    console.print(Panel(table, expand=False, border_style="dim", title="⏱️ Timings"))


def print_summary(
    scored: int,
    skipped: int,
    errors: int,
    folder: Path,
    conn: any,
) -> None:
    rows = all_scores(folder, conn)
    
    # Summary Table
    summary_table = Table(box=None, padding=(0, 2))
    summary_table.add_row("🖼️  Scored", f"[bold]{scored:>4}[/bold] images")
    summary_table.add_row("⏭️  Skipped", f"[dim]{skipped:>4}[/dim] (already in database)")
    
    if errors:
        summary_table.add_row("❌  Errors", f"[red]{errors:>4}[/red] (logged to {LOG_PATH})")
    else:
        summary_table.add_row("✅  Errors", f"[green]{errors:>4}[/green]")
    
    console.print()
    console.print(Panel(summary_table, expand=False, title="📊 Summary", border_style="blue"))

    if errors:
        failures = all_failures(folder, conn)
        if failures:
            error_list = Table(box=None, header_style="bold red")
            error_list.add_column("File")
            error_list.add_column("Error")
            for row in failures:
                error_list.add_row(Path(row['path']).name, f"[red]{row['error']}[/red]")
            console.print(Panel(error_list, title="Recent Failures", border_style="red", expand=False))

    if not rows:
        return

    buckets: dict[int, int] = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for row in rows:
        if row["rating"] in buckets:
            buckets[row["rating"]] += 1

    labels = {5: "8.5+", 4: "7–8.5", 3: "5.5–7", 2: "4–5.5", 1: "<4"}
    
    dist_table = Table(title="Rating Distribution", box=None, padding=(0, 1))
    dist_table.add_column("Rating", justify="left")
    dist_table.add_column("Score Range", justify="right", style="dim")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Bar", justify="left")

    colors = {5: "green", 4: "green", 3: "yellow", 2: "orange3", 1: "red"}

    for stars in (5, 4, 3, 2, 1):
        count = buckets[stars]
        bar = "█" * min(count, 30)
        if count > 30:
            bar += "+"
        dist_table.add_row(
            star_display(stars),
            f"({labels[stars]})",
            str(count),
            f"[{colors[stars]}]{bar}[/]"
        )

    console.print()
    console.print(Panel(dist_table, expand=False, title="📈 Distribution", border_style="cyan"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score images aesthetically and write XMP star ratings."
    )
    parser.add_argument("folder", type=Path, help="Folder of images to classify")
    parser.add_argument(
        "--force", action="store_true", help="Re-score already-processed images"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Scan subdirectories recursively"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show per-phase timing summary after the run",
    )
    args = parser.parse_args()

    folder: Path = args.folder.resolve()

    if not folder.exists():
        console.print(f"[red]Error: folder not found: {folder}[/red]")
        sys.exit(1)

    if not folder.is_dir():
        console.print(f"[red]Error: expected a folder, got a file: {folder}[/red]")
        sys.exit(1)

    check_exiftool()
    logger = setup_logger()

    # Header Panel
    header = Table(box=None, padding=(0, 1))
    header.add_column(justify="left")
    header.add_row("[bold cyan]Image Classifier[/bold cyan]")
    header.add_row("[dim]Score images aesthetically and write XMP star ratings[/dim]")
    
    console.print(Panel(header, border_style="cyan", expand=False))
    
    console.print("[dim]Note: On first run the model (~1.5 GB) will be downloaded to ~/.cache/huggingface.[/dim]")
    
    console.print()
    with console.status("[bold blue]Loading aesthetic model...", spinner="dots"):
        device = get_device()
        device_color = "green" if device.type != "cpu" else "yellow"
        device_label = device.type.upper()
        
        model, preprocessor = load_model(device)
        console.print(f"✨ Model loaded on [{device_color}]{device_label}[/{device_color}]")

    console.print()
    images = scan_images(folder, args.recursive)
    db_path = DB_PATH
    conn = make_connection(db_path)

    to_process = (
        images if args.force else [p for p in images if not is_processed(p, conn)]
    )
    skipped = len(images) - len(to_process)

    scan_table = Table(box=None, padding=(0, 2))
    scan_table.add_row("📂 Folder", f"[bold]{folder}[/bold]")
    scan_table.add_row("🖼️  Total", f"{len(images)} images")
    scan_table.add_row("🔍 Scanned", f"{skipped} already scored, [bold cyan]{len(to_process)}[/bold cyan] to process")
    
    console.print(Panel(scan_table, title="Scanning Info", border_style="dim", expand=False))
    console.print()

    processor = ImageProcessor(model, preprocessor, device, conn, db_path)
    scored = 0
    errors = 0
    all_timings: list[Timings] = []

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        ) as progress:
            task_id = progress.add_task("[cyan]Processing images...", total=len(to_process))
            for path in to_process:
                progress.update(task_id, description=f"[cyan]Processing: [bold]{path.name}[/bold]")

                def progress_callback(p: Path, s: float, r: int) -> None:
                    # Update task description with the result of the last processed image
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Last: {p.name}  [bold yellow]{s:.2f}[/]  {star_display(r)}",
                    )

                try:
                    score, timings = processor.process_image(path, progress_callback)
                    all_timings.append(timings)
                    scored += 1
                except Exception as exc:
                    error_str = f"{type(exc).__name__}: {exc}"
                    log_error(logger, path, exc)
                    processor.handle_failure(path, error_str)
                    errors += 1
                    progress.update(task_id, advance=1)
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]⚠️  Interrupt received. Partial results saved.[/yellow]")

    print_summary(scored, skipped, errors, folder, processor.conn)
    if args.profile:
        print_profile_summary(all_timings)
    processor.conn.close()


if __name__ == "__main__":
    main()
