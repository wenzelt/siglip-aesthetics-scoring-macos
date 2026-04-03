from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from image_classifier.classifier import (
    SUPPORTED_EXTENSIONS,
    ClassifierError,
    Timings,
    get_device,
    load_model,
    score_image,
    score_to_rating,
)
from image_classifier.database import DB_PATH, all_scores, all_failures, is_processed, make_connection, upsert, upsert_failure
from image_classifier.metadata import MetadataError, check_exiftool, write_rating, write_score_tag

if TYPE_CHECKING:
    import logging

LOG_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.log"

console = Console()


def setup_log() -> logging.Logger | None:
    """Open the append-mode log file. Returns None if it cannot be opened."""
    import logging

    logger = logging.getLogger("classify")
    if logger.handlers:
        return logger
    logger.setLevel(logging.ERROR)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(LOG_PATH), mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger
    except OSError:
        print(f"Warning: could not open log file at {LOG_PATH}", file=sys.stderr)
        return None


def log_error(logger: logging.Logger | None, path: Path, exc: Exception) -> None:
    """Write a tab-separated error line to the log."""
    if logger is None:
        return
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    logger.error("%s\t%s\t%s: %s", timestamp, path, type(exc).__name__, exc)


def scan_images(folder: Path, recursive: bool) -> list[Path]:
    """Return all supported image files in folder, skipping hidden paths."""
    glob_fn = folder.rglob if recursive else folder.glob
    return [
        path
        for path in glob_fn("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.parts)
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def star_display(rating: int) -> str:
    """Return a unicode star string for the given 1–5 rating."""
    return "★" * rating + "☆" * (5 - rating)


def print_profile_summary(all_timings: list[Timings]) -> None:
    """Print a table of mean and max milliseconds per phase."""
    if not all_timings:
        return
    n = len(all_timings)
    phases = [
        ("load",       [t.load_ms       for t in all_timings]),
        ("preprocess", [t.preprocess_ms for t in all_timings]),
        ("infer",      [t.infer_ms      for t in all_timings]),
        ("upsert",     [t.upsert_ms     for t in all_timings]),
        ("exiftool",   [t.exiftool_ms   for t in all_timings]),
        ("xattr",      [t.xattr_ms      for t in all_timings]),
        ("total",      [t.total_ms      for t in all_timings]),
    ]
    console.print()
    console.print(f"  [bold]Timing profile[/bold] ({n} images, milliseconds)")
    console.print(f"  {'phase':<12}  {'mean':>8}  {'max':>8}")
    console.rule("  ", characters="─")
    for name, values in phases:
        mean = sum(values) / n
        mx = max(values)
        if name == "total":
            console.print(f"  [bold]{name:<12}[/bold]  [bold]{mean:>8.1f}[/bold]  [bold]{mx:>8.1f}[/bold]")
        else:
            console.print(f"  {name:<12}  {mean:>8.1f}  {mx:>8.1f}")


def print_summary(
    scored: int,
    skipped: int,
    errors: int,
    folder: Path,
    conn: sqlite3.Connection,
) -> None:
    console.rule()
    console.print(f"  Scored:  {scored:>4} images")
    console.print(f"  Skipped: {skipped:>4} (already in database)")
    if errors:
        console.print(f"  Errors:  {errors:>4} (logged to {LOG_PATH} and stored in DB)")
        failures = all_failures(folder, conn)
        for row in failures:
            console.print(f"    [red]✗[/red] {Path(row['path']).name}  {row['error']}")
    else:
        console.print(f"  Errors:  {errors:>4}")

    rows = all_scores(folder, conn)
    if not rows:
        return

    console.print()
    buckets: dict[int, int] = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for row in rows:
        if row["rating"] in buckets:
            buckets[row["rating"]] += 1

    labels = {5: "8.5+", 4: "7–8.5", 3: "5.5–7", 2: "4–5.5", 1: "<4"}
    console.print("  Distribution:")
    for stars in (5, 4, 3, 2, 1):
        count = buckets[stars]
        bar = "█" * min(count, 30)
        console.print(f"  {star_display(stars)}  ({labels[stars]:>6})  {count:>4} images  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score images aesthetically and write XMP star ratings."
    )
    parser.add_argument("folder", type=Path, help="Folder of images to classify")
    parser.add_argument("--force", action="store_true", help="Re-score already-processed images")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively")
    parser.add_argument("--profile", action="store_true", help="Show per-phase timing summary after the run")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()

    if not folder.exists():
        console.print(f"[red]Error: folder not found: {folder}[/red]")
        sys.exit(1)

    if not folder.is_dir():
        console.print(f"[red]Error: expected a folder, got a file: {folder}[/red]")
        sys.exit(1)

    check_exiftool()
    logger = setup_log()

    console.rule("Image Classifier")
    console.print(
        "Note: On first run the model checkpoint (~1.5 GB) will be downloaded "
        "to ~/.cache/huggingface. Subsequent runs use the cache."
    )
    console.print("Loading model...", end=" ")

    device = get_device()
    if device.type == "cpu":
        console.print("[yellow][CPU — MPS not available][/yellow]")
    else:
        console.print(f"[green][{device.type.upper()}][/green]")

    model, preprocessor = load_model(device)
    console.print()

    images = scan_images(folder, args.recursive)
    db_path = DB_PATH
    conn = make_connection(db_path)

    to_process = images if args.force else [p for p in images if not is_processed(p, conn)]
    skipped = len(images) - len(to_process)

    console.print(f"Scanning [bold]{folder}[/bold]")
    console.print(f"  Found {len(images)} images ({skipped} already scored, {len(to_process)} to process)")
    console.print()

    scored = 0
    errors = 0
    all_timings: list[Timings] = []

    with Progress(
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("", total=len(to_process))
        for path in to_process:
            progress.update(task_id, description=path.name)
            try:
                score, timings = score_image(path, model, preprocessor, device)
                rating = score_to_rating(score)

                t = time.perf_counter()
                try:
                    upsert(path, score, rating, conn)
                except sqlite3.OperationalError:
                    conn.close()
                    conn = make_connection(db_path)
                    upsert(path, score, rating, conn)
                timings.upsert_ms = (time.perf_counter() - t) * 1000

                original_mtime_ns = path.stat().st_mtime_ns
                try:
                    t = time.perf_counter()
                    write_rating(path, rating)
                    timings.exiftool_ms = (time.perf_counter() - t) * 1000

                    t = time.perf_counter()
                    write_score_tag(path, score)
                    timings.xattr_ms = (time.perf_counter() - t) * 1000
                finally:
                    os.utime(path, ns=(path.stat().st_atime_ns, original_mtime_ns))

                all_timings.append(timings)
                scored += 1
                progress.update(
                    task_id,
                    advance=1,
                    description=f"{path.name}  {score:.2f}  {star_display(rating)}",
                )
            except Exception as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                log_error(logger, path, exc)
                try:
                    upsert_failure(path, error_str, conn)
                except sqlite3.OperationalError:
                    try:
                        conn.close()
                        conn = make_connection(db_path)
                        upsert_failure(path, error_str, conn)
                    except sqlite3.OperationalError:
                        pass  # DB unavailable; failure already logged to disk
                errors += 1
                progress.update(task_id, advance=1)

    print_summary(scored, skipped, errors, folder, conn)
    if args.profile:
        print_profile_summary(all_timings)
    conn.close()
