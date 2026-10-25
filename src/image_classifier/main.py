from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from image_classifier import __version__
from image_classifier.classifier import Timings, get_device, load_model
from image_classifier.database import (
    DB_PATH,
    is_processed,
    make_connection,
    rating_counts,
)
from image_classifier.logging_utils import LOG_PATH, log_error, setup_logger
from image_classifier.metadata import check_exiftool, finder_tags_supported
from image_classifier.processor import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    ImageProcessor,
    ScoredImage,
)
from image_classifier.report import (
    RunSummary,
    print_profile_summary,
    print_summary,
    star_display,
)
from image_classifier.scanner import is_protected_bundle, scan_images

console = Console()
err_console = Console(stderr=True)

_MAX_FAILURES_SHOWN = 10

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classify",
        description="Score images aesthetically and write XMP star ratings.",
    )
    parser.add_argument("folder", type=Path, help="Folder of images to classify")
    parser.add_argument(
        "--force", action="store_true", help="Re-score already-processed images"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Scan subdirectories recursively"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score images and report the results without modifying any image "
        "or the database",
    )
    parser.add_argument(
        "--no-finder-tags",
        action="store_true",
        help="Write only the XMP star rating, not the macOS Finder score tag",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help="Threads for decoding images and writing metadata (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help="Images scored per model call; lower it if you run out of memory "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        metavar="PATH",
        help="SQLite cache of scored images (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show per-phase timing summary after the run",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _print_header() -> None:
    header = Table(box=None, padding=(0, 1))
    header.add_column(justify="left")
    header.add_row("[bold cyan]Image Classifier[/bold cyan]")
    header.add_row("[dim]Score images aesthetically and write XMP star ratings[/dim]")
    console.print(Panel(header, border_style="cyan", expand=False))


def _print_scan_info(folder: Path, total: int, skipped: int, to_process: int) -> None:
    scan_table = Table(box=None, padding=(0, 2))
    scan_table.add_row("📂 Folder", f"[bold]{folder}[/bold]")
    scan_table.add_row("🖼️  Total", f"{total} images")
    scan_table.add_row(
        "🔍 Scanned",
        f"{skipped} already scored, [bold cyan]{to_process}[/bold cyan] to process",
    )
    console.print(
        Panel(scan_table, title="Scanning Info", border_style="dim", expand=False)
    )
    console.print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        # Ctrl+C while scanning or loading the model; interrupting the scoring
        # loop itself is handled in _process so it can still print a summary.
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        return EXIT_INTERRUPTED


def _run(args: argparse.Namespace) -> int:
    folder: Path = args.folder.resolve()

    if not folder.exists():
        err_console.print(f"[red]Error: folder not found: {folder}[/red]")
        return EXIT_ERROR
    if not folder.is_dir():
        err_console.print(f"[red]Error: expected a folder, got a file: {folder}[/red]")
        return EXIT_ERROR
    if is_protected_bundle(folder):
        err_console.print(
            f"[red]Error: {folder.name} is a photo-library bundle managed by another "
            "app; modifying its contents can corrupt the library.[/red]\n"
            "[dim]Export the photos to a normal folder and classify that instead.[/dim]"
        )
        return EXIT_ERROR

    if not args.dry_run:
        check_exiftool()
    logger = setup_logger()

    _print_header()
    if args.dry_run:
        console.print(
            "[yellow]Dry run: no images or database rows will be written.[/yellow]"
        )
    console.print()

    # Work out what needs doing before paying for the model load: when
    # everything is already scored, the run should finish instantly.
    try:
        images = scan_images(folder, args.recursive)
    except OSError as exc:
        err_console.print(f"[red]Error: cannot read folder {folder}: {exc}[/red]")
        return EXIT_ERROR

    conn = make_connection(args.db)
    processor: ImageProcessor | None = None
    try:
        to_process = (
            images if args.force else [p for p in images if not is_processed(p, conn)]
        )
        skipped = len(images) - len(to_process)
        _print_scan_info(folder, len(images), skipped, len(to_process))

        if not to_process:
            console.print("[green]Nothing to do.[/green]")
            print_summary(
                console,
                RunSummary(
                    scored=0,
                    skipped=skipped,
                    errors=0,
                    rating_counts=rating_counts(folder, conn),
                ),
                LOG_PATH,
            )
            return EXIT_OK

        console.print(
            "[dim]Note: On first run the model (~1.5 GB) will be downloaded to "
            "~/.cache/huggingface.[/dim]"
        )
        try:
            with console.status(
                "[bold blue]Loading aesthetic model...", spinner="dots"
            ):
                device = get_device()
                model, preprocessor = load_model(device)
        except Exception as exc:
            err_console.print(
                f"[red]Error: could not load the aesthetic model: "
                f"{type(exc).__name__}: {exc}[/red]\n"
                "[dim]The first run needs an internet connection to download the "
                "model; interrupted downloads resume on the next attempt.[/dim]"
            )
            return EXIT_ERROR
        device_color = "green" if device.type != "cpu" else "yellow"
        console.print(
            f"✨ Model loaded on [{device_color}]{device.type.upper()}[/{device_color}]"
        )

        finder_tags = finder_tags_supported() and not args.no_finder_tags
        if not finder_tags and not args.no_finder_tags and not args.dry_run:
            console.print(
                "[dim]Finder tags need macOS; writing XMP star ratings only.[/dim]"
            )
        console.print()

        processor = ImageProcessor(
            model,
            preprocessor,
            device,
            conn,
            args.db,
            workers=args.workers,
            batch_size=args.batch_size,
            write_finder_tags=finder_tags,
            dry_run=args.dry_run,
        )
        return _process(
            processor, to_process, folder, skipped, logger, args.dry_run, args.profile
        )
    finally:
        # The processor swaps in a fresh connection if the old one died.
        (processor.conn if processor else conn).close()


def _process(
    processor: ImageProcessor,
    to_process: list[Path],
    folder: Path,
    skipped: int,
    logger: logging.Logger | None,
    dry_run: bool,
    profile: bool,
) -> int:
    scored = 0
    errors = 0
    interrupted = False
    all_timings: list[Timings] = []
    run_ratings: Counter[int] = Counter()
    failures: list[tuple[str, str]] = []

    start = time.perf_counter()
    outcomes = processor.run(to_process)
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
            task_id = progress.add_task(
                "[cyan]Processing images...", total=len(to_process)
            )
            for outcome in outcomes:
                if isinstance(outcome, ScoredImage):
                    scored += 1
                    all_timings.append(outcome.timings)
                    run_ratings[outcome.rating] += 1
                    progress.update(
                        task_id,
                        advance=1,
                        description=(
                            f"Last: {outcome.path.name}  "
                            f"[bold yellow]{outcome.score:.2f}[/]  "
                            f"{star_display(outcome.rating)}"
                        ),
                    )
                else:
                    errors += 1
                    log_error(logger, outcome.path, outcome.error)
                    failures.append((outcome.path.name, outcome.message))
                    progress.update(task_id, advance=1)
    except KeyboardInterrupt:
        interrupted = True
        console.print()
        console.print("[yellow]⚠️  Interrupt received. Partial results saved.[/yellow]")
    finally:
        outcomes.close()
    elapsed = time.perf_counter() - start

    summary = RunSummary(
        scored=scored,
        skipped=skipped,
        errors=errors,
        # A dry run records nothing, so report only what this run scored.
        rating_counts=(
            run_ratings if dry_run else rating_counts(folder, processor.conn)
        ),
        elapsed_s=elapsed,
        dry_run=dry_run,
        failures=failures[-_MAX_FAILURES_SHOWN:][::-1],
        failures_total=errors,
    )
    print_summary(console, summary, LOG_PATH)
    if profile:
        print_profile_summary(console, all_timings, elapsed)
    return EXIT_INTERRUPTED if interrupted else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
