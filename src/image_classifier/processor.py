from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from image_classifier.classifier import (
    Timings,
    load_image,
    preprocess,
    score_batch,
    score_to_rating,
    validate_score,
)
from image_classifier.database import make_connection, upsert_failures, upsert_many
from image_classifier.metadata import read_finder_tags, write_rating, write_score_tag

DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class ScoredImage:
    """An image that was scored, had its metadata written and was recorded."""

    path: Path
    score: float
    rating: int
    timings: Timings


@dataclass(frozen=True)
class FailedImage:
    """An image that could not be scored or written; the run carries on."""

    path: Path
    error: Exception

    @property
    def message(self) -> str:
        return f"{type(self.error).__name__}: {self.error}"


Outcome = ScoredImage | FailedImage


@dataclass
class _Loaded:
    path: Path
    pixel_values: torch.Tensor
    timings: Timings


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _write_metadata(
    path: Path, score: float, rating: int, finder_tags: bool
) -> tuple[float, float]:
    """Write the XMP rating (and Finder tag) to path, preserving its mtime.

    Returns (exiftool_ms, xattr_ms).  Safe to call from worker threads.
    """
    original_mtime_ns = path.stat().st_mtime_ns
    exiftool_ms = xattr_ms = 0.0
    try:
        # Read the user's tags *before* exiftool replaces the file, since the
        # replacement is not guaranteed to carry extended attributes across.
        tags: list[str] = []
        if finder_tags:
            t = time.perf_counter()
            tags = read_finder_tags(path)
            xattr_ms = _ms_since(t)

        t = time.perf_counter()
        write_rating(path, rating)
        exiftool_ms = _ms_since(t)

        if finder_tags:
            t = time.perf_counter()
            write_score_tag(path, score, existing_tags=tags)
            xattr_ms += _ms_since(t)
    finally:
        with contextlib.suppress(OSError):
            os.utime(path, ns=(path.stat().st_atime_ns, original_mtime_ns))
    return exiftool_ms, xattr_ms


class ImageProcessor:
    """Scores images and writes their metadata, in parallel where it pays off.

    Work is done in batches.  While one batch is being scored and written, the
    next batch is already being decoded on worker threads; metadata writes
    (which are subprocess calls) also run on worker threads.  Inference and all
    SQLite access stay on the thread that iterates ``run()``.

    A single bad image never aborts the run: it is reported as a FailedImage
    and recorded in the failures table.  With ``dry_run`` nothing is written
    to the images or the database.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        device: Any,
        conn: sqlite3.Connection,
        db_path: Path,
        *,
        workers: int = DEFAULT_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        write_finder_tags: bool = True,
        dry_run: bool = False,
    ):
        if workers < 1 or batch_size < 1:
            raise ValueError("workers and batch_size must be at least 1")
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.conn = conn
        self.db_path = db_path
        self.workers = workers
        self.batch_size = batch_size
        self.write_finder_tags = write_finder_tags
        self.dry_run = dry_run

    def run(self, paths: Sequence[Path]) -> Generator[Outcome, None, None]:
        """Process paths, yielding one outcome per path as each batch completes."""
        batches = [
            list(paths[i : i + self.batch_size])
            for i in range(0, len(paths), self.batch_size)
        ]
        if not batches:
            return

        loader = ThreadPoolExecutor(self.workers, thread_name_prefix="load")
        writer = ThreadPoolExecutor(self.workers, thread_name_prefix="write")
        try:
            pending = self._submit_loads(loader, batches[0])
            for index in range(len(batches)):
                loaded = [future.result() for future in pending]
                # Start decoding the next batch before this one is scored.
                following = index + 1
                pending = (
                    self._submit_loads(loader, batches[following])
                    if following < len(batches)
                    else []
                )
                yield from self._finish_batch(loaded, writer)
        finally:
            loader.shutdown(wait=True, cancel_futures=True)
            writer.shutdown(wait=True, cancel_futures=True)

    # -- stages -----------------------------------------------------------

    def _submit_loads(
        self, pool: ThreadPoolExecutor, batch: list[Path]
    ) -> list[Future[_Loaded | FailedImage]]:
        return [pool.submit(self._load, path) for path in batch]

    def _load(self, path: Path) -> _Loaded | FailedImage:
        """Decode and preprocess one image (worker thread)."""
        timings = Timings()
        try:
            t = time.perf_counter()
            image_array = load_image(path, self.preprocessor)
            timings.load_ms = _ms_since(t)

            t = time.perf_counter()
            pixel_values = preprocess(image_array, self.preprocessor)
            timings.preprocess_ms = _ms_since(t)
        except Exception as exc:
            return FailedImage(path, exc)
        return _Loaded(path, pixel_values, timings)

    def _infer_each(self, items: list[_Loaded]) -> list[float | Exception]:
        """Score items one at a time, keeping any failure local to its image."""
        results: list[float | Exception] = []
        for item in items:
            try:
                score = score_batch([item.pixel_values], self.model, self.device)[0]
                results.append(validate_score(score))
            except Exception as exc:
                results.append(exc)
        return results

    def _infer(self, items: list[_Loaded]) -> list[float | Exception]:
        """Score items in one forward pass, falling back to one-by-one on error."""
        if not items:
            return []
        start = time.perf_counter()
        try:
            raw = score_batch([i.pixel_values for i in items], self.model, self.device)
            results: list[float | Exception] = []
            for score in raw:
                try:
                    results.append(validate_score(score))
                except Exception as exc:
                    results.append(exc)
        except Exception:
            # One bad tensor (or an out-of-memory batch) must not sink the rest.
            results = self._infer_each(items)
        share = _ms_since(start) / len(items)
        for item in items:
            item.timings.infer_ms = share
        return results

    def _finish_batch(
        self, loaded: list[_Loaded | FailedImage], writer: ThreadPoolExecutor
    ) -> Iterator[Outcome]:
        """Infer, write metadata and record one batch; yield outcomes in order."""
        outcomes: list[Outcome | None] = [None] * len(loaded)
        ready: list[tuple[int, _Loaded]] = []
        for index, item in enumerate(loaded):
            if isinstance(item, FailedImage):
                outcomes[index] = item
            else:
                ready.append((index, item))

        scores = self._infer([item for _, item in ready])

        writes: list[tuple[int, _Loaded, float, int, Future[tuple[float, float]]]] = []
        for (index, item), score in zip(ready, scores, strict=True):
            if isinstance(score, Exception):
                outcomes[index] = FailedImage(item.path, score)
                continue
            rating = score_to_rating(score)
            if self.dry_run:
                outcomes[index] = ScoredImage(item.path, score, rating, item.timings)
            else:
                future = writer.submit(
                    _write_metadata, item.path, score, rating, self.write_finder_tags
                )
                writes.append((index, item, score, rating, future))

        for index, item, score, rating, future in writes:
            try:
                item.timings.exiftool_ms, item.timings.xattr_ms = future.result()
            except Exception as exc:
                outcomes[index] = FailedImage(item.path, exc)
            else:
                outcomes[index] = ScoredImage(item.path, score, rating, item.timings)

        finished = [outcome for outcome in outcomes if outcome is not None]
        if not self.dry_run:
            finished = self._record(finished)
        yield from finished

    # -- persistence ------------------------------------------------------

    def _with_retry(self, operation: Callable[[sqlite3.Connection], None]) -> None:
        """Run a database operation, reconnecting once if the connection died."""
        try:
            operation(self.conn)
        except sqlite3.OperationalError:
            self.conn.close()
            self.conn = make_connection(self.db_path)
            operation(self.conn)

    def _record(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Persist a batch (one transaction for scores, one for failures).

        Metadata is written before the database row, so a failed write never
        leaves a "processed" row that would make the next run skip the image.
        If the database itself is unavailable the scored images are reported
        as failures instead; they simply get scored again next run.
        """
        scored = [o for o in outcomes if isinstance(o, ScoredImage)]
        failed = [o for o in outcomes if isinstance(o, FailedImage)]

        if failed:
            with contextlib.suppress(sqlite3.Error):  # DB unavailable
                self._with_retry(
                    lambda conn: upsert_failures(
                        ((f.path, f.message) for f in failed), conn
                    )
                )

        if not scored:
            return outcomes

        start = time.perf_counter()
        try:
            self._with_retry(
                lambda conn: upsert_many(
                    ((s.path, s.score, s.rating) for s in scored), conn
                )
            )
        except sqlite3.Error as exc:
            return [
                FailedImage(o.path, exc) if isinstance(o, ScoredImage) else o
                for o in outcomes
            ]
        share = _ms_since(start) / len(scored)
        for s in scored:
            s.timings.upsert_ms = share
        return outcomes
