from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

from image_classifier.classifier import Timings, score_image, score_to_rating
from image_classifier.database import make_connection, upsert, upsert_failure
from image_classifier.metadata import write_rating, write_score_tag


class ProgressCallback(Protocol):
    def __call__(self, path: Path, score: float, rating: int) -> None: ...


class ImageProcessor:
    """Handles the processing of individual images and updates database/metadata."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        device: Any,
        conn: sqlite3.Connection,
        db_path: Path,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.conn = conn
        self.db_path = db_path

    def process_image(
        self, path: Path, callback: ProgressCallback | None = None
    ) -> tuple[float, Timings]:
        """Process a single image: score it, update DB, and write metadata."""
        score, timings = score_image(path, self.model, self.preprocessor, self.device)
        rating = score_to_rating(score)

        t = time.perf_counter()
        self._safe_upsert(path, score, rating)
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

        if callback:
            callback(path, score, rating)

        return score, timings

    def handle_failure(self, path: Path, error_str: str) -> None:
        """Record a processing failure in the database."""
        try:
            upsert_failure(path, error_str, self.conn)
        except sqlite3.OperationalError:
            try:
                self.conn.close()
                self.conn = make_connection(self.db_path)
                upsert_failure(path, error_str, self.conn)
            except sqlite3.OperationalError:
                pass  # DB unavailable

    def _safe_upsert(self, path: Path, score: float, rating: int) -> None:
        """Upsert a record, retrying once if the connection is dead."""
        try:
            upsert(path, score, rating, self.conn)
        except sqlite3.OperationalError:
            self.conn.close()
            self.conn = make_connection(self.db_path)
            upsert(path, score, rating, self.conn)
