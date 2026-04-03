from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.db"


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Create the images and failures tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            path         TEXT PRIMARY KEY,
            score        REAL NOT NULL,
            rating       INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failures (
            path      TEXT PRIMARY KEY,
            error     TEXT NOT NULL,
            failed_at TEXT NOT NULL
        )
    """)
    conn.commit()


def make_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema.

    The caller owns the connection lifecycle and is responsible for closing it.
    Pass ':memory:' for an in-memory database (useful in tests).
    """
    path = Path(db_path)
    if (
        str(path) != ":memory:"
    ):  # pathlib.Path(":memory:") stringifies back to ":memory:"
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL mode avoids creating a rollback journal file during writes, which
    # is more resilient when macOS background processes (Spotlight, Time
    # Machine) briefly lock the database directory.
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    _apply_schema(conn)
    return conn


def is_processed(path: Path, conn: sqlite3.Connection) -> bool:
    """Return True if the image at path has already been scored."""
    row = conn.execute(
        "SELECT 1 FROM images WHERE path = ?",
        (str(path.resolve()),),
    ).fetchone()
    return row is not None


def upsert(path: Path, score: float, rating: int, conn: sqlite3.Connection) -> None:
    """Insert or replace a scored image record."""
    processed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT OR REPLACE INTO images (path, score, rating, processed_at) VALUES (?, ?, ?, ?)",
        (str(path.resolve()), score, rating, processed_at),
    )
    conn.commit()


def upsert_failure(path: Path, error: str, conn: sqlite3.Connection) -> None:
    """Insert or replace a failure record for an image that could not be scored."""
    failed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT OR REPLACE INTO failures (path, error, failed_at) VALUES (?, ?, ?)",
        (str(path.resolve()), error, failed_at),
    )
    conn.commit()


def all_failures(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all failure records whose path is inside folder."""
    safe_prefix = (
        str(folder.resolve())
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        + "/"
    )
    return conn.execute(
        "SELECT path, error, failed_at FROM failures WHERE path LIKE ? ESCAPE '\\'",
        (safe_prefix + "%",),
    ).fetchall()


def all_scores(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all scored images whose path is inside folder (trailing-slash safe)."""
    # Escape LIKE wildcards in the folder path itself, then append the real wildcard.
    safe_prefix = (
        str(folder.resolve())
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        + "/"
    )
    return conn.execute(
        "SELECT path, score, rating FROM images WHERE path LIKE ? ESCAPE '\\'",
        (safe_prefix + "%",),
    ).fetchall()
