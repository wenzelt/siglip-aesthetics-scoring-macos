from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.db"


def make_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            path         TEXT PRIMARY KEY,
            score        REAL NOT NULL,
            rating       INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        )
    """)
    conn.commit()
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


def all_scores(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all scored images whose path is inside folder (trailing-slash safe)."""
    prefix = str(folder.resolve()) + "/"
    return conn.execute(
        "SELECT path, score, rating FROM images WHERE path LIKE ?",
        (prefix + "%",),
    ).fetchall()
