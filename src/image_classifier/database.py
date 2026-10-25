from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.db"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    in_memory = str(path) == ":memory:"  # Path(":memory:") stringifies back unchanged
    if not in_memory:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    if not in_memory:
        # WAL mode avoids creating a rollback journal file during writes, which
        # is more resilient when macOS background processes (Spotlight, Time
        # Machine) briefly lock the database directory.  NORMAL sync is safe
        # under WAL (a crash can lose the last commit, never corrupt the file),
        # and the worst case here is re-scoring one batch.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    _apply_schema(conn)
    return conn


def is_processed(path: Path, conn: sqlite3.Connection) -> bool:
    """Return True if the image at path has already been scored."""
    row = conn.execute(
        "SELECT 1 FROM images WHERE path = ?",
        (str(path.resolve()),),
    ).fetchone()
    return row is not None


def upsert_many(
    rows: Iterable[tuple[Path, float, int]], conn: sqlite3.Connection
) -> None:
    """Insert or replace scored image records in one transaction.

    Also clears any failure record for those paths, since they now succeeded.
    """
    processed_at = _now()
    records = [
        (str(path.resolve()), score, rating, processed_at)
        for path, score, rating in rows
    ]
    if not records:
        return
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO images (path, score, rating, processed_at)"
            " VALUES (?, ?, ?, ?)",
            records,
        )
        conn.executemany(
            "DELETE FROM failures WHERE path = ?", [(r[0],) for r in records]
        )


def upsert(path: Path, score: float, rating: int, conn: sqlite3.Connection) -> None:
    """Insert or replace a scored image record."""
    upsert_many([(path, score, rating)], conn)


def upsert_failures(rows: Iterable[tuple[Path, str]], conn: sqlite3.Connection) -> None:
    """Insert or replace failure records in one transaction."""
    failed_at = _now()
    records = [(str(path.resolve()), error, failed_at) for path, error in rows]
    if not records:
        return
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO failures (path, error, failed_at) VALUES (?, ?, ?)",
            records,
        )


def upsert_failure(path: Path, error: str, conn: sqlite3.Connection) -> None:
    """Insert or replace a failure record for an image that could not be scored."""
    upsert_failures([(path, error)], conn)


def _folder_pattern(folder: Path) -> str:
    """LIKE pattern matching every path inside folder (and not its siblings).

    Wildcards in the folder name itself are escaped; use with ESCAPE '\\'.
    """
    escaped = (
        str(folder.resolve())
        .rstrip("/")  # only the filesystem root keeps a trailing slash
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return escaped + "/%"


def all_failures(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return failure records whose path is inside folder, newest first."""
    return conn.execute(
        "SELECT path, error, failed_at FROM failures WHERE path LIKE ? ESCAPE '\\'"
        " ORDER BY failed_at DESC, path",
        (_folder_pattern(folder),),
    ).fetchall()


def all_scores(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all scored images whose path is inside folder (trailing-slash safe)."""
    return conn.execute(
        "SELECT path, score, rating FROM images WHERE path LIKE ? ESCAPE '\\'",
        (_folder_pattern(folder),),
    ).fetchall()


def rating_counts(folder: Path, conn: sqlite3.Connection) -> dict[int, int]:
    """Return {star rating: number of scored images} for paths inside folder."""
    rows = conn.execute(
        "SELECT rating, COUNT(*) FROM images WHERE path LIKE ? ESCAPE '\\'"
        " GROUP BY rating",
        (_folder_pattern(folder),),
    ).fetchall()
    return {int(rating): int(count) for rating, count in rows}
