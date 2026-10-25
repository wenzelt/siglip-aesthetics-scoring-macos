from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from image_classifier.database import (
    all_failures,
    all_scores,
    is_processed,
    make_connection,
    rating_counts,
    upsert,
    upsert_failure,
    upsert_failures,
    upsert_many,
)


def test_schema_created(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='images'"
    ).fetchone()
    assert row is not None


def test_is_processed_false_for_new_path(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    assert is_processed(img, conn) is False


def test_upsert_and_is_processed(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=7.3, rating=4, conn=conn)
    assert is_processed(img, conn) is True


def test_upsert_stores_correct_values(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=6.1, rating=3, conn=conn)
    row = conn.execute(
        "SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)
    ).fetchone()
    assert abs(row["score"] - 6.1) < 0.001
    assert row["rating"] == 3


def test_upsert_replaces_on_force(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=4.0, rating=2, conn=conn)
    upsert(img, score=8.9, rating=5, conn=conn)
    row = conn.execute(
        "SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)
    ).fetchone()
    assert row["rating"] == 5


def test_path_stored_as_resolved_absolute(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=5.0, rating=2, conn=conn)
    rows = conn.execute("SELECT path FROM images").fetchall()
    assert rows[0]["path"] == str(img.resolve())


def test_all_scores_scoped_to_folder(conn, tmp_path):
    folder_a = tmp_path / "vacation"
    folder_b = tmp_path / "vacation-backup"
    folder_a.mkdir()
    folder_b.mkdir()
    img_a = folder_a / "a.jpg"
    img_b = folder_b / "b.jpg"
    img_a.touch()
    img_b.touch()
    upsert(img_a, score=7.0, rating=4, conn=conn)
    upsert(img_b, score=8.0, rating=5, conn=conn)
    results = all_scores(folder_a, conn)
    assert len(results) == 1
    assert results[0]["rating"] == 4


def test_all_scores_returns_all_in_folder(conn, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(3):
        img = folder / f"img_{i}.jpg"
        img.touch()
        upsert(img, score=float(i + 5), rating=3, conn=conn)
    results = all_scores(folder, conn)
    assert len(results) == 3


def test_processed_at_is_iso8601_utc(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=5.5, rating=3, conn=conn)
    row = conn.execute("SELECT processed_at FROM images").fetchone()
    assert row["processed_at"].endswith("Z")
    assert "T" in row["processed_at"]


# --- failures table ---


def test_failures_table_created(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='failures'"
    ).fetchone()
    assert row is not None


def test_upsert_failure_stores_path_and_error(conn, tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()
    upsert_failure(img, "ValueError: mean must have 1 elements", conn)
    row = conn.execute(
        "SELECT path, error FROM failures WHERE path = ?", (str(img.resolve()),)
    ).fetchone()
    assert row is not None
    assert "ValueError" in row["error"]


def test_upsert_failure_overwrites_previous_error(conn, tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()
    upsert_failure(img, "first error", conn)
    upsert_failure(img, "second error", conn)
    rows = conn.execute(
        "SELECT error FROM failures WHERE path = ?", (str(img.resolve()),)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["error"] == "second error"


def test_upsert_failure_records_timestamp(conn, tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()
    upsert_failure(img, "some error", conn)
    row = conn.execute("SELECT failed_at FROM failures").fetchone()
    assert row["failed_at"].endswith("Z")
    assert "T" in row["failed_at"]


def test_all_failures_scoped_to_folder(conn, tmp_path):
    folder_a = tmp_path / "photos"
    folder_b = tmp_path / "other"
    folder_a.mkdir()
    folder_b.mkdir()
    upsert_failure(folder_a / "a.jpg", "err", conn)
    upsert_failure(folder_b / "b.jpg", "err", conn)
    results = all_failures(folder_a, conn)
    assert len(results) == 1
    assert "photos" in results[0]["path"]


def test_all_failures_returns_empty_when_none(conn, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    assert all_failures(folder, conn) == []


def test_upsert_clears_failure_record_on_success(conn, tmp_path):
    """A successful upsert must remove any pre-existing failures row for that path."""
    img = tmp_path / "retry.jpg"
    img.touch()
    upsert_failure(img, "OSError: file unreadable", conn)
    # Confirm the failure row exists before the retry
    row = conn.execute(
        "SELECT 1 FROM failures WHERE path = ?", (str(img.resolve()),)
    ).fetchone()
    assert row is not None, "pre-condition: failure row should exist before upsert"

    upsert(img, score=7.5, rating=4, conn=conn)

    leftover = conn.execute(
        "SELECT 1 FROM failures WHERE path = ?", (str(img.resolve()),)
    ).fetchone()
    assert leftover is None, (
        "upsert() must delete the failures row for a successfully scored image"
    )


# --- batch writes ---


def test_upsert_many_writes_every_row_in_one_call(conn, tmp_path):
    rows = []
    for i in range(5):
        img = tmp_path / f"{i}.jpg"
        img.touch()
        rows.append((img, 5.0 + i / 10, 3))
    upsert_many(rows, conn)
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 5
    assert all(is_processed(path, conn) for path, _, _ in rows)


def test_upsert_many_accepts_a_generator_and_an_empty_batch(conn, tmp_path):
    upsert_many(iter([]), conn)
    img = tmp_path / "a.jpg"
    img.touch()
    upsert_many((row for row in [(img, 6.0, 3)]), conn)
    assert is_processed(img, conn)


def test_upsert_many_clears_failures_for_all_of_its_paths(conn, tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    upsert_failures([(a, "boom"), (b, "boom")], conn)
    upsert_many([(a, 7.0, 4), (b, 7.0, 4)], conn)
    assert conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0


def test_upsert_many_is_atomic(conn, tmp_path):
    """If any row is invalid nothing from the batch is committed."""
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    with pytest.raises(sqlite3.IntegrityError):
        upsert_many([(good, 7.0, 4), (bad, None, 4)], conn)  # score is NOT NULL
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_upsert_failures_writes_every_row(conn, tmp_path):
    upsert_failures([(tmp_path / "a.jpg", "e1"), (tmp_path / "b.jpg", "e2")], conn)
    assert conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 2
    upsert_failures([], conn)


# --- rating_counts ---


def test_rating_counts_groups_by_star_rating_within_the_folder(conn, tmp_path):
    inside = tmp_path / "trip"
    other = tmp_path / "trip-backup"
    inside.mkdir()
    other.mkdir()
    for i, rating in enumerate([5, 5, 3, 1]):
        upsert(inside / f"{i}.jpg", float(rating), rating, conn)
    upsert(other / "x.jpg", 9.0, 5, conn)
    assert rating_counts(inside, conn) == {5: 2, 3: 1, 1: 1}


def test_rating_counts_is_empty_for_an_unscored_folder(conn, tmp_path):
    assert rating_counts(tmp_path, conn) == {}


# --- folder matching ---


@pytest.mark.parametrize("name", ["100%_done", "a_b", "50%", "back\\slash"])
def test_folder_filters_treat_like_wildcards_literally(conn, tmp_path, name):
    folder = tmp_path / name
    sibling = tmp_path / name.replace("_", "X").replace("%", "Y").replace("\\", "Z")
    folder.mkdir()
    sibling.mkdir()
    upsert(folder / "in.jpg", 7.0, 4, conn)
    upsert(sibling / "out.jpg", 7.0, 4, conn)
    assert [Path(r["path"]).name for r in all_scores(folder, conn)] == ["in.jpg"]
    assert rating_counts(folder, conn) == {4: 1}


def test_folder_filter_works_for_the_filesystem_root(conn):
    upsert(Path("/top.jpg"), 7.0, 4, conn)
    assert len(all_scores(Path("/"), conn)) == 1


def test_all_failures_are_listed_newest_first(conn, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    with patch("image_classifier.database._now", return_value="2024-01-01T00:00:00Z"):
        upsert_failure(folder / "old.jpg", "e", conn)
    with patch("image_classifier.database._now", return_value="2025-01-01T00:00:00Z"):
        upsert_failure(folder / "new.jpg", "e", conn)
    assert [Path(r["path"]).name for r in all_failures(folder, conn)] == [
        "new.jpg",
        "old.jpg",
    ]


# --- connection setup ---


def test_file_database_uses_wal_and_creates_parent_folders(tmp_path):
    db = tmp_path / "nested" / "dir" / "classify.db"
    c = make_connection(db)
    try:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert c.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        c.close()
    assert db.exists()


def test_scores_survive_reopening_the_database(tmp_path):
    db = tmp_path / "classify.db"
    img = tmp_path / "a.jpg"
    img.touch()
    first = make_connection(db)
    upsert(img, 8.0, 4, first)
    first.close()
    second = make_connection(db)
    try:
        assert is_processed(img, second)
    finally:
        second.close()
