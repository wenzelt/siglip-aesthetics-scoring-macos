from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from image_classifier.database import all_scores, is_processed, make_connection, upsert


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema applied."""
    c = make_connection(":memory:")
    yield c
    c.close()


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
    row = conn.execute("SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)).fetchone()
    assert abs(row["score"] - 6.1) < 0.001
    assert row["rating"] == 3


def test_upsert_replaces_on_force(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=4.0, rating=2, conn=conn)
    upsert(img, score=8.9, rating=5, conn=conn)
    row = conn.execute("SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)).fetchone()
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
