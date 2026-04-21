from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.classifier import Timings
from image_classifier.processor import ImageProcessor
from image_classifier.metadata import MetadataError

_TIMINGS = Timings(load_ms=10.0, preprocess_ms=20.0, infer_ms=500.0)


@pytest.fixture
def mock_processor():
    model = MagicMock()
    preprocessor = MagicMock()
    device = MagicMock()
    conn = MagicMock()
    db_path = Path(":memory:")
    return ImageProcessor(model, preprocessor, device, conn, db_path)


def test_process_image_calls_dependencies(mock_processor, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    with (
        patch("image_classifier.processor.score_image", return_value=(7.0, _TIMINGS)),
        patch("image_classifier.processor.score_to_rating", return_value=4),
        patch("image_classifier.processor.write_rating") as mock_write_rating,
        patch("image_classifier.processor.write_score_tag") as mock_write_tag,
        patch("image_classifier.processor.upsert") as mock_upsert,
    ):
        score, timings = mock_processor.process_image(img)

    assert score == 7.0
    mock_write_rating.assert_called_once_with(img, 4)
    mock_write_tag.assert_called_once_with(img, 7.0)
    mock_upsert.assert_called_once_with(img, 7.0, 4, mock_processor.conn)


def test_process_image_restores_mtime(mock_processor, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    original_mtime = img.stat().st_mtime_ns

    with (
        patch("image_classifier.processor.score_image", return_value=(7.0, _TIMINGS)),
        patch("image_classifier.processor.write_rating"),
        patch("image_classifier.processor.write_score_tag"),
        patch("image_classifier.processor.upsert"),
    ):
        mock_processor.process_image(img)

    assert img.stat().st_mtime_ns == original_mtime


def test_upsert_called_even_when_metadata_write_fails(mock_processor, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    with (
        patch("image_classifier.processor.score_image", return_value=(7.0, _TIMINGS)),
        patch("image_classifier.processor.score_to_rating", return_value=4),
        patch(
            "image_classifier.processor.write_rating",
            side_effect=MetadataError("exiftool failed"),
        ),
        patch("image_classifier.processor.upsert") as mock_upsert,
    ):
        with pytest.raises(MetadataError):
            mock_processor.process_image(img)

    # In my current implementation of ImageProcessor.process_image,
    # the upsert is called BEFORE write_rating.
    mock_upsert.assert_called_once()


def test_safe_upsert_retries_on_operational_error(mock_processor, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    old_conn = mock_processor.conn
    mock_processor.conn.execute.side_effect = [sqlite3.OperationalError("dead"), None]

    with (
        patch("image_classifier.processor.upsert", side_effect=[sqlite3.OperationalError("dead"), None]),
        patch("image_classifier.processor.make_connection", return_value=MagicMock()) as mock_make_conn,
    ):
        mock_processor._safe_upsert(img, 7.0, 4)

    assert mock_make_conn.called
    assert old_conn.close.called


def test_handle_failure_records_error(mock_processor, tmp_path):
    img = tmp_path / "broken.jpg"
    img.touch()

    with patch("image_classifier.processor.upsert_failure") as mock_upsert_failure:
        mock_processor.handle_failure(img, "RuntimeError: CUDA error")

    mock_upsert_failure.assert_called_once_with(img, "RuntimeError: CUDA error", mock_processor.conn)
