from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.metadata import MetadataError, check_exiftool, write_rating


def test_check_exiftool_exits_when_missing():
    with patch("image_classifier.metadata.shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            check_exiftool()
    assert "brew install exiftool" in str(exc_info.value)


def test_check_exiftool_passes_when_present():
    with patch("image_classifier.metadata.shutil.which", return_value="/usr/local/bin/exiftool"):
        check_exiftool()  # Should not raise


def test_write_rating_calls_exiftool_with_correct_args(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("image_classifier.metadata.subprocess.run", return_value=mock_result) as mock_run:
        write_rating(img, rating=4)
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "exiftool" in call_args
    assert "-XMP:Rating=4" in call_args
    assert "-overwrite_original" in call_args
    assert str(img) in call_args


def test_write_rating_raises_metadata_error_on_nonzero_exit(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Error: something went wrong"
    with patch("image_classifier.metadata.subprocess.run", return_value=mock_result):
        with pytest.raises(MetadataError) as exc_info:
            write_rating(img, rating=3)
    assert "photo.jpg" in str(exc_info.value)
    assert "something went wrong" in str(exc_info.value)


def test_write_rating_all_valid_ratings(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("image_classifier.metadata.subprocess.run", return_value=mock_result) as mock_run:
        for rating in range(1, 6):
            write_rating(img, rating=rating)
            call_args = mock_run.call_args[0][0]
            assert f"-XMP:Rating={rating}" in call_args
