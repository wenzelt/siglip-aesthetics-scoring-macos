from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from image_classifier.metadata import MetadataError, check_exiftool, write_rating, write_score_tag


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


# --- write_score_tag tests ---


def _decode_xattr_call(mock_run):
    """Extract the tags list written by the xattr call."""
    call_args = mock_run.call_args[0][0]
    hex_data = call_args[3]
    return plistlib.loads(bytes.fromhex(hex_data))


def test_write_score_tag_writes_only_numeric_score(tmp_path):
    """Only the numeric score should be written — no star string."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_ok = MagicMock(returncode=0)
    # Simulate no existing tags
    xattr_fail = MagicMock(returncode=1)
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_fail, xattr_ok],
    ) as mock_run:
        write_score_tag(img, rating=3, score=6.4)

    tags = _decode_xattr_call(mock_run)
    assert tags == ["6.4"]


def test_write_score_tag_no_star_string_in_tags(tmp_path):
    """Star strings (★…) must NOT appear in written tags."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_ok = MagicMock(returncode=0)
    xattr_fail = MagicMock(returncode=1)
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_fail, xattr_ok],
    ) as mock_run:
        write_score_tag(img, rating=5, score=9.1)

    tags = _decode_xattr_call(mock_run)
    assert not any("★" in t or "☆" in t for t in tags)


def test_write_score_tag_preserves_user_tags(tmp_path):
    """Existing non-score user tags must be kept; old score tag replaced."""
    img = tmp_path / "photo.jpg"
    img.touch()
    existing_tags = ["vacation", "family", "5.5"]  # "5.5" is old score tag
    plist_hex = plistlib.dumps(existing_tags, fmt=plistlib.FMT_BINARY).hex()

    xattr_read_ok = MagicMock(returncode=0, stdout=plist_hex)
    xattr_write_ok = MagicMock(returncode=0)
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_read_ok, xattr_write_ok],
    ) as mock_run:
        write_score_tag(img, rating=4, score=7.8)

    tags = _decode_xattr_call(mock_run)
    assert "vacation" in tags
    assert "family" in tags
    assert "7.8" in tags
    assert "5.5" not in tags  # old score replaced


def test_write_score_tag_raises_on_xattr_failure(tmp_path):
    """MetadataError is raised when xattr write fails."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_fail_read = MagicMock(returncode=1)
    xattr_fail_write = MagicMock(returncode=1, stderr="permission denied")
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_fail_read, xattr_fail_write],
    ):
        with pytest.raises(MetadataError, match="xattr failed"):
            write_score_tag(img, rating=2, score=4.3)
