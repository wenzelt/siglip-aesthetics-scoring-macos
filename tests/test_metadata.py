from __future__ import annotations

import os
import plistlib
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.metadata import (
    MetadataError,
    _is_score_tag,
    check_exiftool,
    finder_tags_supported,
    read_finder_tags,
    write_rating,
    write_score_tag,
)


def test_check_exiftool_exits_when_missing():
    with patch("image_classifier.metadata.shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            check_exiftool()
    assert "brew install exiftool" in str(exc_info.value)


def test_check_exiftool_passes_when_present():
    with patch(
        "image_classifier.metadata.shutil.which", return_value="/usr/local/bin/exiftool"
    ):
        check_exiftool()  # Should not raise


def test_write_rating_calls_exiftool_with_correct_args(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch(
        "image_classifier.metadata.subprocess.run", return_value=mock_result
    ) as mock_run:
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
    with patch(
        "image_classifier.metadata.subprocess.run", return_value=mock_result
    ) as mock_run:
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
        write_score_tag(img, score=6.4)

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
        write_score_tag(img, score=9.1)

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
        write_score_tag(img, score=7.8)

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
            write_score_tag(img, score=4.3)


# --- subprocess timeout ---


def test_write_rating_includes_timeout(tmp_path):
    """exiftool call must include a timeout to avoid hanging."""
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0),
    ) as mock_run:
        write_rating(img, rating=3)
    assert mock_run.call_args.kwargs.get("timeout") == 30


def test_write_score_tag_includes_timeout(tmp_path):
    """xattr write call must include a timeout to avoid hanging."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_fail = MagicMock(returncode=1)
    xattr_ok = MagicMock(returncode=0)
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_fail, xattr_ok],
    ) as mock_run:
        write_score_tag(img, score=6.0)
    # Last call is the xattr write; check it has a timeout
    assert mock_run.call_args.kwargs.get("timeout") == 30


# --- _is_score_tag false-positive fix ---


def test_is_score_tag_rejects_bare_integer():
    """Bare integers (e.g. user tag '7') must NOT be identified as score tags."""
    assert _is_score_tag("7") is False
    assert _is_score_tag("3") is False
    assert _is_score_tag("10") is False
    assert _is_score_tag("0") is False


def test_is_score_tag_accepts_one_decimal_format():
    """Our score format (e.g. '7.3', '10.0') must be identified as score tags."""
    assert _is_score_tag("7.3") is True
    assert _is_score_tag("10.0") is True
    assert _is_score_tag("0.5") is True


# --- write_score_tag: rating param removed ---


def test_write_score_tag_accepts_score_without_rating(tmp_path):
    """write_score_tag must work when called without a rating argument."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_fail = MagicMock(returncode=1)
    xattr_ok = MagicMock(returncode=0)
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=[xattr_fail, xattr_ok],
    ) as mock_run:
        write_score_tag(img, score=7.2)  # no rating arg
    tags = _decode_xattr_call(mock_run)
    assert tags == ["7.2"]


# --- read_finder_tags: bare except narrowed ---


def test_read_finder_tags_propagates_unexpected_errors(tmp_path):
    """Non-parsing errors (e.g. MemoryError) must not be silently swallowed."""
    img = tmp_path / "photo.jpg"
    img.touch()
    xattr_ok = MagicMock(returncode=0, stdout="62706c6973743030")
    with patch("image_classifier.metadata.subprocess.run", return_value=xattr_ok):
        with patch(
            "image_classifier.metadata.plistlib.loads", side_effect=MemoryError("oom")
        ):
            with pytest.raises(MemoryError):
                read_finder_tags(img)


# --- mtime preservation ---


def test_write_rating_does_not_change_mtime_when_caller_restores(tmp_path):
    """Verify that the os.utime restore pattern used in main.py preserves mtime."""
    img = tmp_path / "photo.jpg"
    img.touch()
    # Set a known mtime in the past so any write is detectable
    past_ns = img.stat().st_mtime_ns - 10_000_000_000  # 10 seconds ago
    os.utime(img, ns=(img.stat().st_atime_ns, past_ns))

    original_mtime_ns = img.stat().st_mtime_ns

    mock_result = MagicMock(returncode=0)
    with patch("image_classifier.metadata.subprocess.run", return_value=mock_result):
        try:
            write_rating(img, rating=3)
        finally:
            os.utime(img, ns=(img.stat().st_atime_ns, original_mtime_ns))

    assert img.stat().st_mtime_ns == original_mtime_ns


def test_write_score_tag_does_not_change_mtime_when_caller_restores(tmp_path):
    """Verify that the os.utime restore pattern used in main.py preserves mtime."""
    img = tmp_path / "photo.jpg"
    img.touch()
    past_ns = img.stat().st_mtime_ns - 10_000_000_000
    os.utime(img, ns=(img.stat().st_atime_ns, past_ns))

    original_mtime_ns = img.stat().st_mtime_ns

    xattr_fail = MagicMock(returncode=1)
    xattr_ok = MagicMock(returncode=0)
    with patch(
        "image_classifier.metadata.subprocess.run", side_effect=[xattr_fail, xattr_ok]
    ):
        try:
            write_score_tag(img, score=7.5)
        finally:
            os.utime(img, ns=(img.stat().st_atime_ns, original_mtime_ns))

    assert img.stat().st_mtime_ns == original_mtime_ns


# --- platform guard ---


def test_finder_tags_supported_on_macos_with_xattr():
    with (
        patch("image_classifier.metadata.sys.platform", "darwin"),
        patch("image_classifier.metadata.shutil.which", return_value="/usr/bin/xattr"),
    ):
        assert finder_tags_supported() is True


def test_finder_tags_unsupported_on_linux_even_if_an_xattr_binary_exists():
    with (
        patch("image_classifier.metadata.sys.platform", "linux"),
        patch("image_classifier.metadata.shutil.which", return_value="/usr/bin/xattr"),
    ):
        assert finder_tags_supported() is False


def test_finder_tags_unsupported_on_macos_without_xattr():
    with (
        patch("image_classifier.metadata.sys.platform", "darwin"),
        patch("image_classifier.metadata.shutil.which", return_value=None),
    ):
        assert finder_tags_supported() is False


# --- subprocess failure modes ---


def test_write_rating_turns_a_hang_into_metadata_error(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=subprocess.TimeoutExpired("exiftool", 30),
    ):
        with pytest.raises(MetadataError, match="timed out"):
            write_rating(img, rating=3)


def test_write_rating_turns_a_missing_binary_into_metadata_error(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=FileNotFoundError("exiftool"),
    ):
        with pytest.raises(MetadataError, match="could not be run"):
            write_rating(img, rating=3)


def test_read_finder_tags_hang_does_not_look_like_no_tags(tmp_path):
    """A timed-out read must fail loudly, not return [] and let the user's tags be overwritten."""
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        side_effect=subprocess.TimeoutExpired("xattr", 30),
    ):
        with pytest.raises(MetadataError, match="timed out"):
            read_finder_tags(img)


# --- read_finder_tags: unexpected plist contents ---


@pytest.mark.parametrize(
    "payload",
    [{"not": "a list"}, "just a string", 42],
)
def test_read_finder_tags_ignores_non_list_plists(tmp_path, payload):
    img = tmp_path / "photo.jpg"
    img.touch()
    hex_out = plistlib.dumps(payload, fmt=plistlib.FMT_BINARY).hex()
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=hex_out),
    ):
        assert read_finder_tags(img) == []


def test_read_finder_tags_drops_non_string_entries(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    hex_out = plistlib.dumps(["Red\n6", 7, "trip"], fmt=plistlib.FMT_BINARY).hex()
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=hex_out),
    ):
        assert read_finder_tags(img) == ["Red\n6", "trip"]


def test_read_finder_tags_ignores_garbage_hex(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0, stdout="zz not hex"),
    ):
        assert read_finder_tags(img) == []


def test_read_finder_tags_handles_xattr_multiline_hex_output(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    raw = plistlib.dumps(["vacation"], fmt=plistlib.FMT_BINARY).hex()
    wrapped = "\n".join(raw[i : i + 32] for i in range(0, len(raw), 32))
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=wrapped),
    ):
        assert read_finder_tags(img) == ["vacation"]


# --- write_score_tag: pre-read tags ---


def test_write_score_tag_uses_supplied_tags_without_reading_again(tmp_path):
    """Tags read before the exiftool rewrite are used as-is; no second xattr read."""
    img = tmp_path / "photo.jpg"
    img.touch()
    with patch(
        "image_classifier.metadata.subprocess.run",
        return_value=MagicMock(returncode=0),
    ) as mock_run:
        write_score_tag(img, score=7.8, existing_tags=["family", "5.5", "Red\n6"])

    assert mock_run.call_count == 1  # only the write
    assert mock_run.call_args[0][0][1] == "-wx"
    assert _decode_xattr_call(mock_run) == ["family", "Red\n6", "7.8"]


# --- _is_score_tag: ranges and legacy stars ---


def test_is_score_tag_accepts_scores_slightly_outside_the_nominal_range():
    assert _is_score_tag("-0.3") is True
    assert _is_score_tag("10.4") is True


def test_is_score_tag_recognises_legacy_star_tags():
    assert _is_score_tag("★★★☆☆") is True


def test_is_score_tag_leaves_user_tags_alone():
    for tag in ("vacation", "2023", "v1.2.3", "Red\n6", "7.55", "a7.3"):
        assert _is_score_tag(tag) is False, tag
