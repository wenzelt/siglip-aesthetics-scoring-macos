from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.classifier import Timings
from image_classifier.main import print_summary, scan_images, star_display

_TIMINGS = Timings(load_ms=10.0, preprocess_ms=20.0, infer_ms=500.0)


# --- star_display ---


def test_star_display_one():
    assert star_display(1) == "★☆☆☆☆"


def test_star_display_five():
    assert star_display(5) == "★★★★★"


def test_star_display_three():
    assert star_display(3) == "★★★☆☆"


# --- scan_images ---


def test_scan_images_returns_supported_extensions(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.mp4").touch()
    (tmp_path / "d.txt").touch()
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert names == {"a.jpg", "b.png"}


def test_scan_images_skips_hidden_files(tmp_path):
    (tmp_path / ".hidden.jpg").touch()
    (tmp_path / "visible.jpg").touch()
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert ".hidden.jpg" not in names
    assert "visible.jpg" in names


def test_scan_images_flat_by_default(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.jpg").touch()
    (sub / "nested.jpg").touch()
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert "top.jpg" in names
    assert "nested.jpg" not in names


def test_scan_images_recursive_flag(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.jpg").touch()
    (sub / "nested.jpg").touch()
    results = scan_images(tmp_path, recursive=True)
    names = {p.name for p in results}
    assert "top.jpg" in names
    assert "nested.jpg" in names


def test_scan_images_case_insensitive_extensions(tmp_path):
    (tmp_path / "photo.JPG").touch()
    (tmp_path / "photo.PNG").touch()
    results = scan_images(tmp_path, recursive=False)
    assert len(results) == 2


# --- CLI integration tests ---


def _make_mock_conn():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    conn.execute.return_value.fetchall.return_value = []
    return conn


def test_main_exits_on_missing_folder(tmp_path):
    missing = tmp_path / "does_not_exist"
    with patch("sys.argv", ["classify", str(missing)]):
        with pytest.raises(SystemExit) as exc_info:
            from image_classifier.main import main

            main()
    assert exc_info.value.code == 1


def test_main_exits_on_file_instead_of_folder(tmp_path):
    f = tmp_path / "image.jpg"
    f.touch()
    with patch("sys.argv", ["classify", str(f)]):
        with pytest.raises(SystemExit) as exc_info:
            from image_classifier.main import main

            main()
    assert exc_info.value.code == 1


def test_main_skips_already_processed_images(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()
    mock_conn.execute.return_value.fetchone.return_value = {"path": str(img)}

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=True),
        patch("image_classifier.main.score_image") as mock_score,
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    mock_score.assert_not_called()


def test_main_force_flag_rescores_all_images(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path), "--force"]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=True),
        patch(
            "image_classifier.main.score_image", return_value=(7.0, _TIMINGS)
        ) as mock_score,
        patch("image_classifier.main.score_to_rating", return_value=4),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.write_score_tag"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    mock_score.assert_called_once()


def test_main_counts_errors_without_crashing(tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    from image_classifier.classifier import ClassifierError

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch(
            "image_classifier.main.score_image",
            side_effect=ClassifierError("bad image"),
        ),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.upsert_failure"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()  # Should complete without raising


def test_main_survives_unexpected_exception(tmp_path):
    """Unexpected exceptions (ValueError, RuntimeError, etc.) must not kill the run."""
    img = tmp_path / "broken.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch(
            "image_classifier.main.score_image",
            side_effect=ValueError("mean must have 1 elements"),
        ),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.upsert_failure") as mock_upsert_failure,
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()  # Must not raise

    mock_upsert_failure.assert_called_once()


def test_upsert_called_even_when_metadata_write_fails(tmp_path):
    """Score must be saved to DB even if exiftool/xattr fails."""
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    from image_classifier.metadata import MetadataError

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch("image_classifier.main.score_image", return_value=(7.0, _TIMINGS)),
        patch("image_classifier.main.score_to_rating", return_value=4),
        patch(
            "image_classifier.main.write_rating",
            side_effect=MetadataError("exiftool failed"),
        ),
        patch("image_classifier.main.write_score_tag"),
        patch("image_classifier.main.upsert") as mock_upsert,
        patch("image_classifier.main.upsert_failure"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    mock_upsert.assert_called_once()


def test_main_records_failure_in_db(tmp_path):
    """Failed images are stored in the failures table for later review."""
    img = tmp_path / "broken.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch(
            "image_classifier.main.score_image", side_effect=RuntimeError("CUDA error")
        ),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.upsert_failure") as mock_upsert_failure,
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    args = mock_upsert_failure.call_args
    assert args[0][0] == img  # first positional arg is the path
    assert "RuntimeError" in args[0][1]  # error string contains exception type


# --- print_summary ---


def test_print_summary_handles_out_of_range_rating(tmp_path):
    """print_summary must not raise KeyError when DB contains an out-of-range rating."""
    mock_conn = MagicMock()
    out_of_range_row = {"rating": 0, "path": str(tmp_path / "x.jpg"), "score": 1.0}
    with patch("image_classifier.main.all_scores", return_value=[out_of_range_row]):
        print_summary(0, 0, 0, tmp_path, mock_conn)  # must not raise


# --- --profile flag ---


def test_main_profile_flag_collects_timings(tmp_path):
    """With --profile, the run completes and timing data is gathered without error."""
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path), "--profile"]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch("image_classifier.main.score_image", return_value=(7.0, _TIMINGS)),
        patch("image_classifier.main.score_to_rating", return_value=4),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.write_score_tag"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.upsert_failure"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()  # must not raise


def test_timings_total_includes_all_phases():
    t = Timings(
        load_ms=10,
        preprocess_ms=20,
        infer_ms=500,
        upsert_ms=2,
        exiftool_ms=50,
        xattr_ms=5,
    )
    assert abs(t.total_ms - 587.0) < 0.01
