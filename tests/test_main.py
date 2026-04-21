from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from image_classifier.classifier import Timings
from image_classifier.main import print_summary, star_display

_TIMINGS = Timings(load_ms=10.0, preprocess_ms=20.0, infer_ms=500.0)


# --- star_display ---


def test_star_display_one():
    assert star_display(1) == "★☆☆☆☆"


def test_star_display_five():
    assert star_display(5) == "★★★★★"


def test_star_display_three():
    assert star_display(3) == "★★★☆☆"


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

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_logger", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=True),
        patch("image_classifier.processor.ImageProcessor.process_image") as mock_process,
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    mock_process.assert_not_called()


def test_main_force_flag_rescores_all_images(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path), "--force"]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_logger", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=True),
        patch(
            "image_classifier.processor.ImageProcessor.process_image", return_value=(7.0, _TIMINGS)
        ) as mock_process,
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()

    mock_process.assert_called_once()


def test_main_counts_errors_without_crashing(tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_logger", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch(
            "image_classifier.processor.ImageProcessor.process_image",
            side_effect=RuntimeError("bad image"),
        ),
        patch("image_classifier.processor.ImageProcessor.handle_failure") as mock_handle_failure,
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main

        main()  # Should complete without raising

    mock_handle_failure.assert_called_once()


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
        patch("image_classifier.main.setup_logger", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch(
            "image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch("image_classifier.processor.ImageProcessor.process_image", return_value=(7.0, _TIMINGS)),
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
