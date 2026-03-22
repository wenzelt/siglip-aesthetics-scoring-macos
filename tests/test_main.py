from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.main import scan_images, star_display


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
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
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
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=True),
        patch("image_classifier.main.score_image", return_value=7.0) as mock_score,
        patch("image_classifier.main.score_to_rating", return_value=4),
        patch("image_classifier.main.write_rating"),
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
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.is_processed", return_value=False),
        patch("image_classifier.main.score_image", side_effect=ClassifierError("bad image")),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier.main import main
        main()  # Should complete without raising
