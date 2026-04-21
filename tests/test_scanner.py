from __future__ import annotations

from image_classifier.scanner import scan_images


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
