from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from image_classifier.scanner import is_protected_bundle, scan_images


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


def test_scan_images_works_inside_hidden_parent_folder(tmp_path):
    folder = tmp_path / ".photos"
    folder.mkdir()
    (folder / "a.jpg").touch()
    (folder / ".hidden.jpg").touch()
    results = scan_images(folder, recursive=True)
    assert {p.name for p in results} == {"a.jpg"}


def test_scan_images_returns_a_sorted_list(tmp_path):
    for name in ["c.jpg", "a.jpg", "b.png"]:
        (tmp_path / name).touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "0.jpg").touch()
    results = scan_images(tmp_path, recursive=True)
    assert results == sorted(results)
    assert [p.name for p in results] == ["a.jpg", "b.png", "c.jpg", "0.jpg"]


def test_scan_images_skips_bmp_which_exiftool_cannot_write(tmp_path):
    (tmp_path / "a.bmp").touch()
    (tmp_path / "b.jpg").touch()
    assert [p.name for p in scan_images(tmp_path, recursive=False)] == ["b.jpg"]


def test_scan_images_returns_absolute_paths_under_the_folder(tmp_path):
    (tmp_path / "a.jpg").touch()
    (result,) = scan_images(tmp_path, recursive=False)
    assert result == tmp_path / "a.jpg"


def test_scan_images_never_descends_into_hidden_folders(tmp_path):
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "x.jpg").touch()
    (tmp_path / "ok.jpg").touch()
    visited = []
    real_scandir = os.scandir

    def spy(path):
        visited.append(Path(path).name)
        return real_scandir(path)

    with patch("image_classifier.scanner.os.scandir", spy):
        results = scan_images(tmp_path, recursive=True)

    assert [p.name for p in results] == ["ok.jpg"]
    assert ".git" not in visited  # pruned, not walked and filtered afterwards


@pytest.mark.parametrize(
    "bundle",
    [
        "Photos Library.photoslibrary",
        "Old.PHOTOSLIBRARY",
        "x.aplibrary",
        "y.photolibrary",
    ],
)
def test_scan_images_skips_photo_library_bundles(tmp_path, bundle):
    inside = tmp_path / bundle / "originals"
    inside.mkdir(parents=True)
    (inside / "secret.jpg").touch()
    (tmp_path / "ok.jpg").touch()
    assert [p.name for p in scan_images(tmp_path, recursive=True)] == ["ok.jpg"]


def test_is_protected_bundle():
    assert is_protected_bundle(Path("/Users/me/Pictures/Photos Library.photoslibrary"))
    assert is_protected_bundle(Path("Library.PhotosLibrary"))
    assert not is_protected_bundle(Path("/Users/me/Pictures/photoslibrary-export"))
    assert not is_protected_bundle(Path("/Users/me/Pictures"))


def test_scan_images_ignores_directories_named_like_images(tmp_path):
    (tmp_path / "album.jpg").mkdir()
    (tmp_path / "real.jpg").touch()
    assert [p.name for p in scan_images(tmp_path, recursive=True)] == ["real.jpg"]


def test_scan_images_includes_symlinked_files_and_ignores_broken_links(tmp_path):
    target = tmp_path / "real.jpg"
    target.touch()
    (tmp_path / "link.jpg").symlink_to(target)
    (tmp_path / "dangling.jpg").symlink_to(tmp_path / "missing.jpg")
    assert [p.name for p in scan_images(tmp_path, recursive=False)] == [
        "link.jpg",
        "real.jpg",
    ]


def test_scan_images_does_not_follow_directory_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.jpg").touch()
    root = tmp_path / "root"
    root.mkdir()
    (root / "loop").symlink_to(outside, target_is_directory=True)
    assert scan_images(root, recursive=True) == []


def test_scan_images_skips_unreadable_subfolders(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "x.jpg").touch()
    (tmp_path / "ok.jpg").touch()
    real_scandir = os.scandir

    def deny_locked(path):
        if Path(path).name == "locked":
            raise PermissionError(13, "denied")
        return real_scandir(path)

    with patch("image_classifier.scanner.os.scandir", deny_locked):
        results = scan_images(tmp_path, recursive=True)

    assert [p.name for p in results] == ["ok.jpg"]


def test_scan_images_raises_for_an_unreadable_root(tmp_path):
    with patch("image_classifier.scanner.os.scandir", side_effect=PermissionError):
        with pytest.raises(PermissionError):
            scan_images(tmp_path, recursive=False)
