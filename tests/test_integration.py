"""End-to-end tests against the real exiftool (and xattr on macOS).

Only the HuggingFace model is replaced, with a stub that scores by brightness.
These are skipped when exiftool is not installed.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from image_classifier.classifier import SUPPORTED_EXTENSIONS
from image_classifier.main import main
from image_classifier.metadata import (
    MetadataError,
    finder_tags_supported,
    read_finder_tags,
    write_rating,
    write_score_tag,
)
from tests.helpers import FakeModel, FakePreprocessor, make_image, requires_exiftool

pytestmark = requires_exiftool

# Brightness chosen to land mid-band: score = level / 255 * 9 + 1.
LIBRARY = {
    "one.jpg": (28, 1),  # ~2.0
    "two.png": (106, 2),  # ~4.7
    "three.webp": (149, 3),  # ~6.2
    "four.tiff": (191, 4),  # ~7.7
    "five.heic": (234, 5),  # ~9.2
}
PAST_NS = 1_600_000_000 * 10**9


def read_rating(path: Path) -> str:
    out = subprocess.run(
        ["exiftool", "-s3", "-XMP:Rating", str(path)], capture_output=True, text=True
    )
    return out.stdout.strip()


def build_library(root: Path) -> Path:
    folder = root / "photos"
    folder.mkdir()
    for name, (level, _) in LIBRARY.items():
        make_image(folder / name, level=level)
        os.utime(folder / name, ns=(PAST_NS, PAST_NS))
    (folder / "corrupt.jpg").write_bytes(b"definitely not a jpeg")
    make_image(folder / "ignored.bmp")  # exiftool cannot write BMP
    make_image(folder / ".hidden.jpg")
    bundle = folder / "Vault.photoslibrary"
    bundle.mkdir()
    make_image(bundle / "inside.jpg")
    return folder


def run_cli(folder: Path, db: Path, *flags: str) -> SimpleNamespace:
    """Run the real CLI with a stub model; returns exit code, model and whether it loaded."""
    model = FakeModel()
    with (
        patch("image_classifier.main.setup_logger", return_value=None),
        patch("image_classifier.main.get_device", return_value=torch.device("cpu")),
        patch(
            "image_classifier.main.load_model",
            return_value=(model, FakePreprocessor()),
        ) as load,
    ):
        code = main([str(folder), "--db", str(db), *flags])
    return SimpleNamespace(code=code, model=model, model_loaded=load.called)


def db_rows(db: Path) -> dict[str, int]:
    with closing(sqlite3.connect(db)) as conn:
        return {
            Path(p).name: r for p, r in conn.execute("SELECT path, rating FROM images")
        }


# --- exiftool really can write every supported format ---


@pytest.mark.parametrize("ext", sorted(SUPPORTED_EXTENSIONS))
def test_exiftool_writes_a_rating_to_every_supported_format(tmp_path, ext):
    path = make_image(tmp_path / f"photo{ext}")
    write_rating(path, 4)
    assert read_rating(path) == "4"


def test_write_rating_reports_exiftool_errors_for_unwritable_files(tmp_path):
    path = tmp_path / "fake.jpg"
    path.write_bytes(b"not a jpeg")
    with pytest.raises(MetadataError, match="Not a valid JPG"):
        write_rating(path, 3)


# --- the whole CLI ---


def test_cli_scores_rates_and_records_a_mixed_library(tmp_path):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"

    result = run_cli(folder, db, "--no-finder-tags")

    assert result.code == 0
    for name, (_, expected) in LIBRARY.items():
        assert read_rating(folder / name) == str(expected), name
    assert db_rows(db) == {name: stars for name, (_, stars) in LIBRARY.items()}
    # Only the five real images reached the model: no BMP, hidden file, bundle or corrupt file.
    assert sum(result.model.batch_sizes) == len(LIBRARY)


def test_cli_preserves_modification_times_and_leaves_no_backups(tmp_path):
    folder = build_library(tmp_path)

    run_cli(folder, tmp_path / "t.db", "--no-finder-tags")

    for name in LIBRARY:
        assert (folder / name).stat().st_mtime_ns == PAST_NS, name
    assert not list(folder.glob("*_original"))


def test_cli_records_corrupt_files_as_failures_without_aborting(tmp_path):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"

    result = run_cli(folder, db, "--no-finder-tags")

    assert result.code == 0
    with closing(sqlite3.connect(db)) as conn:
        failures = [Path(p).name for (p,) in conn.execute("SELECT path FROM failures")]
    assert failures == ["corrupt.jpg"]
    assert "corrupt.jpg" not in db_rows(db)


def test_cli_never_touches_photo_library_bundles_or_unsupported_files(tmp_path):
    folder = build_library(tmp_path)

    run_cli(folder, tmp_path / "t.db", "--recursive", "--no-finder-tags")

    assert read_rating(folder / "Vault.photoslibrary" / "inside.jpg") == ""
    assert read_rating(folder / ".hidden.jpg") == ""


def test_cli_refuses_to_run_on_a_photo_library_bundle_itself(tmp_path):
    folder = build_library(tmp_path)
    result = run_cli(folder / "Vault.photoslibrary", tmp_path / "t.db")
    assert result.code == 1
    assert result.model_loaded is False
    assert read_rating(folder / "Vault.photoslibrary" / "inside.jpg") == ""


def test_cli_second_run_skips_everything_and_never_loads_the_model(tmp_path):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"
    run_cli(folder, db, "--no-finder-tags")
    # Remove the one permanently-failing file so nothing is left to retry.
    (folder / "corrupt.jpg").unlink()

    result = run_cli(folder, db, "--no-finder-tags")

    assert result.code == 0
    assert result.model_loaded is False
    assert result.model.batch_sizes == []


def test_cli_force_rescores_everything(tmp_path):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"
    run_cli(folder, db, "--no-finder-tags")

    result = run_cli(folder, db, "--no-finder-tags", "--force")

    assert sum(result.model.batch_sizes) == len(
        LIBRARY
    )  # corrupt never reaches the model


def test_cli_dry_run_changes_nothing(tmp_path):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"

    result = run_cli(folder, db, "--dry-run")

    assert result.code == 0
    assert sum(result.model.batch_sizes) == len(LIBRARY)  # it did score them...
    for name in LIBRARY:
        assert read_rating(folder / name) == "", name  # ...but wrote nothing
    assert db_rows(db) == {}


@pytest.mark.parametrize("workers,batch", [(1, 1), (2, 3), (8, 16)])
def test_cli_gives_identical_results_for_any_worker_and_batch_settings(
    tmp_path, workers, batch
):
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"

    run_cli(
        folder,
        db,
        "--no-finder-tags",
        "--workers",
        str(workers),
        "--batch-size",
        str(batch),
    )

    assert db_rows(db) == {name: stars for name, (_, stars) in LIBRARY.items()}


# --- Finder tags: only where the platform supports them ---


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Finder tags are supported here; see below"
)
def test_cli_works_on_platforms_without_finder_tags(tmp_path):
    """Regression: on Linux every image used to fail because `xattr` does not exist."""
    assert finder_tags_supported() is False
    folder = build_library(tmp_path)
    db = tmp_path / "t.db"

    result = run_cli(folder, db)  # note: no --no-finder-tags

    assert result.code == 0
    assert db_rows(db) == {name: stars for name, (_, stars) in LIBRARY.items()}


@pytest.mark.skipif(sys.platform != "darwin", reason="Finder tags need macOS")
def test_cli_writes_finder_tags_and_keeps_the_users_own(tmp_path):
    folder = build_library(tmp_path)
    target = folder / "one.jpg"
    # The user tagged this photo, and an older run left a score tag behind.
    write_score_tag(target, 5.5, existing_tags=["Vacation"])
    assert read_finder_tags(target) == ["Vacation", "5.5"]

    run_cli(folder, tmp_path / "t.db")

    tags = read_finder_tags(target)
    assert tags[0] == "Vacation"
    assert "5.5" not in tags
    assert len(tags) == 2  # Vacation + exactly one fresh score tag
