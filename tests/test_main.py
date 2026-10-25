from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import image_classifier.processor as processor_module
from image_classifier import __version__
from image_classifier.classifier import Timings
from image_classifier.database import make_connection, upsert
from image_classifier.main import build_parser, main
from image_classifier.processor import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    FailedImage,
    ImageProcessor,
    ScoredImage,
)
from tests.helpers import FakeModel, FakePreprocessor, make_image


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    """Run main() against a temp folder with the model, device, logging and metadata stubbed."""
    folder = tmp_path / "photos"
    folder.mkdir()
    model = FakeModel()
    env = SimpleNamespace(
        folder=folder,
        db=tmp_path / "classify.db",
        model=model,
        load_model=MagicMock(return_value=(model, FakePreprocessor())),
        check_exiftool=MagicMock(),
        logger=MagicMock(spec=logging.Logger),
        ratings={},
        finder_tags=False,
        capsys=capsys,
    )
    monkeypatch.setattr("image_classifier.main.load_model", env.load_model)
    monkeypatch.setattr("image_classifier.main.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("image_classifier.main.check_exiftool", env.check_exiftool)
    monkeypatch.setattr("image_classifier.main.setup_logger", lambda: env.logger)
    monkeypatch.setattr(
        "image_classifier.main.finder_tags_supported", lambda: env.finder_tags
    )
    monkeypatch.setattr(
        processor_module,
        "write_rating",
        lambda p, r: env.ratings.__setitem__(p.name, r),
    )
    monkeypatch.setattr(processor_module, "read_finder_tags", lambda p: [])
    monkeypatch.setattr(processor_module, "write_score_tag", lambda *a, **k: None)

    def run(*flags, target=None):
        return main([str(target or folder), "--db", str(env.db), *flags])

    env.run = run
    env.image = lambda name, level=149: make_image(folder / name, level=level)
    return env


def output(cli) -> str:
    captured = cli.capsys.readouterr()
    return captured.out + captured.err


# --- argument parsing ---


def test_parser_defaults():
    args = build_parser().parse_args(["pics"])
    assert args.folder == Path("pics")
    assert not (args.force or args.recursive or args.dry_run or args.profile)
    assert not args.no_finder_tags
    assert args.workers == DEFAULT_WORKERS
    assert args.batch_size == DEFAULT_BATCH_SIZE


def test_parser_accepts_every_option():
    args = build_parser().parse_args(
        [
            "pics",
            "--force",
            "--recursive",
            "--dry-run",
            "--no-finder-tags",
            "--profile",
            "--workers",
            "3",
            "--batch-size",
            "16",
            "--db",
            "x.db",
        ]
    )
    assert args.force and args.recursive and args.dry_run and args.profile
    assert args.no_finder_tags
    assert (args.workers, args.batch_size, args.db) == (3, 16, Path("x.db"))


@pytest.mark.parametrize(
    "bad", [["--workers", "0"], ["--batch-size", "-2"], ["--workers", "many"]]
)
def test_parser_rejects_non_positive_or_non_numeric_sizes(bad, capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["pics", *bad])
    assert exc_info.value.code == 2


def test_version_flag_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


# --- folder validation ---


def test_main_exits_on_missing_folder(cli, tmp_path):
    assert cli.run(target=tmp_path / "does_not_exist") == 1
    assert "folder not found" in output(cli)


def test_main_exits_on_file_instead_of_folder(cli):
    f = cli.folder / "image.jpg"
    f.touch()
    assert cli.run(target=f) == 1
    assert "expected a folder" in output(cli)


def test_main_refuses_photo_library_bundles(cli, tmp_path):
    bundle = tmp_path / "Pictures.photoslibrary"
    bundle.mkdir()
    assert cli.run(target=bundle) == 1
    assert "photo-library bundle" in output(cli)
    cli.load_model.assert_not_called()


def test_main_reports_an_unreadable_folder(cli, monkeypatch):
    def deny(folder, recursive):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("image_classifier.main.scan_images", deny)
    assert cli.run() == 1
    assert "cannot read folder" in output(cli)


# --- skipping and forcing ---


def test_main_scores_new_images_and_exits_zero(cli):
    cli.image("a.jpg", level=28)
    cli.image("b.jpg", level=234)

    assert cli.run() == 0

    assert cli.ratings == {"a.jpg": 1, "b.jpg": 5}
    out = output(cli)
    assert "Model loaded on" in out
    assert "Distribution" in out


def test_main_checks_for_exiftool_before_doing_anything(cli):
    cli.image("a.jpg")
    cli.run()
    cli.check_exiftool.assert_called_once()


def test_main_skips_already_scored_images(cli):
    img = cli.image("photo.jpg")
    conn = make_connection(cli.db)
    upsert(img, 7.0, 4, conn)
    conn.close()
    cli.image("new.jpg")

    cli.run()

    assert cli.ratings == {"new.jpg": 3}


def test_main_with_everything_scored_never_loads_the_model(cli):
    img = cli.image("photo.jpg")
    conn = make_connection(cli.db)
    upsert(img, 7.0, 4, conn)
    conn.close()

    assert cli.run() == 0

    cli.load_model.assert_not_called()
    assert "Nothing to do" in output(cli)


def test_main_with_an_empty_folder_exits_cleanly_without_the_model(cli):
    assert cli.run() == 0
    cli.load_model.assert_not_called()


def test_main_force_rescores_already_processed_images(cli):
    img = cli.image("photo.jpg")
    conn = make_connection(cli.db)
    upsert(img, 1.0, 1, conn)
    conn.close()

    cli.run("--force")

    assert cli.ratings == {"photo.jpg": 3}
    conn = make_connection(cli.db)
    try:
        assert conn.execute("SELECT rating FROM images").fetchone()[0] == 3
    finally:
        conn.close()


def test_main_is_not_recursive_unless_asked(cli):
    sub = cli.folder / "sub"
    sub.mkdir()
    make_image(sub / "nested.jpg")
    cli.image("top.jpg")

    cli.run()
    assert set(cli.ratings) == {"top.jpg"}

    cli.run("--recursive")
    assert set(cli.ratings) == {"top.jpg", "nested.jpg"}


# --- model loading ---


def test_main_reports_a_model_load_failure_helpfully(cli):
    cli.image("photo.jpg")
    cli.load_model.side_effect = OSError("Connection refused")

    assert cli.run() == 1

    out = output(cli)
    assert "could not load the aesthetic model" in out
    assert "Connection refused" in out
    assert "internet connection" in out
    assert cli.ratings == {}


# --- errors ---


def test_main_counts_and_logs_errors_without_crashing(cli):
    cli.image("good.jpg")
    (cli.folder / "bad.jpg").write_bytes(b"not a jpeg")

    assert cli.run() == 0

    assert cli.ratings == {"good.jpg": 3}
    cli.logger.error.assert_called_once()
    out = output(cli)
    assert "Recent Failures" in out
    assert "bad.jpg" in out


def test_main_records_failures_so_they_are_retried_next_time(cli):
    (cli.folder / "bad.jpg").write_bytes(b"not a jpeg")
    cli.run()
    conn = sqlite3.connect(cli.db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 1
    finally:
        conn.close()

    cli.load_model.reset_mock()
    cli.run()
    cli.load_model.assert_called_once()  # a failed image is not "done"


def test_main_caps_the_failure_list(cli):
    for i in range(14):
        (cli.folder / f"bad{i:02d}.jpg").write_bytes(b"x")

    cli.run()

    out = output(cli)
    assert "and 4 more" in out


# --- interruption ---


def test_main_handles_keyboard_interrupt_with_a_summary_and_exit_130(cli, monkeypatch):
    cli.image("a.jpg")

    def interrupted(self, paths):
        yield ScoredImage(paths[0], 6.0, 3, Timings())
        raise KeyboardInterrupt

    monkeypatch.setattr(ImageProcessor, "run", interrupted)

    assert cli.run() == 130

    out = output(cli)
    assert "Interrupt received" in out
    assert "Summary" in out


def test_main_ctrl_c_while_loading_the_model_exits_cleanly(cli, monkeypatch):
    """The model download is the longest wait; Ctrl+C there must not dump a traceback."""
    cli.image("a.jpg")
    cli.load_model.side_effect = KeyboardInterrupt
    connections = []
    real = make_connection

    def tracking(path):
        conn = real(path)
        connections.append(conn)
        return conn

    monkeypatch.setattr("image_classifier.main.make_connection", tracking)

    assert cli.run() == 130

    assert "Interrupted" in output(cli)
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")  # connection was still closed


# --- dry run ---


def test_main_dry_run_writes_nothing_and_does_not_need_exiftool(cli):
    cli.image("a.jpg")

    assert cli.run("--dry-run") == 0

    cli.check_exiftool.assert_not_called()
    assert cli.ratings == {}
    out = output(cli)
    assert "Dry run" in out
    assert "Would score" in out
    conn = sqlite3.connect(cli.db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
    finally:
        conn.close()


# --- processor wiring ---


def test_main_passes_tuning_options_to_the_processor(cli, monkeypatch):
    cli.image("a.jpg")
    seen = {}
    real_init = ImageProcessor.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ImageProcessor, "__init__", spy)

    cli.run("--workers", "3", "--batch-size", "5", "--no-finder-tags")

    assert seen == dict(workers=3, batch_size=5, write_finder_tags=False, dry_run=False)


def test_main_writes_finder_tags_when_the_platform_supports_them(cli, monkeypatch):
    cli.image("a.jpg")
    cli.finder_tags = True
    seen = {}
    real_init = ImageProcessor.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(ImageProcessor, "__init__", spy)
    cli.run()
    assert seen["write_finder_tags"] is True


def test_main_explains_when_finder_tags_are_unavailable(cli):
    cli.image("a.jpg")
    cli.finder_tags = False
    cli.run()
    assert "Finder tags need macOS" in output(cli)


def test_main_does_not_mention_finder_tags_when_the_user_opted_out(cli):
    cli.image("a.jpg")
    cli.finder_tags = True
    cli.run("--no-finder-tags")
    assert "Finder tags" not in output(cli)


# --- profile ---


def test_main_profile_flag_prints_the_timing_table(cli):
    cli.image("a.jpg")
    cli.run("--profile")
    out = output(cli)
    assert "Timing Profile (1 images)" in out
    assert "images/s" in out


def test_main_without_profile_flag_prints_no_timing_table(cli):
    cli.image("a.jpg")
    cli.run()
    assert "Timing Profile" not in output(cli)


# --- resource handling ---


def test_main_closes_the_database_connection(cli, monkeypatch):
    cli.image("a.jpg")
    connections = []
    real = make_connection

    def tracking(path):
        conn = real(path)
        connections.append(conn)
        return conn

    monkeypatch.setattr("image_classifier.main.make_connection", tracking)
    cli.run()

    assert connections
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")  # closed


def test_main_closes_the_connection_when_the_model_fails_to_load(cli, monkeypatch):
    cli.image("a.jpg")
    cli.load_model.side_effect = RuntimeError("no model")
    connections = []
    real = make_connection

    def tracking(path):
        conn = real(path)
        connections.append(conn)
        return conn

    monkeypatch.setattr("image_classifier.main.make_connection", tracking)
    cli.run()

    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_failed_image_outcomes_from_the_processor_are_counted(cli, monkeypatch):
    cli.image("a.jpg")

    def always_fails(self, paths):
        for p in paths:
            yield FailedImage(p, RuntimeError("kaput"))

    monkeypatch.setattr(ImageProcessor, "run", always_fails)

    assert cli.run() == 0
    assert "kaput" in output(cli)
