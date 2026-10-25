from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import image_classifier.processor as processor_module
from image_classifier.database import make_connection
from image_classifier.metadata import MetadataError
from image_classifier.processor import (
    FailedImage,
    ImageProcessor,
    ScoredImage,
)
from tests.helpers import FakeModel, make_image

CPU = torch.device("cpu")

# Brightness levels that land mid-band (score = level / 255 * 9 + 1).
DARK, MID, BRIGHT = 28, 149, 234  # ratings 1, 3, 5


@pytest.fixture
def writes(monkeypatch):
    """Replace the metadata writers with recorders; returns what they saw."""
    log = SimpleNamespace(
        order=[], ratings={}, tags={}, threads=set(), fail_for=set(), delay=0.0
    )

    def read_finder_tags(path):
        log.order.append(("read", path.name))
        return ["mine"]

    def write_rating(path, rating):
        log.threads.add(threading.current_thread().name)
        if log.delay:
            time.sleep(log.delay)
        if path.name in log.fail_for:
            raise MetadataError(f"exiftool failed for {path.name}")
        log.order.append(("rating", path.name))
        log.ratings[path.name] = rating

    def write_score_tag(path, score, existing_tags=None):
        log.order.append(("tag", path.name))
        log.tags[path.name] = (score, existing_tags)

    monkeypatch.setattr(processor_module, "read_finder_tags", read_finder_tags)
    monkeypatch.setattr(processor_module, "write_rating", write_rating)
    monkeypatch.setattr(processor_module, "write_score_tag", write_score_tag)
    return log


@pytest.fixture
def make_processor(conn, fake_model, fake_preprocessor):
    def build(**overrides):
        kwargs = dict(
            model=fake_model,
            preprocessor=fake_preprocessor,
            device=CPU,
            conn=conn,
            db_path=Path(":memory:"),
            workers=2,
            batch_size=4,
            write_finder_tags=False,
        )
        kwargs.update(overrides)
        model = kwargs.pop("model")
        preprocessor = kwargs.pop("preprocessor")
        device = kwargs.pop("device")
        conn_ = kwargs.pop("conn")
        db_path = kwargs.pop("db_path")
        return ImageProcessor(model, preprocessor, device, conn_, db_path, **kwargs)

    return build


def library(tmp_path, levels):
    paths = []
    for i, level in enumerate(levels):
        paths.append(make_image(tmp_path / f"img{i:02d}.jpg", level=level))
    return paths


def db_ratings(conn):
    return {
        Path(r["path"]).name: r["rating"] for r in conn.execute("SELECT * FROM images")
    }


def db_failures(conn):
    return {
        Path(r["path"]).name: r["error"] for r in conn.execute("SELECT * FROM failures")
    }


# --- happy path ---


def test_run_scores_writes_metadata_and_records(tmp_path, conn, writes, make_processor):
    paths = library(tmp_path, [DARK, MID, BRIGHT])

    outcomes = list(make_processor().run(paths))

    assert [type(o) for o in outcomes] == [ScoredImage] * 3
    assert [o.rating for o in outcomes] == [1, 3, 5]
    assert outcomes[2].score == pytest.approx(9.2, abs=0.1)
    assert writes.ratings == {"img00.jpg": 1, "img01.jpg": 3, "img02.jpg": 5}
    assert db_ratings(conn) == writes.ratings


def test_run_yields_outcomes_in_input_order(tmp_path, writes, make_processor):
    paths = library(tmp_path, [DARK, MID, BRIGHT, MID, DARK, BRIGHT, MID])

    outcomes = list(make_processor(batch_size=3, workers=4).run(paths))

    assert [o.path for o in outcomes] == paths


def test_run_scores_images_in_batches(tmp_path, writes, fake_model, make_processor):
    paths = library(tmp_path, [MID] * 7)

    list(make_processor(batch_size=3).run(paths))

    assert fake_model.batch_sizes == [3, 3, 1]


def test_run_with_nothing_to_do_does_not_touch_the_model(
    writes, fake_model, make_processor
):
    assert list(make_processor().run([])) == []
    assert fake_model.batch_sizes == []


def test_run_populates_every_timing_phase(tmp_path, writes, make_processor):
    writes.delay = 0.01
    paths = library(tmp_path, [MID, MID])

    outcomes = list(make_processor(write_finder_tags=True).run(paths))

    for outcome in outcomes:
        t = outcome.timings
        assert t.load_ms > 0
        assert t.preprocess_ms > 0
        assert t.infer_ms > 0
        assert t.exiftool_ms >= 9  # slept 10 ms; allow for timer granularity
        assert t.upsert_ms >= 0
        assert t.total_ms == pytest.approx(
            t.load_ms
            + t.preprocess_ms
            + t.infer_ms
            + t.upsert_ms
            + t.exiftool_ms
            + t.xattr_ms
        )


# --- failures stay local ---


def test_unreadable_image_fails_alone(tmp_path, conn, writes, make_processor):
    paths = library(tmp_path, [MID, MID, MID])
    paths[1].write_bytes(b"corrupt")

    outcomes = list(make_processor().run(paths))

    assert [type(o) for o in outcomes] == [ScoredImage, FailedImage, ScoredImage]
    assert outcomes[1].path == paths[1]
    assert outcomes[1].message.startswith("ClassifierError")
    assert set(db_ratings(conn)) == {"img00.jpg", "img02.jpg"}
    assert set(db_failures(conn)) == {"img01.jpg"}
    assert "ClassifierError" in db_failures(conn)["img01.jpg"]


def test_missing_image_fails_alone(tmp_path, writes, make_processor):
    paths = library(tmp_path, [MID])
    ghost = tmp_path / "ghost.jpg"

    outcomes = list(make_processor().run([ghost, *paths]))

    assert isinstance(outcomes[0], FailedImage)
    assert isinstance(outcomes[1], ScoredImage)


def test_failed_metadata_write_leaves_no_processed_row(
    tmp_path, conn, writes, make_processor
):
    """The key invariant: an image is only recorded once its metadata was written,
    so a failure can never cause the next run to skip an image that was not rated."""
    paths = library(tmp_path, [MID, MID, MID])
    writes.fail_for = {"img01.jpg"}

    outcomes = list(make_processor().run(paths))

    assert [type(o) for o in outcomes] == [ScoredImage, FailedImage, ScoredImage]
    assert isinstance(outcomes[1].error, MetadataError)
    assert set(db_ratings(conn)) == {"img00.jpg", "img02.jpg"}
    assert "MetadataError" in db_failures(conn)["img01.jpg"]


def test_a_later_success_clears_the_earlier_failure(
    tmp_path, conn, writes, make_processor
):
    paths = library(tmp_path, [MID])
    writes.fail_for = {"img00.jpg"}
    list(make_processor().run(paths))
    assert set(db_failures(conn)) == {"img00.jpg"}

    writes.fail_for = set()
    list(make_processor().run(paths))

    assert db_failures(conn) == {}
    assert set(db_ratings(conn)) == {"img00.jpg"}


def test_batch_failure_falls_back_to_scoring_one_by_one(
    tmp_path, writes, fake_preprocessor, make_processor
):
    class OneAtATime(FakeModel):
        def __call__(self, pixel_values):
            if pixel_values.shape[0] > 1:
                raise RuntimeError("out of memory")
            return super().__call__(pixel_values)

    model = OneAtATime()
    paths = library(tmp_path, [DARK, MID, BRIGHT])

    outcomes = list(make_processor(model=model, batch_size=3).run(paths))

    assert [o.rating for o in outcomes] == [1, 3, 5]
    assert model.batch_sizes == [1, 1, 1]


def test_one_bad_tensor_does_not_sink_its_batch(tmp_path, writes, make_processor):
    class RejectsWhite(FakeModel):
        def __call__(self, pixel_values):
            if float(pixel_values.float().max()) > 0.99:
                raise RuntimeError("cannot score a white image")
            return super().__call__(pixel_values)

    paths = library(tmp_path, [MID, 255, DARK])

    outcomes = list(make_processor(model=RejectsWhite(), batch_size=3).run(paths))

    assert [type(o) for o in outcomes] == [ScoredImage, FailedImage, ScoredImage]
    assert "white image" in outcomes[1].message


def test_non_finite_score_fails_that_image_only(tmp_path, conn, writes, make_processor):
    class NanForWhite(FakeModel):
        def __call__(self, pixel_values):
            out = super().__call__(pixel_values)
            bad = pixel_values.float().flatten(1).max(dim=1).values > 0.99
            out.logits[bad] = float("nan")
            return out

    paths = library(tmp_path, [MID, 255, MID])

    outcomes = list(make_processor(model=NanForWhite(), batch_size=3).run(paths))

    assert [type(o) for o in outcomes] == [ScoredImage, FailedImage, ScoredImage]
    assert "non-finite" in outcomes[1].message
    assert "img01.jpg" not in writes.ratings  # NaN never became a 5-star rating
    assert set(db_ratings(conn)) == {"img00.jpg", "img02.jpg"}


# --- mtime ---


def test_modification_time_is_restored(tmp_path, writes, make_processor, monkeypatch):
    path = library(tmp_path, [MID])[0]
    past = 1_600_000_000 * 10**9
    os.utime(path, ns=(past, past))

    def rewriting_write_rating(p, rating):  # a real exiftool run bumps the mtime
        os.utime(p, None)

    monkeypatch.setattr(processor_module, "write_rating", rewriting_write_rating)
    list(make_processor().run([path]))

    assert path.stat().st_mtime_ns == past


def test_modification_time_is_restored_even_when_the_write_fails(
    tmp_path, writes, make_processor, monkeypatch
):
    path = library(tmp_path, [MID])[0]
    past = 1_600_000_000 * 10**9
    os.utime(path, ns=(past, past))

    def half_failing_write_rating(p, rating):
        os.utime(p, None)
        raise MetadataError("exiftool died after touching the file")

    monkeypatch.setattr(processor_module, "write_rating", half_failing_write_rating)
    outcomes = list(make_processor().run([path]))

    assert isinstance(outcomes[0], FailedImage)
    assert path.stat().st_mtime_ns == past


# --- Finder tags ---


def test_finder_tags_are_read_before_exiftool_rewrites_the_file(
    tmp_path, writes, make_processor
):
    path = library(tmp_path, [MID])[0]

    list(make_processor(write_finder_tags=True).run([path]))

    assert writes.order == [
        ("read", "img00.jpg"),
        ("rating", "img00.jpg"),
        ("tag", "img00.jpg"),
    ]
    # ... and the pre-read tags are what gets written back.
    assert writes.tags["img00.jpg"][1] == ["mine"]


def test_finder_tags_are_skipped_when_disabled(tmp_path, writes, make_processor):
    path = library(tmp_path, [MID])[0]

    list(make_processor(write_finder_tags=False).run([path]))

    assert writes.order == [("rating", "img00.jpg")]


# --- dry run ---


def test_dry_run_scores_but_writes_nothing(tmp_path, conn, writes, make_processor):
    paths = library(tmp_path, [DARK, BRIGHT])

    outcomes = list(make_processor(dry_run=True, write_finder_tags=True).run(paths))

    assert [o.rating for o in outcomes] == [1, 5]
    assert writes.order == []
    assert db_ratings(conn) == {}
    assert db_failures(conn) == {}


def test_dry_run_still_reports_unreadable_images(
    tmp_path, conn, writes, make_processor
):
    path = library(tmp_path, [MID])[0]
    path.write_bytes(b"corrupt")

    outcomes = list(make_processor(dry_run=True).run([path]))

    assert isinstance(outcomes[0], FailedImage)
    assert db_failures(conn) == {}


# --- threads ---


def test_metadata_is_written_on_worker_threads_but_the_database_on_the_caller(
    tmp_path, writes, make_processor
):
    paths = library(tmp_path, [MID] * 4)
    db_threads = set()
    real_upsert_many = processor_module.upsert_many

    def spy(rows, conn):
        db_threads.add(threading.current_thread())
        return real_upsert_many(rows, conn)

    with patch.object(processor_module, "upsert_many", spy):
        list(make_processor().run(paths))

    assert db_threads == {threading.current_thread()}
    assert threading.current_thread().name not in writes.threads
    assert all(name.startswith("write") for name in writes.threads)


def test_metadata_writes_actually_run_in_parallel(tmp_path, writes, make_processor):
    writes.delay = 0.15
    paths = library(tmp_path, [MID] * 4)

    start = time.perf_counter()
    list(make_processor(workers=4, batch_size=4).run(paths))
    elapsed = time.perf_counter() - start

    assert elapsed < 4 * 0.15 * 0.75  # well under the serial time


# --- the database ---


def test_dead_connection_is_replaced_and_the_batch_retried(
    tmp_path, writes, make_processor
):
    paths = library(tmp_path, [MID, MID])
    old_conn = MagicMock(spec=sqlite3.Connection)
    fresh = make_connection(":memory:")
    real_upsert_many = processor_module.upsert_many
    attempts = []

    def flaky_upsert_many(rows, conn):
        attempts.append(conn)
        if conn is old_conn:
            raise sqlite3.OperationalError("database is locked")
        return real_upsert_many(rows, conn)

    processor = make_processor(conn=old_conn)
    with (
        patch.object(processor_module, "upsert_many", flaky_upsert_many),
        patch.object(processor_module, "make_connection", return_value=fresh) as remake,
    ):
        outcomes = list(processor.run(paths))

    assert all(isinstance(o, ScoredImage) for o in outcomes)
    remake.assert_called_once()
    old_conn.close.assert_called_once()
    assert attempts == [old_conn, fresh]
    assert processor.conn is fresh
    assert set(db_ratings(fresh)) == {"img00.jpg", "img01.jpg"}
    fresh.close()


def test_unavailable_database_reports_scored_images_as_failed(
    tmp_path, writes, make_processor
):
    """Metadata is already written but unrecorded, so they will simply be redone."""
    paths = library(tmp_path, [MID, MID])

    def always_locked(rows, conn):
        raise sqlite3.OperationalError("database is locked")

    processor = make_processor()
    with patch.object(processor_module, "upsert_many", always_locked):
        outcomes = list(processor.run(paths))
    processor.conn.close()  # the retry swapped in a connection of its own

    assert all(isinstance(o, FailedImage) for o in outcomes)
    assert all("database is locked" in o.message for o in outcomes)


def test_failure_records_are_best_effort_when_the_database_is_down(
    tmp_path, writes, make_processor
):
    path = library(tmp_path, [MID])[0]
    path.write_bytes(b"corrupt")

    def always_locked(rows, conn):
        raise sqlite3.OperationalError("database is locked")

    processor = make_processor()
    with patch.object(processor_module, "upsert_failures", always_locked):
        outcomes = list(processor.run([path]))  # must not raise
    processor.conn.close()  # the retry swapped in a connection of its own

    assert isinstance(outcomes[0], FailedImage)


# --- lifecycle ---


def test_closing_the_run_early_stops_further_work(
    tmp_path, conn, writes, fake_model, make_processor
):
    """What happens on Ctrl+C: finished batches are kept, nothing more is started."""
    paths = library(tmp_path, [MID] * 6)
    run = make_processor(batch_size=1).run(paths)

    first = next(run)
    run.close()

    assert isinstance(first, ScoredImage)
    assert fake_model.batch_sizes == [1]
    assert set(db_ratings(conn)) == {"img00.jpg"}


@pytest.mark.parametrize("bad", [{"workers": 0}, {"batch_size": 0}, {"workers": -1}])
def test_invalid_settings_are_rejected(make_processor, bad):
    with pytest.raises(ValueError):
        make_processor(**bad)


def test_failed_image_message_names_the_exception_type():
    failed = FailedImage(Path("x.jpg"), ValueError("boom"))
    assert failed.message == "ValueError: boom"
