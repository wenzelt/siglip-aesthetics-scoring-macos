from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from image_classifier.classifier import Timings
from image_classifier.report import (
    RunSummary,
    _bar,
    format_duration,
    print_profile_summary,
    print_summary,
    star_display,
)


def render(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    fn(console, *args, **kwargs)
    return buf.getvalue()


def summary(**overrides) -> RunSummary:
    fields = dict(scored=10, skipped=2, errors=0, rating_counts={5: 1, 4: 3, 3: 2})
    fields.update(overrides)
    return RunSummary(**fields)


# --- star_display ---


@pytest.mark.parametrize("rating,expected", [(1, "★☆☆☆☆"), (3, "★★★☆☆"), (5, "★★★★★")])
def test_star_display(rating, expected):
    assert star_display(rating) == expected


# --- format_duration ---


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.25, "250 ms"),
        (42.04, "42.0 s"),
        (185, "3m 05s"),
        (3599, "59m 59s"),
        (8040, "2h 14m"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# --- bars ---


def test_bar_is_proportional_to_the_largest_bucket():
    assert len(_bar(50, 100)) == 15
    assert len(_bar(100, 100)) == 30


def test_bar_scales_for_huge_libraries():
    """The old bar capped at 30 *images*, so 100k-image libraries drew identical bars."""
    assert len(_bar(90_000, 100_000)) < len(_bar(100_000, 100_000))


def test_bar_shows_something_for_any_non_empty_bucket_and_nothing_for_empty():
    assert _bar(1, 100_000) == "█"
    assert _bar(0, 100) == ""
    assert _bar(5, 0) == ""


# --- print_summary ---


def test_summary_reports_counts():
    out = render(print_summary, summary())
    assert "Scored" in out and "10" in out
    assert "Skipped" in out and "already in database" in out
    assert "Errors" in out


def test_summary_shows_throughput_when_images_were_scored():
    out = render(print_summary, summary(scored=30, elapsed_s=10.0))
    assert "10.0 s" in out
    assert "3.0 images/s" in out


def test_summary_omits_time_when_nothing_was_scored():
    out = render(print_summary, summary(scored=0, elapsed_s=0.2))
    assert "images/s" not in out


def test_summary_names_the_log_file_when_there_are_errors():
    out = render(
        print_summary,
        summary(errors=2, failures=[("a.jpg", "boom")], failures_total=2),
        Path("/logs/classify.log"),
    )
    assert "/logs/classify.log" in out


def test_summary_lists_failures_and_how_many_are_hidden():
    failures = [(f"bad{i}.jpg", "ClassifierError: nope") for i in range(3)]
    out = render(print_summary, summary(errors=7, failures=failures, failures_total=7))
    assert "Recent Failures" in out
    assert "bad0.jpg" in out
    assert "and 4 more" in out


def test_summary_does_not_claim_hidden_failures_when_all_are_shown():
    out = render(
        print_summary,
        summary(errors=1, failures=[("a.jpg", "boom")], failures_total=1),
    )
    assert "more" not in out


def test_summary_without_errors_shows_no_failure_panel():
    assert "Recent Failures" not in render(print_summary, summary())


def test_summary_shows_each_star_band_with_counts_and_shares():
    out = render(print_summary, summary(rating_counts={5: 1, 4: 3, 3: 6}))
    for label in ["8.5+", "7–8.5", "5.5–7", "4–5.5", "<4"]:
        assert label in out
    assert "60%" in out and "30%" in out and "10%" in out


def test_summary_formats_large_counts_with_separators():
    assert "12,345" in render(print_summary, summary(rating_counts={5: 12_345}))


def test_summary_ignores_out_of_range_ratings_without_crashing():
    out = render(print_summary, summary(rating_counts={0: 3, 9: 1}))
    assert "Distribution" not in out  # nothing valid to chart


def test_summary_with_no_scores_skips_the_distribution():
    assert "Distribution" not in render(print_summary, summary(rating_counts={}))


def test_dry_run_summary_says_nothing_was_written():
    out = render(print_summary, summary(dry_run=True))
    assert "dry run" in out
    assert "Would score" in out
    assert "(this run)" in out


# --- print_profile_summary ---


def test_profile_summary_tabulates_mean_and_max_per_phase():
    timings = [
        Timings(load_ms=10, preprocess_ms=20, infer_ms=30),
        Timings(load_ms=30, preprocess_ms=20, infer_ms=50),
    ]
    out = render(print_profile_summary, timings)
    assert "Timing Profile (2 images)" in out
    load_row = next(line for line in out.splitlines() if "Load" in line)
    assert "20.0" in load_row and "30.0" in load_row
    assert "Total" in out


def test_profile_summary_explains_overlap_and_reports_wall_clock_throughput():
    out = render(print_profile_summary, [Timings(), Timings()], elapsed_s=0.5)
    assert "4.0 images/s" in out


def test_profile_summary_without_timings_prints_nothing():
    assert render(print_profile_summary, []) == ""
