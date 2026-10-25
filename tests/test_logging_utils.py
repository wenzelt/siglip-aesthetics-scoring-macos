from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import image_classifier.logging_utils as logging_utils


@pytest.fixture
def isolated_logger(tmp_path, monkeypatch):
    """Point the module at a temp log file and a clean 'classify' logger."""
    monkeypatch.setattr(logging_utils, "LOG_PATH", tmp_path / "logs" / "classify.log")
    logger = logging.getLogger("classify")
    saved = list(logger.handlers)
    logger.handlers.clear()
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:] = saved


def test_setup_logger_creates_the_log_folder_and_a_rotating_handler(isolated_logger):
    logger = logging_utils.setup_logger()

    assert logger is isolated_logger
    assert logging_utils.LOG_PATH.parent.is_dir()
    (handler,) = logger.handlers
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes > 0 and handler.backupCount > 0


def test_setup_logger_is_idempotent(isolated_logger):
    first = logging_utils.setup_logger()
    second = logging_utils.setup_logger()
    assert first is second
    assert len(first.handlers) == 1


def test_setup_logger_returns_none_and_warns_when_the_log_is_unwritable(
    isolated_logger, monkeypatch, capsys
):
    def deny(*args, **kwargs):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(logging.handlers, "RotatingFileHandler", deny)

    assert logging_utils.setup_logger() is None
    assert "could not open log file" in capsys.readouterr().err


def test_log_error_writes_a_tab_separated_line(isolated_logger):
    logger = logging_utils.setup_logger()

    logging_utils.log_error(logger, Path("/photos/a.jpg"), ValueError("bad pixel"))

    for handler in logger.handlers:
        handler.flush()
    line = logging_utils.LOG_PATH.read_text().strip()
    timestamp, path, error = line.split("\t")
    assert timestamp.endswith("Z")
    assert path == "/photos/a.jpg"
    assert error == "ValueError: bad pixel"


def test_log_error_without_a_logger_does_nothing():
    logging_utils.log_error(
        None, Path("/photos/a.jpg"), ValueError("x")
    )  # must not raise


def test_log_error_uses_the_supplied_logger():
    logger = MagicMock(spec=logging.Logger)
    logging_utils.log_error(logger, Path("a.jpg"), RuntimeError("boom"))
    logger.error.assert_called_once()
