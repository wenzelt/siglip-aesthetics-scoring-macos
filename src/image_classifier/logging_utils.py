from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.log"

_MAX_LOG_BYTES = 5 * 1024 * 1024
_LOG_BACKUPS = 3


def setup_logger() -> logging.Logger | None:
    """Open the rotating log file. Returns None if it cannot be opened."""
    logger = logging.getLogger("classify")
    if logger.handlers:
        return logger
    logger.setLevel(logging.ERROR)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(LOG_PATH),
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger
    except OSError:
        print(f"Warning: could not open log file at {LOG_PATH}", file=sys.stderr)
        return None


def log_error(logger: logging.Logger | None, path: Path, exc: Exception) -> None:
    """Write a tab-separated error line to the log."""
    if logger is None:
        return
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    logger.error("%s\t%s\t%s: %s", timestamp, path, type(exc).__name__, exc)
