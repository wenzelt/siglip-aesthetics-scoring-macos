from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.log"


def setup_logger() -> logging.Logger | None:
    """Open the append-mode log file. Returns None if it cannot be opened."""
    logger = logging.getLogger("classify")
    if logger.handlers:
        return logger
    logger.setLevel(logging.ERROR)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(LOG_PATH), mode="a")
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
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    logger.error("%s\t%s\t%s: %s", timestamp, path, type(exc).__name__, exc)
