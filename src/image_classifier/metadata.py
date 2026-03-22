from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MetadataError(Exception):
    """Raised when exiftool fails to write metadata."""


def check_exiftool() -> None:
    """Exit with install instructions if exiftool is not on PATH."""
    if shutil.which("exiftool") is None:
        raise SystemExit(
            "Error: exiftool is required but not installed.\n"
            "Install it with: brew install exiftool"
        )


def write_rating(path: Path, rating: int) -> None:
    """Write XMP:Rating to the image file via exiftool."""
    result = subprocess.run(
        ["exiftool", f"-XMP:Rating={rating}", "-overwrite_original", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MetadataError(f"exiftool failed for {path}: {result.stderr.strip()}")
