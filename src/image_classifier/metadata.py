from __future__ import annotations

import plistlib
import re
import shutil
import subprocess
from pathlib import Path

_SUBPROCESS_TIMEOUT = 30
_SCORE_TAG_RE = re.compile(r"^\d{1,2}\.\d$")


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
        timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
        raise MetadataError(f"exiftool failed for {path}: {result.stderr.strip()}")


def _read_finder_tags(path: Path) -> list[str]:
    """Read existing macOS Finder tags from extended attributes."""
    result = subprocess.run(
        ["xattr", "-px", "com.apple.metadata:_kMDItemUserTags", str(path)],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
        return []
    try:
        hex_str = "".join(result.stdout.split())
        return plistlib.loads(bytes.fromhex(hex_str))
    except (ValueError, plistlib.InvalidFileException):
        return []


def _is_score_tag(tag: str) -> bool:
    """Return True if tag looks like a score we previously wrote (★… or N.D format)."""
    if tag.startswith("★"):
        return True
    return bool(_SCORE_TAG_RE.match(tag))


def write_score_tag(path: Path, score: float) -> None:
    """Write aesthetic score as a macOS Finder tag, preserving user's own tags.

    Writes a single numeric score tag (e.g. '7.3').
    Any previously written score tags are replaced; other user tags are preserved.
    """
    score_tag = f"{score:.1f}"

    existing = _read_finder_tags(path)
    kept = [t for t in existing if not _is_score_tag(t)]
    tags = kept + [score_tag]

    plist_bytes = plistlib.dumps(tags, fmt=plistlib.FMT_BINARY)
    result = subprocess.run(
        [
            "xattr",
            "-wx",
            "com.apple.metadata:_kMDItemUserTags",
            plist_bytes.hex(),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    if result.returncode != 0:
        raise MetadataError(f"xattr failed for {path}: {result.stderr.strip()}")
