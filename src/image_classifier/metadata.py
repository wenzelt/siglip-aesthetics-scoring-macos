from __future__ import annotations

import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SUBPROCESS_TIMEOUT = 30
_FINDER_TAGS_ATTR = "com.apple.metadata:_kMDItemUserTags"
# One decimal place, optionally negative: the model can stray a little outside 1-10.
_SCORE_TAG_RE = re.compile(r"^-?\d{1,2}\.\d$")


class MetadataError(Exception):
    """Raised when exiftool or xattr fails to write metadata."""


def check_exiftool() -> None:
    """Exit with install instructions if exiftool is not on PATH."""
    if shutil.which("exiftool") is None:
        raise SystemExit(
            "Error: exiftool is required but not installed.\n"
            "Install it with: brew install exiftool"
        )


def finder_tags_supported() -> bool:
    """Return True where Finder tags can be written (macOS with the xattr tool)."""
    return sys.platform == "darwin" and shutil.which("xattr") is not None


def _run(args: list[str], what: str) -> subprocess.CompletedProcess[str]:
    """Run a helper tool, turning hangs and launch failures into MetadataError."""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise MetadataError(f"{what} timed out after {_SUBPROCESS_TIMEOUT}s") from exc
    except OSError as exc:
        raise MetadataError(f"{what} could not be run: {exc}") from exc


def write_rating(path: Path, rating: int) -> None:
    """Write XMP:Rating to the image file via exiftool."""
    result = _run(
        ["exiftool", f"-XMP:Rating={rating}", "-overwrite_original", str(path)],
        "exiftool",
    )
    if result.returncode != 0:
        raise MetadataError(f"exiftool failed for {path}: {result.stderr.strip()}")


def read_finder_tags(path: Path) -> list[str]:
    """Read existing macOS Finder tags from extended attributes."""
    result = _run(["xattr", "-px", _FINDER_TAGS_ATTR, str(path)], "xattr")
    if result.returncode != 0:
        return []
    try:
        hex_str = "".join(result.stdout.split())
        parsed = plistlib.loads(bytes.fromhex(hex_str))
    except (ValueError, plistlib.InvalidFileException):
        return []
    if not isinstance(parsed, list):
        return []
    return [tag for tag in parsed if isinstance(tag, str)]


def _is_score_tag(tag: str) -> bool:
    """Return True if tag looks like a score we previously wrote (★… or N.D format)."""
    if tag.startswith("★"):
        return True
    return bool(_SCORE_TAG_RE.match(tag))


def write_score_tag(
    path: Path, score: float, existing_tags: list[str] | None = None
) -> None:
    """Write aesthetic score as a macOS Finder tag, preserving user's own tags.

    Writes a single numeric score tag (e.g. '7.3').
    Any previously written score tags are replaced; other user tags are preserved.

    Pass existing_tags (from read_finder_tags) when the tags were read before the
    file was rewritten: exiftool replaces the file via a temporary copy, which is
    not guaranteed to carry extended attributes over.
    """
    score_tag = f"{score:.1f}"

    existing = read_finder_tags(path) if existing_tags is None else existing_tags
    kept = [t for t in existing if not _is_score_tag(t)]
    tags = kept + [score_tag]

    plist_bytes = plistlib.dumps(tags, fmt=plistlib.FMT_BINARY)
    result = _run(
        ["xattr", "-wx", _FINDER_TAGS_ATTR, plist_bytes.hex(), str(path)],
        "xattr",
    )
    if result.returncode != 0:
        raise MetadataError(f"xattr failed for {path}: {result.stderr.strip()}")
