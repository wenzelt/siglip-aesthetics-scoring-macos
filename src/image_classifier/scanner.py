from __future__ import annotations

import os
from pathlib import Path

from image_classifier.classifier import SUPPORTED_EXTENSIONS

# Library bundles managed by Apple apps.  Their contents are tracked by the
# app's own database, so rewriting files inside them behind its back can leave
# the library inconsistent.  They are never descended into.
PROTECTED_BUNDLE_SUFFIXES: frozenset[str] = frozenset(
    {".photoslibrary", ".photolibrary", ".aplibrary"}
)


def is_protected_bundle(path: Path) -> bool:
    """Return True if path names a photo-library bundle that must not be modified."""
    return path.suffix.lower() in PROTECTED_BUNDLE_SUFFIXES


def scan_images(folder: Path, recursive: bool) -> list[Path]:
    """Return all supported image files in folder, sorted by path.

    Hidden files and folders (including macOS ``._*`` AppleDouble files) and
    photo-library bundles are skipped.  They are pruned before descending, so
    a large ``.git`` or ``.Trash`` is never walked.  Unreadable subfolders are
    skipped; an unreadable ``folder`` raises OSError.
    """
    found: list[Path] = []
    pending = [folder]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                listing = list(entries)
        except OSError:
            if current == folder:
                raise
            continue
        for entry in listing:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                if recursive and not is_protected_bundle(Path(entry.name)):
                    pending.append(Path(entry.path))
            elif (
                Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS
                and entry.is_file()
            ):
                found.append(Path(entry.path))
    return sorted(found)
