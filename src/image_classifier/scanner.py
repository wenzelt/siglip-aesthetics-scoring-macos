from __future__ import annotations

from pathlib import Path

from image_classifier.classifier import SUPPORTED_EXTENSIONS


def scan_images(folder: Path, recursive: bool) -> list[Path]:
    """Return all supported image files in folder, skipping hidden paths."""
    glob_fn = folder.rglob if recursive else folder.glob
    return [
        path
        for path in glob_fn("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.parts)
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
