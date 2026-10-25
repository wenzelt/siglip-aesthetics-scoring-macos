"""Shared fakes and helpers for the test suite."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

INPUT_SIZE = 16

requires_exiftool = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="exiftool is not installed"
)


class FakePreprocessor:
    """Stands in for the HuggingFace image processor.

    Declares a small input size (as the real one does) and returns a
    (1, 3, H, W) tensor in [0, 1].
    """

    size = {"height": INPUT_SIZE, "width": INPUT_SIZE}
    resample = 3  # PIL bicubic, as in the real processor config

    def __call__(self, images: Any, return_tensors: str = "pt") -> SimpleNamespace:
        array = np.asarray(images)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().unsqueeze(0)
        return SimpleNamespace(pixel_values=tensor / 255.0)


class FakeModel:
    """Scores each image by its mean brightness: black -> 1.0, white -> 10.0.

    Records the batch size of every forward pass.
    """

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.batch_sizes.append(pixel_values.shape[0])
        score = pixel_values.float().mean(dim=(1, 2, 3)) * 9 + 1
        return SimpleNamespace(logits=score.unsqueeze(-1))


def make_image(
    path: Path,
    level: int = 128,
    size: tuple[int, int] = (32, 24),
    mode: str = "RGB",
    **save_kwargs: Any,
) -> Path:
    """Write a flat-colour image of the given brightness (0-255) to path."""
    Image.new(mode, size, color=level if mode in ("L", "P") else (level,) * 3).save(
        path, **save_kwargs
    )
    return path
