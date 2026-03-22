from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".bmp"}
)


class ClassifierError(Exception):
    """Raised when an image cannot be scored."""


def score_to_rating(score: float) -> int:
    """Map a 1–10 aesthetic score to a 1–5 XMP star rating.

    Boundaries are half-open on the right: [low, high).
    The top range [8.5, 10.0] is fully closed.
    """
    if score < 4.0:
        return 1
    elif score < 5.5:
        return 2
    elif score < 7.0:
        return 3
    elif score < 8.5:
        return 4
    else:
        return 5


def get_device() -> torch.device:
    """Return MPS on Apple Silicon, CPU otherwise."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device) -> tuple:
    """Load the aesthetic predictor model and preprocessor.

    On first run this downloads the checkpoint (~1.5 GB) to the
    HuggingFace cache at ~/.cache/huggingface.
    """
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip  # noqa: PLC0415

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(device)
    return model, preprocessor


def score_image(path: Path, model, preprocessor, device: torch.device) -> float:
    """Score a single image. Raises ClassifierError if the file is unreadable."""
    try:
        image = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, FileNotFoundError) as exc:
        raise ClassifierError(str(exc)) from exc

    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .to(device)
    )
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    return float(score)
