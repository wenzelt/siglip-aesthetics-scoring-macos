from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


@dataclass
class Timings:
    """Per-image phase timings in milliseconds."""

    load_ms: float = 0.0
    preprocess_ms: float = 0.0
    infer_ms: float = 0.0
    upsert_ms: float = 0.0
    exiftool_ms: float = 0.0
    xattr_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.load_ms
            + self.preprocess_ms
            + self.infer_ms
            + self.upsert_ms
            + self.exiftool_ms
            + self.xattr_ms
        )


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


def load_model(device: torch.device) -> tuple[Any, Any]:
    """Load the aesthetic predictor model and preprocessor.

    On first run this downloads the checkpoint (~1.5 GB) to the
    HuggingFace cache at ~/.cache/huggingface.

    Applies a compatibility shim for transformers >= 4.41 where SiglipConfig
    no longer exposes hidden_size directly (it moved to vision_config.hidden_size).
    """
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip  # noqa: PLC0415
    from aesthetic_predictor_v2_5.siglip_v2_5 import AestheticPredictorV2_5Model  # noqa: PLC0415
    from transformers.models.siglip.configuration_siglip import SiglipConfig  # noqa: PLC0415

    # Shim: newer transformers resolves "google/siglip-so400m-patch14-384" to
    # SiglipConfig (full model config) rather than SiglipVisionConfig. The model
    # class expects SiglipVisionConfig, so we extract vision_config before init.
    # Guard ensures the patch is applied only once even if load_model is called
    # multiple times (avoids infinite recursion from double-wrapping).
    if not getattr(AestheticPredictorV2_5Model, "_shim_applied", False):
        _orig_init = AestheticPredictorV2_5Model.__init__

        def _patched_init(self, config, *args, **kwargs):  # type: ignore[misc]
            if isinstance(config, SiglipConfig) and hasattr(config, "vision_config"):
                config = config.vision_config
            _orig_init(self, config, *args, **kwargs)

        AestheticPredictorV2_5Model.__init__ = _patched_init  # type: ignore[method-assign]
        AestheticPredictorV2_5Model._shim_applied = True  # type: ignore[attr-defined]

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(device)
    return model, preprocessor


def score_image(
    path: Path, model: Any, preprocessor: Any, device: torch.device
) -> tuple[float, Timings]:
    """Score a single image. Raises ClassifierError if the file is unreadable.

    Returns (score, timings) where timings breaks down milliseconds spent in
    each phase: load, preprocess, and infer.
    """
    timings = Timings()

    t0 = time.perf_counter()
    try:
        with Image.open(path) as img:
            image = img.convert("RGB")
            arr = np.array(image)
            # Guard against images that still end up with wrong channel count
            # after convert("RGB") (e.g. unusual TIFF/HEIC colorspaces).
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] != 3:
                if arr.shape[2] < 3:
                    arr = np.repeat(arr[:, :, :1], 3, axis=-1)
                else:
                    arr = arr[:, :, :3]
            image = arr
    except (UnidentifiedImageError, OSError) as exc:
        raise ClassifierError(str(exc)) from exc
    timings.load_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .to(device)
    )
    timings.preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    timings.infer_ms = (time.perf_counter() - t2) * 1000

    return float(score), timings
