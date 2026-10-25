from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

# Allow Pillow to load images whose scan data is slightly truncated rather than
# raising OSError("image file is truncated").  This tolerates real-world JPEGs
# (e.g. WhatsApp exports) that are structurally valid but missing a few bytes at
# the end of the compressed scan segment.
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


# Every extension here must be one exiftool can *write* XMP into.  BMP is
# deliberately absent: exiftool reports "Writing of BMP files is not yet
# supported", so a BMP could be scored but never rated.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic"}
)


class ClassifierError(Exception):
    """Raised when an image cannot be scored."""


# (inclusive lower bound, star rating, label), best band first.  This is the
# single source of truth for score -> stars and for the report's score ranges.
RATING_BANDS: tuple[tuple[float, int, str], ...] = (
    (8.5, 5, "8.5+"),
    (7.0, 4, "7–8.5"),
    (5.5, 3, "5.5–7"),
    (4.0, 2, "4–5.5"),
    (-math.inf, 1, "<4"),
)


def score_to_rating(score: float) -> int:
    """Map a 1–10 aesthetic score to a 1–5 XMP star rating.

    Boundaries are half-open on the right: [low, high).
    The top range [8.5, 10.0] is fully closed.
    """
    for lower_bound, stars, _label in RATING_BANDS:
        if score >= lower_bound:
            return stars
    # Only NaN compares false against every bound; never guess a rating for it.
    raise ValueError(f"cannot rate a non-numeric score: {score!r}")


def validate_score(score: float) -> float:
    """Return score unchanged, or raise ClassifierError if the model produced NaN/inf."""
    if not math.isfinite(score):
        raise ClassifierError(f"model returned a non-finite score ({score})")
    return score


def get_device() -> torch.device:
    """Return CUDA if present, MPS on Apple Silicon, CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
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
    from aesthetic_predictor_v2_5.siglip_v2_5 import (
        AestheticPredictorV2_5Model,  # noqa: PLC0415
    )
    from transformers.models.siglip.configuration_siglip import (
        SiglipConfig,  # noqa: PLC0415
    )

    # Shim: newer transformers resolves "google/siglip-so400m-patch14-384" to
    # SiglipConfig (full model config) rather than SiglipVisionConfig. The model
    # class expects SiglipVisionConfig, so we extract vision_config before init.
    # Guard ensures the patch is applied only once even if load_model is called
    # multiple times (avoids infinite recursion from double-wrapping).
    if not getattr(AestheticPredictorV2_5Model, "_shim_applied", False):
        _orig_init = AestheticPredictorV2_5Model.__init__

        def _patched_init(self, config, *args, **kwargs):
            if isinstance(config, SiglipConfig) and hasattr(config, "vision_config"):
                config = config.vision_config
            _orig_init(self, config, *args, **kwargs)

        AestheticPredictorV2_5Model.__init__ = _patched_init
        AestheticPredictorV2_5Model._shim_applied = True

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(device)
    return model, preprocessor


def _input_size(preprocessor: Any) -> tuple[int, int] | None:
    """Return the (width, height) the preprocessor resizes to, if it declares one."""
    size = getattr(preprocessor, "size", None)
    if isinstance(size, dict) and "width" in size and "height" in size:
        return int(size["width"]), int(size["height"])
    return None


def _resample_filter(preprocessor: Any) -> Image.Resampling:
    resample = getattr(preprocessor, "resample", None)
    try:
        return Image.Resampling(resample)  # type: ignore[arg-type]
    except ValueError:
        return Image.Resampling.BICUBIC


def load_image(path: Path, preprocessor: Any = None) -> np.ndarray:
    """Decode path into an upright (H, W, 3) uint8 RGB array.

    EXIF orientation is applied so portrait photos are scored the way they are
    displayed, not as the sideways pixels stored in the file.

    When the preprocessor declares its input size, the image is resized to it
    here with the same filter the preprocessor would use.  Resizing a PIL image
    is several times cheaper than handing a 12 MP ndarray to the HuggingFace
    preprocessor, and the preprocessor's own resize is then a no-op, so the
    resulting tensor is bit-identical.

    Raises ClassifierError if the file cannot be decoded.
    """
    try:
        with Image.open(path) as img:
            upright: Image.Image = img
            # A malformed EXIF block must not make an otherwise decodable image fail.
            with contextlib.suppress(Exception):
                upright = ImageOps.exif_transpose(img)
            rgb = upright.convert("RGB")
            size = _input_size(preprocessor)
            if size is not None and rgb.size != size:
                rgb = rgb.resize(size, _resample_filter(preprocessor))
            return np.asarray(rgb)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ClassifierError(str(exc)) from exc


def preprocess(image_array: np.ndarray, preprocessor: Any) -> torch.Tensor:
    """Run the model's preprocessor, returning a (3, H, W) float tensor."""
    pixel_values: torch.Tensor = preprocessor(
        images=image_array, return_tensors="pt"
    ).pixel_values
    return pixel_values[0]


def score_batch(
    pixel_values: Sequence[torch.Tensor], model: Any, device: torch.device
) -> list[float]:
    """Score preprocessed images in a single forward pass, one score per image."""
    batch = torch.stack(list(pixel_values)).to(torch.bfloat16).to(device)
    with torch.inference_mode():
        logits = model(batch).logits
    scores = logits.float().cpu().numpy().reshape(-1)
    if len(scores) != len(pixel_values):
        raise ClassifierError(
            f"model returned {len(scores)} scores for {len(pixel_values)} images"
        )
    return [float(s) for s in scores]


def score_image(
    path: Path, model: Any, preprocessor: Any, device: torch.device
) -> tuple[float, Timings]:
    """Score a single image. Raises ClassifierError if the file is unreadable.

    Returns (score, timings) where timings breaks down milliseconds spent in
    each phase: load, preprocess, and infer.
    """
    timings = Timings()

    t0 = time.perf_counter()
    image_array = load_image(path, preprocessor)
    timings.load_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    pixel_values = preprocess(image_array, preprocessor)
    timings.preprocess_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    score = validate_score(score_batch([pixel_values], model, device)[0])
    timings.infer_ms = (time.perf_counter() - t2) * 1000

    return score, timings
