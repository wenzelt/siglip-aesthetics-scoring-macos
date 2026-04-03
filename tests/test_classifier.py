from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from image_classifier.classifier import (
    SUPPORTED_EXTENSIONS,
    ClassifierError,
    Timings,
    get_device,
    load_model,
    score_image,
    score_to_rating,
)


# --- score_to_rating ---


@pytest.mark.parametrize(
    "score,expected_rating",
    [
        (1.0, 1),  # well below threshold
        (3.9, 1),  # just below 4.0
        (4.0, 2),  # exactly at 4.0 boundary — goes to 2
        (5.4, 2),  # just below 5.5
        (5.5, 3),  # exactly at 5.5 boundary — goes to 3
        (6.9, 3),  # just below 7.0
        (7.0, 4),  # exactly at 7.0 boundary — goes to 4
        (8.4, 4),  # just below 8.5
        (8.5, 5),  # exactly at 8.5 boundary — goes to 5
        (10.0, 5),  # maximum
    ],
)
def test_score_to_rating_boundaries(score, expected_rating):
    assert score_to_rating(score) == expected_rating


# --- get_device ---


def test_get_device_returns_mps_when_available():
    with patch("torch.backends.mps.is_available", return_value=True):
        device = get_device()
    assert device == torch.device("mps")


def test_get_device_falls_back_to_cpu():
    with patch("torch.backends.mps.is_available", return_value=False):
        device = get_device()
    assert device == torch.device("cpu")


# --- SUPPORTED_EXTENSIONS ---


def test_supported_extensions_are_lowercase():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext == ext.lower()
        assert ext.startswith(".")
    assert len(SUPPORTED_EXTENSIONS) == 8


# --- score_image ---


def _make_mock_model(score_value: float):
    """Return a mock model that produces a fixed score."""
    import numpy as np

    mock_output = MagicMock()
    mock_output.logits.squeeze.return_value.float.return_value.cpu.return_value.numpy.return_value = np.float32(
        score_value
    )

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    return mock_model


def _make_mock_preprocessor():
    """Return a mock preprocessor that passes pixel values through."""
    mock_pixel = MagicMock()
    mock_pixel.to.return_value = mock_pixel

    mock_output = MagicMock()
    mock_output.pixel_values.to.return_value = mock_pixel

    mock_preprocessor = MagicMock()
    mock_preprocessor.return_value = mock_output
    return mock_preprocessor


def test_score_image_returns_tuple_of_score_and_timings(tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    model = _make_mock_model(7.25)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    score, timings = score_image(img_path, model, preprocessor, device)
    assert isinstance(score, float)
    assert abs(score - 7.25) < 0.01
    assert isinstance(timings, Timings)


def test_score_image_timings_are_non_negative(tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    _, timings = score_image(
        img_path, _make_mock_model(5.0), _make_mock_preprocessor(), torch.device("cpu")
    )
    assert timings.load_ms >= 0
    assert timings.preprocess_ms >= 0
    assert timings.infer_ms >= 0


def test_score_image_timings_total_matches_sum(tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    _, timings = score_image(
        img_path, _make_mock_model(5.0), _make_mock_preprocessor(), torch.device("cpu")
    )
    assert (
        abs(
            timings.total_ms
            - (timings.load_ms + timings.preprocess_ms + timings.infer_ms)
        )
        < 0.1
    )


def test_score_image_raises_classifier_error_for_unreadable_file(tmp_path):
    bad_path = tmp_path / "corrupt.jpg"
    bad_path.write_bytes(b"not an image")

    model = _make_mock_model(5.0)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    with pytest.raises(ClassifierError):
        score_image(bad_path, model, preprocessor, device)


def test_score_image_raises_classifier_error_for_missing_file(tmp_path):
    missing = tmp_path / "ghost.jpg"
    model = _make_mock_model(5.0)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    with pytest.raises(ClassifierError):
        score_image(missing, model, preprocessor, device)


# --- load_model shim idempotency ---


def _make_load_model_mocks():
    """Return sys.modules patches needed to call load_model without real ML libs."""

    class FakeAestheticModel:
        def __init__(self, config, *args, **kwargs):
            pass

    class FakeSiglipConfig:
        def __init__(self):
            self.vision_config = object()

    fake_ap = MagicMock()
    fake_ap.convert_v2_5_from_siglip.return_value = (MagicMock(), MagicMock())

    fake_siglip_mod = MagicMock()
    fake_siglip_mod.AestheticPredictorV2_5Model = FakeAestheticModel

    fake_config_mod = MagicMock()
    fake_config_mod.SiglipConfig = FakeSiglipConfig

    return fake_ap, fake_siglip_mod, fake_config_mod, FakeAestheticModel


def test_load_model_shim_not_doubled(monkeypatch):
    """Calling load_model twice must not double-wrap AestheticPredictorV2_5Model.__init__."""
    fake_ap, fake_siglip_mod, fake_config_mod, FakeAestheticModel = (
        _make_load_model_mocks()
    )

    monkeypatch.setitem(sys.modules, "aesthetic_predictor_v2_5", fake_ap)
    monkeypatch.setitem(
        sys.modules, "aesthetic_predictor_v2_5.siglip_v2_5", fake_siglip_mod
    )
    monkeypatch.setitem(
        sys.modules, "transformers.models.siglip.configuration_siglip", fake_config_mod
    )

    device = MagicMock()
    device.type = "cpu"

    load_model(device)
    init_after_first = FakeAestheticModel.__init__

    load_model(device)
    init_after_second = FakeAestheticModel.__init__

    assert init_after_first is init_after_second, (
        "load_model applied the shim twice — second call double-wrapped __init__"
    )


# --- HEIC support ---


def _write_heic_fixture(path) -> None:
    """Write a minimal real HEIC file to *path* using pillow_heif."""
    import pillow_heif
    from PIL import Image as PILImage

    pil_img = PILImage.new("RGB", (16, 16), color=(255, 0, 128))
    pillow_heif.from_pillow(pil_img).save(path)


@pytest.mark.heic
def test_score_image_heic_raises_without_registration(tmp_path):
    """score_image must raise ClassifierError when PIL cannot identify the HEIC file.

    This is the RED anchor: it reproduces the exact error from the bug report —
    PIL.Image.open raises UnidentifiedImageError on a .heic path when
    pillow_heif has not been registered. We simulate that condition by
    patching Image.open inside the classifier module to raise the same error
    that Pillow raises in production without the HEIF opener.
    """
    from PIL import UnidentifiedImageError as _UIE

    heic_path = tmp_path / "fixture.heic"
    _write_heic_fixture(heic_path)

    model = _make_mock_model(5.0)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    # Simulate the production failure: Image.open raises UnidentifiedImageError
    # as it does when pillow_heif is not imported/registered.
    def _fake_open(path, *args, **kwargs):
        raise _UIE(f"cannot identify image file '{path}'")

    with patch("image_classifier.classifier.Image.open", side_effect=_fake_open):
        with pytest.raises(ClassifierError, match="cannot identify image file"):
            score_image(heic_path, model, preprocessor, device)


@pytest.mark.heic
def test_score_image_heic_succeeds_with_registration(tmp_path):
    """score_image must open and score a HEIC file when register_heif_opener() is active.

    This is the GREEN proof: importing image_classifier.classifier already calls
    register_heif_opener() at module level, so PIL.Image.open() handles HEIC.
    The test creates a real HEIC fixture (no mocking of the image load) and
    confirms score_image returns a valid (float, Timings) pair.
    """
    heic_path = tmp_path / "fixture.heic"
    _write_heic_fixture(heic_path)

    model = _make_mock_model(6.5)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    score, timings = score_image(heic_path, model, preprocessor, device)

    assert isinstance(score, float), "score_image must return a float score"
    assert abs(score - 6.5) < 0.01, f"expected ~6.5, got {score}"
    assert isinstance(timings, Timings), "score_image must return a Timings instance"
    assert timings.load_ms >= 0, "load_ms must be non-negative"


# --- Bug 1: truncated JPEG tolerance ---


def test_score_image_succeeds_for_truncated_jpeg(tmp_path):
    """score_image must NOT raise ClassifierError for a truncated-but-parseable JPEG.

    A real JPEG header followed by valid scan data that is simply cut short is
    exactly the kind of file Pillow rejects by default (LOAD_TRUNCATED_IMAGES=False)
    but should tolerate when the flag is True.

    We create a minimal valid JPEG via PIL, then chop the last 512 bytes off,
    which makes Pillow raise ImageFile.IsTruncatedError (a subclass of OSError)
    unless LOAD_TRUNCATED_IMAGES is enabled.
    """
    import io

    from PIL import Image as PILImage

    # Build a real JPEG in memory, then truncate it.
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(200, 100, 50)).save(buf, format="JPEG")
    full_bytes = buf.getvalue()
    # Drop the last 32 bytes so Pillow sees an incomplete scan segment.
    # Dropping 32 bytes is enough to trigger OSError without the flag, but
    # recoverable enough for Pillow to produce pixel data when the flag is set.
    assert len(full_bytes) > 64, "sanity: JPEG must be larger than our truncation"
    truncated_bytes = full_bytes[:-32]

    img_path = tmp_path / "truncated.jpg"
    img_path.write_bytes(truncated_bytes)

    model = _make_mock_model(5.0)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    # Must not raise — truncated images should be loaded tolerantly.
    score, timings = score_image(img_path, model, preprocessor, device)
    assert isinstance(score, float)
    assert isinstance(timings, Timings)


# --- Bug 2: grayscale JPEG produces 3-channel array ---


def test_score_image_palette_with_transparency_produces_3channel_array(tmp_path):
    """score_image must pass a (H, W, 3) uint8 array to the preprocessor for a
    palette image with transparency (mode 'P' + 'transparency').

    The WhatsApp JPEG IMG-20170123-WA0003.jpg is a grayscale/palette image.
    Pillow's img.convert("RGB") on a 'P' image that carries an alpha channel
    internally goes through 'RGBA' first, yielding a (H, W, 4) array.  The
    existing guard in classifier.py slices arr[:, :, :3] for 4-channel arrays,
    BUT the guard for arr.shape[2] < 3 uses np.repeat on axis 0 of the slice —
    a palette image with no transparency that Pillow converts to 1-channel first
    can still produce ndim==3 with shape[2]==1 that the guard handles, BUT a
    palette image converted internally through RGBA produces shape[2]==4, which
    the existing `arr[:, :, :3]` guard *does* handle.

    The critical sub-case that was broken: a palette image saved as PNG (which
    preserves the palette + transparency info) opened and converted.  We test
    that path directly with a synthetic 'P'-mode PNG to confirm exactly 3
    channels reach the preprocessor regardless of intermediate Pillow mode.
    """
    import numpy as np
    from PIL import Image as PILImage

    # Build a palette image with a transparency entry — this is what forces
    # Pillow to route through RGBA when convert("RGB") is called.
    base = PILImage.new("RGBA", (64, 64), color=(100, 150, 200, 128))
    palette_img = base.convert("P")  # This sets mode='P' with palette + alpha info

    img_path = tmp_path / "palette_transparency.png"
    palette_img.save(img_path, format="PNG")

    captured_arrays: list[np.ndarray] = []

    def _capturing_preprocessor(images, **kwargs):  # type: ignore[override]
        arr = images if isinstance(images, np.ndarray) else np.array(images)
        captured_arrays.append(arr)
        return _make_mock_preprocessor()(images=images, **kwargs)

    model = _make_mock_model(5.0)
    device = torch.device("cpu")

    score_image(img_path, model, _capturing_preprocessor, device)

    assert len(captured_arrays) == 1, "preprocessor must be called exactly once"
    arr = captured_arrays[0]
    assert arr.ndim == 3, f"expected 3-D array (H, W, C), got shape {arr.shape}"
    assert arr.shape[2] == 3, f"expected 3 channels, got {arr.shape[2]}"
