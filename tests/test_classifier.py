from __future__ import annotations

import io
import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from image_classifier.classifier import (
    RATING_BANDS,
    SUPPORTED_EXTENSIONS,
    ClassifierError,
    Timings,
    get_device,
    load_image,
    load_model,
    preprocess,
    score_batch,
    score_image,
    score_to_rating,
    validate_score,
)
from tests.helpers import INPUT_SIZE, FakeModel, FakePreprocessor, make_image

CPU = torch.device("cpu")


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
        (-0.4, 1),  # the model can stray below its nominal range
        (11.2, 5),  # ... and above it
    ],
)
def test_score_to_rating_boundaries(score, expected_rating):
    assert score_to_rating(score) == expected_rating


def test_score_to_rating_refuses_nan():
    """NaN compares false against every bound; it must not silently become 5 stars."""
    with pytest.raises(ValueError):
        score_to_rating(float("nan"))


def test_rating_bands_cover_every_star_best_first():
    assert [stars for _, stars, _ in RATING_BANDS] == [5, 4, 3, 2, 1]
    bounds = [lower for lower, _, _ in RATING_BANDS]
    assert bounds == sorted(bounds, reverse=True)
    assert bounds[-1] == -math.inf


# --- validate_score ---


def test_validate_score_passes_finite_values_through():
    assert validate_score(6.5) == 6.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_score_rejects_non_finite(bad):
    with pytest.raises(ClassifierError, match="non-finite"):
        validate_score(bad)


# --- get_device ---


def test_get_device_prefers_cuda():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert get_device() == torch.device("cuda")


def test_get_device_returns_mps_when_available():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert get_device() == torch.device("cpu")


# --- SUPPORTED_EXTENSIONS ---


def test_supported_extensions_are_lowercase():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext == ext.lower()
        assert ext.startswith(".")
    assert len(SUPPORTED_EXTENSIONS) == 7


def test_bmp_is_not_supported_because_exiftool_cannot_write_it():
    assert ".bmp" not in SUPPORTED_EXTENSIONS


# --- load_image ---


def test_load_image_returns_rgb_uint8_array(tmp_path):
    path = make_image(tmp_path / "a.jpg", size=(40, 30))
    arr = load_image(path)
    assert arr.shape == (30, 40, 3)
    assert arr.dtype == np.uint8


def test_load_image_applies_exif_orientation(tmp_path):
    """A 40x30 JPEG flagged Orientation=6 is a portrait photo and must load as 30x40."""
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW to display
    path = make_image(tmp_path / "phone.jpg", size=(40, 30), exif=exif)
    assert load_image(path).shape == (40, 30, 3)


def test_load_image_survives_malformed_exif(tmp_path):
    path = make_image(tmp_path / "a.jpg", size=(40, 30))
    with patch(
        "image_classifier.classifier.ImageOps.exif_transpose",
        side_effect=ValueError("bad exif"),
    ):
        assert load_image(path).shape == (30, 40, 3)


def test_load_image_fits_to_the_preprocessor_input_size(tmp_path):
    path = make_image(tmp_path / "big.jpg", size=(100, 60))
    arr = load_image(path, FakePreprocessor())
    assert arr.shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_load_image_ignores_a_preprocessor_without_a_size(tmp_path):
    path = make_image(tmp_path / "a.jpg", size=(40, 30))
    assert load_image(path, MagicMock()).shape == (30, 40, 3)


def test_load_image_falls_back_to_bicubic_for_an_unknown_resample_setting(tmp_path):
    path = make_image(tmp_path / "big.jpg", size=(100, 60))
    odd = SimpleNamespace(size={"height": 8, "width": 8}, resample="not-a-filter")
    assert load_image(path, odd).shape == (8, 8, 3)


def test_prefit_resize_gives_the_same_tensor_as_the_preprocessors_own_resize(tmp_path):
    """Pre-resizing in PIL must not change what the model sees (bit-identical)."""
    from transformers import SiglipImageProcessor

    processor = SiglipImageProcessor(size={"height": 64, "width": 64})
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((90, 130, 3)) * 255).astype("uint8")).save(
        tmp_path / "noise.png"
    )

    prefit = preprocess(load_image(tmp_path / "noise.png", processor), processor)
    with Image.open(tmp_path / "noise.png") as img:
        reference = preprocess(np.asarray(img.convert("RGB")), processor)

    assert prefit.shape == (3, 64, 64)
    assert torch.equal(prefit, reference)


@pytest.mark.parametrize("mode", ["L", "LA", "P", "RGBA", "CMYK"])
def test_load_image_always_yields_three_channels(tmp_path, mode):
    """Whatever mode the file is in, the preprocessor must receive (H, W, 3)."""
    path = tmp_path / f"{mode}.tiff"
    Image.new(mode, (20, 10)).save(path)
    assert load_image(path).shape == (10, 20, 3)


@pytest.mark.filterwarnings("ignore:Palette images with Transparency")
def test_load_image_handles_palette_png_with_transparency(tmp_path):
    base = Image.new("RGBA", (64, 64), color=(100, 150, 200, 128))
    path = tmp_path / "palette_transparency.png"
    base.convert("P").save(path, format="PNG")
    assert load_image(path).shape == (64, 64, 3)


def test_load_image_tolerates_truncated_jpeg(tmp_path):
    """A JPEG with its last bytes cut off still yields pixels (LOAD_TRUNCATED_IMAGES)."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(buf, format="JPEG")
    full = buf.getvalue()
    assert len(full) > 64
    path = tmp_path / "truncated.jpg"
    path.write_bytes(full[:-32])
    assert load_image(path).shape == (64, 64, 3)


def test_load_image_raises_classifier_error_for_unreadable_file(tmp_path):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ClassifierError):
        load_image(bad)


def test_load_image_raises_classifier_error_for_missing_file(tmp_path):
    with pytest.raises(ClassifierError):
        load_image(tmp_path / "ghost.jpg")


def test_load_image_turns_decompression_bombs_into_classifier_errors(tmp_path):
    path = make_image(tmp_path / "a.jpg")
    with patch(
        "image_classifier.classifier.Image.open",
        side_effect=Image.DecompressionBombError("too many pixels"),
    ):
        with pytest.raises(ClassifierError, match="too many pixels"):
            load_image(path)


# --- preprocess / score_batch ---


def test_preprocess_returns_unbatched_tensor(fake_preprocessor):
    arr = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    assert preprocess(arr, fake_preprocessor).shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_score_batch_returns_one_score_per_image_in_one_forward_pass():
    model = FakeModel()
    dark = torch.zeros(3, 4, 4)
    bright = torch.ones(3, 4, 4)
    scores = score_batch([dark, bright, dark], model, CPU)
    assert model.batch_sizes == [3]
    assert scores == pytest.approx([1.0, 10.0, 1.0], abs=0.1)
    assert all(isinstance(s, float) for s in scores)


def test_score_batch_rejects_a_model_returning_the_wrong_number_of_scores():
    model = MagicMock(
        return_value=SimpleNamespace(logits=torch.zeros(2, 1)),
    )
    with pytest.raises(ClassifierError, match="2 scores for 3 images"):
        score_batch([torch.zeros(3, 4, 4)] * 3, model, CPU)


def test_score_batch_runs_the_model_in_bfloat16():
    model = FakeModel()
    seen = []
    original = model.__call__

    def spy(pixel_values):
        seen.append(pixel_values.dtype)
        return original(pixel_values)

    score_batch([torch.zeros(3, 4, 4)], spy, CPU)
    assert seen == [torch.bfloat16]


# --- score_image ---


def test_score_image_returns_tuple_of_score_and_timings(
    tmp_path, fake_model, fake_preprocessor
):
    path = make_image(tmp_path / "test.jpg", level=255)
    score, timings = score_image(path, fake_model, fake_preprocessor, CPU)
    assert isinstance(score, float)
    assert score == pytest.approx(10.0, abs=0.1)
    assert isinstance(timings, Timings)


def test_score_image_timings_are_non_negative_and_total_matches_sum(
    tmp_path, fake_model, fake_preprocessor
):
    path = make_image(tmp_path / "test.jpg")
    _, timings = score_image(path, fake_model, fake_preprocessor, CPU)
    assert timings.load_ms >= 0
    assert timings.preprocess_ms >= 0
    assert timings.infer_ms >= 0
    assert timings.total_ms == pytest.approx(
        timings.load_ms + timings.preprocess_ms + timings.infer_ms
    )


def test_score_image_raises_classifier_error_for_unreadable_file(
    tmp_path, fake_model, fake_preprocessor
):
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ClassifierError):
        score_image(bad, fake_model, fake_preprocessor, CPU)


def test_score_image_raises_classifier_error_for_missing_file(
    tmp_path, fake_model, fake_preprocessor
):
    with pytest.raises(ClassifierError):
        score_image(tmp_path / "ghost.jpg", fake_model, fake_preprocessor, CPU)


def test_score_image_rejects_non_finite_model_output(tmp_path, fake_preprocessor):
    nan_model = MagicMock(
        return_value=SimpleNamespace(logits=torch.full((1, 1), float("nan")))
    )
    path = make_image(tmp_path / "a.jpg")
    with pytest.raises(ClassifierError, match="non-finite"):
        score_image(path, nan_model, fake_preprocessor, CPU)


def test_timings_total_includes_all_phases():
    t = Timings(
        load_ms=10,
        preprocess_ms=20,
        infer_ms=500,
        upsert_ms=2,
        exiftool_ms=50,
        xattr_ms=5,
    )
    assert abs(t.total_ms - 587.0) < 0.01


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


def test_load_model_shim_unwraps_a_full_siglip_config(monkeypatch):
    """The patched __init__ must hand the model's vision_config, not the full SiglipConfig."""
    received = []

    class FakeAestheticModel:
        def __init__(self, config, *args, **kwargs):
            received.append(config)

    class FakeSiglipConfig:
        def __init__(self):
            self.vision_config = object()

    fake_ap = MagicMock()
    fake_ap.convert_v2_5_from_siglip.return_value = (MagicMock(), MagicMock())
    fake_siglip_mod = MagicMock(AestheticPredictorV2_5Model=FakeAestheticModel)
    fake_config_mod = MagicMock(SiglipConfig=FakeSiglipConfig)
    monkeypatch.setitem(sys.modules, "aesthetic_predictor_v2_5", fake_ap)
    monkeypatch.setitem(
        sys.modules, "aesthetic_predictor_v2_5.siglip_v2_5", fake_siglip_mod
    )
    monkeypatch.setitem(
        sys.modules, "transformers.models.siglip.configuration_siglip", fake_config_mod
    )
    device = MagicMock(type="cpu")
    load_model(device)

    full = FakeSiglipConfig()
    FakeAestheticModel(full)
    already_vision = object()
    FakeAestheticModel(already_vision)

    assert received == [full.vision_config, already_vision]


def test_load_model_returns_the_model_in_bfloat16_on_the_device(monkeypatch):
    model = MagicMock()
    model.to.return_value = model
    fake_ap = MagicMock()
    fake_ap.convert_v2_5_from_siglip.return_value = (model, "preprocessor")
    fake_siglip_mod = MagicMock(AestheticPredictorV2_5Model=type("M", (), {}))
    monkeypatch.setitem(sys.modules, "aesthetic_predictor_v2_5", fake_ap)
    monkeypatch.setitem(
        sys.modules, "aesthetic_predictor_v2_5.siglip_v2_5", fake_siglip_mod
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.siglip.configuration_siglip",
        MagicMock(SiglipConfig=type("C", (), {})),
    )
    device = MagicMock(type="cpu")

    result_model, result_preprocessor = load_model(device)

    assert result_model is model and result_preprocessor == "preprocessor"
    assert [c.args[0] for c in model.to.call_args_list] == [torch.bfloat16, device]


# --- HEIC support ---


def _write_heic_fixture(path) -> None:
    """Write a minimal real HEIC file to *path* using pillow_heif."""
    import pillow_heif

    pil_img = Image.new("RGB", (16, 16), color=(255, 0, 128))
    pillow_heif.from_pillow(pil_img).save(path)


@pytest.mark.heic
def test_load_image_heic_raises_without_registration(tmp_path):
    """load_image must raise ClassifierError when PIL cannot identify the HEIC file.

    Reproduces the original bug report: Image.open raises UnidentifiedImageError
    on a .heic path when pillow_heif has not been registered.
    """
    heic_path = tmp_path / "fixture.heic"
    _write_heic_fixture(heic_path)

    def _fake_open(path, *args, **kwargs):
        raise UnidentifiedImageError(f"cannot identify image file '{path}'")

    with patch("image_classifier.classifier.Image.open", side_effect=_fake_open):
        with pytest.raises(ClassifierError, match="cannot identify image file"):
            load_image(heic_path)


@pytest.mark.heic
def test_score_image_heic_succeeds_with_registration(
    tmp_path, fake_model, fake_preprocessor
):
    """Importing image_classifier.classifier registers the HEIF opener, so a real
    HEIC file (no mocking of the image load) scores like any other image."""
    heic_path = tmp_path / "fixture.heic"
    _write_heic_fixture(heic_path)

    score, timings = score_image(heic_path, fake_model, fake_preprocessor, CPU)

    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0
    assert timings.load_ms >= 0
