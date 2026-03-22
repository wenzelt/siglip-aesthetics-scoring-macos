from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from image_classifier.classifier import (
    SUPPORTED_EXTENSIONS,
    ClassifierError,
    get_device,
    score_image,
    score_to_rating,
)


# --- score_to_rating ---

@pytest.mark.parametrize("score,expected_rating", [
    (1.0, 1),   # well below threshold
    (3.9, 1),   # just below 4.0
    (4.0, 2),   # exactly at 4.0 boundary — goes to 2
    (5.4, 2),   # just below 5.5
    (5.5, 3),   # exactly at 5.5 boundary — goes to 3
    (6.9, 3),   # just below 7.0
    (7.0, 4),   # exactly at 7.0 boundary — goes to 4
    (8.4, 4),   # just below 8.5
    (8.5, 5),   # exactly at 8.5 boundary — goes to 5
    (10.0, 5),  # maximum
])
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
    mock_output.logits.squeeze.return_value.float.return_value.cpu.return_value.numpy.return_value = np.float32(score_value)

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


def test_score_image_returns_float(tmp_path):
    from PIL import Image as PILImage

    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    model = _make_mock_model(7.25)
    preprocessor = _make_mock_preprocessor()
    device = torch.device("cpu")

    result = score_image(img_path, model, preprocessor, device)
    assert isinstance(result, float)
    assert abs(result - 7.25) < 0.01


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
