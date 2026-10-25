from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import image_classifier


def test_version_comes_from_the_installed_package_metadata():
    with patch("importlib.metadata.version", return_value="9.9.9"):
        reloaded = importlib.reload(image_classifier)
    try:
        assert reloaded.__version__ == "9.9.9"
    finally:
        importlib.reload(image_classifier)


def test_version_falls_back_when_the_package_is_not_installed():
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        reloaded = importlib.reload(image_classifier)
    try:
        assert reloaded.__version__ == "0+unknown"
    finally:
        importlib.reload(image_classifier)
