from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from image_classifier.database import make_connection
from tests.helpers import FakeModel, FakePreprocessor


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """In-memory SQLite connection with schema applied."""
    c = make_connection(":memory:")
    yield c
    c.close()


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def fake_preprocessor() -> FakePreprocessor:
    return FakePreprocessor()
