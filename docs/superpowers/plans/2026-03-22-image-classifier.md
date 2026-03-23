# Image Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that scores images aesthetically, caches results in SQLite, and writes XMP star ratings to image metadata for native Finder sorting.

**Architecture:** Four focused modules (database, metadata, classifier, main) wired together in a single sequential loop driven by a `rich` progress bar. Model loads once per run; SQLite prevents re-processing. All state lives in `~/.local/share/image-classifier/`.

**Tech Stack:** Python 3.11+, uv, aesthetic-predictor-v2-5, PyTorch (MPS), Pillow, rich, exiftool (system), sqlite3 (stdlib), pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, `classify` entry point |
| `src/image_classifier/__init__.py` | Empty package marker |
| `src/image_classifier/database.py` | SQLite connection, schema, CRUD operations |
| `src/image_classifier/metadata.py` | exiftool detection and XMP Rating writes |
| `src/image_classifier/classifier.py` | Model loading, device detection, image scoring, score→rating mapping |
| `src/image_classifier/main.py` | CLI entry point, folder scanning, progress loop, summary |
| `tests/test_database.py` | Unit tests for all DB operations (in-memory SQLite) |
| `tests/test_metadata.py` | Unit tests for exiftool integration (mocked subprocess) |
| `tests/test_classifier.py` | Unit tests for scoring, rating mapping, device detection (mocked model) |
| `tests/test_main.py` | Integration tests for CLI wiring, flags, error counting (all deps mocked) |
| `README.md` | Installation and usage instructions |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/image_classifier/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p src/image_classifier tests
touch src/image_classifier/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "image-classifier"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "aesthetic-predictor-v2-5",
    "torch>=2.0",
    "Pillow",
    "rich",
]

[project.scripts]
classify = "image_classifier.main:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/image_classifier"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install system dependency**

```bash
brew install exiftool
```

- [ ] **Step 4: Sync dependencies**

```bash
uv sync
```

Expected: uv resolves and installs all packages. `aesthetic-predictor-v2-5` pulls in torch, transformers, etc. This may take a few minutes on first run.

- [ ] **Step 5: Verify install**

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected: `True` on Apple Silicon M1.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "chore: scaffold project with uv and pyproject.toml"
```

---

## Task 2: Database Module

**Files:**
- Create: `src/image_classifier/database.py`
- Create: `tests/test_database.py`

### Step-by-step (TDD)

- [ ] **Step 1: Write failing tests**

Create `tests/test_database.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from image_classifier.database import all_scores, is_processed, make_connection, upsert


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema applied."""
    c = make_connection(":memory:")
    yield c
    c.close()


def test_schema_created(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='images'"
    ).fetchone()
    assert row is not None


def test_is_processed_false_for_new_path(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    assert is_processed(img, conn) is False


def test_upsert_and_is_processed(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=7.3, rating=4, conn=conn)
    assert is_processed(img, conn) is True


def test_upsert_stores_correct_values(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=6.1, rating=3, conn=conn)
    row = conn.execute("SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)).fetchone()
    assert abs(row["score"] - 6.1) < 0.001
    assert row["rating"] == 3


def test_upsert_replaces_on_force(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=4.0, rating=2, conn=conn)
    upsert(img, score=8.9, rating=5, conn=conn)
    row = conn.execute("SELECT score, rating FROM images WHERE path = ?", (str(img.resolve()),)).fetchone()
    assert row["rating"] == 5


def test_path_stored_as_resolved_absolute(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=5.0, rating=2, conn=conn)
    rows = conn.execute("SELECT path FROM images").fetchall()
    assert rows[0]["path"] == str(img.resolve())


def test_all_scores_scoped_to_folder(conn, tmp_path):
    folder_a = tmp_path / "vacation"
    folder_b = tmp_path / "vacation-backup"
    folder_a.mkdir()
    folder_b.mkdir()
    img_a = folder_a / "a.jpg"
    img_b = folder_b / "b.jpg"
    img_a.touch()
    img_b.touch()
    upsert(img_a, score=7.0, rating=4, conn=conn)
    upsert(img_b, score=8.0, rating=5, conn=conn)
    results = all_scores(folder_a, conn)
    assert len(results) == 1
    assert results[0]["rating"] == 4


def test_all_scores_returns_all_in_folder(conn, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(3):
        img = folder / f"img_{i}.jpg"
        img.touch()
        upsert(img, score=float(i + 5), rating=3, conn=conn)
    results = all_scores(folder, conn)
    assert len(results) == 3


def test_processed_at_is_iso8601_utc(conn, tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    upsert(img, score=5.5, rating=3, conn=conn)
    row = conn.execute("SELECT processed_at FROM images").fetchone()
    assert row["processed_at"].endswith("Z")
    assert "T" in row["processed_at"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_database.py -v
```

Expected: `ImportError` — `database` module doesn't exist yet.

- [ ] **Step 3: Write `src/image_classifier/database.py`**

```python
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.db"


def make_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            path         TEXT PRIMARY KEY,
            score        REAL NOT NULL,
            rating       INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def is_processed(path: Path, conn: sqlite3.Connection) -> bool:
    """Return True if the image at path has already been scored."""
    row = conn.execute(
        "SELECT 1 FROM images WHERE path = ?",
        (str(path.resolve()),),
    ).fetchone()
    return row is not None


def upsert(path: Path, score: float, rating: int, conn: sqlite3.Connection) -> None:
    """Insert or replace a scored image record."""
    processed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT OR REPLACE INTO images (path, score, rating, processed_at) VALUES (?, ?, ?, ?)",
        (str(path.resolve()), score, rating, processed_at),
    )
    conn.commit()


def all_scores(folder: Path, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all scored images whose path is inside folder (trailing-slash safe)."""
    prefix = str(folder.resolve()) + "/"
    return conn.execute(
        "SELECT path, score, rating FROM images WHERE path LIKE ?",
        (prefix + "%",),
    ).fetchall()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_database.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Check coverage**

```bash
uv run pytest tests/test_database.py --cov=image_classifier.database --cov-report=term-missing
```

Expected: 90%+ coverage.

- [ ] **Step 6: Commit**

```bash
git add src/image_classifier/database.py tests/test_database.py
git commit -m "feat: add SQLite database module with TDD"
```

---

## Task 3: Metadata Module

**Files:**
- Create: `src/image_classifier/metadata.py`
- Create: `tests/test_metadata.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_metadata.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_classifier.metadata import MetadataError, check_exiftool, write_rating


def test_check_exiftool_exits_when_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            check_exiftool()
    assert "brew install exiftool" in str(exc_info.value)


def test_check_exiftool_passes_when_present():
    with patch("shutil.which", return_value="/usr/local/bin/exiftool"):
        check_exiftool()  # Should not raise


def test_write_rating_calls_exiftool_with_correct_args(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        write_rating(img, rating=4)
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "exiftool" in call_args
    assert "-XMP:Rating=4" in call_args
    assert "-overwrite_original" in call_args
    assert str(img) in call_args


def test_write_rating_raises_metadata_error_on_nonzero_exit(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Error: something went wrong"
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(MetadataError) as exc_info:
            write_rating(img, rating=3)
    assert "photo.jpg" in str(exc_info.value)


def test_write_rating_all_valid_ratings(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        for rating in range(1, 6):
            write_rating(img, rating=rating)
            call_args = mock_run.call_args[0][0]
            assert f"-XMP:Rating={rating}" in call_args
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_metadata.py -v
```

Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Write `src/image_classifier/metadata.py`**

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MetadataError(Exception):
    """Raised when exiftool fails to write metadata."""


def check_exiftool() -> None:
    """Exit with install instructions if exiftool is not on PATH."""
    if shutil.which("exiftool") is None:
        raise SystemExit(
            "Error: exiftool is required but not installed.\n"
            "Install it with: brew install exiftool"
        )


def write_rating(path: Path, rating: int) -> None:
    """Write XMP:Rating to the image file via exiftool."""
    result = subprocess.run(
        ["exiftool", f"-XMP:Rating={rating}", "-overwrite_original", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MetadataError(f"exiftool failed for {path}: {result.stderr.strip()}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_metadata.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/image_classifier/metadata.py tests/test_metadata.py
git commit -m "feat: add metadata module with exiftool XMP Rating writes"
```

---

## Task 4: Classifier Module

**Files:**
- Create: `src/image_classifier/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
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
    """Return a mock preprocessor."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_classifier.py -v
```

Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Write `src/image_classifier/classifier.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_classifier.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Check coverage**

```bash
uv run pytest tests/test_classifier.py --cov=image_classifier.classifier --cov-report=term-missing
```

Expected: 85%+ (load_model excluded from unit tests as it downloads a real model).

- [ ] **Step 6: Commit**

```bash
git add src/image_classifier/classifier.py tests/test_classifier.py
git commit -m "feat: add classifier module with MPS device detection and score mapping"
```

---

## Task 5: Main Entry Point

**Files:**
- Create: `src/image_classifier/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_main.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from image_classifier.main import scan_images, score_to_rating, star_display


# --- star_display ---

def test_star_display_one():
    assert star_display(1) == "★☆☆☆☆"


def test_star_display_five():
    assert star_display(5) == "★★★★★"


def test_star_display_three():
    assert star_display(3) == "★★★☆☆"


# --- scan_images ---

def test_scan_images_returns_supported_extensions(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.mp4").touch()  # not an image
    (tmp_path / "d.txt").touch()  # not an image
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert names == {"a.jpg", "b.png"}


def test_scan_images_skips_hidden_files(tmp_path):
    (tmp_path / ".hidden.jpg").touch()
    (tmp_path / "visible.jpg").touch()
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert ".hidden.jpg" not in names
    assert "visible.jpg" in names


def test_scan_images_flat_by_default(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.jpg").touch()
    (sub / "nested.jpg").touch()
    results = scan_images(tmp_path, recursive=False)
    names = {p.name for p in results}
    assert "top.jpg" in names
    assert "nested.jpg" not in names


def test_scan_images_recursive_flag(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.jpg").touch()
    (sub / "nested.jpg").touch()
    results = scan_images(tmp_path, recursive=True)
    names = {p.name for p in results}
    assert "top.jpg" in names
    assert "nested.jpg" in names


def test_scan_images_case_insensitive_extensions(tmp_path):
    (tmp_path / "photo.JPG").touch()
    (tmp_path / "photo.PNG").touch()
    results = scan_images(tmp_path, recursive=False)
    assert len(results) == 2


# --- CLI integration tests ---

def _make_mock_conn():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    conn.execute.return_value.fetchall.return_value = []
    return conn


def test_main_exits_on_missing_folder(tmp_path):
    missing = tmp_path / "does_not_exist"
    with patch("sys.argv", ["classify", str(missing)]):
        with pytest.raises(SystemExit) as exc_info:
            from image_classifier.main import main
            main()
    assert exc_info.value.code == 1


def test_main_exits_on_file_instead_of_folder(tmp_path):
    f = tmp_path / "image.jpg"
    f.touch()
    with patch("sys.argv", ["classify", str(f)]):
        with pytest.raises(SystemExit) as exc_info:
            from image_classifier.main import main
            main()
    assert exc_info.value.code == 1


def test_main_skips_already_processed_images(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()
    # Simulate image already in DB
    mock_conn.execute.return_value.fetchone.return_value = {"path": str(img)}

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.score_image") as mock_score,
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier import main as main_module
        import importlib
        importlib.reload(main_module)
        main_module.main()

    mock_score.assert_not_called()


def test_main_force_flag_rescores_all_images(tmp_path):
    img = tmp_path / "photo.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    with (
        patch("sys.argv", ["classify", str(tmp_path), "--force"]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.score_image", return_value=7.0) as mock_score,
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier import main as main_module
        import importlib
        importlib.reload(main_module)
        main_module.main()

    mock_score.assert_called_once()


def test_main_counts_errors_without_crashing(tmp_path):
    img = tmp_path / "bad.jpg"
    img.touch()

    mock_conn = _make_mock_conn()

    from image_classifier.classifier import ClassifierError

    with (
        patch("sys.argv", ["classify", str(tmp_path)]),
        patch("image_classifier.main.check_exiftool"),
        patch("image_classifier.main.setup_log", return_value=None),
        patch("image_classifier.main.get_device", return_value=MagicMock()),
        patch("image_classifier.main.load_model", return_value=(MagicMock(), MagicMock())),
        patch("image_classifier.main.make_connection", return_value=mock_conn),
        patch("image_classifier.main.score_image", side_effect=ClassifierError("bad image")),
        patch("image_classifier.main.write_rating"),
        patch("image_classifier.main.upsert"),
        patch("image_classifier.main.all_scores", return_value=[]),
    ):
        from image_classifier import main as main_module
        import importlib
        importlib.reload(main_module)
        # Should complete without raising
        main_module.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py -v
```

Expected: `ImportError` — main module doesn't exist yet.

- [ ] **Step 3: Write `src/image_classifier/main.py`**

```python
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from image_classifier.classifier import (
    SUPPORTED_EXTENSIONS,
    ClassifierError,
    get_device,
    load_model,
    score_image,
    score_to_rating,
)
from image_classifier.database import DB_PATH, all_scores, is_processed, make_connection, upsert
from image_classifier.metadata import MetadataError, check_exiftool, write_rating

if TYPE_CHECKING:
    import logging
    import sqlite3

LOG_PATH = Path.home() / ".local" / "share" / "image-classifier" / "classify.log"

console = Console()


def setup_log() -> logging.Logger | None:
    """Open the append-mode log file. Returns None if it cannot be opened."""
    import logging

    logger = logging.getLogger("classify")
    logger.setLevel(logging.ERROR)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(LOG_PATH), mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger
    except OSError:
        console.print(
            f"[yellow]Warning: could not open log file at {LOG_PATH}[/yellow]",
            file=sys.stderr,
        )
        return None


def log_error(logger: logging.Logger | None, path: Path, exc: Exception) -> None:
    """Write a tab-separated error line to the log."""
    if logger is None:
        return
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    logger.error("%s\t%s\t%s: %s", timestamp, path, type(exc).__name__, exc)


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


def star_display(rating: int) -> str:
    """Return a unicode star string for the given 1–5 rating."""
    return "★" * rating + "☆" * (5 - rating)


def print_summary(
    scored: int,
    skipped: int,
    errors: int,
    folder: Path,
    conn: sqlite3.Connection,
) -> None:
    console.rule()
    console.print(f"  Scored:  {scored:>4} images")
    console.print(f"  Skipped: {skipped:>4} (already in database)")
    error_line = f"  Errors:  {errors:>4} (see {LOG_PATH})" if errors else f"  Errors:  {errors:>4}"
    console.print(error_line)

    rows = all_scores(folder, conn)
    if not rows:
        return

    console.print()
    buckets: dict[int, int] = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for row in rows:
        buckets[row["rating"]] = buckets.get(row["rating"], 0) + 1

    labels = {5: "8.5+", 4: "7–8.5", 3: "5.5–7", 2: "4–5.5", 1: "<4"}
    console.print("  Distribution:")
    for stars in (5, 4, 3, 2, 1):
        count = buckets[stars]
        bar = "█" * min(count, 30)
        console.print(f"  {star_display(stars)}  ({labels[stars]:>6})  {count:>4} images  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score images aesthetically and write XMP star ratings."
    )
    parser.add_argument("folder", type=Path, help="Folder of images to classify")
    parser.add_argument("--force", action="store_true", help="Re-score already-processed images")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()

    if not folder.exists():
        console.print(f"[red]Error: folder not found: {folder}[/red]")
        sys.exit(1)

    if not folder.is_dir():
        console.print(f"[red]Error: expected a folder, got a file: {folder}[/red]")
        sys.exit(1)

    check_exiftool()
    logger = setup_log()

    console.rule("Image Classifier")
    console.print(
        "Note: On first run the model checkpoint (~1.5 GB) will be downloaded "
        "to ~/.cache/huggingface. Subsequent runs use the cache."
    )
    console.print("Loading model...", end=" ")

    device = get_device()
    if device.type == "cpu":
        console.print("[yellow][CPU — MPS not available][/yellow]")
    else:
        console.print(f"[green][{device.type.upper()}][/green]")

    model, preprocessor = load_model(device)
    console.print()

    images = scan_images(folder, args.recursive)
    conn = make_connection()

    to_process = images if args.force else [p for p in images if not is_processed(p, conn)]
    skipped = len(images) - len(to_process)

    console.print(f"Scanning [bold]{folder}[/bold]")
    console.print(f"  Found {len(images)} images ({skipped} already scored, {len(to_process)} to process)")
    console.print()

    scored = 0
    errors = 0

    with Progress(
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("", total=len(to_process))
        for path in to_process:
            progress.update(task_id, description=path.name)
            try:
                score = score_image(path, model, preprocessor, device)
                rating = score_to_rating(score)
                upsert(path, score, rating, conn)
                write_rating(path, rating)
                scored += 1
                progress.update(
                    task_id,
                    advance=1,
                    description=f"{path.name}  {score:.2f}  {star_display(rating)}",
                )
            except (ClassifierError, MetadataError, OSError) as exc:
                log_error(logger, path, exc)
                errors += 1
                progress.update(task_id, advance=1)

    print_summary(scored, skipped, errors, folder, conn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_main.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite with coverage**

```bash
uv run pytest --cov=image_classifier --cov-report=term-missing
```

Expected: 80%+ overall coverage.

- [ ] **Step 6: Commit**

```bash
git add src/image_classifier/main.py tests/test_main.py
git commit -m "feat: add main entry point with rich progress and CLI flags"
```

---

## Task 6: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# image-classifier

Score images aesthetically and write XMP star ratings so macOS Finder can sort them natively.

Uses [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5) (SigLIP-based, scores 1–10). Optimised for Apple Silicon (MPS).

## Install

```bash
# System dependency
brew install exiftool

# Project
uv sync
```

First run downloads the model checkpoint (~1.5 GB) to `~/.cache/huggingface`.

## Usage

```bash
# Score all images in a folder (skips already-scored)
uv run classify ~/Photos/vacation

# Re-score everything (overwrites existing XMP ratings)
uv run classify ~/Photos/vacation --force

# Include subdirectories
uv run classify ~/Photos --recursive
```

## Star rating scale

| Score | Stars | Meaning |
|-------|-------|---------|
| 8.5–10 | ★★★★★ | Exceptional |
| 7–8.5 | ★★★★☆ | Great |
| 5.5–7 | ★★★☆☆ | Good |
| 4–5.5 | ★★☆☆☆ | Below average |
| < 4 | ★☆☆☆☆ | Poor |

## Files created

| Path | Contents |
|------|----------|
| `~/.local/share/image-classifier/classify.db` | SQLite cache of scores |
| `~/.local/share/image-classifier/classify.log` | Error log (tab-separated, append mode) |

## Sort in Finder

After running, open the folder in Finder → View → as List → right-click the column header → enable **Rating**. Sort by Rating descending to see your best shots first.

## Development

```bash
uv run pytest
uv run pytest --cov=image_classifier --cov-report=term-missing
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install and usage instructions"
```

---

## Task 7: Smoke Test End-to-End

- [ ] **Step 1: Point at a real folder with a few images**

```bash
uv run classify ~/Pictures --recursive
```

Expected: Progress bar appears, images scored, XMP ratings written.

- [ ] **Step 2: Verify XMP Rating in Finder**

Open the folder in Finder → View as List → right-click column header → add Rating column. Confirm stars appear.

- [ ] **Step 3: Verify SQLite cache works (re-run is fast)**

```bash
uv run classify ~/Pictures --recursive
```

Expected: All images skipped ("already in database"), no model inference runs.

- [ ] **Step 4: Verify exiftool on a specific file**

```bash
exiftool -XMP:Rating ~/Pictures/some_photo.jpg
```

Expected: `Rating: N` where N is 1–5.
