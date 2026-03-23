# Image Classifier — Design Spec

**Date:** 2026-03-22
**Status:** Approved

---

## Overview

A command-line tool that scores images in a folder using the `aesthetic-predictor-v2-5` library, persists results to SQLite so classification only runs once, and writes XMP star ratings directly into each image's metadata so macOS Finder can sort by aesthetic score natively.

---

## Goals

- Score images once; never re-process unless explicitly requested
- Write XMP Rating (1–5 stars) to image metadata for native Finder sorting
- Show clear progress while the model runs
- Surface errors without crashing the entire run
- Stay simple: no unnecessary abstractions, no interactive review UI

---

## Non-Goals

- Interactive image review inside the tool
- Moving or copying files
- Support for video or non-image files
- Cloud storage or remote processing

---

## Usage

```bash
# Install system dependency
brew install exiftool

# Install and sync project
uv sync

# Score all unprocessed images in a folder (skips already-scored)
uv run classify ~/Photos/vacation

# Force re-score all images (re-runs model AND re-writes XMP metadata for every image)
uv run classify ~/Photos/vacation --force

# Scan subdirectories recursively
uv run classify ~/Photos --recursive
```

---

## Architecture

Four modules, each with one responsibility:

### `classifier.py`
- Loads `aesthetic-predictor-v2-5` model once at startup
- Detects device: MPS (Apple Silicon) → CPU fallback; prints a one-time notice if falling back to CPU
- First-run model download: on first use, the underlying HuggingFace checkpoint (~1–2 GB) is downloaded to the HuggingFace cache (`~/.cache/huggingface`). This is printed as a notice before the progress bar starts. Subsequent runs use the cached model.
- Accepts a `Path`, returns a `float` score (1.0–10.0)
- Skips unreadable/corrupt files by raising a typed `ClassifierError` for the caller to handle

### `database.py`
- Thin wrapper around `sqlite3` (stdlib, no ORM)
- Paths are stored as **absolute resolved paths** (`Path.resolve()`) before any lookup or write, ensuring cache hits regardless of how the user invokes the tool (relative path, symlink, `~` expansion, etc.)
- DB file is stored in the **user's home data directory** (`~/.local/share/image-classifier/classify.db`) rather than alongside the scanned folder, avoiding failures on read-only volumes, network mounts, or SD cards
- Schema:
  ```sql
  CREATE TABLE IF NOT EXISTS images (
      path         TEXT PRIMARY KEY,
      score        REAL NOT NULL,
      rating       INTEGER NOT NULL,
      processed_at TEXT NOT NULL
  );
  ```
- Operations: `is_processed(path: Path) -> bool`, `upsert(path: Path, score: float, rating: int)`, `all_scores(folder: Path) -> list[sqlite3.Row]`
- `all_scores` is scoped to images whose resolved path starts with the given folder's resolved path (normalized with a trailing `/` before comparison, e.g. `/home/user/Photos/` will not accidentally match `/home/user/Photos-backup/img.jpg`), so the distribution histogram reflects all previously-scored images in that folder (not just the current run)
- `sqlite3.Row` objects are used as return types (support key-based access, e.g., `row["score"]`)
- `processed_at` is stored as an ISO 8601 UTC string using `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")` (avoids the deprecated `datetime.utcnow()`)

### `metadata.py`
- Checks for `exiftool` on PATH at startup; exits immediately with `brew install exiftool` message if missing
- Writes `XMP:Rating` via `exiftool` subprocess call
- Treats non-zero exit codes as errors: logged to `classify.log`, non-fatal
- Files that error during scoring are **not tagged** (no XMP Rating is written or cleared for them)
- On a `--force` run where a previously-tagged image subsequently errors, the existing XMP Rating is **left unchanged** (tool does not clear metadata it cannot re-compute)

### `main.py`
- CLI entry point via `argparse`; flags: `--force`, `--recursive`
- Wires classifier → database → metadata in a single explicit sequential loop (single-threaded by design — the MPS model is not thread-safe and parallelism is out of scope)
- Drives `rich` progress bar: current file, live score, elapsed time
- Progress bar denominator: number of images to process in this run (skipped files are excluded). On `--force`, denominator is the total image count in the scanned folder(s).
- Prints summary table and distribution histogram on completion
- Logs errors to `~/.local/share/image-classifier/classify.log` (same location as the DB, for consistency and to avoid write failures on read-only targets or network mounts). Log file is opened in **append mode** at startup — subsequent runs accumulate into the same file.

---

## Score → Star Rating Mapping

Boundary rule: ranges are **half-open on the right** — i.e., `[low, high)`. The top range `[8.5, 10.0]` is fully closed.

| Aesthetic Score    | XMP Rating | Display   |
|--------------------|-----------|-----------|
| [1.0, 4.0)         | 1         | ★☆☆☆☆    |
| [4.0, 5.5)         | 2         | ★★☆☆☆    |
| [5.5, 7.0)         | 3         | ★★★☆☆    |
| [7.0, 8.5)         | 4         | ★★★★☆    |
| [8.5, 10.0]        | 5         | ★★★★★    |

Threshold rationale: the library's documentation states 5.5+ is a "great" aesthetic score, which maps naturally to 3+ stars.

Unrated/error cases: files that fail scoring receive no XMP Rating tag. They are not written to the database. They appear in the error count in the summary.

---

## Supported Image Formats

The tool scans for files with these extensions (case-insensitive):
`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`, `.heic`, `.bmp`

All formats are opened via Pillow and converted to RGB before scoring. Unsupported formats are silently skipped (not counted as errors).

---

## Scanning Behavior

- **Default:** flat scan — only files directly in the specified folder, no subdirectories
- **`--recursive` flag:** uses `Path.rglob()` to include all nested subdirectories
- Hidden files and directories (names starting with `.`) are skipped

---

## Terminal Output

```
Image Classifier
────────────────────────────────────────────────
Note: Downloading model checkpoint (~1.5 GB) on first run...
Loading model... [MPS]

Scanning /Users/you/Photos/vacation
  Found 247 images (12 already scored, 235 to process)

Processing ━━━━━━━━━━━━━━━━━━░░░░░ 142/235  60%
  Current: IMG_4821.jpg
  Score:   7.34  ★★★★☆

────────────────────────────────────────────────
Done in 4m 32s
  Scored:   235 images
  Skipped:   12 (already in database)
  Errors:     1 (see classify.log)

  Distribution:
  ★★★★★  (8.5+)    18 images  ████
  ★★★★☆  (7–8.5)   41 images  ████████
  ★★★☆☆  (5.5–7)   89 images  ████████████████
  ★★☆☆☆  (4–5.5)   63 images  ████████████
  ★☆☆☆☆  (<4)      24 images  █████
```

---

## Project Structure

```
image-classifier/
├── src/
│   └── image_classifier/
│       ├── __init__.py
│       ├── main.py          # CLI entry point
│       ├── classifier.py    # Model loading + scoring
│       ├── database.py      # SQLite operations
│       └── metadata.py      # exiftool XMP writing
├── tests/
│   ├── test_classifier.py
│   ├── test_database.py
│   ├── test_metadata.py
│   └── test_main.py
├── pyproject.toml
└── README.md
```

---

## Dependencies

**Runtime:**
- `aesthetic-predictor-v2-5`
- `torch>=2.0` (MPS support requires 2.0+; device detected via `torch.backends.mps.is_available()`)
- `Pillow`
- `rich`

**System (via Homebrew):**
- `exiftool`

**Dev:**
- `pytest`
- `pytest-cov`

---

## `pyproject.toml` Structure

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
    "pytest",
    "pytest-cov",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Design Principles (Zen of Python Applied)

- **Explicit over implicit** — device selection, path handling, and error states are always named and visible
- **Simple over complex** — no ORM, no async, no plugin system; plain functions and stdlib where possible
- **Flat over nested** — `main.py` is a linear loop, not a class hierarchy
- **Errors should never pass silently** — all exceptions are caught, logged, and counted; the summary always shows error count
- **One obvious way** — `uv run classify <path>` is the single entry point; no aliases or subcommands beyond the two defined flags
- **Readability counts** — `pathlib.Path` everywhere, type hints on all public functions, no abbreviations in names
- **Single-threaded by design** — MPS model inference is not thread-safe; sequential processing is deliberate and not to be parallelized

---

## Testing Strategy

- **`test_database.py`** — unit tests with an in-memory SQLite DB; test upsert, skip logic, schema creation, path resolution
- **`test_metadata.py`** — mock `subprocess.run`; test exiftool argument construction, error handling, no-write-on-error behavior
- **`test_classifier.py`** — mock the model; test score→rating mapping (including exact boundary values), device detection logic, file-skip behavior, `ClassifierError` propagation
- **`test_main.py`** — integration-level tests mocking all four modules; test `--force` flag wiring, `--recursive` scanning, error counting, skip counting, summary output
- Target: 80%+ coverage

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `exiftool` not installed | Exit at startup with clear `brew install exiftool` message |
| Unreadable/corrupt image | Log to `classify.log`, increment error count, continue |
| Model fails on an image | Log to `classify.log`, increment error count, continue |
| MPS unavailable | Fall back to CPU automatically, print one-time notice before progress bar |
| DB write fails | Raise immediately — DB integrity is required, cannot continue |
| Target folder is read-only | Not an issue — DB is stored in `~/.local/share/image-classifier/` |
| `classify.log` write fails | Print warning to stderr, continue (log is best-effort) |
| Target folder does not exist | Exit immediately with clear message: `Error: folder not found: <path>` |
| Target path is a file, not a folder | Exit immediately with clear message: `Error: expected a folder, got a file: <path>` |

---

## `classify.log` Format

Location: `~/.local/share/image-classifier/classify.log`
Mode: append across runs (never truncated). If the log file cannot be opened at startup, the handle is treated as `None` and all subsequent log writes are silently skipped for the duration of that run.
Format: plain text, tab-separated, one line per error:

```
<ISO 8601 UTC timestamp>\t<absolute image path>\t<exception type>: <message>
```

Example:
```
2026-03-22T14:03:11Z	/Users/you/Photos/vacation/IMG_0042.jpg	OSError: cannot identify image file
```

Timestamps use `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")` (avoids the deprecated `datetime.utcnow()`).
