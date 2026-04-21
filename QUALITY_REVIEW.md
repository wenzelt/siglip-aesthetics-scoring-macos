# Code Quality Review: Image Classifier

## Overview
This review assesses the `image-classifier` project across five axes: correctness, readability, architecture, security, and performance.

## Findings

### 1. Correctness
- **Status:** Good.
- **Observations:**
    - Handles truncated JPEGs and unusual channel counts in images.
    - Correctly uses `register_heif_opener()` for HEIC support.
    - Uses half-open intervals for `score_to_rating`.
    - Restores file modification time after metadata updates.

### 2. Readability & Simplicity
- **Status:** Acceptable, but `main.py` is bloated.
- **Observations:**
    - Well-commented code, especially complex shims and timing logic.
    - Clear function names and docstrings in most modules.
    - `main.py` handles CLI parsing, logging, scanning, and processing, which makes it harder to read and test.

### 3. Architecture
- **Status:** Needs Improvement (Refactoring).
- **Observations:**
    - `main.py` has too many responsibilities (God Object pattern).
    - Database connection management in `main.py` is brittle (manual retry on `OperationalError`).
    - Logic for "process one image" is coupled with the progress bar in `main.py`.

### 4. Security
- **Status:** Good.
- **Observations:**
    - Uses parameterized SQL queries everywhere.
    - Uses list-based arguments for `subprocess.run` (prevents shell injection).
    - `trust_remote_code=True` in `load_model` is necessary for the specific HF model but should be noted.

### 5. Performance
- **Status:** Good.
- **Observations:**
    - Uses `bfloat16` and `inference_mode` for GPU/MPS efficiency.
    - Uses WAL mode for SQLite.
    - Timing profiles are captured and reported.

## Proposed Improvements

1.  **Refactor `main.py`:**
    - Move logging setup to `logging.py`.
    - Move image scanning to `scanner.py` or `utils.py`.
    - Create a `Processor` class or a separate module to handle the image processing loop, decoupling it from CLI and UI (rich).
2.  **Robust Database Management:**
    - Centralize database operations and improve connection handling (e.g., using a context manager or a more robust retry mechanism).
3.  **Enhance `Timings` Integration:**
    - Let the `Processor` manage `Timings` more cleanly.
4.  **Add Type Hints:**
    - Ensure all functions have complete type hints (some are missing or use `Any`).

## Action Plan
1. [x] Create `logging_utils.py` and move logging logic there.
2. [x] Create `scanner.py` and move `scan_images` there.
3. [x] Create `processor.py` to handle the core processing loop and timing logic.
4. [x] Refactor `main.py` to use these new modules.
5. [x] Update tests to reflect these changes.
