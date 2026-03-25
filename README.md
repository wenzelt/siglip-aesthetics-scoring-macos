# image-classifier

> Score every photo in your library aesthetically. Writes XMP star ratings directly into image metadata so macOS Finder sorts your best shots to the top — automatically.

Built on [aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5) (SigLIP-based, 1–10 scale). Optimised for Apple Silicon (MPS).

---

## How it works

```
Photos on disk
      │
      ▼
  PIL decode  ──▶  SigLIP preprocessor  ──▶  Aesthetic model  ──▶  Score (1–10)
                                                                          │
                        ┌─────────────────────────────────────────────────┘
                        ▼
              SQLite cache          XMP:Rating (exiftool)       Finder tag (xattr)
        ~/.local/share/…/classify.db   embedded in file          visible in Finder
```

1. Scans a folder (optionally recursive) for supported image formats
2. Skips files already in the SQLite cache (resumable across runs)
3. Scores each image with the SigLIP aesthetic model on MPS/CPU
4. Persists the score to SQLite **before** writing metadata — so a crash or exiftool failure never loses a result
5. Writes an XMP star rating (1–5 ★) and a numeric Finder tag (e.g. `7.3`) to each file
6. Any per-image failure is caught, logged, and stored in a `failures` table — the run continues

---

## Star rating scale

| Score  | Stars   | Label         |
|--------|---------|---------------|
| 8.5–10 | ★★★★★ | Exceptional   |
| 7–8.5  | ★★★★☆ | Great         |
| 5.5–7  | ★★★☆☆ | Good          |
| 4–5.5  | ★★☆☆☆ | Below average |
| < 4    | ★☆☆☆☆ | Poor          |

---

## Requirements

- macOS (Finder integration) or Linux (scores only)
- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv)
- [exiftool](https://exiftool.org/)

---

## Install

```bash
brew install exiftool
uv sync
```

First run downloads the model checkpoint (~1.5 GB) to `~/.cache/huggingface`. Subsequent runs use the cache.

---

## Usage

```bash
# Score all images in a folder (skips already-scored)
uv run classify ~/Photos/vacation

# Include subdirectories
uv run classify ~/Photos --recursive

# Re-score everything (overwrites existing ratings)
uv run classify ~/Photos/vacation --force
```

### Example output

```
──────────────────── Image Classifier ────────────────────
Note: On first run the model checkpoint (~1.5 GB) will be
downloaded to ~/.cache/huggingface. Subsequent runs use the cache.
Loading model... [MPS]

Scanning /Volumes/Data/Nextcloud/Photos
  Found 47482 images (17818 already scored, 29664 to process)

████████████████████████████████████  100% IMG_9823.HEIC  0:42:17

──────────────────────────────────────────────────────────
  Scored:   29664 images
  Skipped:  17818 (already in database)
  Errors:       3 (logged to ~/.local/share/image-classifier/classify.log)

  Distribution:
  ★★★★★  ( 8.5+)   1823 images  ████████████████████
  ★★★★☆  (7–8.5)   9412 images  ██████████████████████████████
  ★★★☆☆  (5.5–7)  18901 images  ██████████████████████████████
  ★★☆☆☆  (4–5.5)  11203 images  ██████████████████████████████
  ★☆☆☆☆  (  <4)    6143 images  █████████████████
```

---

## Sort in Finder

After running: open folder in Finder → **View → as List** → right-click the column header → enable **Rating**. Sort descending to see your best shots first.

---

## Supported formats

`.jpg` · `.jpeg` · `.png` · `.tiff` · `.tif` · `.webp` · `.heic` · `.bmp`

---

## Data files

| Path | Contents |
|------|----------|
| `~/.local/share/image-classifier/classify.db` | SQLite cache — `images` table (scores) + `failures` table (errors for review) |
| `~/.local/share/image-classifier/classify.log` | Error log, tab-separated, append mode |

### Review failures

```bash
sqlite3 ~/.local/share/image-classifier/classify.db \
  "SELECT path, error, failed_at FROM failures ORDER BY failed_at DESC"
```

---

## Development

### Setup

```bash
uv sync
```

### Run tests

```bash
uv run pytest
```

```
................................................................   [100%]
================================ tests coverage ================================

Name                              Stmts   Miss  Cover
-----------------------------------------------------
image_classifier/__init__.py          0      0   100%
image_classifier/classifier.py       55      7    87%
image_classifier/database.py         34      1    97%
image_classifier/main.py            115     19    83%
image_classifier/metadata.py         38      2    95%
-----------------------------------------------------
TOTAL                               242     29    88%

64 passed in 1.01s
```

### Project layout

```
src/image_classifier/
├── classifier.py   # model loading, image scoring, score→rating mapping
├── database.py     # SQLite helpers (images + failures tables)
├── main.py         # CLI entrypoint, progress loop, summary
└── metadata.py     # exiftool (XMP rating) + xattr (Finder tag) writers
tests/
├── test_classifier.py
├── test_database.py
├── test_main.py
└── test_metadata.py
```

### Lint & type-check

```bash
uv run ruff check src/
uv run mypy src/
```
