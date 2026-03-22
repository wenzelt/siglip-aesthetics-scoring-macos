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

## Sort in Finder

After running, open the folder in Finder → View → as List → right-click the column header → enable **Rating**. Sort by Rating descending to see your best shots first.

## Files created

| Path | Contents |
|------|----------|
| `~/.local/share/image-classifier/classify.db` | SQLite cache of scores |
| `~/.local/share/image-classifier/classify.log` | Error log (tab-separated, append mode) |

## Development

```bash
uv run pytest
uv run pytest --cov=image_classifier --cov-report=term-missing
```
