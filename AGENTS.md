# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python pipeline for stance classification of MBG-related social media comments.

- `pipelines/` contains the staged workflow:
  - `01_collect_data.py` for scraping
  - `02_preprocess_data.py` for cleaning and normalization
  - `03_prepare_labeling.py` for dataset sampling and export
  - `04_finetuning_bert.py` and `05_evaluation.py` for modeling and evaluation
- `main.py` and `pipelines/main.py` are the primary entry points.
- `data/` stores generated artifacts:
  - `data/scrapped/` for raw collection output
  - `data/preprocessed/` for cleaned data
  - `data/labelled/` for labeling assets and guides
- `notebook/` contains exploratory analysis.

## Build, Test, and Development Commands

Use `uv` and `make` for local work.

- `make install` installs dependencies via `uv sync`.
- `make all` runs the full pipeline: collect -> preprocess -> label.
- `make collect`, `make preprocess`, `make label` run individual stages.
- `make collect-tiktok` and `make collect-youtube` limit scraping to one platform.
- `make status` shows pipeline state.
- `uv run python pipelines/main.py --stage all` runs the pipeline directly.

## Coding Style & Naming Conventions

- Target Python 3.12.
- Follow standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes.
- Keep stage files numbered in execution order, matching the existing `01_...` pattern.
- Prefer clear, descriptive names for generated files, especially in `data/` where outputs are versioned or auto-incremented.

## Testing Guidelines

There is no formal test suite checked in yet. If you add tests, place them under a `tests/` directory and name files `test_*.py`.

- Prioritize small unit tests for preprocessing, file selection, and label-sampling logic.
- For pipeline changes, verify with `make status` and one targeted stage run before committing.

## Commit & Pull Request Guidelines

Recent commits use short Conventional Commit-style prefixes such as `feat:`, `refactor:`, and `chore:`. Keep subject lines imperative and specific, e.g. `feat: add preprocessing check for empty rows`.

- PRs should summarize the pipeline stage affected, data artifacts changed, and verification performed.
- Include screenshots or sample outputs only when the change affects labeling or visualization.
- Mention any required environment variables such as `APIFY_TOKEN` or `YOUTUBE_API_KEY`.

## Configuration & Data Safety

Do not commit secrets or local `.env` files. Generated CSV, JSON, and log files under `data/` should be treated as derived artifacts unless intentionally curated.
