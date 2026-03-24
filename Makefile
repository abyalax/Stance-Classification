# Stance Classification Pipeline — Makefile
# Usage: make <command>

PYTHON = uv run python
PIPELINE = pipelines/main.py

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────

## Install semua dependencies
install:
	uv sync

# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

## Jalankan full pipeline (collect → preprocess → label)
all:
	$(PYTHON) $(PIPELINE) --stage all

## Stage 1: Scraping komentar for all platform
collect:
	$(PYTHON) $(PIPELINE) --stage collect --platform all

## Stage 1: Scraping komentar on youtube only
collect-youtube:
	$(PYTHON) $(PIPELINE) --stage collect --platform youtube

## Stage 1: Scraping komentar on tiktok only
collect-tiktok:
	$(PYTHON) $(PIPELINE) --stage collect --platform tiktok

## Force rerun Stage 1: Scraping komentar Instagram
collect-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform all

## Stage 1: Scraping komentar on youtube only
collect-youtube-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform youtube

## Stage 1: Scraping komentar on tiktok only
collect-tiktok-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform tiktok

## Stage 2: Preprocessing & cleaning teks
preprocess:
	$(PYTHON) $(PIPELINE) --stage preprocess

## Force rerun Stage 2: Preprocessing & cleaning teks
preprocess-force:
	$(PYTHON) $(PIPELINE) --stage preprocess --force

## Stage 3: Persiapan dataset untuk labeling di Label Studio
label:
	$(PYTHON) $(PIPELINE) --stage label

## Force rerun Stage 3: Persiapan dataset untuk labeling di Label Studio
label-force:
	$(PYTHON) $(PIPELINE) --stage label --force

## Cek status tiap stage pipeline
status:
	$(PYTHON) $(PIPELINE) --status

## Force rerun semua stage dari awal
force:
	$(PYTHON) $(PIPELINE) --stage all --force

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

## Hapus semua output data (hati-hati!)
clean:
	@echo "⚠️  Ini akan menghapus semua data di folder data/ (kecuali session)."
	@read -p "Lanjutkan? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	find data/ -type f ! -name "instagram_session" -delete
	@echo "✅ Data dibersihkan."

## Tampilkan semua command yang tersedia
help:
	@echo ""
	@echo "Stance Classification Pipeline"
	@echo "════════════════════════════════════════"
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Contoh urutan pertama kali:"
	@echo "  make install → make login → make all"
	@echo ""

.PHONY: install all collect collect-force preprocess preprocess-force label label-force status force clean help
