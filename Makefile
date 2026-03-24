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
	uv add pandas instaloader Pillow matplotlib seaborn

# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

## Jalankan full pipeline (collect → preprocess → label)
all:
	$(PYTHON) $(PIPELINE) --stage all

## Stage 1: Scraping komentar Instagram
collect:
	$(PYTHON) $(PIPELINE) --stage collect

## Stage 2: Preprocessing & cleaning teks
preprocess:
	$(PYTHON) $(PIPELINE) --stage preprocess

## Stage 3: Persiapan dataset untuk labeling di Label Studio
label:
	$(PYTHON) $(PIPELINE) --stage label

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

.PHONY: login login-brave login-chrome install all collect preprocess label status force clean help
