# Stance Classification Pipeline — Makefile
# Usage: make <command>

PYTHON = uv run python
PIPELINE = pipelines/main.py
TOPICS ?= 5
WORDS ?= 10

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────

## Install semua dependencies
## command: make install
install:
	uv sync

# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

## Jalankan full pipeline (collect → preprocess → label)
## command: make all
all:
	$(PYTHON) $(PIPELINE) --stage all

## Stage 1: Scraping komentar for all platform
## command: make collect
collect:
	$(PYTHON) $(PIPELINE) --stage collect --platform all

## Stage 1: Scraping komentar on youtube only
## command: make collect-youtube
collect-youtube:
	$(PYTHON) $(PIPELINE) --stage collect --platform youtube

## Stage 1: Scraping komentar on tiktok only
## command: make collect-tiktok
collect-tiktok:
	$(PYTHON) $(PIPELINE) --stage collect --platform tiktok

## Force rerun Stage 1: Scraping komentar Instagram
## command: make collect-force 
collect-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform all

## Stage 1: Scraping komentar on youtube only
## command: make collect-youtube-force
collect-youtube-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform youtube

## Stage 1: Scraping komentar on tiktok only
## command: make collect-tiktok-force 
collect-tiktok-force:
	$(PYTHON) $(PIPELINE) --stage collect --force --platform tiktok

## Stage 2: Preprocessing & cleaning teks
## command: make preprocess 
preprocess:
	$(PYTHON) $(PIPELINE) --stage preprocess

## Force rerun Stage 2: Preprocessing & cleaning teks

preprocess-force:
	$(PYTHON) $(PIPELINE) --stage preprocess --force

## Stage 3: Persiapan dataset untuk labeling di Label Studio
## command: make label
label:
	$(PYTHON) $(PIPELINE) --stage label

## Force rerun Stage 3: Persiapan dataset untuk labeling di Label Studio
## command: make label-force
label-force:
	$(PYTHON) $(PIPELINE) --stage label --force

## Stage UAS: Topic modeling LDA dan NMF
## command: make topic
topic:
	$(PYTHON) pipelines/04_topic_modeling.py --topics $(TOPICS) --words $(WORDS)

## Stage UAS: Topic modeling dengan jumlah topik khusus
## command: make topic-topics TOPICS=5
topic-topics:
	$(PYTHON) pipelines/04_topic_modeling.py --topics $(TOPICS) --words $(WORDS)

## Cek status tiap stage pipeline
## command: make status
status:
	$(PYTHON) $(PIPELINE) --status

## Force rerun semua stage dari awal
## command: make force
force:
	$(PYTHON) $(PIPELINE) --stage all --force

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

## Hapus semua output data (hati-hati!)
## command: make clean
clean:
	@echo "⚠️  Ini akan menghapus semua data di folder data/ (kecuali session)."
	$(PYTHON) -c "confirm = input('Lanjutkan? [y/N] '); exit(0 if confirm.lower() == 'y' else 1)"
	$(PYTHON) -c "import os, glob; [os.remove(f) for f in glob.glob('data/**/*', recursive=True) if os.path.isfile(f) and 'instagram_session' not in f]"
	@echo "✅ Data dibersihkan."

## Tampilkan semua command yang tersedia
## command: make help
help:
	@echo ""
	@echo "Stance Classification Pipeline"
	@$(PYTHON) -c "import re; lines = [line[3:].strip() for line in open('Makefile', encoding='utf-8') if line.startswith('##')]; [print('  ' + line + ('\n' if 'command:' in line else '')) for line in lines]"
	@echo ""
	@echo "Contoh urutan pertama kali:"
	@echo "  make install -> make all"
	@echo ""

.PHONY: install all collect collect-force preprocess preprocess-force label label-force topic topic-topics status force clean help
