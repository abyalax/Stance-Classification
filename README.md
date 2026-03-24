# Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG)

Pipeline untuk klasifikasi stance opini publik terhadap Program Makan Bergizi Gratis (MBG) pada komentar media sosial (TikTok + YouTube) menggunakan IndoBERT.

---

## 📊 Research Context

**Judul Penelitian**: Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG) pada Komentar Media Sosial Menggunakan IndoBERT

**Konteks Isu**: Presiden Prabowo Subianto memastikan program **Makan Bergizi Gratis (MBG)** tetap berjalan di tengah tekanan efisiensi anggaran akibat dampak konflik Timur Tengah terhadap harga energi global (22 Maret 2026).

**Data Sources**:
- **TikTok**: 3.903 komentar dari video pernyataan Prabowo MBG
- **YouTube**: 1.283 komentar dari video pernyataan Prabowo MBG  
- **Total**: 5.186 komentar

**Label Schema**: 8 kelas stance (FAVOR, AGAINST, PRO_GOV, CONTRA_GOV, CONDITIONAL, SUGGESTION, DISCUSSION, OFF_TOPIC)

---

## 🏗️ Struktur Proyek

```
stance-classification/
├── pipelines/
│   ├── main.py                  # Entry point pipeline
│   ├── 01_collect_data.py       # Data collection (TikTok + YouTube)
│   ├── 02_preprocess_data.py    # Preprocessing & cleaning
│   └── 03_prepare_labeling.py   # Labeling preparation
├── data/
│   ├── scrapped/                 # Raw data from collection
│   │   ├── tiktok_comments_raw.csv
│   │   └── youtube_comments_raw.csv
│   ├── preprocessed/             # Processed data
│   │   ├── comments_processed.csv
│   │   ├── preprocessing_analysis.csv
│   │   └── preprocessing_analysis_detailed.json
│   └── pipeline_status.json      # Pipeline status tracking
├── logs/                        # Log files
│   ├── collect_data.log
│   ├── preprocessing.log
│   └── pipeline.log
├── Makefile                     # Shortcut commands
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Setup Awal

### 1. Install Dependencies

```bash
# Install core dependencies
make install
```
or Explicit with uv

```bash
# Install core dependencies
uv sync
```

### 2. Environment Variables

Buat file `.env` di root directory:

```bash
# TikTok scraping (Apify)
APIFY_TOKEN=your_apify_token_here

# YouTube scraping (Google API)
YOUTUBE_API_KEY=your_youtube_api_key_here
```

**Cara mendapatkan API keys**:
- **Apify Token**: https://console.apify.com/settings/integrations
- **YouTube API Key**: https://console.cloud.google.com/apis/credentials

---

## 🎯 Menjalankan Pipeline

### Quick Start (Recommended)

```bash
# Full pipeline dari awal
make all

# Atau step-by-step
make collect      # Stage 1: scrap data dari TikTok + YouTube
make preprocess   # Stage 2: preprocessing & cleaning  
make label        # Stage 3: persiapan labeling
make status       # Cek status pipeline
```

### Platform-Specific Collection

```bash
# Collect dari semua platform
make collect

# Platform spesifik
make collect-tiktok    # TikTok only
make collect-youtube   # YouTube only

# Force rerun (ignore cache)
make collect-force
make preprocess-force
make label-force
```

### Manual Execution

```bash
# Full pipeline
uv run python pipelines/main.py --stage all

# Per stage
uv run python pipelines/main.py --stage collect --platform all
uv run python pipelines/main.py --stage preprocess
uv run python pipelines/main.py --stage label

# Force rerun
uv run python pipelines/main.py --stage all --force
```

---

## 📋 Detail Pipeline

### Stage 1 — Data Collection (`01_collect_data.py`)

**Multi-platform scraping dengan robust error handling**:

**TikTok Scraping**:
- Method: Apify Actor (`BDec00yAmCm1QbMEI`)
- Output: `data/scrapped/tiktok_comments_raw.csv`
- Fitur: Auto-increment filename, retry logic, comprehensive logging

**YouTube Scraping**:
- Method: YouTube Data API v3 (official)
- Output: `data/scrapped/youtube_comments_raw.csv`
- Fitur: Rate limiting, metadata extraction, error handling

**Field Extraction**:
- TikTok: `text`, `createTimeISO`, `replyCommentTotal`, `diggCount`
- YouTube: `text`, `likes_count`, `published_at`, `reply_count`

### Stage 2 — Preprocessing (`02_preprocess_data.py`)

**Platform-agnostic preprocessing dengan stance-aware normalization**:

**Class Hierarchy**:
- `BasePreprocessor`: Core preprocessing methods
- `TiktokPreprocessor`: TikTok-specific data loading
- `YoutubePreprocessor`: YouTube-specific data loading  
- `DataPreprocessor`: Main orchestrator

**Preprocessing Steps**:
1. **Text Cleaning**: Remove URLs, mentions, hashtags, emojis, excessive characters
2. **Slang Normalization**: 200+ Indonesian slang words (including punctuation patterns)
3. **Stopwords Removal**: Custom stopwords yang **preserve negators** (`tidak`, `bukan`, `jangan`, `mungkin`)
4. **Spam Detection**: Rule-based spam filtering
5. **Quality Analysis**: Per-platform dan combined statistics

**Output Files**:
- `comments_processed.csv`: 4 text variations (raw, clean, normalized, final)
- `preprocessing_analysis.csv`: Summary statistics
- `preprocessing_analysis_detailed.json`: Platform-specific analysis

**Key Features**:
- **Stance-aware**: Negators penting untuk sentiment analysis tidak dihapus
- **Punctuation handling**: Normalisasi slang seperti `tlg.di` → `tolong.di`
- **Platform merging**: Combine TikTok + YouTube dengan `source` column
- **Auto-increment**: Prevent file overwriting

### Stage 3 — Labeling Preparation (`03_prepare_labeling.py`)

Persiapan dataset untuk anotasi manual di Label Studio:
- Stratified sampling per platform
- Export format JSON untuk Label Studio import
- Annotation guidelines documentation

---

## 🏷️ Label Schema

| Label | Definisi | Contoh |
|---|---|---|
| `FAVOR` | Mendukung program MBG dan/atau pernyataan Prabowo | "MBG bagus, lanjutkan!" |
| `AGAINST` | Menentang/mengkritik program MBG dan/atau pernyataan Prabowo | "MBG pemborosan, hentikan!" |
| `PRO_GOV` | Mendukung kebijakan/program pemerintah secara umum | "Pemerintah sudah benar" |
| `CONTRA_GOV` | Menentang/mengkritik kebijakan pemerintah secara umum | "Pemerintah tidak peka" |
| `CONDITIONAL` | Mendukung sebagian, disertai syarat atau catatan kritis | "Boleh MBG asal..." |
| `SUGGESTION` | Memberikan saran atau masukan konstruktif | "Sebaiknya MBG difokuskan..." |
| `DISCUSSION` | Reply/diskusi lateral antar pengguna | "Saya setuju dengan komentar..." |
| `OFF_TOPIC` | Tidak relevan dengan isu yang dibahas | "Promo produk nih" |

> **Skema**: Single-label — satu komentar = satu label dominan

---

## 📈 Pipeline Status & Monitoring

**Current Status** (as of latest run):
```bash
make status
```

**Log Files**:
- `logs/collect_data.log`: Data collection logs
- `logs/preprocessing.log`: Preprocessing logs  
- `logs/pipeline.log`: Overall pipeline logs

**Quality Metrics**:
- **Total Collected**: 5.186 comments (3.903 TikTok + 1.283 YouTube)
- **After Preprocessing**: ~3.500+ comments (spam filtered)
- **Slang Coverage**: 200+ Indonesian slang patterns
- **Negator Preservation**: ✅ Critical for stance classification

---

## 🛠️ Available Commands

### Pipeline Commands
```bash
make all              # Full pipeline
make collect          # Collect all platforms
make collect-tiktok   # TikTok only
make collect-youtube  # YouTube only
make preprocess       # Preprocessing
make label            # Labeling preparation
make status           # Check pipeline status
```

### Force Rerun
```bash
make collect-force    # Force collect
make preprocess-force # Force preprocessing
make label-force      # Force labeling
make force            # Force all stages
```

### Utilities
```bash
make clean            # Delete all output data
make help             # Show all commands
```

---

## 🔧 Troubleshooting

### API Key Issues
```bash
# Missing APIFY_TOKEN
Error: APIFY_TOKEN not found in .env - Tiktok scraping will be disabled

# Missing YOUTUBE_API_KEY  
Error: YOUTUBE_API_KEY not found in .env - YouTube scraping will be disabled
```

**Solution**: Tambahkan API keys ke file `.env`

### Pipeline Status Issues
```bash
# "Data collection must be completed first"
make collect-force    # Force rerun collection
```

### Preprocessing Issues
```bash
# Check preprocessing logs
tail -f logs/preprocessing.log

# Force rerun preprocessing
make preprocess-force
```

### Rate Limiting
- **YouTube**: 10.000 API units/hari (gratis)
- **TikTok**: Tergantung Apify credit
- **Solution**: Tunggu beberapa menit, gunakan `--force` untuk retry

---

## 📚 Research Notes

### Key Design Decisions
- **Dual Platform**: TikTok + YouTube untuk analisis komparatif
- **Stance Classification**: Lebih kaya dari sentiment analysis
- **IndoBERT**: Pre-trained Bahasa Indonesia, lebih relevan
- **Negator Preservation**: Critical untuk stance detection
- **Platform-Specific Preprocessing**: Handle different data formats

### Expected Deliverables
1. **Model**: IndoBERT fine-tuned untuk stance classification
2. **Dataset**: Multi-platform labeled dataset
3. **Analysis**: Distribusi stance opini publik terhadap MBG
4. **Insights**: Perbandingan TikTok vs YouTube stance patterns

### Next Steps (After Labeling)
1. Manual labeling di Label Studio (300-500 samples)
2. Fine-tuning IndoBERT dengan labeled data
3. Model evaluation & confusion matrix analysis
4. Stance distribution analysis & visualization

---

## 📄 Documentation

- **Research Context**: `research_context.md` - Detail penelitian lengkap
- **Annotation Guidelines**: Referensi untuk labeling
- **Pipeline Logs**: `logs/` folder untuk debugging
- **Data Analysis**: `data/preprocessed/preprocessing_analysis_*.json`

---

*Last updated: 2026-03-24*  
*Pipeline Status: ✅ Data Collection & Preprocessing Completed*
