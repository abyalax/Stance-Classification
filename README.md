# Stance Classification Pipeline

Pipeline untuk klasifikasi stance opini publik terhadap isu ketahanan energi dan kebijakan fiskal Indonesia pada komentar Instagram @ferryirwandi.

## Struktur Proyek

```
stance-classification/
├── pipelines/
│   └── main.py                  # Entry point pipeline
├── data/
│   ├── comments_raw.csv         # Output: raw comments
│   ├── comments_processed.csv   # Output: preprocessed comments
│   ├── posts_metadata.csv       # Output: post metadata
│   ├── scraping_summary.json    # Output: scraping summary
│   ├── preprocessing_analysis.json
│   ├── data_quality_report.png
│   ├── instagram_session        # Session file (tidak di-commit ke git)
│   └── labeling_package/
│       ├── labeling_dataset.csv
│       ├── labeling_dataset.json
│       ├── label_studio_config.json
│       ├── annotation_guidelines.md
│       └── labeling_analysis.json
├── 01_collect_data.py
├── 02_preprocess_data.py
├── 03_prepare_labeling.py
├── Makefile                     # Shortcut commands
├── requirements.txt
└── README.md
```

---

## Setup Awal

### 1. Install Dependencies

```bash
uv add pandas instaloader
# atau dari requirements.txt
uv add -r requirements.txt
```

### 2. Login Instagram (wajib, jalankan sekali)

Akses komentar Instagram **memerlukan autentikasi**. Gunakan akun dummy/alternatif, bukan akun utama.

```bash
make login
```

Command ini akan meminta username dan password secara interaktif, lalu menyimpan session ke `data/instagram_session`. Session ini akan digunakan otomatis oleh pipeline.

> ⚠️ File `data/instagram_session` jangan di-commit ke Git. Pastikan sudah ada di `.gitignore`.

---

## Menjalankan Pipeline

### Shortcut via Makefile

```bash
make login        # Login Instagram & simpan session (jalankan sekali)
make collect      # Stage 1: scraping komentar
make preprocess   # Stage 2: preprocessing teks
make label        # Stage 3: persiapan labeling
make all          # Jalankan full pipeline (collect → preprocess → label)
make status       # Cek status tiap stage
make force        # Force rerun semua stage dari awal
make clean        # Hapus semua output data (hati-hati)
make help         # Tampilkan semua command
```

### Manual via Python

```bash
# Full pipeline
uv run python pipelines/main.py --stage all

# Per stage
uv run python pipelines/main.py --stage collect
uv run python pipelines/main.py --stage preprocess
uv run python pipelines/main.py --stage label

# Cek status
uv run python pipelines/main.py --status

# Force rerun
uv run python pipelines/main.py --stage all --force
```

---

## Detail Tiap Stage

### Stage 1 — Data Collection (`01_collect_data.py`)

Scraping komentar dari 2 post target @ferryirwandi:
- Post 1: Dampak kenaikan harga minyak terhadap fiskal Indonesia
- Post 2: Narasi penyebab — konflik global & aktor geopolitik

Fitur:
- Session-based authentication (wajib login)
- Rate limiting protection + random delay
- Retry logic (hingga 10x) dengan exponential backoff
- Safe attribute access untuk kompatibilitas antar versi instaloader

Output: `data/comments_raw.csv`, `data/posts_metadata.csv`, `data/scraping_summary.json`

---

### Stage 2 — Preprocessing (`02_preprocess_data.py`)

Pembersihan dan normalisasi teks komentar:
- Hapus URL, mention, hashtag, emoji
- Normalisasi slang Bahasa Indonesia
- Deteksi dan filter spam
- Analisis kualitas data

Output: `data/comments_processed.csv`, `data/preprocessing_analysis.json`, `data/data_quality_report.png`

---

### Stage 3 — Labeling Preparation (`03_prepare_labeling.py`)

Persiapan dataset untuk anotasi manual di Label Studio:
- Stratified sampling (300–500 komentar)
- Export format JSON siap import ke Label Studio
- Konfigurasi Label Studio (XML)
- Annotation guidelines

Output: `data/labeling_package/`

---

## Label Schema

| Label | Definisi |
|---|---|
| `FAVOR` | Mendukung argumen/posisi Ferry Irwandi |
| `AGAINST` | Menentang/mengkritik argumen Ferry Irwandi |
| `PRO_GOV` | Mendukung kebijakan/program pemerintah Indonesia |
| `CONTRA_GOV` | Menentang/mengkritik kebijakan pemerintah Indonesia |
| `CONDITIONAL` | Mendukung sebagian, disertai syarat atau catatan kritis |
| `SUGGESTION` | Memberikan saran atau masukan konstruktif berbasis data |
| `DISCUSSION` | Reply/diskusi lateral antar netizen |
| `OFF_TOPIC` | Tidak relevan dengan isu yang dibahas |

> Skema: **single-label** — satu komentar = satu label paling dominan.

---

## Alur Kerja Lengkap (End-to-End)

```
make login
    ↓
make collect          → ~4.581 komentar dari 2 post
    ↓
make preprocess       → cleaning, normalisasi, filter spam
    ↓
make label            → sampling + export ke Label Studio
    ↓
[Manual] Import labeling_dataset.json ke Label Studio
    ↓
[Manual] Anotasi 300–500 sampel oleh 2–3 annotator
    ↓
[Manual] Hitung Cohen's Kappa (target κ ≥ 0.6)
    ↓
[Next] Fine-tuning IndoBERT
```

---

## Logging

| File | Isi |
|---|---|
| `scraping.log` | Log stage collect |
| `preprocessing.log` | Log stage preprocess |
| `labeling.log` | Log stage label |
| `pipeline.log` | Log keseluruhan pipeline |

---

## Troubleshooting

### `Login required` saat collect
Session tidak ditemukan atau sudah expired.
```bash
make login   # buat ulang session
```

### `403 Forbidden` dari graphql
Normal terjadi di awal — instaloader akan retry otomatis. Jika terus gagal, tunggu beberapa menit sebelum coba lagi (Instagram rate limit).

### `AttributeError` pada comment object
Versi instaloader tidak kompatibel. Pastikan menggunakan versi terbaru:
```bash
uv add instaloader --upgrade
```

### Session expired di tengah scraping
Instagram session bisa expire dalam beberapa jam. Jika terjadi di tengah proses:
```bash
make login   # login ulang
make collect # jalankan ulang collect
```

---

## Catatan Penting

- Gunakan **akun Instagram dummy** untuk login, bukan akun utama
- Jangan commit `data/instagram_session` ke Git
- Scraping 4.500+ komentar membutuhkan waktu ~15–30 menit tergantung koneksi dan rate limit
- Untuk keperluan akademik, dokumentasikan metode scraping di bab metodologi skripsi
