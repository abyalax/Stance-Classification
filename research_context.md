# Research Context Documentation
## Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG)

---

## 1. Identitas Penelitian

| Item | Detail |
|---|---|
| **Judul** | Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG) pada Komentar Media Sosial Menggunakan IndoBERT |
| **Jenjang** | Skripsi S1 |
| **Metode Utama** | Stance Classification dengan Fine-tuned IndoBERT |
| **Platform Data** | TikTok + YouTube |
| **Total Data** | 5.186 komentar (3.903 TikTok + 1.283 YouTube) |

---

## 2. Latar Belakang & Konteks Isu

Presiden Prabowo Subianto memastikan program **Makan Bergizi Gratis (MBG)** tetap berjalan di tengah tekanan efisiensi anggaran akibat dampak konflik Timur Tengah terhadap harga energi global. Pernyataan ini disampaikan dalam sesi diskusi bersama jurnalis dan pengamat yang ditayangkan di YouTube pada **22 Maret 2026**.

Isu ini memicu perdebatan publik yang luas karena menyentuh beberapa dimensi sekaligus:
- **Fiskal** — ruang APBN yang semakin sempit akibat subsidi energi membengkak
- **Program sosial** — efektivitas dan efisiensi MBG di lapangan
- **Insiden** — kasus keracunan massal terkait program MBG
- **Prioritas kebijakan** — apakah MBG tetap relevan saat ekonomi global tertekan

---

## 3. Sumber Data

### Platform & Volume

| Platform | Sumber | Jumlah Komentar |
|---|---|---|
| **TikTok** | Video pernyataan Prabowo MBG (22 Maret) | 3.903 |
| **YouTube** | Video pernyataan Prabowo MBG (22 Maret) | 1.283 |
| **Total** | | **5.186** |

### Tools Scraping

| Platform | Method | Tool |
|---|---|---|
| TikTok | Browser automation via cloud | Apify actor `BDec00yAmCm1QbMEI` |
| YouTube | Official API | YouTube Data API v3 (gratis, quota 10.000 unit/hari) |

### Output File

| File | Isi |
|---|---|
| `tiktok_comments_raw.csv` | 3.903 komentar mentah dari TikTok |
| `youtube_comments_raw.csv` | 1.283 komentar mentah dari YouTube |

---

## 4. Objektif Penelitian

1. Membangun model stance classification berbasis IndoBERT untuk komentar media sosial berbahasa Indonesia
2. Menganalisis distribusi stance opini publik terhadap program MBG
3. Membandingkan pola stance antara platform TikTok vs YouTube terhadap isu yang sama
4. Mengidentifikasi dominasi opini: pro/kontra pemerintah, saran konstruktif, diskusi kritis

---

## 5. Label Schema (Final)

| Label | Definisi Singkat |
|---|---|
| `FAVOR` | Mendukung program MBG dan/atau pernyataan Prabowo |
| `AGAINST` | Menentang/mengkritik program MBG dan/atau pernyataan Prabowo |
| `PRO_GOV` | Mendukung kebijakan/program pemerintah secara umum |
| `CONTRA_GOV` | Menentang/mengkritik kebijakan pemerintah secara umum |
| `CONDITIONAL` | Mendukung sebagian, disertai syarat atau catatan kritis |
| `SUGGESTION` | Memberikan saran atau masukan konstruktif berbasis data |
| `DISCUSSION` | Reply/diskusi lateral antar pengguna |
| `OFF_TOPIC` | Tidak relevan dengan isu yang dibahas |

> Skema: **single-label**, satu komentar = satu label dominan.

---

## 6. Pipeline Penelitian

```
1. Data Collection ✅
   ├── TikTok  : 3.903 komentar via Apify
   └── YouTube : 1.283 komentar via YouTube Data API v3

2. Preprocessing
   ├── Merge & deduplication (TikTok + YouTube)
   ├── Cleaning: remove spam, URL, emoji, duplikat
   ├── Normalisasi slang Bahasa Indonesia
   └── Estimasi data bersih: ~4.000–4.500 komentar

3. Labelling
   ├── Manual: 300–500 sampel (stratified sampling per platform)
   ├── Tools: Label Studio (self-hosted)
   ├── Annotator: minimal 2–3 orang
   ├── Validasi: Cohen's Kappa (target κ ≥ 0.6)
   └── Auto-label sisanya: IndoBERT existing sebagai silver label

4. Fine-tuning IndoBERT
   ├── Model base: IndoBERT (indobenchmark/indobert-base-p1)
   ├── Split: 70% train / 15% val / 15% test
   └── Task: multi-class classification (8 label)

5. Evaluasi Model
   ├── Metrics: F1-score (macro & weighted), Precision, Recall
   └── Confusion matrix per label

6. Analisis & Visualisasi Hasil
   ├── Distribusi stance keseluruhan
   ├── Perbandingan stance TikTok vs YouTube
   └── Insight opini publik terhadap program MBG
```

---

## 7. Output & Kontribusi Penelitian

### Output Teknis
- Model IndoBERT fine-tuned untuk stance classification komentar media sosial Bahasa Indonesia
- Dataset berlabel multi-platform (TikTok + YouTube) sebagai aset penelitian

### Output Analitis
- Distribusi stance opini publik terhadap program MBG
- Perbandingan pola stance antar platform (TikTok vs YouTube)
- Insight: apakah publik lebih banyak mendukung, mengkritik, atau memberi saran terhadap MBG

### Kontribusi Akademik
- Benchmark stance classification untuk domain kebijakan sosial berbahasa Indonesia
- Studi komparatif opini publik lintas platform media sosial
- Validasi efektivitas IndoBERT pada konteks diskusi kebijakan publik

---

## 8. Deliverables Penelitian

| Deliverable | Status |
|---|---|
| Annotation Guideline (`.docx`) | ✅ Selesai |
| Research Context Documentation (`.md`) | ✅ Dokumen ini |
| Scraping script — TikTok (Apify) | ✅ Selesai |
| Scraping script — YouTube (Official API) | ✅ Selesai |
| Raw data TikTok (3.903 komentar) | ✅ Selesai |
| Raw data YouTube (1.283 komentar) | ✅ Selesai |
| Preprocessing script | 🔲 Belum |
| Label Studio config (XML) | ✅ Tercantum di Annotation Guideline |
| Fine-tuning script IndoBERT | 🔲 Belum |
| Evaluasi & visualisasi | 🔲 Belum |

---

## 9. Catatan & Keputusan Desain

| Keputusan | Alasan |
|---|---|
| Ganti platform Instagram → TikTok + YouTube | Instagram memblokir scraping secara agresif sejak 2024 |
| Ganti topik geopolitik → MBG | MBG lebih focused, lebih relevan publik, lebih banyak source |
| Dual platform (TikTok + YouTube) | Memperkaya data, memungkinkan analisis komparatif antar platform |
| YouTube pakai Official API (bukan Apify) | Gratis, reliable, tidak ada risiko block atau credit habis |
| Stance classification, bukan sentiment analysis | Dataset kaya — komentar berisi saran, kritik bersyarat, diskusi multi-dimensi |
| Single-label per komentar | Feasible untuk S1, mengurangi kompleksitas training |
| 8 label | Mencerminkan kekayaan diskusi: saran, bersyarat, pro/kontra pemerintah |
| IndoBERT bukan mBERT | Pre-trained pada corpus Bahasa Indonesia, lebih relevan secara linguistik |
| Studi kasus spesifik (pernyataan Prabowo 22 Maret) | Konteks isu koheren, memudahkan interpretasi hasil |

---

*Dokumen ini adalah living document — update setiap ada keputusan desain baru.*
*Last updated: 2026-03-24*
