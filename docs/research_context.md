# Research Context Documentation

## Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG)

---

## 1. Identitas Penelitian

| Item              | Detail                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Judul**         | Stance Classification Opini Publik terhadap Program Makan Bergizi Gratis (MBG) pada Komentar Media Sosial Menggunakan IndoBERT |
| **Jenjang**       | Skripsi S1                                                                                                                     |
| **Metode Utama**  | Stance Classification dengan Fine-tuned IndoBERT                                                                               |
| **Platform Data** | TikTok + YouTube                                                                                                               |
| **Total Data**    | 5.186 komentar (3.903 TikTok + 1.283 YouTube)                                                                                  |

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

| Platform    | Sumber                                  | Jumlah Komentar |
| ----------- | --------------------------------------- | --------------- |
| **TikTok**  | Video pernyataan Prabowo MBG (22 Maret) | 3.903           |
| **YouTube** | Video pernyataan Prabowo MBG (22 Maret) | 1.283           |
| **Total**   |                                         | **5.186**       |

### Tools Scraping

| Platform | Method                       | Tool                                                 |
| -------- | ---------------------------- | ---------------------------------------------------- |
| TikTok   | Browser automation via cloud | Apify actor `BDec00yAmCm1QbMEI`                      |
| YouTube  | Official API                 | YouTube Data API v3 (gratis, quota 10.000 unit/hari) |

### Output File

| File                       | Isi                                |
| -------------------------- | ---------------------------------- |
| `tiktok_comments_raw.csv`  | 3.903 komentar mentah dari TikTok  |
| `youtube_comments_raw.csv` | 1.283 komentar mentah dari YouTube |

### Struktur Folder Data

```
data/
├── scrapped/                 # Raw data dari collection
│   ├── tiktok_comments_raw.csv
│   └── youtube_comments_raw.csv
├── preprocessed/             # Data setelah preprocessing
│   ├── comments_processed.csv
│   ├── labeling_dataset.json
│   └── preprocessing_analysis.csv
├── labelled/                 # Data berlabel setelah manual annotation
│   ├── labeled_dataset.csv
│   └── labeling_statistics.json
└── pipeline_status.json      # Status tracking
```

---

## 4. Objektif Penelitian

1. Membangun model stance classification berbasis IndoBERT untuk komentar media sosial berbahasa Indonesia
2. Menganalisis distribusi stance opini publik terhadap program MBG
3. Membandingkan pola stance antara platform TikTok vs YouTube terhadap isu yang sama
4. Mengidentifikasi dominasi opini: pro/kontra pemerintah, saran konstruktif, diskusi kritis

---

## 5. Label Schema (Final)

| Label         | Definisi Singkat                                             |
| ------------- | ------------------------------------------------------------ |
| `FAVOR`       | Mendukung program MBG dan/atau pernyataan Prabowo            |
| `AGAINST`     | Menentang/mengkritik program MBG dan/atau pernyataan Prabowo |
| `PRO_GOV`     | Mendukung kebijakan/program pemerintah secara umum           |
| `CONTRA_GOV`  | Menentang/mengkritik kebijakan pemerintah secara umum        |
| `CONDITIONAL` | Mendukung sebagian, disertai syarat atau catatan kritis      |
| `SUGGESTION`  | Memberikan saran atau masukan konstruktif berbasis data      |
| `DISCUSSION`  | Reply/diskusi lateral antar pengguna                         |
| `OFF_TOPIC`   | Tidak relevan dengan isu yang dibahas                        |

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
   ├── Output: `data/labelled/labeled_dataset.csv`
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

## 9. Annotation Guidelines

### Label Schema

#### FAVOR

**Definition:** Comments that support, agree with, or reinforce the MBG program and/or Prabowo's statements.

**Examples:**

- "MBG bagus, lanjutkan!"
- "Setuju pak Prabowo, program ini sangat membantu"
- "Program MBG akan sangat bermanfaat untuk rakyat"

**Keywords:** setuju, benar, tepat, bagus, mendukung, support, betul, mbg bagus, lanjutkan, program baik, bantu

#### AGAINST

**Definition:** Comments that oppose, contradict, or criticize the MBG program and/or Prabowo's statements.

**Examples:**

- "MBG pemborosan, hentikan!"
- "Tidak setuju dengan program MBG"
- "Anggaran MBG sebaiknya dialihkan"

**Keywords:** tidak setuju, keliru, berbeda, kontra, hentikan, pemborosan, tidak efektif, gagal, salah alih

#### PRO_GOV

**Definition:** Comments that support Indonesian government policies or programs in general.

**Examples:**

- "Pemerintah sudah berusaha maksimal"
- "Program pemerintah akan sangat membantu"
- "Presiden tepat dalam kebijakan ini"

**Keywords:** pemerintah, presiden, kementerian, kebijakan, program, dukung, prabowo, negara, bantuan

#### CONTRA_GOV

**Definition:** Comments that criticize Indonesian government policies or programs in general.

**Examples:**

- "Pemerintah kurang responsif"
- "Kebijakan ini memberatkan rakyat"
- "Harus ada evaluasi total pemerintah"

**Keywords:** pemerintah, presiden, kementerian, kebijakan, program, kritik, tidak peka, salah urus, korupsi

#### CONDITIONAL

**Definition:** Comments that provide partial support with conditions or critical notes.

**Examples:**

- "Boleh MBG asal tidak memberatkan"
- "Setuju program ini tapi perlu perbaikan"
- "Program baik jika tepat sasaran"

**Keywords:** tapi, namun, walaupun, meskipun, asal, syarat, jika, boleh asal, setuju tapi

#### SUGGESTION

**Definition:** Comments that provide constructive suggestions or solutions.

**Examples:**

- "Sebaiknya MBG difokuskan pada anak sekolah"
- "Harus ada evaluasi ulang program"
- "Sarankan untuk perbaiki tata kelola"

**Keywords:** sarankan, usulkan, sebaiknya, harus, perlu, solusi, fokuskan, evaluasi, perbaiki

#### DISCUSSION

**Definition:** Comments that are replies to other users or lateral discussions.

**Examples:**

- "@username saya setuju dengan pendapatmu"
- "Balasan untuk komentar di atas"
- "Menanggapi @username tentang MBG"

**Keywords:** @, reply, balas, komen, pendapat, setuju dengan

#### OFF_TOPIC

**Definition:** Comments that are not relevant to the discussed issues.

**Examples:**

- "Jualan produk herbal"
- "Follow akun saya"
- "Promo jasa kebersihan"

**Keywords:** jualan, promo, iklan, follow, like, subscribe, promosi

### Annotation Rules

#### Primary Rules

1. **Single Label:** Each comment gets exactly ONE label
2. **Context First:** Consider context of MBG program and Prabowo's statements
3. **Main Stance:** Focus on the dominant stance, not minor mentions
4. **Explicit vs Implicit:** Both explicit and implicit stances count
5. **Platform Context:** Consider both TikTok and YouTube contexts

#### Decision Hierarchy

1. If it's clearly OFF_TOPIC (spam, ads) → OFF_TOPIC
2. If it's a reply to another user → DISCUSSION
3. If it discusses government policies generally → PRO_GOV or CONTRA_GOV
4. If it discusses MBG program specifically → FAVOR or AGAINST
5. If it has conditions → CONDITIONAL
6. If it provides solutions → SUGGESTION

#### Edge Cases

- **Mixed Stance:** Choose the dominant stance
- **Unclear:** Use your best judgment, add notes
- **Short Comments:** Analyze based on available content
- **Sarcasm:** Consider the intended meaning
- **Platform Differences:** Account for platform-specific language styles

### Quality Standards

- **Consistency:** Apply criteria consistently across all annotations
- **Accuracy:** Read carefully before labeling
- **Documentation:** Add notes for ambiguous cases
- **Confidence:** Rate your confidence (1-5 scale)

### Process

1. Read the original comment
2. Read the cleaned and final versions if needed
3. Consider the platform context (TikTok/YouTube)
4. Apply decision hierarchy
5. Select the best label
6. Rate confidence
7. Add notes if necessary

---

## 8. Deliverables Penelitian

| Deliverable                              | Status         |
| ---------------------------------------- | -------------- |
| Annotation Guideline (`.docx`)           | ✅ Selesai     |
| Research Context Documentation (`.md`)   | ✅ Dokumen ini |
| Scraping script — TikTok (Apify)         | ✅ Selesai     |
| Scraping script — YouTube (Official API) | ✅ Selesai     |
| Raw data TikTok (3.903 komentar)         | ✅ Selesai     |
| Raw data YouTube (1.283 komentar)        | ✅ Selesai     |
| Preprocessing script                     | ✅ Selesai     |
| Labeling preparation script              | ✅ Selesai     |
| Label Studio config (JSON)               | ✅ Selesai     |
| Labeled dataset (manual annotation)      | 🔲 Pending     |
| Fine-tuning script IndoBERT              | 🔲 Belum       |
| Evaluasi & visualisasi                   | 🔲 Belum       |

---

## 9. Catatan & Keputusan Desain

| Keputusan                                          | Alasan                                                                        |
| -------------------------------------------------- | ----------------------------------------------------------------------------- |
| Ganti platform Instagram → TikTok + YouTube        | Instagram memblokir scraping secara agresif sejak 2024                        |
| Ganti topik geopolitik → MBG                       | MBG lebih focused, lebih relevan publik, lebih banyak source                  |
| Dual platform (TikTok + YouTube)                   | Memperkaya data, memungkinkan analisis komparatif antar platform              |
| YouTube pakai Official API (bukan Apify)           | Gratis, reliable, tidak ada risiko block atau credit habis                    |
| Stance classification, bukan sentiment analysis    | Dataset kaya — komentar berisi saran, kritik bersyarat, diskusi multi-dimensi |
| Single-label per komentar                          | Feasible untuk S1, mengurangi kompleksitas training                           |
| 8 label                                            | Mencerminkan kekayaan diskusi: saran, bersyarat, pro/kontra pemerintah        |
| IndoBERT bukan mBERT                               | Pre-trained pada corpus Bahasa Indonesia, lebih relevan secara linguistik     |
| Studi kasus spesifik (pernyataan Prabowo 22 Maret) | Konteks isu koheren, memudahkan interpretasi hasil                            |

---

_Dokumen ini adalah living document — update setiap ada keputusan desain baru._
_Last updated: 2026-03-24_
