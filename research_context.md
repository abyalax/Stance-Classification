# Research Context Documentation
## Stance Classification Opini Publik terhadap Isu Ketahanan Energi dan Kebijakan Fiskal Indonesia

---

## 1. Identitas Penelitian

| Item | Detail |
|---|---|
| **Judul** | Stance Classification Opini Publik terhadap Isu Ketahanan Energi dan Kebijakan Fiskal Indonesia pada Komentar Instagram Menggunakan IndoBERT (Studi Kasus: Akun @ferryirwandi) |
| **Jenjang** | Skripsi S1 |
| **Metode Utama** | Stance Classification dengan Fine-tuned IndoBERT |
| **Platform Data** | Instagram (@ferryirwandi) |
| **Total Data** | 4.581 komentar dari 2 post |

---

## 2. Sumber Data

### Post 1 — Dampak Kenaikan Harga Minyak terhadap Indonesia
Analisis dampak kenaikan harga minyak ($105–110/barel) terhadap Indonesia dalam 3 dimensi utama:

- **Shock Fiskal** — Subsidi energi membengkak ±Rp330T/tahun (selisih dari asumsi APBN $70–80)
- **Neraca Pembayaran** — Tambahan impor energi ~$10 miliar/tahun (Indonesia = net importer)
- **Rupiah & Moneter** — Capital outflow + demand USD naik → inflasi & tekanan suku bunga

**Risiko yang disoroti:**
- Defisit APBN berpotensi tembus >3% PDB
- Program MBG (~Rp335T) mempersempit ruang fiskal jika tidak efisien

**Strategi kebijakan yang diusulkan Ferry:**
- *Short term:* tahan harga BBM/listrik + stabilisasi rupiah
- *Mid term:* reprioritasi APBN + subsidi lebih targeted
- *Long term:* kurangi impor energi (biofuel, efisiensi)

---

### Post 2 — Narasi Penyebab: Energi & Konflik Global
Analisis berbasis report LNG & capital flow:

- Konflik global sebagai driver kenaikan harga energi (oil & gas)
- Disrupsi jalur kunci (Selat Hormuz) → potensi lonjakan harga global
- Klaim: ada aktor global termasuk Donald Trump yang dianggap diuntungkan/mendorong eskalasi

> **Catatan penting:** Ferry sendiri mengakui bahwa insight energi bersifat faktual (supply disruption → harga naik), namun klaim aktor & niat politik bersifat **opini/spekulatif**.

---

### Relasi Antar Post

```
Post 2 (Penyebab)                    Post 1 (Dampak)
Supply disruption energi global  →   Fiskal, Rupiah, Neraca Pembayaran Indonesia
Konflik global / aktor geopolitik →  Tekanan APBN & program pemerintah (MBG)
```

---

## 3. Objektif Penelitian

1. Membangun model stance classification berbasis IndoBERT untuk komentar Instagram berbahasa Indonesia
2. Menganalisis distribusi stance opini publik terhadap isu ketahanan energi dan kebijakan fiskal Indonesia
3. Mengidentifikasi pola opini netizen: pro/kontra pemerintah, dukungan/sanggahan terhadap analisis Ferry, saran konstruktif

---

## 4. Label Schema (Final)

| Label | Definisi Singkat |
|---|---|
| `FAVOR` | Mendukung argumen/posisi Ferry Irwandi |
| `AGAINST` | Menentang/mengkritik argumen Ferry Irwandi |
| `PRO_GOV` | Mendukung kebijakan/program pemerintah Indonesia |
| `CONTRA_GOV` | Menentang/mengkritik kebijakan pemerintah Indonesia |
| `CONDITIONAL` | Mendukung sebagian, disertai syarat atau catatan kritis |
| `SUGGESTION` | Memberikan saran atau masukan konstruktif berbasis data |
| `DISCUSSION` | Reply/diskusi lateral antar netizen |
| `OFF_TOPIC` | Tidak relevan dengan isu yang dibahas |

> Skema: **single-label**, satu komentar = satu label dominan.

---

## 5. Pipeline Penelitian

```
1. Data Collection
   └── Scraping 4.581 komentar dari 2 post @ferryirwandi
   └── Link Post 1 : https://www.instagram.com/p/DVscrIakZKd/
   └── Link Post 2 : https://www.instagram.com/p/DVtbswgk4h6/
       (tools: Instaloader)

2. Preprocessing
   └── Cleaning: remove spam, URL, duplikat
   └── Normalisasi slang Bahasa Indonesia
   └── Estimasi data bersih: ~3.500–4.000 komentar

3. Labelling
   └── Manual: 300–500 sampel (stratified sampling)
   └── Tools: Label Studio (self-hosted)
   └── Annotator: minimal 2–3 orang
   └── Validasi: Cohen's Kappa (target κ ≥ 0.6)
   └── Auto-label sisanya: IndoBERT existing sebagai silver label

4. Fine-tuning IndoBERT
   └── Model base: IndoBERT (indobenchmark/indobert-base-p1)
   └── Split: 70% train / 15% val / 15% test
   └── Task: multi-class classification (8 label)

5. Evaluasi Model
   └── Metrics: F1-score (macro & weighted), Precision, Recall
   └── Confusion matrix per label

6. Analisis & Visualisasi Hasil
   └── Distribusi stance keseluruhan
   └── Perbandingan distribusi stance Post 1 vs Post 2
   └── Insight opini publik terhadap isu fiskal & energi
```

---

## 6. Output & Kontribusi Penelitian

### Output Teknis
- Model IndoBERT fine-tuned untuk stance classification komentar Instagram Bahasa Indonesia
- Dataset berlabel (annotation + komentar) sebagai aset penelitian

### Output Analitis
- Distribusi stance opini publik terhadap isu ketahanan energi & kebijakan fiskal Indonesia
- Perbandingan stance antara isu penyebab (Post 2) vs dampak (Post 1)
- Insight: dominasi label apa yang muncul (kritik konstruktif, dukungan bersyarat, dll)

### Kontribusi Akademik
- Benchmark stance classification untuk domain ekonomi-fiskal berbahasa Indonesia
- Validasi efektivitas IndoBERT pada konteks diskusi kebijakan publik di Instagram

---

## 7. Deliverables Penelitian

| Deliverable | Status |
|---|---|
| Annotation Guideline (`.docx`) | ✅ Selesai |
| Research Context Documentation (`.md`) | ✅ Dokumen ini |
| Scraping script | 🔲 Belum |
| Preprocessing script | 🔲 Belum |
| Label Studio config (XML) | ✅ Tercantum di Annotation Guideline |
| Fine-tuning script IndoBERT | 🔲 Belum |
| Evaluasi & visualisasi | 🔲 Belum |

---

## 8. Catatan & Keputusan Desain

| Keputusan | Alasan |
|---|---|
| Stance classification, bukan sentiment analysis | Dataset terlalu kaya untuk sekadar positif/negatif/netral |
| Single-label per komentar | Feasible untuk S1, mengurangi kompleksitas training |
| 8 label (bukan 3) | Mencerminkan kekayaan diskusi: saran, bersyarat, pro/kontra pemerintah |
| IndoBERT bukan mBERT | Pre-trained pada corpus Bahasa Indonesia, lebih relevan secara linguistik |
| Studi kasus 2 post saja | Fokus pada konteks isu yang koheren dan terdefinisi |
| Tidak membedakan klaim faktual vs spekulatif | Di luar scope S1, cukup labeli stance-nya |

---

*Dokumen ini adalah living document — update setiap ada keputusan desain baru.*
*Last updated: 2026-03-23*
