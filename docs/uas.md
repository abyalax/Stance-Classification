# Context Proyek UAS Data Mining

> Perbandingan Latent Dirichlet Allocation (LDA) dan Non-negative Matrix Factorization (NMF) untuk Identifikasi Topik Opini Publik terhadap Program Makan Bergizi Gratis (MBG) pada Komentar TikTok dan YouTube

---

## 1. Latar Belakang

Program Makan Bergizi Gratis (MBG) merupakan salah satu program prioritas pemerintah yang memunculkan beragam tanggapan publik di media sosial. Komentar pengguna pada platform TikTok dan YouTube mengandung opini, kritik, dukungan, saran, serta diskusi terkait implementasi program tersebut.

Sebagian besar penelitian opini publik menggunakan pendekatan sentiment analysis atau stance classification yang memerlukan proses pelabelan data secara manual. Pada penelitian ini dipilih pendekatan **Topic Modeling** karena:

- Dataset belum memiliki label anotasi.
- Jumlah data relatif besar (>4000 komentar).
- Tujuan penelitian lebih berfokus pada identifikasi tema pembahasan utama masyarakat.
- Topic Modeling merupakan metode unsupervised learning sehingga tidak memerlukan proses labeling.

Penelitian ini membandingkan dua algoritma topic modeling yang umum digunakan, yaitu:

1. Latent Dirichlet Allocation (LDA)
2. Non-negative Matrix Factorization (NMF)

---

## 2. Tujuan Penelitian

### Tujuan Utama

Mengidentifikasi topik-topik dominan yang muncul dalam opini publik terhadap Program Makan Bergizi Gratis (MBG) pada platform TikTok dan YouTube.

### Tujuan Khusus

1. Menghasilkan kumpulan topik utama yang dibahas masyarakat terkait MBG.
2. Membandingkan performa algoritma LDA dan NMF dalam membentuk topik yang koheren.
3. Menganalisis perbedaan pola diskusi antara pengguna TikTok dan YouTube.
4. Menentukan algoritma topic modeling yang paling sesuai untuk dataset komentar MBG.

---

## 3. Rumusan Masalah

1. Topik apa saja yang dominan dibahas masyarakat terkait Program Makan Bergizi Gratis?
2. Bagaimana perbedaan hasil topic modeling menggunakan LDA dan NMF?
3. Algoritma manakah yang menghasilkan topik paling koheren dan mudah diinterpretasikan?
4. Apakah terdapat perbedaan fokus pembahasan antara komentar TikTok dan YouTube?

---

## 4. Dataset Penelitian

## Sumber Data

### TikTok

Video sumber:

https://www.tiktok.com/@suaradotcom/video/7620063435551788304

Konteks:

Pernyataan Presiden Prabowo mengenai keberlanjutan Program Makan Bergizi Gratis (MBG) di tengah isu efisiensi anggaran.

Metode pengambilan data:

- Apify Actor
- Komentar utama dan balasan komentar

Jumlah komentar awal:

- ± 3903 komentar

Contoh sample hasil scrapping

```csv
videoWebUrl,submittedVideoUrl,input,cid,createTime,createTimeISO,text,diggCount,likedByAuthor,pinnedByAuthor,repliesToId,replyCommentTotal,uid,uniqueId,avatarThumbnail,mentions,detailedMentions
https://www.tiktok.com/@suaradotcom/video/7620063435551788304,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,7620097982969266962,1774192328,2026-03-22T15:12:08.000Z,Kepala keras sekali,363,False,False,,25.0,7290098237809345537,user4401039229213,https://p16-common-sign.tiktokcdn-us.com/musically-maliva-obj/1594805258216454~tplv-tiktokx-cropcenter:100:100.jpg?dr=9640&refresh_token=fd447206&x-expires=1774429200&x-signature=YRTKwNUURw%2ByfYjJCQun4SRR4qE%3D&t=4d5b0474&ps=13740610&shp=30310797&shcp=ff37627b&idc=useast5,[],[]
https://www.tiktok.com/@suaradotcom/video/7620063435551788304,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,7620082012879160071,1774188692,2026-03-22T14:11:32.000Z,"Proyeknya beliau, tidak mungkin beliau menghentikan MBG😅",1001,False,False,,29.0,7552430766997128210,user3436883098108,https://p16-common-sign.tiktokcdn-us.com/musically-maliva-obj/1594805258216454~tplv-tiktokx-cropcenter:100:100.jpg?dr=9640&refresh_token=fd447206&x-expires=1774429200&x-signature=YRTKwNUURw%2ByfYjJCQun4SRR4qE%3D&t=4d5b0474&ps=13740610&shp=30310797&shcp=ff37627b&idc=useast5,[],[]
https://www.tiktok.com/@suaradotcom/video/7620063435551788304,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858,7620634247766786824,1774317194,2026-03-24T01:53:14.000Z,"dan anehnya kita yg minta MBG distop malah dicap negatif, padahal membantu pemerintah yg sedang efisiensi",4,False,False,,0.0,6796478012344042497,herur.m,https://p16-common-sign.tiktokcdn-us.com/tos-useast2a-avt-0068-giso/515920743d2c8218d34a3071f970236e~tplv-tiktokx-cropcenter:100:100.jpg?dr=9640&refresh_token=e40823da&x-expires=1774429200&x-signature=0jXgkTuoTbfSGTQ5QRl4DCJtFVs%3D&t=4d5b0474&ps=13740610&shp=30310797&shcp=ff37627b&idc=useast5,[],[]
```

---

#### YouTube

Video sumber:

https://www.youtube.com/watch?v=S4iey1sr1Cg

Konteks:

Pernyataan Presiden Prabowo terkait Program Makan Bergizi Gratis (MBG).

Metode pengambilan data:

- YouTube Data API v3

Jumlah komentar awal:

- ± 1283 komentar

Contoh sample hasil scrapping

```csv
comment_id,text,author,likes_count,published_at,updated_at,reply_count,is_reply,parent_id,video_id,source
UgzY0dDeSE6uPXpq5Tx4AaABAg,"Jelas keliatan banget, mash soal Asal Bapak Senang "" dia hanya melihat stunting, tapi yg keracunan di diamkan, semacam "" korban jdi tersangka, yg maling tetap Benar """,@Felix_Bro-e9j,68,2026-03-21T03:52:43Z,2026-03-21T03:52:43Z,10,False,,S4iey1sr1Cg,youtube
UgzY0dDeSE6uPXpq5Tx4AaABAg.AUaxsAK08atAUb0s0HP-H7,Padahal stunting hanya bisa di cegah dr kandungan sampai   batita 3 tahun,@DoraJarajakabw,8,2026-03-21T04:27:39Z,2026-03-21T04:27:39Z,0,True,UgzY0dDeSE6uPXpq5Tx4AaABAg,S4iey1sr1Cg,youtube
UgzY0dDeSE6uPXpq5Tx4AaABAg.AUaxsAK08atAUbQmS81sh3,"​@DoraJarajakabw betul. Lebih bener program jokowi dulu PMT untuk balita stunting, lewat kader posyandu jd yg diberi bisa tepat sasaran krn di posyandu akan tau mana anak stunting dan tdk. Yg masak kader posyandu ya tinggal diberi penyuluhan lg dan diawasi biar masaknya bener2 yg berkualitas.",@RatnaKurniasari-x9r,8,2026-03-21T08:14:05Z,2026-03-21T08:14:05Z,0,True,UgzY0dDeSE6uPXpq5Tx4AaABAg,S4iey1sr1Cg,youtube
UgzY0dDeSE6uPXpq5Tx4AaABAg.AUaxsAK08atAUezG219VSk,ttg keracunan dah dengar . tp dia blng cuma 0.0001% aja,@Nixie_Sebong,2,2026-03-22T17:21:49Z,2026-03-22T17:21:49Z,0,True,UgzY0dDeSE6uPXpq5Tx4AaABAg,S4iey1sr1Cg,youtube
```

---

### Dataset Gabungan

Dataset TikTok dan YouTube digabungkan menjadi satu dataset terintegrasi.

Jumlah data setelah preprocessing dan spam filtering:

```text
4131 komentar
```

Contoh Sample data

```csv
created_at,raw_text,likes_count,reply_count,source,is_spam,clean_text,normalized_text,final_text,text_length,word_count,clean_text_length,clean_word_count
2026-03-22T22:25:43.000Z,siapa yg setuju MBG distop?,3942,69.0,tiktok,False,setuju makan bergizi gratis distop?,siapa yg setuju makan bergizi gratis distop?,setuju makan bergizi gratis distop?,27.0,5.0,35,5
2026-03-22T14:01:13.000Z,"Susah sekali bapak Presiden ini di kritiknya, padahal banyak sekali masyarakat dan tokoh nasional minta di hentikan program MBG nya tuuk di evaluasi",2816,68.0,tiktok,False,"susah presiden kritiknya, masyarakat tokoh nasional hentikan program makan bergizi gratis nya tuuk evaluasi","susah sekali bapak presiden ini di kritiknya, padahal banyak sekali masyarakat dan tokoh nasional minta di hentikan program makan bergizi gratis nya tuuk di evaluasi","susah presiden kritiknya, masyarakat tokoh nasional hentikan program makan bergizi gratis nya tuuk evaluasi",148.0,23.0,107,14
2026-03-22T13:19:14.000Z,"hentikan aj pak mending dialihkan ke program beasiswa kuliah gratis 🙏",1999,60.0,tiktok,False,hentikan mending dialihkan program beasiswa kuliah gratis,hentikan saja pak mending dialihkan ke program beasiswa kuliah gratis,hentikan mending dialihkan program beasiswa kuliah gratis,69.0,11.0,57,7
```

---

## 5. Struktur Dataset Final

Dataset hasil preprocessing memiliki kolom:

| Kolom             | Deskripsi                         |
| ----------------- | --------------------------------- |
| created_at        | Waktu komentar dibuat             |
| raw_text          | Komentar asli                     |
| likes_count       | Jumlah like komentar              |
| reply_count       | Jumlah balasan komentar           |
| source            | Platform asal (TikTok / YouTube)  |
| is_spam           | Indikator spam                    |
| clean_text        | Hasil text cleaning               |
| normalized_text   | Hasil normalisasi slang           |
| final_text        | Teks akhir setelah preprocessing  |
| text_length       | Panjang karakter teks asli        |
| word_count        | Jumlah kata teks asli             |
| clean_text_length | Panjang karakter setelah cleaning |
| clean_word_count  | Jumlah kata setelah cleaning      |

Kolom utama yang digunakan dalam penelitian:

```text
final_text
source
```

Kolom pendukung analisis:

```text
likes_count
reply_count
word_count
clean_word_count
```

---

## 6. Kondisi Codebase Saat Ini

### Arsitektur Proyek

root/
`└── 📁data
        └── 📁labelled
                ├── labeling_analysis.json
                ├── labeling_dataset.csv
                ├── labeling_dataset.json
                ├── LABELING_GUIDE.md
                ├── README.md
        └── 📁preprocessed
                ├── comments_processed_1.csv
                ├── comments_processed_2.csv
                ├── comments_processed_3.csv
                ├── comments_processed.csv
                ├── preprocessing_analysis_1.csv
                ├── preprocessing_analysis_2.csv
                ├── preprocessing_analysis_3.csv
                ├── preprocessing_analysis_detailed.json
                ├── preprocessing_analysis.csv
        └── 📁scrapped
                ├── comment_post_1.csv
                ├── comment_post.csv
                ├── comments_extracted.csv
                ├── tiktok_comments_raw_1.csv
                ├── tiktok_comments_raw.csv
                ├── youtube_comments_raw.csv
        └── 📁topic_modeling
                └── 📁plots
                ├── lda_top_words.png
                ├── nmf_top_words.png
                ├── platform_topic_distribution.png
                ├── document_topics.csv
                ├── metrics.csv
                ├── metrics.json
                ├── platform_topic_distribution.csv
                ├── summary.md
                ├── topics.csv
        ├── pipeline_status.json
        └── preprocessing_analysis.json
   └── 📁notebook
        └── visualization.ipynb # saya ingin di sini bisa import code di pipelines dan jalankan step by step, just for wrapper and visualization, inti algoritma ada di dalam folder pipelines
   └── 📁pipelines
        └── 📁__pycache__
        ├── __init__.py
        ├── 01_collect_data.py
        ├── 02_preprocess_data.py
        ├── 03_prepare_labeling.py
        ├── 04_finetuning_bert.py
        ├── 04_topic_modeling.py
        ├── 05_evaluation.py
        └── main.py`

---

### Pipeline yang Sudah Selesai

#### Stage 1 — Data Collection

TikTok:

- Apify Actor

YouTube:

- YouTube Data API v3

Output:

```text
data/scrapped/
```

---

#### Stage 2 — Preprocessing

Tahapan:

1. Remove URL
2. Remove mention
3. Remove hashtag
4. Remove emoji
5. Remove duplicate characters
6. Slang normalization
7. Stopword removal
8. Spam filtering

Output:

```text
data/preprocessed/comments_processed.csv
```

---

#### Stage 3 — Label Preparation

Sudah tersedia namun tidak digunakan dalam penelitian ini karena pendekatan yang dipilih adalah unsupervised learning.

---

## 7. Perubahan Scope Penelitian

### Scope Lama

```text
Stance Classification
Menggunakan IndoBERT
Membutuhkan Labeling Manual
```

Status:

```text
Tidak digunakan
```

Alasan:

- Belum tersedia label final.
- Membutuhkan biaya dan waktu anotasi tambahan.
- Tidak sesuai target penyelesaian UAS.

---

### Scope Baru

```text
Topic Modeling
Unsupervised Learning
Tanpa Labeling
```

Status:

```text
Digunakan
```

---

## 8. Algoritma yang Digunakan

### Model 1 — Latent Dirichlet Allocation (LDA)

Jenis:

```text
Probabilistic Topic Modeling
```

Input:

```text
CountVectorizer
```

Output:

```text
Kumpulan topik
Distribusi kata pada tiap topik
Distribusi topik pada tiap dokumen
```

Karakteristik:

- Pendekatan probabilistik.
- Satu komentar dapat mengandung beberapa topik sekaligus.
- Sangat populer dalam penelitian topic modeling.

---

### Model 2 — Non-negative Matrix Factorization (NMF)

Jenis:

```text
Matrix Factorization Topic Modeling
```

Input:

```text
TF-IDF Vectorizer
```

Output:

```text
Kumpulan topik
Bobot kata dominan pada tiap topik
```

Karakteristik:

- Cocok untuk data TF-IDF.
- Umumnya menghasilkan topik yang lebih jelas.
- Interpretasi topik lebih mudah.

---

## 9. Workflow Penelitian

```text
Dataset Komentar
        │
        ▼
Preprocessing
        │
        ▼
Dataset Final (4060 komentar)
        │
        ├─────────────► CountVectorizer
        │                     │
        │                     ▼
        │                   LDA
        │
        └─────────────► TF-IDF
                              │
                              ▼
                             NMF
                              │
                              ▼
                     Evaluasi dan Perbandingan
```

---

## 10. Evaluasi Model

Karena penelitian menggunakan unsupervised learning, tidak digunakan:

```text
Accuracy
Precision
Recall
F1 Score
```

Sebagai gantinya digunakan:

### Topic Coherence

Mengukur keterkaitan makna kata-kata dalam suatu topik.

Interpretasi:

```text
Semakin tinggi semakin baik.
```

---

### Topic Diversity

Mengukur tingkat perbedaan antar topik.

Interpretasi:

```text
Semakin tinggi semakin baik.
```

---

### Interpretability Analysis

Analisis manual terhadap kata dominan setiap topik.

Contoh:

```text
Topic 1
gizi
stunting
anak
sekolah

Topic 2
keracunan
makanan
pengawasan
kualitas
```

---

## 11. Tools dan Library

Bahasa:

```text
Python 3.12
```

Library:

```text
pandas
numpy
scikit-learn
gensim
matplotlib
wordcloud
jupyter
```

Environment:

```text
uv
```

Opsional:

```text
Docker
```

---

## 12. Rencana Output Penelitian

### Output 1

Topik dominan opini publik MBG.

Contoh:

- Gizi dan Stunting
- Keracunan Makanan
- Anggaran dan Efisiensi
- Dukungan Program
- Politik dan Pemerintahan

---

### Output 2

Perbandingan kualitas LDA dan NMF.

Contoh:

| Model | Topic Coherence |
| ----- | --------------- |
| LDA   | 0.43            |
| NMF   | 0.58            |

---

### Output 3

Visualisasi topik.

Bentuk:

- Word Cloud
- Top Words per Topic
- Topic Distribution

---

### Output 4

Analisis TikTok vs YouTube.

Contoh:

TikTok lebih banyak membahas:

- Politik
- Dukungan terhadap Prabowo

YouTube lebih banyak membahas:

- Implementasi MBG
- Kasus keracunan
- Efektivitas program

---

## 13. Outcome Penelitian

Penelitian ini diharapkan menghasilkan:

1. Identifikasi tema-tema utama yang dibahas masyarakat terkait Program Makan Bergizi Gratis.
2. Perbandingan empiris performa LDA dan NMF pada dataset komentar media sosial berbahasa Indonesia.
3. Analisis karakteristik diskusi publik pada platform TikTok dan YouTube.
4. Rekomendasi algoritma topic modeling yang paling sesuai untuk analisis opini publik MBG.

Kesimpulan akhir penelitian akan menunjukkan:

- Topik dominan yang menjadi perhatian masyarakat.
- Perbedaan pola diskusi antar platform.
- Algoritma terbaik (LDA atau NMF) berdasarkan kualitas topik yang dihasilkan.
