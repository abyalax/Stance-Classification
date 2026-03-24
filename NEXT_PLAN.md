# Resumable YouTube Comments Scraper - Implementation Plan

## Overview
Plan untuk membuat sistem scraper YouTube comments yang **resumable dengan pagination**, dengan dukungan **jeda waktu multi-hari** antara fetch sessions.

---

## 1. Gunakan `nextPageToken` sebagai Checkpoint Utama

### Konsep
- Setiap respons dari `commentThreads().list()` mengembalikan `nextPageToken` untuk halaman berikutnya.
- **Simpan token ini ke file seperti pipeline_status.json** setelah setiap halaman selesai diproses.
- Token ini adalah *stateless pointer* — selama token masih valid (biasanya 1-3 hari), masih bisa melanjutkan dari titik itu.

### Catatan dari Dokumentasi
- *"A call to this method has a quota cost of 1 unit."*
- Artinya: setiap request (termasuk resume) menghitung kuota.
---

## 2. Simpan Metadata Tambahan untuk Resume yang Aman

### Data yang Perlu Disimpan
Selain `nextPageToken`, simpan juga:
- `video_id` — untuk memastikan resume di video yang benar
- `page_number` — untuk logging dan debugging
- `total_comments_so_far` — untuk memantau progress
- `timestamp` — untuk mengetahui kapan terakhir checkpoint dibuat
- `total_available_comments` — jumlah total komentar di video (dari respons pertama)

### Struktur Checkpoint (JSON)
```json
{
  "video_id": "S4iey1sr1Cg",
  "next_page_token": "EAEaQ...",
  "page": 12,
  "total_collected": 1283,
  "total_available": 6394,
  "last_updated": "2026-03-24T17:09:11Z",
  "status": "in_progress"
}
```

### Lokasi Penyimpanan
- File: `data/{video_id}_checkpoint.json`
---

## 3. Handle Token Kadaluarsa (Expired)

### Masalah
Karena resume setelah beberapa hari, **token bisa kadaluarsa**.

### Strategi Fallback

#### ✅ Opsi A: Mulai Ulang dari Awal (Recommended)
- Jika `nextPageToken` gagal digunakan (error 400/403), **mulai ulang dari awal** (`pageToken=None`).
- **Jangan buang data lama** — gabungkan dengan data baru.
- Gunakan `order="relevance"` atau `order="time"` **konsisten** agar hasilnya tidak tumpang tindih.

#### ✅ Opsi B: Simpan Checkpoint Terakhir (Advanced)
- Simpan `publishedAt` komentar terakhir yang diambil.
- Saat resume, jika token gagal, filter data baru berdasarkan timestamp.
- **Catatan**: YouTube API tidak mendukung `publishedAfter` untuk `commentThreads.list()` — ini hanya opsional jika ada fitur tambahan.

#### ✅ Opsi C: Hybrid Approach
- Coba gunakan token lama terlebih dahulu.
- Jika gagal, mulai ulang dari awal.
- Gunakan `comment_id` untuk deduplikasi otomatis.

---

## 4. Simpan Data Sementara Secara Incremental

### Strategi Penyimpanan

#### Per-Page Files
- Setiap halaman, simpan data ke file terpisah.
- Struktur: `data/scrapped/video_{video_id}/page_{page_number}.json`
- Contoh:
  ```
  data/
  └──scrapped/
    └── video_S4iey1sr1Cg/
        ├── page_1.json
        ├── page_2.json
        ├── page_11.json
        └── page_12.json
  ```

#### Partial CSV
- Simpan juga ke CSV incremental.
- File: `data/scrapped/youtube_comments_raw_{increment}.csv`
- Setiap halaman, **append** data baru (jangan overwrite).

### Keuntungan
- Mencegah kehilangan data jika proses gagal di tengah jalan.
- Mudah untuk debugging (lihat data per halaman).
- Saat resume, tinggal lanjutkan dari halaman berikutnya.

---

## 5. Validasi dan Deduplikasi

### Deduplikasi Strategy
- Gunakan `comment_id` sebagai unique key.
- Saat resume, cek apakah `comment_id` sudah ada di data lama.
- Jika ada duplikasi, skip atau merge (update timestamp jika ada perubahan).

### Implementasi
- Simpan set of `comment_id` di memory.
- Sebelum append ke data final, cek keberadaannya.

---

## 6. Logging dan Monitoring

### Log yang Perlu Dicatat
- **Checkpoint events**: Kapan checkpoint dibuat, di halaman berapa.
- **Resume events**: Kapan resume dimulai, dari halaman berapa, token status.
- **Error events**: Token kadaluarsa, API error, quota error.
- **Timing**: Durasi per halaman, total waktu scraping.

### Format Log
```
2026-03-24 17:09:11 - INFO - [S4iey1sr1Cg] Page 11: 1283 comments collected
2026-03-24 17:09:11 - INFO - [S4iey1sr1Cg] Checkpoint saved: page=11, token=EAEaQ...
2026-03-25 10:00:00 - INFO - [S4iey1sr1Cg] Resuming from checkpoint: page=12
2026-03-25 10:00:01 - ERROR - [S4iey1sr1Cg] Token expired, restarting from page 1
```

---

## 7. Alur Kerja Resume (Step-by-Step)

### Session Pertama (Fresh Start)
1. Cek apakah ada file checkpoint untuk `video_id`.
2. Tidak ada → mulai dari `pageToken=None`, `page=1`.
3. Fetch halaman pertama → simpan data.
4. Simpan checkpoint dengan `nextPageToken`.
5. Loop: fetch halaman berikutnya → simpan data → update checkpoint.
6. Saat selesai (tidak ada `nextPageToken`), ubah status checkpoint ke `"completed"`.

### Session Kedua (Resume setelah jeda)
1. Cek apakah ada file checkpoint untuk `video_id`.
2. Ada → baca `nextPageToken`, `page`, `total_collected`.
3. **Coba gunakan `nextPageToken`**:
   - ✅ Berhasil → lanjutkan dari halaman berikutnya.
   - ❌ Gagal (token expired) → mulai ulang dari `pageToken=None`, tapi **jangan hapus data lama**.
4. Fetch halaman → gabungkan dengan data lama → deduplikasi.
5. Update checkpoint.
6. Loop sampai selesai.

### Session Ketiga+ (Jika diperlukan)
- Ulangi proses Session Kedua.

---

## 8. Risiko & Mitigasi

| Risiko | Penyebab | Mitigasi |
|--------|---------|----------|
| Token kadaluarsa | Jeda > 3 hari | Mulai ulang dari awal, gabungkan data lama |
| Kuota habis | Terlalu banyak request | Tambah delay (`time.sleep(1)`), monitor kuota |
| Duplikasi data | Resume dengan data lama | Gunakan `comment_id` sebagai unique key, deduplikasi |
| Data tidak lengkap | Proses terhenti | Simpan incremental per halaman |
| Perubahan data YouTube | Komentar dihapus/diubah | Gunakan `order="relevance"` konsisten, accept eventual consistency |
| Checkpoint corrupt | File error | Backup checkpoint, validasi JSON saat load |

---

## 9. Validasi & Testing Plan

### Unit Tests
- Test checkpoint save/load.
- Test deduplikasi logic.
- Test fallback saat token expired.

### Integration Tests
- Test resume setelah 1 hari.
- Test resume setelah 5 hari (token expired).
- Test multi-video scraping.

### Manual Testing
- Jalankan scraper, hentikan di halaman 5.
- Resume, pastikan lanjut dari halaman 6.
- Hentikan lagi, resume, pastikan tidak ada duplikasi.

---

## 10. Kesimpulan

✅ **Bisa membuat sistem resumable** dengan:
- Menyimpan `nextPageToken` + metadata ke file checkpoint.
- Menangani token kadaluarsa dengan fallback ke awal.
- Menyimpan data incremental per halaman untuk keamanan.
- Menggabungkan data lama dan baru saat resume.
- Deduplikasi menggunakan `comment_id`.

✅ **Dokumentasi YouTube API mendukung ini** — `nextPageToken` dirancang untuk pagination, dan tidak ada batasan resume selama kuota tersedia.

✅ **Aman dan valid** — selama Anda menghormati kuota dan tidak spam.

---

## 11. Next Steps

1. Implementasi checkpoint save/load logic.
2. Implementasi deduplikasi logic.
3. Implementasi fallback saat token expired.
4. Tambahkan comprehensive logging.
5. Testing di berbagai skenario (token valid, token expired, etc).
```