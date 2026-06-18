# Topic Modeling Evaluation Summary

## Dataset

- Total komentar digunakan: 3611
- TikTok: 2694
- YouTube: 917

## Model Comparison

| Model | Topics | Coherence NPMI | Topic Diversity |
| --- | ---: | ---: | ---: |
| LDA | 5 | 0.1256 | 0.8200 |
| NMF | 5 | 0.1691 | 0.8400 |

## Recommendation

Model terbaik pada konfigurasi ini adalah NMF karena memiliki coherence NPMI 0.1691 dan topic diversity 0.8400.

## Topics

### LDA

- Topic 0 (Penghentian atau evaluasi program): rakyat, negara, stop, anggaran, benar, uang, presiden, bagus, jangan, bukan
- Topic 1 (Anak, sekolah, dan manfaat gizi): anak, baik, lapangan, indonesia, presiden, prabowo, masyarakat, kasih, gizi, sehat
- Topic 2 (Anak, sekolah, dan manfaat gizi): anak, sekolah, prabowo, desa, daerah, stunting, anak anak, ekonomi, makanan, lanjutkan
- Topic 3 (Implementasi dapur, menu, dan sekolah): dikorupsi, sekolah, setuju, menu, guru, dana, uang, dapur, mending, langsung
- Topic 4 (Korupsi dan penyalahgunaan program): korupsi, presiden, ladang, periode, jangan, kepala, uang, ladang korupsi, keras, bagus

### NMF

- Topic 0 (Korupsi dan penyalahgunaan program): korupsi, ladang, ladang korupsi, uang, uang korupsi, lahan, bukan, lahan korupsi, gimana, sarang korupsi
- Topic 1 (Anak, sekolah, dan manfaat gizi): anak, prabowo, sekolah, rakyat, uang, anak anak, indonesia, kasih, baik, makanan
- Topic 2 (Penghentian atau evaluasi program): stop, setuju, stop stop, pendidikan, dikorupsi, jangan, mending, pa, mending stop, ganti
- Topic 3 (Dukungan politik terhadap Prabowo): periode, setuju, lanjutkan, prabowo, beliau, ngapain, baik, lanjutkan prabowo, ganti, mantap
- Topic 4 (Kritik terhadap presiden atau pemerintah): presiden, rakyat, ganti, indonesia, ganti presiden, hentikan, salah, keras, omon, sehat

## Output Files

- `topics.csv`: top words per topic for LDA and NMF
- `metrics.csv`: coherence and diversity comparison
- `document_topics.csv`: dominant topic for each comment
- `platform_topic_distribution.csv`: TikTok vs YouTube topic distribution
- `topic_size.csv`: document count per topic
- `topic_similarity.csv`: pairwise topic similarity
- `plots/`: visualizations for top words, platform distribution, and topic size
