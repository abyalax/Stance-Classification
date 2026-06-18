import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/topic_modeling.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

TOPIC_STOPWORDS = {
    "aja",
    "bapak",
    "bener",
    "bergizi",
    "bgm",
    "bgt",
    "bpk",
    "dah",
    "dong",
    "ga",
    "gak",
    "gk",
    "gratis",
    "iya",
    "jd",
    "kak",
    "kayak",
    "kok",
    "lah",
    "liat",
    "makan",
    "mbg",
    "nya",
    "orang",
    "pak",
    "program",
    "saja",
    "sih",
    "tau",
    "tdk",
    "tidak",
    "ya",
    "yaa",
    "yang",
}


class TopicModelingPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        n_topics: int = 5,
        n_words: int = 10,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.output_dir = self.data_dir / "topic_modeling"
        self.models_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.n_topics = n_topics
        self.n_words = n_words
        self.random_state = random_state

    def _latest_preprocessed_file(self) -> Path:
        files = list(self.preprocessed_dir.glob("comments_processed*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No preprocessed comments files found in {self.preprocessed_dir}"
            )

        def version(path: Path) -> int:
            suffix = path.stem.split("_")[-1]
            return int(suffix) if suffix.isdigit() else 0

        return max(files, key=version)

    def load_dataset(self) -> pd.DataFrame:
        input_file = self._latest_preprocessed_file()
        df = pd.read_csv(input_file, encoding="utf-8")
        logger.info(f"Loaded {len(df)} records from {input_file}")

        required_columns = {"final_text", "source"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.copy()
        df["final_text"] = df["final_text"].fillna("").astype(str).str.strip()
        df = df[df["final_text"].str.len() > 0].reset_index(drop=True)
        df["source"] = df["source"].fillna("unknown").astype(str).str.lower()

        logger.info(f"Using {len(df)} non-empty comments for topic modeling")
        logger.info(f"Source distribution: {df['source'].value_counts().to_dict()}")
        return df

    def _build_vectorizers(self) -> Tuple[CountVectorizer, TfidfVectorizer]:
        common_params = {
            "max_df": 0.9,
            "min_df": 5,
            "max_features": 3000,
            "ngram_range": (1, 2),
            "token_pattern": r"(?u)\b\w\w+\b",
            "stop_words": list(TOPIC_STOPWORDS),
        }
        return CountVectorizer(**common_params), TfidfVectorizer(**common_params)

    def _extract_topics(
        self, model_name: str, model, feature_names: np.ndarray
    ) -> pd.DataFrame:
        rows = []
        for topic_idx, topic_weights in enumerate(model.components_):
            top_indices = topic_weights.argsort()[::-1][: self.n_words]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [float(topic_weights[i]) for i in top_indices]

            rows.append(
                {
                    "model": model_name,
                    "topic_id": topic_idx,
                    "topic_label": self._infer_topic_label(top_words),
                    "top_words": ", ".join(top_words),
                    "top_weights": json.dumps(top_weights),
                }
            )
        return pd.DataFrame(rows)

    def _infer_topic_label(self, words: List[str]) -> str:
        word_set = set(words)

        if word_set & {"korupsi", "ladang", "lahan", "sarang"}:
            return "Korupsi dan penyalahgunaan program"
        if word_set & {"dapur", "menu", "guru", "dana"}:
            return "Implementasi dapur, menu, dan sekolah"
        if word_set & {"anak", "sekolah", "stunting", "gizi", "makanan"}:
            return "Anak, sekolah, dan manfaat gizi"
        if word_set & {"stop", "mending", "evaluasi", "dikorupsi"}:
            return "Penghentian atau evaluasi program"
        if word_set & {"periode", "lanjutkan", "beliau", "mantap"}:
            return "Dukungan politik terhadap Prabowo"
        if word_set & {"ganti", "omon", "salah", "keras"}:
            return "Kritik terhadap presiden atau pemerintah"
        if word_set & {"hentikan"}:
            return "Penghentian atau evaluasi program"
        if word_set & {"anggaran", "uang", "negara", "rakyat"}:
            return "Anggaran dan alokasi dana"
        return "Tema umum diskusi MBG"

    def _assign_document_topics(
        self, df: pd.DataFrame, model_name: str, doc_topic_matrix: np.ndarray
    ) -> pd.DataFrame:
        result = df[["source", "final_text"]].copy()
        result["model"] = model_name
        result["dominant_topic"] = doc_topic_matrix.argmax(axis=1)
        result["dominant_topic_score"] = doc_topic_matrix.max(axis=1)
        return result

    def save_topics(self, topics: pd.DataFrame, assignments: pd.DataFrame):
        topics.to_csv(self.output_dir / "topics.csv", index=False, encoding="utf-8")
        assignments.to_csv(
            self.output_dir / "document_topics.csv", index=False, encoding="utf-8"
        )
        logger.info("Saved topics and document topics")

    def save_models(
        self,
        lda: LatentDirichletAllocation,
        nmf: NMF,
        count_vectorizer: CountVectorizer,
        tfidf_vectorizer: TfidfVectorizer,
    ):
        with open(self.models_dir / "lda.pkl", "wb") as f:
            pickle.dump(lda, f)
        with open(self.models_dir / "nmf.pkl", "wb") as f:
            pickle.dump(nmf, f)
        with open(self.models_dir / "count_vectorizer.pkl", "wb") as f:
            pickle.dump(count_vectorizer, f)
        with open(self.models_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        logger.info("Saved models and vectorizers")

    def run(self) -> Dict:
        df = self.load_dataset()
        count_vectorizer, tfidf_vectorizer = self._build_vectorizers()

        count_matrix = count_vectorizer.fit_transform(df["final_text"])
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["final_text"])

        logger.info(f"Count matrix shape: {count_matrix.shape}")
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method="batch",
            random_state=self.random_state,
            max_iter=30,
        )
        nmf = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            init="nndsvda",
            max_iter=500,
        )

        lda_doc_topics = lda.fit_transform(count_matrix)
        nmf_doc_topics = nmf.fit_transform(tfidf_matrix)

        count_features = np.array(count_vectorizer.get_feature_names_out())
        tfidf_features = np.array(tfidf_vectorizer.get_feature_names_out())

        lda_topics = self._extract_topics("lda", lda, count_features)
        nmf_topics = self._extract_topics("nmf", nmf, tfidf_features)
        topics = pd.concat([lda_topics, nmf_topics], ignore_index=True)

        lda_assignments = self._assign_document_topics(df, "lda", lda_doc_topics)
        nmf_assignments = self._assign_document_topics(df, "nmf", nmf_doc_topics)
        assignments = pd.concat([lda_assignments, nmf_assignments], ignore_index=True)

        self.save_topics(topics, assignments)
        self.save_models(lda, nmf, count_vectorizer, tfidf_vectorizer)

        logger.info("Topic modeling completed successfully")
        return {
            "records": len(df),
            "topics_file": str(self.output_dir / "topics.csv"),
            "document_topics_file": str(self.output_dir / "document_topics.csv"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run LDA and NMF topic modeling for MBG comments"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics")
    parser.add_argument("--words", type=int, default=10, help="Top words per topic")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    pipeline = TopicModelingPipeline(
        data_dir=args.data_dir,
        n_topics=args.topics,
        n_words=args.words,
        random_state=args.random_state,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
