import argparse
import json
import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)


class TopicModelEvaluator:
    def __init__(
        self,
        data_dir: str = "data",
    ):
        self.data_dir = Path(data_dir)
        self.topic_dir = self.data_dir / "topic_modeling"
        self.plot_dir = self.topic_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _latest_preprocessed_file(self) -> Path:
        files = list((self.data_dir / "preprocessed").glob("comments_processed*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No preprocessed comments files found in {self.data_dir / 'preprocessed'}"
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

        logger.info(f"Using {len(df)} non-empty comments for evaluation")
        logger.info(f"Source distribution: {df['source'].value_counts().to_dict()}")
        return df

    def load_topics(self) -> pd.DataFrame:
        topics_file = self.topic_dir / "topics.csv"
        if not topics_file.exists():
            raise FileNotFoundError(f"Topics file not found: {topics_file}")
        topics = pd.read_csv(topics_file, encoding="utf-8")
        logger.info(f"Loaded topics from {topics_file}")
        return topics

    def load_document_topics(self) -> pd.DataFrame:
        doc_topics_file = self.topic_dir / "document_topics.csv"
        if not doc_topics_file.exists():
            raise FileNotFoundError(f"Document topics file not found: {doc_topics_file}")
        assignments = pd.read_csv(doc_topics_file, encoding="utf-8")
        logger.info(f"Loaded document topics from {doc_topics_file}")
        return assignments

    def load_models(self) -> Tuple:
        with open(self.topic_dir / "models" / "lda.pkl", "rb") as f:
            lda = pickle.load(f)
        with open(self.topic_dir / "models" / "nmf.pkl", "rb") as f:
            nmf = pickle.load(f)
        with open(self.topic_dir / "models" / "count_vectorizer.pkl", "rb") as f:
            count_vectorizer = pickle.load(f)
        with open(self.topic_dir / "models" / "tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        logger.info("Loaded models and vectorizers")
        return lda, nmf, count_vectorizer, tfidf_vectorizer

    def evaluate_topic_diversity(self, topics: pd.DataFrame) -> Dict[str, float]:
        diversity = {}
        for model_name in ["lda", "nmf"]:
            model_topics = topics[topics["model"] == model_name]
            words = []
            for text in model_topics["top_words"]:
                words.extend([word.strip() for word in text.split(",") if word.strip()])
            diversity[model_name] = len(set(words)) / len(words) if words else 0.0
        logger.info(f"Topic diversity: {diversity}")
        return diversity

    def evaluate_topic_coherence(
        self, matrix, feature_names: np.ndarray, topics: pd.DataFrame
    ) -> Dict[str, float]:
        coherence = {}
        for model_name in ["lda", "nmf"]:
            model_topics = topics[topics["model"] == model_name]
            binary = matrix.copy()
            binary.data = np.ones_like(binary.data)
            n_docs = binary.shape[0]
            if n_docs == 0:
                coherence[model_name] = 0.0
                continue

            term_to_idx = {term: idx for idx, term in enumerate(feature_names)}
            scores = []

            for _, row in model_topics.iterrows():
                words = [word.strip() for word in row["top_words"].split(",")]
                indices = [term_to_idx[word] for word in words if word in term_to_idx]
                if len(indices) < 2:
                    continue

                sub = binary[:, indices]
                doc_freq = np.asarray(sub.sum(axis=0)).ravel()
                cooccur = (sub.T @ sub).toarray()

                for i, j in combinations(range(len(indices)), 2):
                    pair_count = cooccur[i, j]
                    if pair_count <= 0:
                        continue

                    p_ij = pair_count / n_docs
                    p_i = doc_freq[i] / n_docs
                    p_j = doc_freq[j] / n_docs
                    if p_i <= 0 or p_j <= 0:
                        continue

                    pmi = np.log(p_ij / (p_i * p_j))
                    npmi = pmi / (-np.log(p_ij))
                    scores.append(npmi)

            coherence[model_name] = float(np.mean(scores)) if scores else 0.0
        logger.info(f"Topic coherence: {coherence}")
        return coherence

    def evaluate_topic_size(self, assignments: pd.DataFrame) -> pd.DataFrame:
        topic_size = (
            assignments.groupby(["model", "dominant_topic"])
            .size()
            .reset_index(name="count")
        )
        topic_size = topic_size.sort_values(["model", "dominant_topic"])
        topic_size.to_csv(self.topic_dir / "topic_size.csv", index=False, encoding="utf-8")
        logger.info(f"Saved topic size to {self.topic_dir / 'topic_size.csv'}")
        return topic_size

    def evaluate_platform_distribution(self, assignments: pd.DataFrame) -> pd.DataFrame:
        counts = (
            assignments.groupby(["model", "source", "dominant_topic"])
            .size()
            .reset_index(name="count")
        )
        totals = counts.groupby(["model", "source"])["count"].transform("sum")
        counts["percentage"] = (counts["count"] / totals * 100).round(2)
        counts.to_csv(
            self.topic_dir / "platform_topic_distribution.csv",
            index=False,
            encoding="utf-8",
        )
        logger.info(f"Saved platform distribution to {self.topic_dir / 'platform_topic_distribution.csv'}")
        return counts

    def evaluate_topic_similarity(self, lda, nmf) -> pd.DataFrame:
        similarity_data = []
        
        for model_name, model in [("lda", lda), ("nmf", nmf)]:
            sim_matrix = cosine_similarity(model.components_)
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    if i < j:
                        similarity_data.append({
                            "model": model_name,
                            "topic_i": i,
                            "topic_j": j,
                            "similarity": float(sim_matrix[i, j])
                        })
        
        similarity_df = pd.DataFrame(similarity_data)
        similarity_df.to_csv(
            self.topic_dir / "topic_similarity.csv",
            index=False,
            encoding="utf-8"
        )
        logger.info(f"Saved topic similarity to {self.topic_dir / 'topic_similarity.csv'}")
        return similarity_df

    def plot_top_words(self, model_name: str, model, feature_names: np.ndarray, n_topics: int, n_words: int):
        fig, axes = plt.subplots(
            n_topics,
            1,
            figsize=(10, max(3 * n_topics, 8)),
            constrained_layout=True,
        )
        if n_topics == 1:
            axes = [axes]

        for topic_idx, ax in enumerate(axes):
            weights = model.components_[topic_idx]
            top_indices = weights.argsort()[::-1][: n_words]
            words = [feature_names[i] for i in top_indices][::-1]
            values = [weights[i] for i in top_indices][::-1]

            sns.barplot(x=values, y=words, ax=ax, color="#2f6f73")
            ax.set_title(f"{model_name.upper()} Topic {topic_idx}")
            ax.set_xlabel("Weight")
            ax.set_ylabel("")

        output_path = self.plot_dir / f"{model_name}_top_words.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        logger.info(f"Saved plot: {output_path}")

    def plot_platform_distribution(self, distribution: pd.DataFrame):
        models = sorted(distribution["model"].unique())
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(7 * len(models), 5),
            sharey=True,
            constrained_layout=True,
        )
        if len(models) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, models):
            model_data = distribution[distribution["model"] == model_name].copy()
            model_data["dominant_topic"] = model_data["dominant_topic"].astype(int)
            sns.barplot(
                data=model_data,
                x="dominant_topic",
                y="percentage",
                hue="source",
                ax=ax,
            )
            ax.set_title(f"{model_name.upper()} Topic Distribution by Platform")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Percentage")

        output_path = self.plot_dir / "platform_topic_distribution.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        logger.info(f"Saved plot: {output_path}")

    def plot_topic_size(self, topic_size: pd.DataFrame):
        models = sorted(topic_size["model"].unique())
        
        for model_name in models:
            model_data = topic_size[topic_size["model"] == model_name].copy()
            model_data["dominant_topic"] = model_data["dominant_topic"].astype(int)
            
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            sns.barplot(
                data=model_data,
                x="dominant_topic",
                y="count",
                ax=ax,
                color="#2f6f73"
            )
            ax.set_title(f"{model_name.upper()} Topic Size")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Count")
            
            output_path = self.plot_dir / f"{model_name}_topic_size.png"
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
            logger.info(f"Saved plot: {output_path}")

    def generate_summary(
        self,
        df: pd.DataFrame,
        topics: pd.DataFrame,
        metrics: pd.DataFrame,
        distribution: pd.DataFrame,
        topic_size: pd.DataFrame,
    ):
        best_row = metrics.sort_values(
            ["topic_coherence_npmi", "topic_diversity"], ascending=False
        ).iloc[0]

        lines = [
            "# Topic Modeling Evaluation Summary",
            "",
            "## Dataset",
            "",
            f"- Total komentar digunakan: {len(df)}",
            f"- TikTok: {(df['source'] == 'tiktok').sum()}",
            f"- YouTube: {(df['source'] == 'youtube').sum()}",
            "",
            "## Model Comparison",
            "",
            "| Model | Topics | Coherence NPMI | Topic Diversity |",
            "| --- | ---: | ---: | ---: |",
            *[
                (
                    f"| {row['model'].upper()} | {int(row['n_topics'])} | "
                    f"{row['topic_coherence_npmi']:.4f} | {row['topic_diversity']:.4f} |"
                )
                for _, row in metrics.iterrows()
            ],
            "",
            "## Recommendation",
            "",
            (
                f"Model terbaik pada konfigurasi ini adalah {best_row['model'].upper()} "
                f"karena memiliki coherence NPMI {best_row['topic_coherence_npmi']:.4f} "
                f"dan topic diversity {best_row['topic_diversity']:.4f}."
            ),
            "",
            "## Topics",
            "",
        ]

        for model_name in ["lda", "nmf"]:
            lines.append(f"### {model_name.upper()}")
            lines.append("")
            for _, row in topics[topics["model"] == model_name].iterrows():
                lines.append(
                    f"- Topic {row['topic_id']} ({row['topic_label']}): "
                    f"{row['top_words']}"
                )
            lines.append("")

        lines.extend(
            [
                "## Output Files",
                "",
                "- `topics.csv`: top words per topic for LDA and NMF",
                "- `metrics.csv`: coherence and diversity comparison",
                "- `document_topics.csv`: dominant topic for each comment",
                "- `platform_topic_distribution.csv`: TikTok vs YouTube topic distribution",
                "- `topic_size.csv`: document count per topic",
                "- `topic_similarity.csv`: pairwise topic similarity",
                "- `plots/`: visualizations for top words, platform distribution, and topic size",
                "",
            ]
        )

        output_path = self.topic_dir / "summary.md"
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved summary: {output_path}")

    def run(self, n_topics: int = 5, n_words: int = 10) -> Dict:
        df = self.load_dataset()
        topics = self.load_topics()
        assignments = self.load_document_topics()
        lda, nmf, count_vectorizer, tfidf_vectorizer = self.load_models()

        count_matrix = count_vectorizer.transform(df["final_text"])
        tfidf_matrix = tfidf_vectorizer.transform(df["final_text"])

        count_features = np.array(count_vectorizer.get_feature_names_out())
        tfidf_features = np.array(tfidf_vectorizer.get_feature_names_out())

        diversity = self.evaluate_topic_diversity(topics)
        coherence_lda = self.evaluate_topic_coherence(count_matrix, count_features, topics[topics["model"] == "lda"])
        coherence_nmf = self.evaluate_topic_coherence(tfidf_matrix, tfidf_features, topics[topics["model"] == "nmf"])
        coherence = {"lda": coherence_lda.get("lda", 0.0), "nmf": coherence_nmf.get("nmf", 0.0)}

        topic_size = self.evaluate_topic_size(assignments)
        distribution = self.evaluate_platform_distribution(assignments)
        similarity = self.evaluate_topic_similarity(lda, nmf)

        metrics = pd.DataFrame([
            {
                "model": "lda",
                "n_topics": n_topics,
                "topic_coherence_npmi": coherence["lda"],
                "topic_diversity": diversity["lda"]
            },
            {
                "model": "nmf",
                "n_topics": n_topics,
                "topic_coherence_npmi": coherence["nmf"],
                "topic_diversity": diversity["nmf"]
            }
        ])

        metrics.to_csv(self.topic_dir / "metrics.csv", index=False, encoding="utf-8")
        metrics.to_json(
            self.topic_dir / "metrics.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
        logger.info(f"Saved metrics to {self.topic_dir / 'metrics.csv'} and {self.topic_dir / 'metrics.json'}")

        self.plot_top_words("lda", lda, count_features, n_topics, n_words)
        self.plot_top_words("nmf", nmf, tfidf_features, n_topics, n_words)
        self.plot_platform_distribution(distribution)
        self.plot_topic_size(topic_size)

        self.generate_summary(df, topics, metrics, distribution, topic_size)

        logger.info("Evaluation completed successfully")
        return {
            "records": len(df),
            "metrics_file": str(self.topic_dir / "metrics.csv"),
            "summary_file": str(self.topic_dir / "summary.md"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LDA and NMF topic models for MBG comments"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics")
    parser.add_argument("--words", type=int, default=10, help="Top words per topic")
    args = parser.parse_args()

    evaluator = TopicModelEvaluator(data_dir=args.data_dir)
    evaluator.run(n_topics=args.topics, n_words=args.words)


if __name__ == "__main__":
    main()
