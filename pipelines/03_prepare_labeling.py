import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/labeling.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LabelingPreparator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir) / "labelled"
        self.data_dir.mkdir(exist_ok=True)

        # Label schema from research context - updated for MBG program
        self.label_schema = {
            "FAVOR": "Mendukung program MBG dan/atau pernyataan Prabowo",
            "AGAINST": "Menentang/mengkritik program MBG dan/atau pernyataan Prabowo",
            "PRO_GOV": "Mendukung kebijakan/program pemerintah secara umum",
            "CONTRA_GOV": "Menentang/mengkritik kebijakan pemerintah secara umum",
            "CONDITIONAL": "Mendukung sebagian, disertai syarat atau catatan kritis",
            "SUGGESTION": "Memberikan saran atau masukan konstruktif berbasis data",
            "DISCUSSION": "Reply/diskusi lateral antar pengguna",
            "OFF_TOPIC": "Tidak relevan dengan isu yang dibahas",
        }

        # Keywords for each label (for initial sampling) - updated for MBG context
        self.label_keywords = {
            "FAVOR": [
                "setuju",
                "benar",
                "tepat",
                "bagus",
                "mendukung",
                "support",
                "betul",
                "sama",
                "mbg bagus",
                "lanjutkan",
                "program baik",
                "bantu",
            ],
            "AGAINST": [
                "salah",
                "tidak setuju",
                "keliru",
                "berbeda",
                "kontra",
                "hentikan",
                "pemborosan",
                "tidak efektif",
                "gagal",
                "salah alih",
            ],
            "PRO_GOV": [
                "pemerintah",
                "presiden",
                "kementerian",
                "kebijakan",
                "program",
                "dukung",
                "negara",
                "bantuan",
            ],
            "CONTRA_GOV": [
                "pemerintah",
                "presiden",
                "kementerian",
                "kebijakan",
                "program",
                "kritik",
                "tidak peka",
                "salah urus",
                "korupsi",
            ],
            "CONDITIONAL": [
                "tapi",
                "namun",
                "walaupun",
                "meskipun",
                "asal",
                "syarat",
                "jika",
                "boleh asal",
                "setuju tapi",
            ],
            "SUGGESTION": [
                "sarankan",
                "usulkan",
                "sebaiknya",
                "harus",
                "perlu",
                "solusi",
                "fokuskan",
                "evaluasi",
                "perbaiki",
            ],
            "DISCUSSION": ["@", "reply", "balas", "komen", "pendapat", "setuju dengan"],
            "OFF_TOPIC": [
                "jualan",
                "promo",
                "iklan",
                "follow",
                "like",
                "subscribe",
                "promosi",
            ],
        }

    def load_processed_data(self) -> pd.DataFrame:
        """Load preprocessed comments data - updated for multi-platform"""
        # Load from preprocessed directory (where the processed data is stored)
        preprocessed_dir = Path("data") / "preprocessed"

        # Find the latest processed file
        processed_files = list(preprocessed_dir.glob("comments_processed*.csv"))

        if not processed_files:
            raise FileNotFoundError(
                f"No processed comments files found in {preprocessed_dir}"
            )

        # Get the latest file (highest number)
        latest_file = max(
            processed_files,
            key=lambda x: (
                int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0
            ),
        )

        df = pd.read_csv(latest_file, encoding="utf-8")
        logger.info(f"Loaded {len(df)} processed comments from {latest_file.name}")

        return df

    def create_labeling_dataset(
        self,
        df: pd.DataFrame,
        sample_size: int = 500,
        stratify_by: str = "source",  # Updated to use platform source
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Create stratified sample for manual labeling - updated for multi-platform"""

        logger.info(f"Creating labeling dataset with {sample_size} samples")

        # Filter for high-quality comments - updated column names
        df_quality = df[
            (df["clean_word_count"] >= 3)  # At least 3 words
            & (df["clean_word_count"] <= 50)  # Max 50 words
            & (df["is_spam"] == False)  # Non-spam comments only
        ].copy()

        logger.info(f"Filtered to {len(df_quality)} high-quality comments")

        # Stratified sampling by platform (source)
        if stratify_by in df_quality.columns:
            samples_per_source = sample_size // df_quality[stratify_by].nunique()

            sampled_comments = []
            for source in df_quality[stratify_by].unique():
                source_comments = df_quality[df_quality[stratify_by] == source]

                if len(source_comments) <= samples_per_source:
                    sampled_comments.append(source_comments)
                else:
                    # Random sample from this source
                    sample = source_comments.sample(
                        n=samples_per_source, random_state=random_state
                    )
                    sampled_comments.append(sample)

            df_sampled = pd.concat(sampled_comments, ignore_index=True)
        else:
            # Simple random sampling
            df_sampled = df_quality.sample(
                n=min(sample_size, len(df_quality)), random_state=random_state
            )

        # Add labeling columns
        df_sampled["label"] = ""
        df_sampled["annotator_id"] = ""
        df_sampled["labeling_date"] = ""
        df_sampled["confidence"] = ""
        df_sampled["notes"] = ""

        logger.info(f"Created labeling dataset with {len(df_sampled)} comments")

        return df_sampled

    def export_for_label_studio(
        self, df: pd.DataFrame, output_format: str = "json"
    ) -> str:
        """Export data for Label Studio import"""

        output_file = self.data_dir / f"labeling_dataset.{output_format}"

        if output_format == "json":
            # Label Studio JSON format
            tasks = []
            for idx, row in df.iterrows():
                # Handle NaN values
                reply_count = row["reply_count"]
                if pd.isna(reply_count):
                    reply_count = 0

                task = {
                    "id": idx,
                    "data": {
                        "raw_text": str(row["raw_text"]),
                        "clean_text": str(row["clean_text"]),
                        "final_text": str(row["final_text"]),
                        "source": str(row["source"]),
                        "created_at": str(row["created_at"]),
                        "likes_count": (
                            int(row["likes_count"])
                            if not pd.isna(row["likes_count"])
                            else 0
                        ),
                        "reply_count": int(reply_count),
                    },
                }
                tasks.append(task)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

        elif output_format == "csv":
            # Simple CSV format
            df.to_csv(output_file, index=False, encoding="utf-8")

        logger.info(f"Exported {len(df)} samples to {output_file}")
        return str(output_file)

    def analyze_labeling_readiness(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset readiness for labeling - updated for multi-platform"""

        analysis = {
            "total_comments": len(df),
            "platform_distribution": df["source"].value_counts().to_dict(),
            "text_length_stats": {
                "mean": df["clean_word_count"].mean(),
                "median": df["clean_word_count"].median(),
                "min": df["clean_word_count"].min(),
                "max": df["clean_word_count"].max(),
            },
            "spam_ratio": df["is_spam"].mean(),
            "quality_indicators": {
                "comments_3_50_words": (
                    (df["clean_word_count"] >= 3) & (df["clean_word_count"] <= 50)
                ).sum(),
                "non_spam": (df["is_spam"] == False).sum(),
                "unique_texts": df["final_text"].nunique(),
            },
        }

        return analysis

    def save_labeling_package(self, df_sampled: pd.DataFrame, analysis: Dict):
        """Save complete labeling package"""

        # Create labeling directory
        labeling_dir = self.data_dir
        labeling_dir.mkdir(exist_ok=True)

        # Save sampled dataset
        sampled_file = labeling_dir / "labeling_dataset.csv"
        df_sampled.to_csv(sampled_file, index=False, encoding="utf-8")

        # Export for Label Studio
        json_file = self.export_for_label_studio(df_sampled, "json")

        # Save analysis
        analysis_file = labeling_dir / "labeling_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Labeling package saved to {labeling_dir}")
        logger.info("Files created:")
        logger.info(f"  - {sampled_file}")
        logger.info(f"  - {json_file}")
        logger.info(f"  - {analysis_file}")


def main():
    """Main labeling preparation function"""
    try:
        # Initialize preparator
        preparator = LabelingPreparator()

        # Load processed data
        df = preparator.load_processed_data()

        # Analyze readiness
        analysis = preparator.analyze_labeling_readiness(df)
        logger.info("Data analysis completed")

        # Create labeling dataset
        df_sampled = preparator.create_labeling_dataset(df, sample_size=500)

        # Save labeling package
        preparator.save_labeling_package(df_sampled, analysis)

        logger.info("Labeling preparation completed successfully!")
        logger.info(f"Ready for manual annotation of {len(df_sampled)} comments")

        return df_sampled, analysis

    except Exception as e:
        logger.error(f"Labeling preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
