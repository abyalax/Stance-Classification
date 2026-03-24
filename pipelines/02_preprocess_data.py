import re
import logging
import json
import nltk

import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─── Base Preprocessor ─────────────────────────────────────────────────────────


class BasePreprocessor:
    """Base class for platform-specific preprocessors"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir) / "preprocessed"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Download required NLTK data
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        # Initialize Indonesian stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        # Load custom stopwords (exclude important negators)
        self.stop_words = self._load_custom_stopwords()

        # Load slang dictionary
        self.slang_dict = self._load_slang_dict()

        # Patterns for cleaning
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.mention_pattern = re.compile(r"@[A-Za-z0-9_]+")
        self.hashtag_pattern = re.compile(r"#[A-Za-z0-9_]+")
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+"
        )
        self.excessive_pattern = re.compile(r"(.)\1{2,}")  # Excessive characters
        self.number_pattern = re.compile(r"\d+")  # Numbers

        # Additional custom stopwords for social media (exclude important negators)
        self.custom_stopwords = {
            "yg",
            "yang",
            "dgn",
            "dengan",
            "untuk",
            "untk",
            "pd",
            "pada",
            "dll",
            "dsb",
            "dan",
            "atau",
            "tapi",
            "tp",
            "jg",
            "juga",
            "bgt",
            "banget",
            "sih",
            "dong",
            "deh",
            "nih",
            "toh",
            "kok",
            "yah",
            "ya",
            "kan",
            "lah",
            "tuh",
            "mah",
            "sdh",
            "sudah",
            "blm",
            "belum",
            "lg",
            "lagi",
            "aja",
            "saja",
            "klo",
            "kalau",
            "kalo",
        }

    def _load_custom_stopwords(self) -> set:
        """Load Indonesian stopwords with important negators excluded"""
        # Standard Indonesian stopwords
        standard_stopwords = set(stopwords.words("indonesian"))

        # IMPORTANT: Keep these negators and stance-related words
        important_words = {
            "tidak",
            "bukan",
            "jangan",
            "belum",
            "tak",
            "gak",
            "ga",
            "nggak",
            "tdk",
            "g",
            "jgn",
            "jngn",
            "setuju",
            "pro",
            "kontra",
            "mendukung",
            "menolak",
            "menyetujui",
            "baik",
            "buruk",
            "bagus",
            "jelek",
            "salah",
            "benar",
            "mungkin",
        }

        # Remove important words from stopwords
        custom_stopwords = standard_stopwords - important_words

        # Add social media specific stopwords
        social_stopwords = {
            "yg",
            "yang",
            "dgn",
            "dengan",
            "untuk",
            "untk",
            "pd",
            "pada",
            "dll",
            "dsb",
            "dan",
            "atau",
            "tapi",
            "tp",
            "jg",
            "juga",
            "bgt",
            "banget",
            "sih",
            "dong",
            "deh",
            "nih",
            "toh",
            "kok",
            "yah",
            "ya",
            "kan",
            "lah",
            "tuh",
            "mah",
            "sdh",
            "sudah",
            "blm",
            "belum",
            "lg",
            "lagi",
            "aja",
            "saja",
            "klo",
            "kalau",
            "kalo",
        }

        return custom_stopwords.union(social_stopwords)

    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, hashtags, emojis, etc."""

        # Remove URLs
        text = self.url_pattern.sub(" ", text)

        # Remove mentions
        text = self.mention_pattern.sub(" ", text)

        # Remove hashtags
        text = self.hashtag_pattern.sub(" ", text)

        # Remove emojis
        text = self.emoji_pattern.sub(" ", text)

        # Remove excessive characters
        text = self.excessive_pattern.sub(r"\1\1", text)

        # Remove numbers
        text = self.number_pattern.sub(" ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _load_slang_dict(self) -> Dict[str, str]:
        """Load Indonesian slang normalization dictionary"""
        slang_dict = {
            # Common abbreviations
            "apk": "aplikasi",
            "aj": "saja",
            "bgt": "banget",
            "bgtt": "banget",
            "bgts": "banget",
            "brp": "berapa",
            "bs": "bisa",
            "bsh": "bisa",
            "dy": "dia",
            "dr": "dari",
            "dri": "dari",
            "dg": "dengan",
            "dgn": "dengan",
            "dlm": "dalam",
            "dpt": "dapat",
            "emang": "memang",
            "g": "tidak",
            "gini": "begini",
            "gitu": "begitu",
            "gt": "gitu",
            "gtau": "tidak tahu",
            "ga tau": "tidak tahu",
            "gak tau": "tidak tahu",
            "hr": "hari",
            "hrs": "harus",
            "jg": "juga",
            "jgn": "jangan",
            "jk": "jika",
            "kalo": "kalau",
            "klu": "kalau",
            "klw": "kalau",
            "keknya": "sepertinya",
            "km": "kamu",
            "kmu": "kamu",
            "knp": "kenapa",
            "krn": "karena",
            "karna": "karena",
            "lbh": "lebih",
            "mngkn": "mungkin",
            "mrk": "mereka",
            "msk": "masuk",
            "msh": "masih",
            "nggak": "tidak",
            "ngak": "tidak",
            "gak": "tidak",
            "ga": "tidak",
            "pd": "pada",
            "pdk": "pendapat",
            "pnya": "punya",
            "py": "punya",
            "sdh": "sudah",
            "sdhkan": "sudah",
            "skrg": "sekarang",
            "skrang": "sekarang",
            "sm": "sama",
            "smp": "sampai",
            "spt": "seperti",
            "sprti": "seperti",
            "seumur2": "seumur hidup",
            "sy": "saya",
            "sya": "saya",
            "sllu": "selalu",
            "tlg": "tolong",
            "tlong": "tolong",
            "tp": "tapi",
            "ttg": "tentang",
            "utk": "untuk",
            "untk": "untuk",
            "wkt": "waktu",
            "wktu": "waktu",
            "ya": "iya",
            "yah": "iya",
            "yuk": "ayo",
            # Economic/financial terms
            "bbm": "bahan bakar minyak",
            "pertalite": "pertalite",
            "pertamax": "pertamax",
            "solar": "solar",
            "minyak": "minyak",
            "bensin": "bensin",
            "apbn": "anggaran pendapatan dan belanja negara",
            "defisit": "defisit",
            "inflasi": "inflasi",
            "rupiah": "rupiah",
            "kurs": "kurs",
            "subsidy": "subsidi",
            "subsidi": "subsidi",
            "fiskal": "fiskal",
            "moneter": "moneter",
            "neraca": "neraca",
            "pembayaran": "pembayaran",
            "impor": "impor",
            "ekspor": "ekspor",
            "mbg": "makan bergizi gratis",
            "makan bergizi gratis": "makan bergizi gratis",
            # Common expressions
            "hadeh": "hadeh",
            "aduh": "aduh",
            "wih": "wih",
            "wow": "wow",
            "hmm": "hmm",
            "hemm": "hmm",
            "gitu": "begitu",
            "gini": "begini",
            "begini": "begini",
            "begitu": "begitu",
            "dong": "dong",
            "donk": "dong",
            "dunk": "dong",
            "kan": "kan",
            "kok": "kok",
            "ko": "kok",
            "loh": "loh",
            "lho": "loh",
            "toh": "toh",
            "mah": "mah",
            "nih": "ini",
            "nie": "ini",
            "nihh": "ini",
        }
        return slang_dict

    def clean_text(self, text: str) -> str:
        """Clean individual text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = self.url_pattern.sub(" ", text)

        # Remove mentions
        text = self.mention_pattern.sub(" ", text)

        # Remove hashtags (keep the text without #)
        text = self.hashtag_pattern.sub(" ", text)

        # Remove emojis
        text = self.emoji_pattern.sub(" ", text)

        # Remove excessive characters (more than 2 consecutive)
        text = self.excessive_pattern.sub(r"\1\1", text)

        # Remove numbers
        text = self.number_pattern.sub(" ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_slang(self, text: str) -> str:
        """Normalize Indonesian slang words"""
        # First, handle specific patterns with punctuation
        text = re.sub(r"\btlg\.di\b", "tolong.di", text, flags=re.IGNORECASE)
        text = re.sub(r"\bklo\.\b", "kalau.", text, flags=re.IGNORECASE)
        text = re.sub(r"\btp\.i\b", "tapi.i", text, flags=re.IGNORECASE)
        text = re.sub(r"\bjg\.n\b", "juga.n", text, flags=re.IGNORECASE)

        # Then handle normal slang words
        words = text.split()
        normalized_words = []

        for word in words:
            # Clean word from punctuation at edges
            clean_word = word.lower().strip(".,!?;:\"'()[]{}")

            if clean_word in self.slang_dict:
                # Check if original word had punctuation and preserve it
                if word.endswith("."):
                    normalized_words.append(self.slang_dict[clean_word] + ".")
                elif word.endswith(","):
                    normalized_words.append(self.slang_dict[clean_word] + ",")
                else:
                    normalized_words.append(self.slang_dict[clean_word])
            else:
                normalized_words.append(word)

        return " ".join(normalized_words)

    def remove_stopwords(self, text: str) -> str:
        """Remove Indonesian stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

    def stem_text(self, text: str) -> str:
        """Stem Indonesian text"""
        return self.stemmer.stem(text)

    def preprocess_text(
        self,
        text: str,
        clean: bool = True,
        normalize: bool = True,
        remove_stopwords: bool = True,
        stem: bool = False,
    ) -> str:
        """Complete preprocessing pipeline"""
        if not isinstance(text, str) or pd.isna(text):
            return ""

        processed_text = text

        if clean:
            processed_text = self.clean_text(processed_text)

        if normalize:
            processed_text = self.normalize_slang(processed_text)

        if remove_stopwords:
            processed_text = self.remove_stopwords(processed_text)

        if stem:
            processed_text = self.stem_text(processed_text)

        return processed_text.strip()

    def detect_spam(self, text: str) -> bool:
        """Detect potential spam comments"""
        if not isinstance(text, str) or pd.isna(text):
            return True

        text_lower = text.lower()

        # Very short comments (less than 3 characters)
        if len(text.strip()) < 3:
            return True

        # Only numbers or symbols
        if re.match(r"^[\d\s\W]+$", text):
            return True

        # Repetitive characters
        if self.excessive_pattern.search(text):
            return True

        # Common spam patterns
        spam_patterns = [
            r"^(ok|sip|mantap|nice|good|great|wow|wih)+\s*$",
            r"^(follow|like|comment|subscribe)+\s*$",
            r"^(cek|cekibrot|cekkidmat)+\s*$",
            r"^(promo|jual|beli|order)+\s*$",
            r"^\w+\s*\w*\s*$",  # Single word repeated
        ]

        for pattern in spam_patterns:
            if re.match(pattern, text_lower):
                return True

        return False


# ─── TikTok Preprocessor ─────────────────────────────────────────────────────


class TiktokPreprocessor(BasePreprocessor):
    """TikTok-specific data preprocessor"""

    def __init__(self, data_dir: str = "data"):
        # Initialize raw data directory for loading
        self.raw_data_dir = Path(data_dir) / "scrapped"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parent for preprocessing methods (output to preprocessed)
        super().__init__(data_dir)

    def load_raw_data(self) -> pd.DataFrame:
        """Load TikTok raw data and extract relevant fields"""
        tiktok_file = None

        # Find the latest TikTok comments file
        for file in sorted(
            self.raw_data_dir.glob("tiktok_comments_raw*.csv"), reverse=True
        ):
            if file.exists():
                tiktok_file = file
                break

        if not tiktok_file:
            raise FileNotFoundError("TikTok comments file not found")

        # Load relevant columns: text, createTimeISO, replyCommentTotal, diggCount
        df = pd.read_csv(
            tiktok_file,
            encoding="utf-8",
            usecols=["text", "createTimeISO", "replyCommentTotal", "diggCount"],
        )

        # Standardize field names
        df = df.rename(
            columns={
                "text": "raw_text",
                "createTimeISO": "created_at",
                "replyCommentTotal": "reply_count",
                "diggCount": "likes_count",
            }
        )

        # Add source column
        df["source"] = "tiktok"

        logger.info(f"Loaded {len(df)} TikTok comments from {tiktok_file}")
        return df

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess TikTok dataset"""
        logger.info("Starting TikTok data preprocessing...")

        df_processed = df.copy()

        # Add preprocessing columns
        df_processed["is_spam"] = df_processed["raw_text"].apply(self.detect_spam)
        df_processed["clean_text"] = df_processed["raw_text"].apply(
            self.preprocess_text
        )
        df_processed["normalized_text"] = df_processed["raw_text"].apply(
            lambda x: self.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=False, stem=False
            )
        )
        df_processed["final_text"] = df_processed["raw_text"].apply(
            lambda x: self.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=True, stem=False
            )
        )

        # Add text statistics
        df_processed["text_length"] = df_processed["raw_text"].str.len()
        df_processed["word_count"] = df_processed["raw_text"].str.split().str.len()
        df_processed["clean_text_length"] = df_processed["clean_text"].str.len()
        df_processed["clean_word_count"] = (
            df_processed["clean_text"].str.split().str.len()
        )

        # Filter out spam and empty comments
        df_filtered = df_processed[
            (~df_processed["is_spam"]) & (df_processed["clean_text"].str.len() > 0)
        ].copy()

        logger.info(
            f"Filtered out {len(df_processed) - len(df_filtered)} spam/empty comments"
        )
        logger.info(f"Remaining TikTok comments: {len(df_filtered)}")

        return df_filtered

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze TikTok data quality"""
        analysis = {
            "platform": "tiktok",
            "total_comments": len(df),
            "avg_text_length": df["raw_text"].str.len().mean(),
            "avg_word_count": df["raw_text"].str.split().str.len().mean(),
            "avg_likes_per_comment": df["likes_count"].mean(),
            "max_likes": df["likes_count"].max(),
            "min_likes": df["likes_count"].min(),
            "avg_reply_count": df["reply_count"].mean(),
            "comments_with_no_likes": (df["likes_count"] == 0).sum(),
            "comments_with_likes": (df["likes_count"] > 0).sum(),
            "comments_with_replies": (df["reply_count"] > 0).sum(),
        }
        return analysis


# ─── YouTube Preprocessor ─────────────────────────────────────────────────────


class YoutubePreprocessor(BasePreprocessor):
    """YouTube-specific data preprocessor"""

    def __init__(self, data_dir: str = "data"):
        # Initialize raw data directory for loading
        self.raw_data_dir = Path(data_dir) / "scrapped"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parent for preprocessing methods (output to preprocessed)
        super().__init__(data_dir)

    def load_raw_data(self) -> pd.DataFrame:
        """Load YouTube raw data and extract relevant fields"""
        youtube_file = None

        # Find the latest YouTube comments file
        for file in sorted(
            self.raw_data_dir.glob("youtube_comments_raw*.csv"), reverse=True
        ):
            if file.exists():
                youtube_file = file
                break

        if not youtube_file:
            raise FileNotFoundError("YouTube comments file not found")

        # Load relevant columns: text, likes_count, published_at, reply_count
        df = pd.read_csv(
            youtube_file,
            encoding="utf-8",
            usecols=["text", "likes_count", "published_at", "reply_count"],
        )

        # Standardize field names
        df = df.rename(
            columns={
                "text": "raw_text",
                "published_at": "created_at",
                "likes_count": "likes_count",  # Already standardized
                "reply_count": "reply_count",  # Already standardized
            }
        )

        # Add source column
        df["source"] = "youtube"

        logger.info(f"Loaded {len(df)} YouTube comments from {youtube_file}")
        return df

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess YouTube dataset"""
        logger.info("Starting YouTube data preprocessing...")

        df_processed = df.copy()

        # Add preprocessing columns
        df_processed["is_spam"] = df_processed["raw_text"].apply(self.detect_spam)
        df_processed["clean_text"] = df_processed["raw_text"].apply(
            self.preprocess_text
        )
        df_processed["normalized_text"] = df_processed["raw_text"].apply(
            lambda x: self.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=False, stem=False
            )
        )
        df_processed["final_text"] = df_processed["raw_text"].apply(
            lambda x: self.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=True, stem=False
            )
        )

        # Add text statistics
        df_processed["text_length"] = df_processed["raw_text"].str.len()
        df_processed["word_count"] = df_processed["raw_text"].str.split().str.len()
        df_processed["clean_text_length"] = df_processed["clean_text"].str.len()
        df_processed["clean_word_count"] = (
            df_processed["clean_text"].str.split().str.len()
        )

        # Filter out spam and empty comments
        df_filtered = df_processed[
            (~df_processed["is_spam"]) & (df_processed["clean_text"].str.len() > 0)
        ].copy()

        logger.info(
            f"Filtered out {len(df_processed) - len(df_filtered)} spam/empty comments"
        )
        logger.info(f"Remaining YouTube comments: {len(df_filtered)}")

        return df_filtered

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze YouTube data quality"""
        analysis = {
            "platform": "youtube",
            "total_comments": len(df),
            "avg_text_length": df["raw_text"].str.len().mean(),
            "avg_word_count": df["raw_text"].str.split().str.len().mean(),
            "avg_likes_per_comment": df["likes_count"].mean(),
            "max_likes": df["likes_count"].max(),
            "min_likes": df["likes_count"].min(),
            "avg_reply_count": df["reply_count"].mean(),
            "comments_with_no_likes": (df["likes_count"] == 0).sum(),
            "comments_with_likes": (df["likes_count"] > 0).sum(),
            "comments_with_replies": (df["reply_count"] > 0).sum(),
        }
        return analysis


# ─── Combined Preprocessor ────────────────────────────────────────────────────


class DataPreprocessor:
    """Main preprocessor that handles both TikTok and YouTube data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir) / "preprocessed"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tiktok_processor = TiktokPreprocessor(data_dir)
        self.youtube_processor = YoutubePreprocessor(data_dir)

    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine data from both platforms"""
        all_data = []

        try:
            tiktok_data = self.tiktok_processor.load_raw_data()
            all_data.append(tiktok_data)
            logger.info(f"Loaded TikTok data: {len(tiktok_data)} comments")
        except FileNotFoundError:
            logger.warning("TikTok data file not found, skipping...")

        try:
            youtube_data = self.youtube_processor.load_raw_data()
            all_data.append(youtube_data)
            logger.info(f"Loaded YouTube data: {len(youtube_data)} comments")
        except FileNotFoundError:
            logger.warning("YouTube data file not found, skipping...")

        if not all_data:
            raise FileNotFoundError("No data files found for any platform")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined total: {len(combined_df)} comments")

        return combined_df

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the combined dataset"""
        logger.info("Starting combined data preprocessing...")

        # Use base preprocessing methods
        base_processor = BasePreprocessor(self.data_dir)

        df_processed = df.copy()

        # Add preprocessing columns
        df_processed["is_spam"] = df_processed["raw_text"].apply(
            base_processor.detect_spam
        )
        df_processed["clean_text"] = df_processed["raw_text"].apply(
            base_processor.preprocess_text
        )
        df_processed["normalized_text"] = df_processed["raw_text"].apply(
            lambda x: base_processor.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=False, stem=False
            )
        )
        df_processed["final_text"] = df_processed["raw_text"].apply(
            lambda x: base_processor.preprocess_text(
                x, clean=True, normalize=True, remove_stopwords=True, stem=False
            )
        )

        # Add text statistics
        df_processed["text_length"] = df_processed["raw_text"].str.len()
        df_processed["word_count"] = df_processed["raw_text"].str.split().str.len()
        df_processed["clean_text_length"] = df_processed["clean_text"].str.len()
        df_processed["clean_word_count"] = (
            df_processed["clean_text"].str.split().str.len()
        )

        # Filter out spam and empty comments
        df_filtered = df_processed[
            (~df_processed["is_spam"]) & (df_processed["clean_text"].str.len() > 0)
        ].copy()

        logger.info(
            f"Filtered out {len(df_processed) - len(df_filtered)} spam/empty comments"
        )
        logger.info(f"Remaining comments: {len(df_filtered)}")

        return df_filtered

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze combined data quality"""
        platform_stats = {}

        for platform in df["source"].unique():
            platform_data = df[df["source"] == platform]
            platform_stats[platform] = {
                "total_comments": len(platform_data),
                "avg_text_length": platform_data["raw_text"].str.len().mean(),
                "avg_word_count": platform_data["raw_text"]
                .str.split()
                .str.len()
                .mean(),
                "avg_likes_per_comment": platform_data["likes_count"].mean(),
                "max_likes": platform_data["likes_count"].max(),
                "min_likes": platform_data["likes_count"].min(),
                "avg_reply_count": platform_data["reply_count"].mean(),
                "comments_with_no_likes": (platform_data["likes_count"] == 0).sum(),
                "comments_with_likes": (platform_data["likes_count"] > 0).sum(),
                "comments_with_replies": (platform_data["reply_count"] > 0).sum(),
            }

        overall_analysis = {
            "overall": {
                "total_comments": len(df),
                "platforms": list(df["source"].unique()),
                "avg_text_length": df["raw_text"].str.len().mean(),
                "avg_word_count": df["raw_text"].str.split().str.len().mean(),
                "avg_likes_per_comment": df["likes_count"].mean(),
                "max_likes": df["likes_count"].max(),
                "min_likes": df["likes_count"].min(),
                "avg_reply_count": df["reply_count"].mean(),
                "comments_with_no_likes": (df["likes_count"] == 0).sum(),
                "comments_with_likes": (df["likes_count"] > 0).sum(),
                "comments_with_replies": (df["reply_count"] > 0).sum(),
            },
            "by_platform": platform_stats,
        }

        return overall_analysis

    def save_processed_data(self, df: pd.DataFrame, analysis: Dict):
        """Save processed data and analysis"""
        # Save processed data with increment
        self.save_to_csv_with_increment(df, "comments_processed.csv")

        # Save analysis
        self.save_to_csv_with_increment(
            pd.DataFrame([analysis["overall"]]), "preprocessing_analysis.csv"
        )

        # Save detailed platform analysis as JSON
        detailed_analysis_file = self.data_dir / "preprocessing_analysis_detailed.json"
        with open(detailed_analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved detailed analysis to {detailed_analysis_file}")

    def save_to_csv_with_increment(self, df: pd.DataFrame, base_filename: str) -> Path:
        """Save data to CSV with auto-increment if file already exists."""
        if df.empty:
            logger.info("No data to save.")
            return self.data_dir / base_filename

        output_path = self.data_dir / base_filename
        counter = 1
        while output_path.exists():
            name = base_filename.rsplit(".", 1)[0]
            output_path = self.data_dir / f"{name}_{counter}.csv"
            counter += 1

        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Data saved to {output_path}")
        logger.info(f"Total records: {len(df)}")
        return output_path


# ─── Main Function ─────────────────────────────────────────────────────────


def main():
    """Main preprocessing function"""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Load raw data
        df_raw = preprocessor.load_raw_data()

        # Preprocess data
        df_processed = preprocessor.preprocess_dataset(df_raw)

        # Analyze data quality
        analysis = preprocessor.analyze_data_quality(df_processed)

        # Save results
        preprocessor.save_processed_data(df_processed, analysis)

        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Processed {len(df_processed)} comments")

        return df_processed, analysis

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
