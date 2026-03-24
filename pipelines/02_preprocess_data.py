import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize Indonesian stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Indonesian stopwords
        self.stop_words = set(stopwords.words('indonesian'))
        
        # Additional custom stopwords for social media
        self.custom_stopwords = {
            'yg', 'yang', 'dgn', 'dengan', 'untuk', 'untk', 'pd', 'pada', 'dll', 
            'dsb', 'dan', 'atau', 'tapi', 'tp', 'jg', 'juga', 'gk', 'gak', 'nggak',
            'ga', 'tdk', 'tidak', 'bgt', 'banget', 'sangat', 'krn', 'karena', 'jika',
            'jk', 'klo', 'kalau', 'kalo', 'ya', 'iyah', 'iya', 'ok', 'oke', 'sip',
            'mas', 'mba', 'mbak', 'pak', 'bapak', 'bu', 'ibu', 'kak', 'kakak',
            'bro', 'sis', 'agan', 'sist', 'om', 'tante', 'dewasa'
        }
        self.stop_words.update(self.custom_stopwords)
        
        # Patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+"
        )
        self.excessive_pattern = re.compile(r'(.)\1{2,}')  # Excessive characters
        self.number_pattern = re.compile(r'\d+')  # Numbers
        
        # Slang normalization dictionary
        self.slang_dict = self._load_slang_dict()
        
    def _load_slang_dict(self) -> Dict[str, str]:
        """Load Indonesian slang normalization dictionary"""
        # Common Indonesian slang words
        slang_dict = {
            # Common abbreviations
            'apk': 'aplikasi',
            'bgt': 'banget',
            'bgtt': 'banget',
            'bgts': 'banget',
            'brp': 'berapa',
            'bs': 'bisa',
            'bsh': 'bisa',
            'dr': 'dari',
            'dri': 'dari',
            'dg': 'dengan',
            'dgn': 'dengan',
            'dlm': 'dalam',
            'dpt': 'dapat',
            'gini': 'begini',
            'gitu': 'begitu',
            'gt': 'gitu',
            'gtau': 'tidak tahu',
            'ga tau': 'tidak tahu',
            'gak tau': 'tidak tahu',
            'hr': 'hari',
            'hrs': 'harus',
            'jg': 'juga',
            'jgn': 'jangan',
            'jk': 'jika',
            'kalo': 'kalau',
            'klu': 'kalau',
            'klw': 'kalau',
            'km': 'kamu',
            'kmu': 'kamu',
            'knp': 'kenapa',
            'krn': 'karena',
            'karna': 'karena',
            'lbh': 'lebih',
            'mngkn': 'mungkin',
            'mrk': 'mereka',
            'msk': 'masuk',
            'msh': 'masih',
            'nggak': 'tidak',
            'ngak': 'tidak',
            'gak': 'tidak',
            'ga': 'tidak',
            'pd': 'pada',
            'pdk': 'pendapat',
            'pnya': 'punya',
            'py': 'punya',
            'sdh': 'sudah',
            'sdhkan': 'sudah',
            'skrg': 'sekarang',
            'skrang': 'sekarang',
            'sm': 'sama',
            'smp': 'sampai',
            'spt': 'seperti',
            'sprti': 'seperti',
            'sy': 'saya',
            'sya': 'saya',
            'tlg': 'tolong',
            'tlong': 'tolong',
            'tp': 'tapi',
            'ttg': 'tentang',
            'utk': 'untuk',
            'untk': 'untuk',
            'wkt': 'waktu',
            'wktu': 'waktu',
            'ya': 'iya',
            'yah': 'iya',
            'yuk': 'ayo',
            
            # Economic/financial terms
            'bbm': 'bahan bakar minyak',
            'pertalite': 'pertalite',
            'pertamax': 'pertamax',
            'solar': 'solar',
            'minyak': 'minyak',
            'bensin': 'bensin',
            'apbn': 'anggaran pendapatan dan belanja negara',
            'defisit': 'defisit',
            'inflasi': 'inflasi',
            'rupiah': 'rupiah',
            'kurs': 'kurs',
            'subsidy': 'subsidi',
            'subsidi': 'subsidi',
            'fiskal': 'fiskal',
            'moneter': 'moneter',
            'neraca': 'neraca',
            'pembayaran': 'pembayaran',
            'impor': 'impor',
            'ekspor': 'ekspor',
            'mbg': 'makan bergizi gratis',
            'makan bergizi gratis': 'makan bergizi gratis',
            
            # Common expressions
            'hadeh': 'hadeh',
            'aduh': 'aduh',
            'wih': 'wih',
            'wow': 'wow',
            'hmm': 'hmm',
            'hemm': 'hmm',
            'gitu': 'begitu',
            'gini': 'begini',
            'begini': 'begini',
            'begitu': 'begitu',
            'dong': 'dong',
            'donk': 'dong',
            'dunk': 'dong',
            'kan': 'kan',
            'kok': 'kok',
            'ko': 'kok',
            'loh': 'loh',
            'lho': 'loh',
            'toh': 'toh',
            'mah': 'mah',
            'nih': 'ini',
            'nie': 'ini',
            'nihh': 'ini',
            'dong': 'dong',
        }
        
        return slang_dict
    
    def clean_text(self, text: str) -> str:
        """Clean individual text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions
        text = self.mention_pattern.sub(' ', text)
        
        # Remove hashtags (keep the text without #)
        text = self.hashtag_pattern.sub(' ', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove excessive characters (more than 2 consecutive)
        text = self.excessive_pattern.sub(r'\1\1', text)
        
        # Remove numbers
        text = self.number_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_slang(self, text: str) -> str:
        """Normalize Indonesian slang words"""
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word in self.slang_dict:
                normalized_words.append(self.slang_dict[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Indonesian stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """Stem Indonesian text"""
        return self.stemmer.stem(text)
    
    def preprocess_text(self, text: str, 
                       clean: bool = True, 
                       normalize: bool = True, 
                       remove_stopwords: bool = True, 
                       stem: bool = False) -> str:
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
        if re.match(r'^[\d\s\W]+$', text):
            return True
        
        # Repetitive characters
        if self.excessive_pattern.search(text):
            return True
        
        # Common spam patterns
        spam_patterns = [
            r'^(ok|sip|mantap|nice|good|great|wow|wih)+\s*$',
            r'^(follow|like|comment|subscribe)+\s*$',
            r'^(cek|cekibrot|cekkidmat)+\s*$',
            r'^(promo|jual|beli|order)+\s*$',
            r'^\w+\s*\w*\s*$'  # Single word repeated
        ]
        
        for pattern in spam_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw comments data and extract only text and likesCount columns"""
        comments_file = self.data_dir / "comment_post.csv"
        
        if not comments_file.exists():
            raise FileNotFoundError(f"Raw comments file not found: {comments_file}")
        
        # Load only the columns we need: text and likesCount
        df = pd.read_csv(comments_file, encoding='utf-8', usecols=['text', 'likesCount'])
        logger.info(f"Loaded {len(df)} comments from {comments_file}")
        logger.info(f"Extracted columns: {list(df.columns)}")
        
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the entire dataset"""
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Add preprocessing columns
        df_processed['is_spam'] = df_processed['text'].apply(self.detect_spam)
        df_processed['text_clean'] = df_processed['text'].apply(self.preprocess_text)
        df_processed['text_normalized'] = df_processed['text'].apply(
            lambda x: self.preprocess_text(x, clean=True, normalize=True, remove_stopwords=False, stem=False)
        )
        df_processed['text_final'] = df_processed['text'].apply(
            lambda x: self.preprocess_text(x, clean=True, normalize=True, remove_stopwords=True, stem=False)
        )
        
        # Add text statistics
        df_processed['text_length'] = df_processed['text'].str.len()
        df_processed['text_word_count'] = df_processed['text'].str.split().str.len()
        df_processed['clean_text_length'] = df_processed['text_clean'].str.len()
        df_processed['clean_text_word_count'] = df_processed['text_clean'].str.split().str.len()
        
        # Filter out spam and empty comments
        df_filtered = df_processed[
            (~df_processed['is_spam']) & 
            (df_processed['text_clean'].str.len() > 0)
        ].copy()
        
        logger.info(f"Filtered out {len(df_processed) - len(df_filtered)} spam/empty comments")
        logger.info(f"Remaining comments: {len(df_filtered)}")
        
        return df_filtered
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality after preprocessing"""
        analysis = {
            'total_comments': len(df),
            'avg_text_length': df['text'].str.len().mean(),
            'avg_word_count': df['text'].str.split().str.len().mean(),
            'avg_likes_per_comment': df['likesCount'].mean(),
            'max_likes': df['likesCount'].max(),
            'min_likes': df['likesCount'].min(),
            'comments_with_no_likes': (df['likesCount'] == 0).sum(),
            'comments_with_likes': (df['likesCount'] > 0).sum()
        }
        
        return analysis
    
    def save_processed_data(self, df: pd.DataFrame, analysis: Dict):
        """Save processed data and analysis"""
        # Save extracted data (text and likesCount only)
        extracted_file = self.data_dir / "comments_extracted.csv"
        df.to_csv(extracted_file, index=False, encoding='utf-8')
        logger.info(f"Saved extracted data to {extracted_file}")
        
        # Save analysis
        analysis_file = self.data_dir / "preprocessing_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved analysis to {analysis_file}")
    
    def main(self):
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
