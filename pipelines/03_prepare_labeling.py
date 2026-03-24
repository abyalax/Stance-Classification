import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LabelingPreparator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Label schema from research context
        self.label_schema = {
            'FAVOR': 'Mendukung argumen/posisi Ferry Irwandi',
            'AGAINST': 'Menentang/mengkritik argumen Ferry Irwandi',
            'PRO_GOV': 'Mendukung kebijakan/program pemerintah Indonesia',
            'CONTRA_GOV': 'Menentang/mengkritik kebijakan pemerintah Indonesia',
            'CONDITIONAL': 'Mendukung sebagian, disertai syarat atau catatan kritis',
            'SUGGESTION': 'Memberikan saran atau masukan konstruktif berbasis data',
            'DISCUSSION': 'Reply/diskusi lateral antar netizen',
            'OFF_TOPIC': 'Tidak relevan dengan isu yang dibahas'
        }
        
        # Keywords for each label (for initial sampling)
        self.label_keywords = {
            'FAVOR': ['setuju', 'benar', 'tepat', 'bagus', 'mendukung', 'support', 'betul', 'sama'],
            'AGAINST': ['salah', 'tidak', 'keliru', 'salah', 'berbeda', 'tidak setuju', 'kontra'],
            'PRO_GOV': ['pemerintah', 'presiden', 'kementerian', 'kebijakan', 'program', 'dukung'],
            'CONTRA_GOV': ['pemerintah', 'presiden', 'kementerian', 'kebijakan', 'program', 'kritik'],
            'CONDITIONAL': ['tapi', 'namun', 'walaupun', 'meskipun', 'asal', 'syarat', 'jika'],
            'SUGGESTION': ['sarankan', 'usulkan', 'sebaiknya', 'harus', 'perlu', 'solusi'],
            'DISCUSSION': ['@', 'reply', 'balas', 'komen', 'pendapat'],
            'OFF_TOPIC': ['jualan', 'promo', 'iklan', 'follow', 'like', 'subscribe']
        }
        
    def load_processed_data(self) -> pd.DataFrame:
        """Load preprocessed comments data"""
        processed_file = self.data_dir / "comments_processed.csv"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed comments file not found: {processed_file}")
        
        df = pd.read_csv(processed_file, encoding='utf-8')
        logger.info(f"Loaded {len(df)} processed comments")
        
        return df
    
    def create_labeling_dataset(self, 
                               df: pd.DataFrame, 
                               sample_size: int = 500,
                               stratify_by: str = 'post_shortcode',
                               random_state: int = 42) -> pd.DataFrame:
        """Create stratified sample for manual labeling"""
        
        logger.info(f"Creating labeling dataset with {sample_size} samples")
        
        # Filter for high-quality comments
        df_quality = df[
            (df['clean_text_word_count'] >= 3) &  # At least 3 words
            (df['clean_text_word_count'] <= 50) &  # Max 50 words
            (~df['is_reply'])  # Focus on original comments
        ].copy()
        
        logger.info(f"Filtered to {len(df_quality)} high-quality comments")
        
        # Stratified sampling by post
        if stratify_by in df_quality.columns:
            samples_per_post = sample_size // df_quality[stratify_by].nunique()
            
            sampled_comments = []
            for post_code in df_quality[stratify_by].unique():
                post_comments = df_quality[df_quality[stratify_by] == post_code]
                
                if len(post_comments) <= samples_per_post:
                    sampled_comments.append(post_comments)
                else:
                    # Random sample from this post
                    sample = post_comments.sample(
                        n=samples_per_post, 
                        random_state=random_state
                    )
                    sampled_comments.append(sample)
            
            df_sampled = pd.concat(sampled_comments, ignore_index=True)
        else:
            # Simple random sampling
            df_sampled = df_quality.sample(
                n=min(sample_size, len(df_quality)), 
                random_state=random_state
            )
        
        # Add labeling columns
        df_sampled['label'] = ''
        df_sampled['annotator_id'] = ''
        df_sampled['labeling_date'] = ''
        df_sampled['confidence'] = ''
        df_sampled['notes'] = ''
        
        logger.info(f"Created labeling dataset with {len(df_sampled)} comments")
        
        return df_sampled
    
    def create_label_studio_config(self) -> Dict:
        """Create Label Studio configuration for stance classification"""
        
        config = {
            "version": "1.0",
            "type": "image",  # Using image type for better UI
            "settings": {
                "inline": False,
                "singlePanel": False,
                "showLabels": True,
                "showEmptyLabel": True
            },
            "controls": [
                {
                    "type": "choices",
                    "name": "stance",
                    "choices": [
                        {"value": "FAVOR", "text": "FAVOR - Mendukung argumen Ferry"},
                        {"value": "AGAINST", "text": "AGAINST - Menentang argumen Ferry"},
                        {"value": "PRO_GOV", "text": "PRO_GOV - Mendukung pemerintah"},
                        {"value": "CONTRA_GOV", "text": "CONTRA_GOV - Menentang pemerintah"},
                        {"value": "CONDITIONAL", "text": "CONDITIONAL - Dukung bersyarat"},
                        {"value": "SUGGESTION", "text": "SUGGESTION - Saran konstruktif"},
                        {"value": "DISCUSSION", "text": "DISCUSSION - Diskusi antar netizen"},
                        {"value": "OFF_TOPIC", "text": "OFF_TOPIC - Tidak relevan"}
                    ],
                    "required": True,
                    "multiple": False,
                    "showInline": True
                },
                {
                    "type": "rating",
                    "name": "confidence",
                    "title": "Confidence Level",
                    "required": True,
                    "min": 1,
                    "max": 5,
                    "step": 1
                },
                {
                    "type": "textarea",
                    "name": "notes",
                    "title": "Notes (Optional)",
                    "required": False,
                    "rows": 3
                }
            ],
            "view": {
                "type": "html",
                "template": "<div style='padding: 20px; font-family: Arial, sans-serif;'>"
                          "<h3>Comment Analysis</h3>"
                          "<div style='background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px;'>"
                          "<strong>Username:</strong> ${owner_username}<br>"
                          "<strong>Date:</strong> ${created_at}<br>"
                          "<strong>Likes:</strong> ${likes_count}<br>"
                          "<strong>Post:</strong> ${post_shortcode}"
                          "</div>"
                          "<div style='background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px;'>"
                          "<strong>Original Text:</strong><br>${text}"
                          "</div>"
                          "<div style='background-color: #f0f8e8; padding: 15px; margin: 10px 0; border-radius: 5px;'>"
                          "<strong>Cleaned Text:</strong><br>${text_clean}"
                          "</div>"
                          "</div>"
            }
        }
        
        return config
    
    def export_for_label_studio(self, df: pd.DataFrame, output_format: str = 'json') -> str:
        """Export data for Label Studio import"""
        
        output_file = self.data_dir / f"labeling_dataset.{output_format}"
        
        if output_format == 'json':
            # Label Studio JSON format
            tasks = []
            for idx, row in df.iterrows():
                task = {
                    "id": idx,
                    "data": {
                        "text": row['text'],
                        "text_clean": row['text_clean'],
                        "owner_username": row['owner_username'],
                        "created_at": row['created_at'],
                        "likes_count": row['likes_count'],
                        "post_shortcode": row['post_shortcode'],
                        "comment_id": row['comment_id']
                    }
                }
                tasks.append(task)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            # Simple CSV format
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(df)} samples to {output_file}")
        return str(output_file)
    
    def create_annotation_guidelines(self) -> str:
        """Create detailed annotation guidelines"""
        
        guidelines = """
# STANCE CLASSIFICATION ANNOTATION GUIDELINES

## Overview
This document provides guidelines for annotating Instagram comments related to Indonesian energy security and fiscal policy discussions from @ferryirwandi's posts.

## Label Schema

### FAVOR
**Definition:** Comments that support, agree with, or reinforce Ferry Irwandi's arguments/positions.

**Examples:**
- "Setuju pak Ferry, subsidi memang harus ditinjau ulang"
- "Analisisnya tepat sekali"
- "Betul sekali, ini yang saya pikirkan juga"

**Keywords:** setuju, benar, tepat, bagus, mendukung, support, betul, sama

### AGAINST  
**Definition:** Comments that oppose, contradict, or criticize Ferry Irwandi's arguments/positions.

**Examples:**
- "Saya tidak setuju dengan analisis ini"
- "Data yang dipakai kurang akurat"
- "Ini terlalu berlebihan"

**Keywords:** salah, tidak, keliru, berbeda, tidak setuju, kontra

### PRO_GOV
**Definition:** Comments that support Indonesian government policies or programs.

**Examples:**
- "Pemerintah sudah berusaha maksimal"
- "Program MBG akan sangat membantu"
- "Kebijakan ini tepat untuk rakyat"

**Keywords:** pemerintah, presiden, kementerian, kebijakan, program, dukung

### CONTRA_GOV
**Definition:** Comments that criticize Indonesian government policies or programs.

**Examples:**
- "Pemerintah kurang responsif"
- "Kebijakan ini memberatkan rakyat"
- "Harus ada evaluasi total"

**Keywords:** pemerintah, presiden, kementerian, kebijakan, program, kritik

### CONDITIONAL
**Definition:** Comments that provide partial support with conditions or critical notes.

**Examples:**
- "Boleh saja tapi dengan syarat..."
- "Setuju asal tidak memberatkan"
- "Benar tapi perlu ditambahkan..."

**Keywords:** tapi, namun, walaupun, meskipun, asal, syarat, jika

### SUGGESTION
**Definition:** Comments that provide constructive suggestions or solutions.

**Examples:**
- "Sebaiknya fokus pada energi terbarukan"
- "Harus ada solusi jangka panjang"
- "Sarankan untuk evaluasi ulang"

**Keywords:** sarankan, usulkan, sebaiknya, harus, perlu, solusi

### DISCUSSION
**Definition:** Comments that are replies to other users or lateral discussions.

**Examples:**
- "@username saya setuju dengan pendapatmu"
- "Balasan untuk komentar di atas"
- "Menanggapi @username..."

**Keywords:** @, reply, balas, komen, pendapat

### OFF_TOPIC
**Definition:** Comments that are not relevant to the discussed issues.

**Examples:**
- "Jualan produk herbal"
- "Follow akun saya"
- "Promo jasa kebersihan"

**Keywords:** jualan, promo, iklan, follow, like, subscribe

## Annotation Rules

### Primary Rules
1. **Single Label:** Each comment gets exactly ONE label
2. **Context First:** Consider the context of Ferry's posts
3. **Main Stance:** Focus on the dominant stance, not minor mentions
4. **Explicit vs Implicit:** Both explicit and implicit stances count

### Decision Hierarchy
1. If it's clearly OFF_TOPIC (spam, ads) → OFF_TOPIC
2. If it's a reply to another user → DISCUSSION
3. If it discusses government policies → PRO_GOV or CONTRA_GOV
4. If it discusses Ferry's analysis → FAVOR or AGAINST
5. If it has conditions → CONDITIONAL
6. If it provides solutions → SUGGESTION

### Edge Cases
- **Mixed Stance:** Choose the dominant stance
- **Unclear:** Use your best judgment, add notes
- **Short Comments:** Analyze based on available content
- **Sarcasm:** Consider the intended meaning

## Quality Standards
- **Consistency:** Apply criteria consistently across all annotations
- **Accuracy:** Read carefully before labeling
- **Documentation:** Add notes for ambiguous cases
- **Confidence:** Rate your confidence (1-5 scale)

## Process
1. Read the original comment
2. Read the cleaned version if needed
3. Consider the post context
4. Apply decision hierarchy
5. Select the best label
6. Rate confidence
7. Add notes if necessary
"""
        
        return guidelines
    
    def analyze_labeling_readiness(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset readiness for labeling"""
        
        analysis = {
            'total_comments': len(df),
            'unique_users': df['owner_username'].nunique(),
            'posts_distribution': df['post_shortcode'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df['clean_text_word_count'].mean(),
                'median': df['clean_text_word_count'].median(),
                'min': df['clean_text_word_count'].min(),
                'max': df['clean_text_word_count'].max()
            },
            'reply_ratio': df['is_reply'].mean(),
            'quality_indicators': {
                'comments_3_50_words': ((df['clean_text_word_count'] >= 3) & 
                                       (df['clean_text_word_count'] <= 50)).sum(),
                'non_replies': (~df['is_reply']).sum(),
                'unique_texts': df['text_clean'].nunique()
            }
        }
        
        return analysis
    
    def save_labeling_package(self, df_sampled: pd.DataFrame, analysis: Dict):
        """Save complete labeling package"""
        
        # Create labeling directory
        labeling_dir = self.data_dir / "labeling_package"
        labeling_dir.mkdir(exist_ok=True)
        
        # Save sampled dataset
        sampled_file = labeling_dir / "labeling_dataset.csv"
        df_sampled.to_csv(sampled_file, index=False, encoding='utf-8')
        
        # Export for Label Studio
        json_file = self.export_for_label_studio(df_sampled, 'json')
        
        # Save Label Studio config
        config_file = labeling_dir / "label_studio_config.json"
        config = self.create_label_studio_config()
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Save annotation guidelines
        guidelines_file = labeling_dir / "annotation_guidelines.md"
        guidelines = self.create_annotation_guidelines()
        with open(guidelines_file, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        
        # Save analysis
        analysis_file = labeling_dir / "labeling_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Labeling package saved to {labeling_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - {sampled_file}")
        logger.info(f"  - {json_file}")
        logger.info(f"  - {config_file}")
        logger.info(f"  - {guidelines_file}")
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
