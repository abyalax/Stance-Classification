#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Stance Classification Research
This script orchestrates the complete pipeline from data collection to labeling preparation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import json
import importlib.util

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Import modules directly
spec1 = importlib.util.spec_from_file_location("collect_data", Path(__file__).parent / "01_collect_data.py")
collect_data = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(collect_data)

spec2 = importlib.util.spec_from_file_location("preprocess_data", Path(__file__).parent / "02_preprocess_data.py")
preprocess_data = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocess_data)

spec3 = importlib.util.spec_from_file_location("prepare_labeling", Path(__file__).parent / "03_prepare_labeling.py")
prepare_labeling = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(prepare_labeling)

InstagramScraper = collect_data.InstagramScraper
DataPreprocessor = preprocess_data.DataPreprocessor
LabelingPreparator = prepare_labeling.LabelingPreparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.scraper = InstagramScraper(data_dir)
        self.preprocessor = DataPreprocessor(data_dir)
        self.labeling_preparator = LabelingPreparator(data_dir)
        
        # Pipeline status tracking
        self.status_file = self.data_dir / "pipeline_status.json"
        self.pipeline_status = self._load_pipeline_status()
        
    def _load_pipeline_status(self) -> Dict:
        """Load pipeline status from file"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'data_collection': {'completed': False, 'timestamp': None},
            'preprocessing': {'completed': False, 'timestamp': None},
            'labeling_preparation': {'completed': False, 'timestamp': None}
        }
    
    def _save_pipeline_status(self):
        """Save pipeline status to file"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_status, f, indent=2, ensure_ascii=False)
    
    def run_data_collection(self, force: bool = False) -> bool:
        """Run data collection stage"""
        if self.pipeline_status['data_collection']['completed'] and not force:
            logger.info("Data collection already completed. Use --force to rerun.")
            return True
        
        try:
            logger.info("Starting data collection stage...")
            
            # Run scraping
            results = self.scraper.scrape_posts()
            
            # Update status
            self.pipeline_status['data_collection'] = {
                'completed': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_comments': len(results['comments']),
                'total_posts': len(results['posts'])
            }
            self._save_pipeline_status()
            
            logger.info("Data collection completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return False
    
    def run_preprocessing(self, force: bool = False) -> bool:
        """Run preprocessing stage"""
        if self.pipeline_status['preprocessing']['completed'] and not force:
            logger.info("Preprocessing already completed. Use --force to rerun.")
            return True
        
        if not self.pipeline_status['data_collection']['completed']:
            logger.error("Data collection must be completed first.")
            return False
        
        try:
            logger.info("Starting preprocessing stage...")
            
            # Load raw data
            df_raw = self.preprocessor.load_raw_data()
            
            # Preprocess data
            df_processed = self.preprocessor.preprocess_dataset(df_raw)
            
            # Analyze and save
            analysis = self.preprocessor.analyze_data_quality(df_processed)
            self.preprocessor.save_processed_data(df_processed, analysis)
            
            # Update status
            self.pipeline_status['preprocessing'] = {
                'completed': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_processed': len(df_processed),
                'spam_filtered': len(df_raw) - len(df_processed)
            }
            self._save_pipeline_status()
            
            logger.info("Preprocessing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return False
    
    def run_labeling_preparation(self, force: bool = False) -> bool:
        """Run labeling preparation stage"""
        if self.pipeline_status['labeling_preparation']['completed'] and not force:
            logger.info("Labeling preparation already completed. Use --force to rerun.")
            return True
        
        if not self.pipeline_status['preprocessing']['completed']:
            logger.error("Preprocessing must be completed first.")
            return False
        
        try:
            logger.info("Starting labeling preparation stage...")
            
            # Load processed data
            df = self.labeling_preparator.load_processed_data()
            
            # Analyze readiness
            analysis = self.labeling_preparator.analyze_labeling_readiness(df)
            
            # Create labeling dataset
            df_sampled = self.labeling_preparator.create_labeling_dataset(df, sample_size=500)
            
            # Save labeling package
            self.labeling_preparator.save_labeling_package(df_sampled, analysis)
            
            # Update status
            self.pipeline_status['labeling_preparation'] = {
                'completed': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'sample_size': len(df_sampled),
                'labeling_package_created': True
            }
            self._save_pipeline_status()
            
            logger.info("Labeling preparation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Labeling preparation failed: {e}")
            return False
    
    def run_full_pipeline(self, force: bool = False) -> bool:
        """Run complete pipeline"""
        logger.info("Starting full pipeline execution...")
        
        stages = [
            ("Data Collection", self.run_data_collection),
            ("Preprocessing", self.run_preprocessing),
            ("Labeling Preparation", self.run_labeling_preparation)
        ]
        
        for stage_name, stage_func in stages:
            logger.info(f"Running stage: {stage_name}")
            if not stage_func(force=force):
                logger.error(f"Pipeline failed at stage: {stage_name}")
                return False
            logger.info(f"Stage completed: {stage_name}")
        
        logger.info("Full pipeline completed successfully!")
        return True
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return self.pipeline_status
    
    def print_pipeline_status(self):
        """Print pipeline status summary"""
        print("\n" + "="*50)
        print("PIPELINE STATUS SUMMARY")
        print("="*50)
        
        for stage, status in self.pipeline_status.items():
            status_symbol = "✅" if status['completed'] else "❌"
            stage_name = stage.replace('_', ' ').title()
            
            print(f"{status_symbol} {stage_name}")
            if status['completed']:
                print(f"   Completed: {status['timestamp']}")
                for key, value in status.items():
                    if key not in ['completed', 'timestamp']:
                        print(f"   {key}: {value}")
            else:
                print("   Status: Not completed")
            print()
        
        print("="*50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stance Classification Pipeline Orchestrator")
    parser.add_argument(
        '--stage', 
        choices=['collect', 'preprocess', 'label', 'all'],
        default='all',
        help='Pipeline stage to run (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory path (default: data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rerun completed stages'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.data_dir)
    
    # Show status if requested
    if args.status:
        orchestrator.print_pipeline_status()
        return
    
    # Run requested stage
    success = False
    
    if args.stage == 'collect':
        success = orchestrator.run_data_collection(force=args.force)
    elif args.stage == 'preprocess':
        success = orchestrator.run_preprocessing(force=args.force)
    elif args.stage == 'label':
        success = orchestrator.run_labeling_preparation(force=args.force)
    elif args.stage == 'all':
        success = orchestrator.run_full_pipeline(force=args.force)
    
    # Print final status
    orchestrator.print_pipeline_status()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()