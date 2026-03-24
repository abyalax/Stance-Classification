from apify_client import ApifyClient
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Load environment variables from .env file
load_dotenv()

# Initialize the ApifyClient with your API token from environment
api_token = os.getenv("APIFY_TOKEN")
if not api_token:
    raise ValueError("APIFY_TOKEN not found in environment variables. Please set it in your .env file.")
client = ApifyClient(api_token)

# Prepare the Actor input
run_input = {
    "postUrls": [
        "https://www.instagram.com/p/DVscrIakZKd/",
        "https://www.instagram.com/p/DVtbswgk4h6/",
    ],
    "maxCommentsPerPost": 2000,
    "sortOrder": "popular",
}


class InstagramScraper:
    def __init__(self, data_dir: str = "data"):
        """Initialize Instagram scraper"""
        self.data_dir = Path(data_dir)
    
    def scrape_posts(self) -> Dict[str, List[Any]]:
        """Scrape Instagram posts and return data in expected format"""
        # Run the Actor and wait for it to finish
        run = client.actor("499mNnuVGkU2S5rh1").call(run_input=run_input)

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

        # Collect all comments from the dataset
        comments_data = []
        posts_data = []
        
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            comments_data.append(item)
            print(item)
            
            # Extract post information
            if 'postUrl' in item:
                post_info = {
                    'url': item.get('postUrl'),
                    'owner': item.get('ownerUsername'),
                    'timestamp': item.get('timestamp'),
                    'likes': item.get('likesCount', 0)
                }
                if post_info not in posts_data:
                    posts_data.append(post_info)

        # Save data to CSV file
        if comments_data:
            df = pd.DataFrame(comments_data)
            output_path = self.data_dir / "comment_post.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nData saved to {output_path}")
            print(f"Total comments collected: {len(comments_data)}")
        else:
            print("No comments data collected.")

        return {
            'comments': comments_data,
            'posts': posts_data
        }


# For standalone execution (keeping original functionality)
if __name__ == "__main__":
    scraper = InstagramScraper()
    results = scraper.scrape_posts()