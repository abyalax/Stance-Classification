import os
import logging
import time

import pandas as pd

from googleapiclient.discovery import build
from apify_client import ApifyClient
from typing import Dict, List, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# ─── Clients ──────────────────────────────────────────────────────────────────

try:
    apify_token = os.getenv("APIFY_TOKEN")
    if not apify_token:
        logger.warning(
            "APIFY_TOKEN not found in .env - Tiktok scraping will be disabled"
        )
        raise ValueError(
            "APIFY_TOKEN not found in .env",
            "Get token on https://console.apify.com/settings/integrations",
        )
    apify_client = ApifyClient(apify_token)
except Exception as e:
    logger.error(f"Failed to initialize Apify client: {e}")
    apify_client = None

try:
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        logger.warning(
            "YOUTUBE_API_KEY not found in .env - YouTube scraping will be disabled"
        )
        youtube_api_key = None
except Exception as e:
    logger.error(f"Error checking YouTube API key: {e}")
    youtube_api_key = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def save_to_csv(data: List[Dict], data_dir: Path, base_filename: str) -> Path:
    """Save data to CSV with auto-increment if file already exists."""
    if not data:
        logger.info("No data to save.")
        return data_dir / base_filename

    df = pd.DataFrame(data)
    output_path = data_dir / base_filename

    counter = 1
    while output_path.exists():
        name = base_filename.rsplit(".", 1)[0]
        output_path = data_dir / f"{name}_{counter}.csv"
        counter += 1

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Data saved to {output_path}")
    logger.info(f"Total collected: {len(data)} items")
    return output_path


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    import re

    patterns = [
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtube\.com/shorts/([^?&]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract video ID from URL: {url}")


# ─── TikTok Scraper (Apify) ───────────────────────────────────────────────────

TIKTOK_RUN_INPUT = {
    "postURLs": [
        "https://www.tiktok.com/@suaradotcom/video/7620063435551788304?is_from_webapp=1&sender_device=pc&web_id=7607072056891491858"
    ],
    "commentsPerPost": 5000,
    "maxRepliesPerComment": 100,
    "resultsPerPage": 100,
    "profileScrapeSections": ["videos"],
    "profileSorting": "latest",
    "excludePinnedPosts": False,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/collect_data.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TiktokScraper:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir) / "scrapped"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not apify_client:
            raise ValueError("Apify client not initialized - check APIFY_TOKEN")

        self.apify_client = apify_client

    def scrape_posts(self) -> Dict[str, List[Any]]:
        """Scrape TikTok comments via Apify and save to CSV."""
        logger.info("\n" + "=" * 50)
        logger.info("Tiktok Scrapping")
        logger.info("=" * 50)

        run = self.apify_client.actor("BDec00yAmCm1QbMEI").call(
            run_input=TIKTOK_RUN_INPUT
        )

        raw_data = []
        for item in self.apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            raw_data.append(item)

        save_to_csv(raw_data, self.data_dir, "tiktok_comments_raw.csv")
        return {"raw_data": raw_data}


# ─── YouTube Scraper (Official API v3) ────────────────────────────────────────

YOUTUBE_TARGET_URLS = [
    "https://youtu.be/S4iey1sr1Cg",
]

YOUTUBE_MAX_COMMENTS = 5000  # per video, set None for unlimited


class YoutubeScraper:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir) / "scrapped"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not youtube_api_key:
            raise ValueError("YouTube API key not available")

        self.youtube = build("youtube", "v3", developerKey=youtube_api_key)

    def _get_video_metadata(self, video_id: str) -> Dict:
        """Fetch video title and comment count."""
        response = (
            self.youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        )

        if not response.get("items"):
            return {"title": "Unknown", "comment_count": 0}

        item = response["items"][0]
        return {
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "comment_count": int(item["statistics"].get("commentCount", 0)),
            "published_at": item["snippet"]["publishedAt"],
        }

    def _scrape_video_comments(
        self, video_id: str, max_comments: int = None
    ) -> List[Dict]:
        """Scrape all comments from a single video using pagination."""
        comments = []
        next_page_token = None
        page = 1

        while True:
            try:
                response = (
                    self.youtube.commentThreads()
                    .list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=next_page_token,
                        textFormat="plainText",
                        order="time",  # Determinisitic and measureable, instead of arg "relevance" cause can break pagination
                    )
                    .execute()
                )
            except Exception as e:
                logger.error(f"  API error on page {page}: {e}")
                break

            for thread in response.get("items", []):
                top = thread["snippet"]["topLevelComment"]["snippet"]

                comments.append(
                    {
                        "comment_id": thread["id"],
                        "text": top["textDisplay"],
                        "author": top["authorDisplayName"],
                        "likes_count": top["likeCount"],
                        "published_at": top["publishedAt"],
                        "updated_at": top["updatedAt"],
                        "reply_count": thread["snippet"]["totalReplyCount"],
                        "is_reply": False,
                        "parent_id": None,
                        "video_id": video_id,
                        "source": "youtube",
                    }
                )

                # Include replies if available in response
                if "replies" in thread:
                    for reply in thread["replies"]["comments"]:
                        rs = reply["snippet"]
                        comments.append(
                            {
                                "comment_id": reply["id"],
                                "text": rs["textDisplay"],
                                "author": rs["authorDisplayName"],
                                "likes_count": rs["likeCount"],
                                "published_at": rs["publishedAt"],
                                "updated_at": rs["updatedAt"],
                                "reply_count": 0,
                                "is_reply": True,
                                "parent_id": thread["id"],
                                "video_id": video_id,
                                "source": "youtube",
                            }
                        )

            logger.info(f"  Page {page}: {len(comments)} comments so far...")

            if max_comments and len(comments) >= max_comments:
                comments = comments[:max_comments]
                logger.info(f"  Reached limit of {max_comments} comments.")
                break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            page += 1
            time.sleep(0.5)

        return comments

    def scrape_comments(self) -> Dict[str, List[Any]]:
        """Scrape comments from all target YouTube URLs."""
        logger.info("\n" + "=" * 50)
        logger.info("Youtube Scrapping (Official API v3)")
        logger.info("=" * 50)

        all_comments = []

        for url in YOUTUBE_TARGET_URLS:
            video_id = extract_video_id(url)
            logger.info(f"\nVideo ID : {video_id}")

            meta = self._get_video_metadata(video_id)
            logger.info(f"Title    : {meta['title']}")
            logger.info(f"Channel  : {meta['channel']}")
            logger.info(f"Comments : {meta['comment_count']} total")

            comments = self._scrape_video_comments(
                video_id, max_comments=YOUTUBE_MAX_COMMENTS
            )
            all_comments.extend(comments)
            logger.info(f"Scraped  : {len(comments)} comments from {video_id}")

        save_to_csv(all_comments, self.data_dir, "youtube_comments_raw.csv")
        return {"raw_data": all_comments}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape comments from TikTok and/or YouTube"
    )
    parser.add_argument(
        "--platform",
        choices=["tiktok", "youtube", "all"],
        default="all",
        help="Platform to scrape (default: all)",
    )
    args = parser.parse_args()

    if args.platform in ("tiktok", "all"):
        try:
            if not apify_token:
                logger.warning(
                    "Apify Token key not available - skipping Tiktok scraping"
                )
                if args.platform == "tiktok":
                    sys.exit(1)
            else:
                tiktok = TiktokScraper()
                tiktok.scrape_posts()
        except Exception as e:
            logger.error(f"TikTok scraping failed: {e}")
            if args.platform == "tiktok":
                sys.exit(1)

    if args.platform in ("youtube", "all"):
        try:
            if not youtube_api_key:
                logger.warning(
                    "YouTube API key not available - skipping YouTube scraping"
                )
                if args.platform == "youtube":
                    sys.exit(1)
            else:
                youtube = YoutubeScraper()
                youtube.scrape_comments()
        except Exception as e:
            logger.error(f"YouTube scraping failed: {e}")
            if args.platform == "youtube":
                sys.exit(1)
