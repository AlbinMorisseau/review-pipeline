import requests
import csv
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
from src.utils import setup_logger
import re
import polars as pl

class AccessibleGoScraper:
    """Scraper to extract reveiws from AccessibleGO."""
    
    # base URLs for community sources (feed and archives)
    FEED_API_URL = "https://community.accessiblego.com/internal_api/home_page_posts"
    ARCHIVE_API_BASE = "https://community.accessiblego.com/internal_api/spaces"
    
    # Mapping to know which spaces of the API have to be extracted
    ARCHIVES = {
        'hotel-accessibility': {
            'space_id': 666302,
            'category': 'hotels',
            'name': 'Hotel Accessibility'
        },
        'air-travel': {
            'space_id': 666303,
            'category': 'air-travel',
            'name': 'Air Travel'
        },
        'cruises': {
            'space_id': 666304,
            'category': 'cruises',
            'name': 'Cruises'
        },
        'accessible-transportation': {
            'space_id': 666306,
            'category': 'transportation',
            'name': 'Accessible Transportation'
        }
    }
    
    def __init__(self, output_file: str = "data/accessiblego.csv"):
        self.output_file = output_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://community.accessiblego.com/'
        })
    
    def fetch_feed_page(self, page: int) -> Optional[Dict]:
        """Extract the data of the feed page"""
        try:
            params = {'page': page, 'sort': 'latest'}
            response = self.session.get(self.FEED_API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Feed page error{page}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON error feed page {page}: {e}")
            return None
    
    def fetch_archive_page(self, space_id: int, page: int) -> Optional[Dict]:
        """Extract the reviews of a specific archive page via its space_id."""
        try:
            url = f"{self.ARCHIVE_API_BASE}/{space_id}/posts"
            params = {
                'page': page,
                'include_top_pinned_post': 'true',
                'used_on': 'posts',
                'per_page': 15
            }
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Archive error space_id={space_id} page {page}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON error archive space_id={space_id} page {page}: {e}")
            return None
    
    def clean_html(self, html_content: str) -> str:
        """CLean HTML content and extract text only"""
        if not html_content:
            return ""
        
        # supression of HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # Decoding HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&apos;', "'")
        
        # Standardisation of spaces and lines
        text = re.sub(r'\n\s*\n', '\n', text)  # Reducing multiple lines
        text = ' '.join(text.split())  # Normaliser spaces
        
        return text.strip()
    
    def extract_text_from_tiptap(self, tiptap_body: Optional[Dict]) -> str:
        """Extract the text from the tiptap content recursively."""
        
        if not tiptap_body or not isinstance(tiptap_body, dict):
            return ""
        
        text_parts = []
        
        def extract_recursive(node):
            """Extract the text recursively"""
            if not node:
                return
            
            if isinstance(node, dict):
                # Direct text
                if node.get('type') == 'text' and 'text' in node:
                    text_parts.append(node['text'])
                
                # Exploration of children
                if 'content' in node:
                    extract_recursive(node['content'])
                
                # Other possible keys
                for key in ['body', 'children', 'elements']:
                    if key in node:
                        extract_recursive(node[key])
            
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(tiptap_body)
        
        # Join and clean
        text = ' '.join(text_parts)
        text = ' '.join(text.split())
        return text.strip()
    
    def extract_text_content(self, record: Dict) -> str:
        """
        Extract text from a record by trying different sources.
        Optimised priority order for archives and feed.
        """
        # 1. tiptap_body (main feed)
        if 'tiptap_body' in record and record['tiptap_body']:
            text = self.extract_text_from_tiptap(record['tiptap_body'])
            if text:
                return text
        
        # 2. body_trix_content (solution de repli)
        if 'body_trix_content' in record and record['body_trix_content']:
            text = self.clean_html(record['body_trix_content'])
            if text:
                return text
            
        # 3. truncated_content (archives - main source according to your example)
        if 'truncated_content' in record and record['truncated_content']:
            text = self.clean_html(record['truncated_content'])
            if text:
                return text
        
        # 4. body_for_editor (fallback)
        if 'body_for_editor' in record and record['body_for_editor']:
            text = self.clean_html(record['body_for_editor'])
            if text:
                return text
        
        # 5. body_text (fallback)
        if 'body_text' in record and record['body_text']:
            text = self.clean_html(record['body_text'])
            if text:
                return text
        
        # 6. Title/name as a last resort
        if 'display_title' in record and record['display_title']:
            return record['display_title'].strip()
        
        if 'name' in record and record['name']:
            return record['name'].strip()
        
        return ""
    
    def parse_date(self, date_str: str) -> str:
        """Parse an ISO date and return it in YYYY-MM-DD format."""
        if not date_str:
            return ''
        
        try:
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_obj.strftime('%Y-%m-%d')
        except (ValueError, AttributeError):
            return date_str
    
    def parse_feed_record(self, record: Dict) -> Optional[Dict]:
        """Parse a record from the main feed."""
        try:
            space_name = record.get('space_name', '')
            
            # Date extraction
            published_at = record.get('published_at', '')
            date = self.parse_date(published_at)
            
            # Text reviews extraction
            text = self.extract_text_content(record)
            
            if not text:
                return None
            
            return {
                'date': date,
                'category': space_name if space_name else 'unknown',
                'text': text,
                'post_id': record.get('id', ''),
                'source': 'feed'
            }
        
        except Exception as e:
            logger.warning(f"Parsing error feed record {record.get('id', 'unknown')}: {e}")
            return None
    
    def parse_archive_record(self, record: Dict, category: str) -> Optional[Dict]:
        """Parse an archive record."""
        try:
            # Date extraction
            published_at = record.get('published_at', '')
            date = self.parse_date(published_at)
            
            # Text Extraction
            text = self.extract_text_content(record)
            
            if not text:
                return None
            
            return {
                'date': date,
                'category': category,
                'text': text,
                'post_id': record.get('id', ''),
                'source': 'archive'
            }
        
        except Exception as e:
            logger.warning(f"Parsing error archive record {record.get('id', 'unknown')}: {e}")
            return None
    
    def scrape_feed(self, start_page: int = 1, max_pages: Optional[int] = None) -> List[Dict]:
        """Scrape the main feed."""
        reviews = []
        current_page = start_page
        pages_scraped = 0
        
        logger.info("=" * 60)
        logger.info(f"Scraping of main FEED (page {start_page}...)")
        logger.info("=" * 60)
        
        while True:
            if max_pages and pages_scraped >= max_pages:
                break
            
            logger.info(f"Feed - Page {current_page}...")
            data = self.fetch_feed_page(current_page)
            
            if not data:
                break
            
            records = data.get('records', [])
            if not records:
                logger.info("Feed - No record, end")
                break
            
            page_reviews = 0
            for record in records:
                parsed = self.parse_feed_record(record)
                if parsed:
                    reviews.append(parsed)
                    page_reviews += 1
            
            logger.info(f"Feed - Page {current_page}: {page_reviews}/{len(records)} reviews extracted")
            
            if not data.get('has_next_page', False):
                logger.info("Feed - Last page reached")
                break
            
            current_page += 1
            pages_scraped += 1
            time.sleep(1)  # Rate limiting to do some ethical parsing
        
        logger.info(f"Feed ended: {len(reviews)} reviews in {pages_scraped} pages")
        return reviews
    
    def scrape_archive(self, archive_key: str, start_page: int = 1, 
                      max_pages: Optional[int] = None) -> List[Dict]:
        """Scrape a specific archive"""
        if archive_key not in self.ARCHIVES:
            logger.error(f"Archive inconnue: {archive_key}")
            return []
        
        config = self.ARCHIVES[archive_key]
        space_id = config['space_id']
        category = config['category']
        name = config['name']
        
        reviews = []
        current_page = start_page
        pages_scraped = 0
        
        logger.info("=" * 60)
        logger.info(f"Scraping ARCHIVE: {name} (space_id={space_id}, category={category})")
        logger.info("=" * 60)
        
        while True:
            if max_pages and pages_scraped >= max_pages:
                break
            
            logger.info(f"Archive {name} - Page {current_page}...")
            data = self.fetch_archive_page(space_id, current_page)
            
            if not data:
                break
            
            records = data.get('records', [])
            if not records:
                logger.info(f"Archive {name} - No records, end")
                break
            
            page_reviews = 0
            for record in records:
                parsed = self.parse_archive_record(record, category)
                if parsed:
                    reviews.append(parsed)
                    page_reviews += 1
            
            logger.info(f"Archive {name} - Page {current_page}: {page_reviews}/{len(records)} reviews")
            
            if not data.get('has_next_page', False):
                logger.info(f"Archive {name} - Last page reached")
                break
            
            current_page += 1
            pages_scraped += 1
            time.sleep(1)  # Rate limiting to do some ethical parsing
        
        logger.info(f"Archive {name} ended: {len(reviews)} reviews in {pages_scraped} pages")
        return reviews
    
    def scrape_all_archives(self, start_page: int = 1, max_pages: Optional[int] = None) -> List[Dict]:
        """Scrape all archives."""
        all_archive_reviews = []
        
        for archive_key in self.ARCHIVES.keys():
            reviews = self.scrape_archive(archive_key, start_page, max_pages)
            all_archive_reviews.extend(reviews)
            time.sleep(2)  # Pause between archives to do some ethical parsing
        
        return all_archive_reviews
    
    def save_to_csv(self, reviews: List[Dict]):
        """Save the reviews in a CSV file."""
        if not reviews:
            logger.warning("No reviews to save")
            return
        
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['date', 'category', 'review']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for review in reviews:
                    writer.writerow({
                        'date': review['date'],
                        'category': review['category'],
                        'review': review['text']
                    })
            
            logger.info(f"OK - {len(reviews)} reviews saved in {self.output_file}")
        
        except IOError as e:
            logger.error(f"CSV saving error: {e}")
    
    def print_statistics(self, reviews: List[Dict]):
        """Print detailled statistics of the scrapping."""
        if not reviews:
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATS")
        logger.info("=" * 60)
        
        # Par catégorie
        category_counts = {}
        source_counts = {}
        
        for review in reviews:
            cat = review['category']
            src = review.get('source', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1
        
        logger.info(f"\nTotal: {len(reviews)} reviews")
        
        logger.info("\nPer category:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {cat}: {count} reviews")
        
        logger.info("\nPer source:")
        for src, count in sorted(source_counts.items()):
            logger.info(f"  - {src}: {count} reviews")
    
    def clean_reviews_text(self, reviews: List[Dict]) -> List[Dict]:
        """
        Clean text reviews by deleting skip lines.
        """
        cleaned_reviews = []
        
        for review in reviews:
            cleaned_review = review.copy()
            
            # Replaces return line characters by space character
            if 'text' in cleaned_review and cleaned_review['text']:
                text = cleaned_review['text'].replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                # Normalise multiple spaces
                text = ' '.join(text.split())
                cleaned_review['text'] = text.strip()
            
            cleaned_reviews.append(cleaned_review)
        
        logger.info(f"Cleaning done: {len(cleaned_reviews)} reviews cleaned")
        return cleaned_reviews


    def verify_csv_integrity(self):
        """
       Veryfiing CSV integrity by comparing the number of reveiws extracted and the one within the CSV
        """
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = sum(1 for _ in reader)
                # -1 for the head
                actual_reviews = lines - 1
            
            logger.info(f"Verification CSV: {actual_reviews} reviex rows in {self.output_file}")
            return actual_reviews
        
        except IOError as e:
            logger.error(f"Error during CSV verification: {e}")
            return None


    # Method to  run the pipeline
    def run(self, scrape_feed: bool = True, scrape_archives: bool = True,
            start_page: int = 1, max_pages: Optional[int] = None):
        """Execute full scrapping of accessibleGO"""
        logger.info("=" * 60)
        logger.info("BEGINING OF ACCESSIBLEGO SCRAPPING")
        logger.info("=" * 60)
        
        all_reviews = []
        
        # feed scrapping
        if scrape_feed:
            feed_reviews = self.scrape_feed(start_page, max_pages)
            all_reviews.extend(feed_reviews)
        
        # archives scrapping
        if scrape_archives:
            archive_reviews = self.scrape_all_archives(start_page, max_pages)
            all_reviews.extend(archive_reviews)
        
        # Cleaning of the skip lines in reviews
        logger.info("\nCleaning of the skip lines in reviews...")
        all_reviews = self.clean_reviews_text(all_reviews)
        
        # saving
        self.save_to_csv(all_reviews)
        
        # Verification of CSV integrity
        actual_count = self.verify_csv_integrity()
        if actual_count and actual_count != len(all_reviews):
            logger.warning(f"WARNING: Differences detected")
            logger.warning(f"   Reviews collectd: {len(all_reviews)}")
            logger.warning(f"   CVS rows: {actual_count}")
        else:
            logger.info(f"✓ Intégrity verified: {len(all_reviews)} reviews = {actual_count} row CSV")
        
        # stats printing
        self.print_statistics(all_reviews)
        
        logger.info("\n" + "=" * 60)
        logger.info("SCRAPING ENDED")
        logger.info("=" * 60)

        # Format
        df_reviews = pl.read_csv(self.output_file)
        df_reviews = df_reviews.with_row_index("id")
        df_reviews.write_csv(self.output_file)


if __name__ == "__main__":
    logger = setup_logger()
    scraper = AccessibleGoScraper(output_file="data/accessiblego.csv")
    scraper.run(scrape_feed=True, scrape_archives=True)
