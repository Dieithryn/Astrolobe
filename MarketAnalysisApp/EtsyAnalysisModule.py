"""
Etsy 3D Print Market Research Tool - Web Scraping Version (Firefox)
Analyzes cosplay and 3D printing markets on Etsy using web scraping + DeepSeek AI

NO API KEY REQUIRED - Uses web scraping instead

Requirements:
pip install requests beautifulsoup4 pandas selenium webdriver-manager ollama
"""

import os
import json
import time
import re
import pandas as pd
import random
import requests
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:118.0) Gecko/20100101 Firefox/118.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
]

# Selenium for dynamic content
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.firefox import GeckoDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: selenium not installed. Using basic scraping only.")

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. AI analysis will be disabled.")


def random_user_agent():
    return random.choice(USER_AGENTS)


def get_firefox_driver(headless: bool = False):
    options = Options()
    options.headless = headless
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference("useAutomationExtension", False)
    options.set_preference("general.useragent.override", random_user_agent())
    options.set_preference("media.navigator.enabled", False)
    options.set_preference("privacy.trackingprotection.enabled", False)

    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)

    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def scroll_page_human(driver):
    for _ in range(random.randint(2, 4)):
        driver.execute_script("window.scrollBy(0, arguments[0]);", random.randint(200, 600))
        time.sleep(random.uniform(1.5, 3.5))


class EtsyAPIClient:
    """Uses Etsy's official API first before falling back to scraping."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openapi.etsy.com/v3/application"

    def search_listings(self, query: str, limit: int = 100):
        """Returns Etsy API search results or None if API fails."""
        try:
            url = f"{self.base_url}/listings/active"
            params = {
                "api_key": self.api_key,
                "keywords": query,
                "limit": limit,
                "includes": "images"
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"⚠️ Etsy API error: {response.status_code} → {response.text[:200]}")
                return None

            data = response.json()

            if "results" not in data or len(data["results"]) == 0:
                print("⚠️ Etsy API returned no results")
                return None

            # Normalize to match your scraper structure
            normalized = []
            for item in data["results"]:
                normalized.append({
                    "listing_id": item.get("listing_id"),
                    "title": item.get("title"),
                    "price": float(item.get("price", 0)),
                    "url": item.get("url"),
                    "image": item.get("images", [{}])[0].get("url_fullxfull"),
                    "source": "etsy_api"
                })

            return normalized

        except Exception as e:
            print(f"⚠️ Etsy API failed: {e}")
            return None

class EtsyWebScraper:
    """Scrapes Etsy listings without API access"""
    
    def __init__(self, use_selenium: bool = False):
        self.base_url = "https://www.etsy.com"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": random_user_agent(),
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive"
        }
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.driver = None
        if self.use_selenium:
            self._init_selenium()

    def _init_selenium(self):
        """Initialize patched Selenium WebDriver using Firefox"""
        try:
            self.driver = get_firefox_driver(headless=False)
            print("✅ Selenium WebDriver (Firefox) initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize Selenium (Firefox): {e}")
            self.use_selenium = False

    def search_listings(self, keywords: str, max_pages: int = 5) -> List[Dict]:

        """Search Etsy for listings by keywords"""
        all_listings = []
        search_url = f"{self.base_url}/search?q={quote_plus(keywords)}"
        print(f"  Searching: {keywords}")
        for page in range(1, max_pages + 1):
            page_url = f"{search_url}&page={page}"
            try:
                if self.use_selenium:
                    listings = self._scrape_with_selenium(page_url)
                else:
                    listings = self._scrape_with_requests(page_url)
                if not listings:
                    break
                all_listings.extend(listings)
                print(f"    Page {page}: Found {len(listings)} listings")
                time.sleep(2)
            except Exception as e:
                print(f"    Error on page {page}: {e}")
                break
        return all_listings

    def _scrape_with_requests(self, url: str) -> List[Dict]:
        """Scrape using requests + BeautifulSoup"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_search_results(soup, url)
        except Exception as e:
            print(f"    Request error: {e}")
            return []

    def _scrape_with_selenium(self, url: str) -> List[Dict]:
        """Scrape using Selenium"""
        listings = []
        try:
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-search-results-container]"))
            )
            scroll_page_human(self.driver)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            listings = self._parse_search_results(soup, url)
        except Exception as e:
            print(f"⚠️ Selenium scraping error at {url}: {e}")
        return listings

    def _parse_search_results(self, soup: BeautifulSoup, search_url: str) -> List[Dict]:
        results = []
        cards = soup.select('li[data-listing-id]')
        for card in cards:
            data = self._extract_listing_data(card, search_url)
            if data:
                results.append(data)
        return results

    def _extract_listing_data(self, card, search_url: str) -> Optional[Dict]:
        try:
            title_elem = card.select_one('h3') or card.select_one('h2') or card.select_one('[class*="title"]')
            title = title_elem.get_text(strip=True) if title_elem else None
            price_elem = card.select_one('span[class*="currency-value"]') or card.select_one('p[class*="price"]')
            price = self._parse_price(price_elem.get_text(strip=True)) if price_elem else 0
            link_elem = card.select_one('a[href*="/listing/"]')
            url, listing_id = None, None
            if link_elem and link_elem.get('href'):
                url = urljoin(self.base_url, link_elem.get('href'))
                match = re.search(r'/listing/(\d+)', url)
                listing_id = match.group(1) if match else None
            shop_elem = card.select_one('span[class*="shop"] p, p[class*="shop"]')
            shop_name = shop_elem.get_text(strip=True) if shop_elem else None
            rating_elem = card.select_one('[class*="rating"], [aria-label*="rating"]')
            rating = None
            if rating_elem:
                text = rating_elem.get('aria-label', '') or rating_elem.get_text()
                match = re.search(r'(\d+\.?\d*)\s*out of', text)
                if match:
                    rating = float(match.group(1))
            if not title or price is None:
                return None
            return {
                'listing_id': listing_id,
                'title': title,
                'price': price,
                'currency': 'USD',
                'url': url,
                'shop_name': shop_name,
                'rating': rating,
                'search_url': search_url,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception:
            return None

    def _parse_price(self, price_text: str) -> float:
        match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
        return float(match.group()) if match else 0.0

    def get_listing_details(self, listing_url: str) -> Optional[Dict]:
        try:
            if self.use_selenium:
                self.driver.get(listing_url)
                time.sleep(2)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                response = self.session.get(listing_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
            details = {}
            desc_elem = soup.select_one('div[data-product-details-description]')
            if desc_elem:
                details['description'] = desc_elem.get_text(strip=True)[:500]
            tag_elems = soup.select('a[href*="/market/"]')
            details['tags'] = [tag.get_text(strip=True) for tag in tag_elems[:10]]
            fav_elem = soup.select_one('[data-favorers-count]')
            if fav_elem:
                details['num_favorers'] = int(fav_elem.get('data-favorers-count', 0))
            time.sleep(1)
            return details
        except Exception as e:
            print(f"    Error fetching details: {e}")
            return None

    def close(self):
        if self.driver:
            self.driver.quit()

class MarketAnalyzer:
    """Analyzes scraped Etsy market data"""
    
class MarketAnalyzer:
    def __init__(self, scraper=None, api_key=None):
        self.scraper = scraper
        self.api = EtsyAPIClient(api_key) if api_key else None
        self.data_cache = []

    
    def fetch_listings(self, search_term: str, max_pages: int = 3) -> List[Dict]:
        """
        Fetch listings using API first, then fallback to scraper if needed.
        """
        listings = []

        # 1️⃣ Try API first
        if self.api:
            try:
                print(f"🔹 Trying Etsy API for '{search_term}'...")
                listings = self.api.search_listings(search_term, max_pages=max_pages)
                if listings:
                    print(f"    ✅ Found {len(listings)} listings via API")
                    return listings
            except Exception as e:
                print(f"    ⚠️ API failed: {e} — falling back to scraping")

        # 2️⃣ Fallback to scraper
        if self.scraper:
            print(f"🔹 Using web scraping for '{search_term}'...")
            listings = self.scraper.search_listings(search_term, max_pages=max_pages)
            print(f"    ✅ Found {len(listings)} listings via scraping")

        return listings

    def collect_cosplay_data(self, fandoms: List[str], max_pages_per_search: int = 3) -> pd.DataFrame:
        """Collect data on cosplay-related 3D prints"""
        all_listings = []
        
        search_templates = [
            "{fandom} cosplay 3d print",
            "{fandom} cosplay prop",
            "{fandom} 3d model stl",
            "{fandom} cosplay accessory"
        ]
        
        for fandom in fandoms:
            print(f"\n📊 Searching for {fandom} cosplay items...")
            
            for template in search_templates:
                search_term = template.format(fandom=fandom)
                listings = self.scraper.search_listings(search_term, max_pages=max_pages_per_search)
                
                # Add metadata
                for listing in listings:
                    listing['category'] = fandom
                    listing['search_term'] = search_term
                
                all_listings.extend(listings)
                time.sleep(1)
        
        df = pd.DataFrame(all_listings)
        self.data_cache = df
        return df
    
    def collect_product_opportunity_data(self, product_types: List[str], max_pages: int = 3) -> pd.DataFrame:
        """Collect data on simple 3D printable products"""
        all_listings = []
        
        for product in product_types:
            print(f"\n📊 Searching for {product}...")
            listings = self.scraper.search_listings(product, max_pages=max_pages)
            
            for listing in listings:
                listing['category'] = 'product'
                listing['search_term'] = product
            
            all_listings.extend(listings)
            time.sleep(1)
        
        df = pd.DataFrame(all_listings)
        return df
    
    def analyze_pricing(self, df: pd.DataFrame) -> Dict:
        """Analyze pricing patterns"""
        if df.empty:
            return {}
        
        analysis = {
            "overall_stats": {
                "mean_price": df["price"].mean(),
                "median_price": df["price"].median(),
                "min_price": df["price"].min(),
                "max_price": df["price"].max(),
                "std_dev": df["price"].std(),
                "total_listings": len(df)
            },
            "by_category": {}
        }
        
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            analysis["by_category"][str(category)] = {
                "count": len(cat_df),
                "mean_price": cat_df["price"].mean(),
                "median_price": cat_df["price"].median(),
                "price_range": f"${cat_df['price'].min():.2f} - ${cat_df['price'].max():.2f}"
            }
        
        return analysis
    
    def identify_gaps(self, df: pd.DataFrame) -> Dict:
        """Identify potential market gaps"""
        gaps = {
            "low_competition_categories": [],
            "high_price_categories": [],
            "category_counts": {}
        }
        
        if df.empty:
            return gaps
        
        # Count listings per category
        category_counts = df["category"].value_counts().to_dict()
        gaps["category_counts"] = category_counts
        
        # Find low competition (fewer listings)
        for category, count in category_counts.items():
            if count < 30:  # Less than 30 listings = low competition
                gaps["low_competition_categories"].append({
                    "category": category,
                    "listing_count": count,
                    "avg_price": df[df["category"] == category]["price"].mean()
                })
        
        # Find high-price categories (opportunity for budget alternatives)
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            avg_price = cat_df["price"].mean()
            if avg_price > 25:  # Above $25 average
                gaps["high_price_categories"].append({
                    "category": category,
                    "avg_price": avg_price,
                    "listing_count": len(cat_df)
                })
        
        return gaps


class DeepSeekAnalyzer:
    """Integrates with locally-run DeepSeek for AI analysis"""
    
    def __init__(self, model_name: str = "deepseek-r1:14b"):
        self.model_name = model_name
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not installed. Install with: pip install ollama")
        
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama. Make sure it's running: {e}")
    
    def analyze_market_opportunities(self, market_data: pd.DataFrame, pricing_analysis: Dict, gaps: Dict) -> str:
        """Use DeepSeek to analyze market data"""
        
        prompt = f"""You are a market research analyst specializing in 3D printing and cosplay merchandise.

Analyze the following Etsy market data scraped from live listings:

MARKET DATA SUMMARY:
- Total listings analyzed: {len(market_data)}
- Categories: {', '.join(market_data['category'].unique()[:10])}
- Date collected: {datetime.now().strftime('%Y-%m-%d')}

PRICING ANALYSIS:
{json.dumps(pricing_analysis, indent=2)}

MARKET GAPS:
{json.dumps(gaps, indent=2)}

TOP SELLERS (by price):
{market_data.nlargest(10, 'price')[['title', 'price', 'category']].to_string()}

Please provide:
1. Top 5 underserved cosplay communities with high potential
2. 10 specific 3D printable products for quick revenue generation (under $20)
3. Optimal pricing strategy based on competition levels
4. Specific niche opportunities with low competition
5. Product recommendations for each identified gap

Be specific with character names, game titles, and actual product ideas."""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating AI analysis: {e}"


class ReportGenerator:
    """Generates comprehensive market research reports"""
    
    def __init__(self, output_dir: str = "market_research_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, market_data: pd.DataFrame, pricing_analysis: Dict, 
                       gaps: Dict, ai_analysis: Optional[str] = None) -> str:
        """Generate comprehensive markdown report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"etsy_market_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Etsy 3D Print Market Research Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: Web Scraping (No API)\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Listings Analyzed:** {len(market_data)}\n")
            f.write(f"- **Categories Covered:** {len(market_data['category'].unique())}\n")
            f.write(f"- **Average Price:** ${pricing_analysis.get('overall_stats', {}).get('mean_price', 0):.2f}\n")
            f.write(f"- **Price Range:** ${pricing_analysis.get('overall_stats', {}).get('min_price', 0):.2f} - ${pricing_analysis.get('overall_stats', {}).get('max_price', 0):.2f}\n\n")
            
            # Market Gaps
            f.write("## Market Gap Analysis\n\n")
            
            f.write("### Low Competition Categories\n\n")
            for gap in gaps.get("low_competition_categories", [])[:10]:
                f.write(f"- **{gap['category']}**: {gap['listing_count']} listings, avg ${gap['avg_price']:.2f}\n")
            f.write("\n")
            
            f.write("### High Price Categories (Budget Alternative Opportunities)\n\n")
            for gap in gaps.get("high_price_categories", [])[:10]:
                f.write(f"- **{gap['category']}**: ${gap['avg_price']:.2f} avg, {gap['listing_count']} listings\n")
            f.write("\n")
            
            # Pricing by Category
            f.write("## Pricing Analysis by Category\n\n")
            for category, data in pricing_analysis.get("by_category", {}).items():
                f.write(f"### {category}\n\n")
                f.write(f"- Listings: {data['count']}\n")
                f.write(f"- Mean Price: ${data['mean_price']:.2f}\n")
                f.write(f"- Median Price: ${data['median_price']:.2f}\n")
                f.write(f"- Range: {data['price_range']}\n\n")
            
            # AI Analysis
            if ai_analysis:
                f.write("## AI-Powered Market Insights\n\n")
                f.write(ai_analysis)
                f.write("\n\n")
            
            # Sample Listings
            f.write("## Sample High-Value Listings\n\n")
            top_listings = market_data.nlargest(20, 'price')
            for _, listing in top_listings.iterrows():
                f.write(f"### {listing['title']}\n")
                f.write(f"- Price: ${listing['price']:.2f}\n")
                f.write(f"- Category: {listing['category']}\n")
                if listing.get('url'):
                    f.write(f"- [View Listing]({listing['url']})\n")
                f.write("\n")
        
        print(f"\n✅ Report saved to: {filepath}")
        return filepath
    
    def export_data(self, market_data: pd.DataFrame) -> str:
        """Export raw data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"etsy_raw_data_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        market_data.to_csv(filepath, index=False)
        print(f"✅ Raw data exported to: {filepath}")
        return filepath


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Etsy 3D Print Market Research Tool (Web Scraping)")
    print("=" * 60)
    print("\n⚠️  Note: Web scraping is slower than API access")
    print("    Be patient and respectful of Etsy's servers\n")
    
    # Initialize scraper (try Selenium first, fall back to requests)
    use_selenium = SELENIUM_AVAILABLE
    scraper = EtsyWebScraper(use_selenium=use_selenium)
    ETSY_API_KEY = "6dbv7wmh0dgwuko2e9g0fglc"
    analyzer = MarketAnalyzer(scraper=scraper, api_key=ETSY_API_KEY)
    
    # Define research parameters
    cosplay_fandoms = [
        "Baldurs Gate",
        "Elden Ring",
        "Genshin Impact",
        "Critical Role",
        "Dungeons Dragons",
        "Warhammer 40k",
        "Chainsaw Man",
        "Cyberpunk 2077",
        "Final Fantasy XIV",
        "Honkai Star Rail"
    ]
    
    simple_products = [
        "3d printed phone charm gaming",
        "3d printed keychain anime",
        "3d printed dice holder",
        "3d printed miniature stand",
        "3d printed cable holder gaming"
    ]
    
    # Collect data
    print("\n🔍 Phase 1: Collecting cosplay market data...")
    print("    (This may take 10-15 minutes)")
    cosplay_data = analyzer.collect_cosplay_data(cosplay_fandoms, max_pages_per_search=2)
    
    print("\n🔍 Phase 2: Collecting simple product data...")
    product_data = analyzer.collect_product_opportunity_data(simple_products, max_pages=2)
    
    # Combine datasets
    all_data = pd.concat([cosplay_data, product_data], ignore_index=True)
    
    # Remove duplicates
    all_data = all_data.drop_duplicates(subset=['listing_id'], keep='first')
    
    print(f"\n📊 Total unique listings collected: {len(all_data)}")
    
    # Analyze data
    print("\n📈 Phase 3: Analyzing pricing and market gaps...")
    pricing_analysis = analyzer.analyze_pricing(all_data)
    gaps = analyzer.identify_gaps(all_data)
    
    # AI Analysis
    ai_analysis = None
    if OLLAMA_AVAILABLE:
        try:
            print("\n🤖 Phase 4: Running DeepSeek AI analysis...")
            deepseek = DeepSeekAnalyzer()
            ai_analysis = deepseek.analyze_market_opportunities(all_data, pricing_analysis, gaps)
        except Exception as e:
            print(f"⚠️ AI analysis skipped: {e}")
    
    # Generate reports
    print("\n📝 Phase 5: Generating reports...")
    report_gen = ReportGenerator()
    report_path = report_gen.generate_report(all_data, pricing_analysis, gaps, ai_analysis)
    data_path = report_gen.export_data(all_data)
    
    # Cleanup
    scraper.close()
    
    print("\n" + "=" * 60)
    print("✅ Market research complete!")
    print(f"📄 Report: {report_path}")
    print(f"💾 Data: {data_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()