import os
import json
import time
import re
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter
import configparser

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. AI analysis will be disabled.")


class RedditConfig:
    """Manages Reddit API configuration and search parameters"""
    
    def __init__(self, config_file: str = "reddit_config.ini"):
        self.config = configparser.ConfigParser()
        
        if not os.path.exists(config_file):
            self._create_default_config(config_file)
            print(f"⚠️ Created default config file: {config_file}")
            print("Please edit it with your Reddit API credentials")
            raise FileNotFoundError(f"Please configure {config_file} with your credentials")
        
        self.config.read(config_file)
    
    def _create_default_config(self, config_file: str):
        """Create default configuration file"""
        self.config['REDDIT'] = {
            'client_id': 'YOUR_CLIENT_ID_HERE',
            'client_secret': 'YOUR_CLIENT_SECRET_HERE',
            'user_agent': 'CosplayDemandAnalyzer/1.0'
        }
        
        with open(config_file, 'w') as f:
            self.config.write(f)
    
    def get_reddit_credentials(self) -> Dict[str, str]:
        """Get Reddit API credentials"""
        return {
            'client_id': self.config['REDDIT']['client_id'],
            'client_secret': self.config['REDDIT']['client_secret'],
            'user_agent': self.config['REDDIT']['user_agent']
        }


class SearchConfig:
    """Manages subreddit targeting and search criteria"""
    
    def __init__(self, config_file: str = "search_config.json"):
        self.config_file = config_file
        
        if not os.path.exists(config_file):
            self._create_default_search_config()
            print(f"✅ Created default search config: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def _create_default_search_config(self):
        """Create default search configuration"""
        default_config = {
            "subreddits": {
                "gaming_general": [
                    "gaming",
                    "cosplay",
                    "cosplayers",
                    "3Dprinting"
                ],
                "specific_games": [
                    "Genshin_Impact",
                    "ffxiv",
                    "Eldenring",
                    "BaldursGate3",
                    "cyberpunkgame",
                    "Warhammer40k",
                    "DnD",
                    "criticalrole",
                    "HonkaiStarRail",
                    "ZZZ_Official"
                ],
                "anime_manga": [
                    "anime",
                    "manga",
                    "ChainsawMan",
                    "OnePiece",
                    "JujutsuKaisen",
                    "DemonSlayer"
                ],
                "maker_communities": [
                    "cosplayprops",
                    "functionalprint",
                    "3Dprintedtabletop"
                ]
            },
            "search_keywords": {
                "demand_signals": [
                    "where can i find",
                    "looking for",
                    "need help finding",
                    "does anyone have",
                    "can't find",
                    "anyone selling",
                    "stl file",
                    "3d model",
                    "how to make",
                    "tutorial for",
                    "cosplay help",
                    "prop help"
                ],
                "quality_indicators": [
                    "accurate",
                    "screen accurate",
                    "game accurate",
                    "high quality",
                    "detailed",
                    "realistic",
                    "premium"
                ],
                "frustration_signals": [
                    "can't afford",
                    "too expensive",
                    "sold out",
                    "discontinued",
                    "hard to find",
                    "nowhere to buy",
                    "wish someone made"
                ]
            },
            "post_relevance_criteria": {
                "min_upvotes": 3,
                "min_comments": 2,
                "max_age_days": 365,
                "exclude_keywords": [
                    "spam",
                    "nsfw",
                    "18+",
                    "nude"
                ]
            },
            "deepseek_prompts": {
                "demand_analysis": "Analyze these Reddit posts for cosplay/3D printing demand. Identify: 1) Most requested characters/props, 2) Pain points (price, availability, quality), 3) Underserved fandoms, 4) Specific product opportunities. Be concrete with character/game names.",
                "sentiment_analysis": "Analyze sentiment in these cosplay community discussions. Identify: 1) What frustrates cosplayers most, 2) What they're willing to pay for, 3) Quality vs price preferences, 4) Gap between what exists and what's wanted.",
                "market_validation": "Compare Etsy supply data with Reddit demand signals. Identify: 1) High demand + low supply opportunities, 2) Overserved markets to avoid, 3) Price sensitivity by category, 4) Quality expectations by community."
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.config = default_config
    
    def get_all_subreddits(self) -> List[str]:
        """Get flattened list of all subreddits"""
        all_subs = []
        for category, subs in self.config['subreddits'].items():
            all_subs.extend(subs)
        return list(set(all_subs))  # Remove duplicates
    
    def get_subreddits_by_category(self, category: str) -> List[str]:
        """Get subreddits for a specific category"""
        return self.config['subreddits'].get(category, [])
    
    def add_subreddit(self, category: str, subreddit: str):
        """Add a new subreddit to track"""
        if category not in self.config['subreddits']:
            self.config['subreddits'][category] = []
        
        if subreddit not in self.config['subreddits'][category]:
            self.config['subreddits'][category].append(subreddit)
            self._save_config()
            print(f"✅ Added r/{subreddit} to {category}")
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)


class RedditScraper:
    """Scrapes Reddit for cosplay demand signals"""
    
    def __init__(self, reddit_config: RedditConfig, search_config: SearchConfig):
        credentials = reddit_config.get_reddit_credentials()
        
        self.reddit = praw.Reddit(
            client_id=credentials['client_id'],
            client_secret=credentials['client_secret'],
            user_agent=credentials['user_agent']
        )
        
        self.search_config = search_config
        self.posts_data = []
    
    def scrape_subreddit(self, subreddit_name: str, time_filter: str = 'year', 
                        limit: int = 100, search_query: Optional[str] = None) -> List[Dict]:
        """Scrape posts from a subreddit"""
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            print(f"  📡 Scraping r/{subreddit_name}...")
            
            if search_query:
                # Search for specific keywords
                submissions = subreddit.search(
                    search_query, 
                    time_filter=time_filter, 
                    limit=limit
                )
            else:
                # Get hot/top posts
                submissions = subreddit.hot(limit=limit)
            
            for submission in submissions:
                post_data = self._extract_post_data(submission, subreddit_name)
                if post_data and self._is_relevant_post(post_data):
                    posts.append(post_data)
            
            print(f"    Found {len(posts)} relevant posts")
            
        except Exception as e:
            print(f"    ⚠️ Error scraping r/{subreddit_name}: {e}")
        
        return posts
    
    def _extract_post_data(self, submission, subreddit_name: str) -> Dict:
        """Extract relevant data from a Reddit post"""
        try:
            # Get top comments for context
            submission.comments.replace_more(limit=0)
            top_comments = [
                {
                    'body': comment.body,
                    'score': comment.score
                }
                for comment in submission.comments[:10]
                if hasattr(comment, 'body')
            ]
            
            return {
                'post_id': submission.id,
                'subreddit': subreddit_name,
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'author': str(submission.author),
                'url': f"https://reddit.com{submission.permalink}",
                'link_flair': submission.link_flair_text,
                'top_comments': top_comments,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    Error extracting post data: {e}")
            return None
    
    def _is_relevant_post(self, post_data: Dict) -> bool:
        """Check if post meets relevance criteria"""
        criteria = self.search_config.config['post_relevance_criteria']
        
        # Check minimum engagement
        if post_data['score'] < criteria['min_upvotes']:
            return False
        
        if post_data['num_comments'] < criteria['min_comments']:
            return False
        
        # Check age
        post_age = datetime.now() - datetime.fromtimestamp(post_data['created_utc'])
        if post_age.days > criteria['max_age_days']:
            return False
        
        # Check for excluded keywords
        text = f"{post_data['title']} {post_data['selftext']}".lower()
        for keyword in criteria['exclude_keywords']:
            if keyword.lower() in text:
                return False
        
        # Check for demand signals
        demand_signals = self.search_config.config['search_keywords']['demand_signals']
        has_demand_signal = any(signal.lower() in text for signal in demand_signals)
        
        # Check for 3D printing or cosplay relevance
        relevant_terms = ['3d print', 'stl', 'cosplay', 'prop', 'costume', 'armor', 'weapon']
        has_relevant_term = any(term in text for term in relevant_terms)
        
        return has_demand_signal or has_relevant_term
    
    def search_across_subreddits(self, search_terms: List[str], 
                                subreddits: List[str], 
                                time_filter: str = 'year',
                                posts_per_sub: int = 50) -> List[Dict]:
        """Search for specific terms across multiple subreddits"""
        all_posts = []
        
        for subreddit in subreddits:
            for term in search_terms:
                posts = self.scrape_subreddit(
                    subreddit, 
                    time_filter=time_filter,
                    limit=posts_per_sub,
                    search_query=term
                )
                all_posts.extend(posts)
                time.sleep(2)  # Be respectful of rate limits
        
        return all_posts
    
    def get_demand_signals(self, subreddits: Optional[List[str]] = None,
                          time_filter: str = 'year',
                          posts_per_sub: int = 100) -> pd.DataFrame:
        """Scrape for demand signals across subreddits"""
        
        if subreddits is None:
            subreddits = self.search_config.get_all_subreddits()
        
        all_posts = []
        demand_keywords = self.search_config.config['search_keywords']['demand_signals']
        
        print(f"\n🔍 Scraping {len(subreddits)} subreddits for demand signals...")
        
        # First pass: Get general posts
        for subreddit in subreddits:
            posts = self.scrape_subreddit(
                subreddit,
                time_filter=time_filter,
                limit=posts_per_sub
            )
            all_posts.extend(posts)
            time.sleep(2)
        
        # Second pass: Targeted keyword searches
        print(f"\n🎯 Searching for specific demand signals...")
        targeted_posts = self.search_across_subreddits(
            demand_keywords[:5],  # Top 5 demand signals
            subreddits,
            time_filter=time_filter,
            posts_per_sub=20
        )
        all_posts.extend(targeted_posts)
        
        # Remove duplicates
        unique_posts = {post['post_id']: post for post in all_posts}
        df = pd.DataFrame(list(unique_posts.values()))
        
        print(f"\n✅ Collected {len(df)} unique posts")
        
        return df


class DemandAnalyzer:
    """Analyzes Reddit data for market demand patterns"""
    
    def __init__(self, search_config: SearchConfig):
        self.search_config = search_config
    
    def extract_mentioned_characters(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract and count character/franchise mentions"""
        mentions = Counter()
        
        # Common patterns for character mentions
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?:\s+cosplay|\s+costume|\s+armor)',
            r'cosplay(?:ing)?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for _, row in df.iterrows():
            text = f"{row['title']} {row['selftext']}"
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                mentions.update(matches)
        
        return dict(mentions.most_common(50))
    
    def identify_pain_points(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify common pain points from posts"""
        pain_points = {
            'price': [],
            'availability': [],
            'quality': [],
            'complexity': []
        }
        
        frustration_signals = self.search_config.config['search_keywords']['frustration_signals']
        
        for _, row in df.iterrows():
            text = f"{row['title']} {row['selftext']}".lower()
            
            if any(signal in text for signal in ['expensive', 'afford', 'cheap', 'budget']):
                pain_points['price'].append(row['title'])
            
            if any(signal in text for signal in ['sold out', 'find', 'where', 'nowhere']):
                pain_points['availability'].append(row['title'])
            
            if any(signal in text for signal in ['quality', 'accurate', 'detailed', 'better']):
                pain_points['quality'].append(row['title'])
            
            if any(signal in text for signal in ['hard to make', 'complicated', 'difficult', 'help']):
                pain_points['complexity'].append(row['title'])
        
        # Limit to top 20 per category
        return {k: v[:20] for k, v in pain_points.items()}
    
    def get_trending_topics(self, df: pd.DataFrame, min_mentions: int = 3) -> List[Dict]:
        """Identify trending characters/games with high engagement"""
        
        # Group by subreddit and analyze engagement
        trending = []
        
        for subreddit in df['subreddit'].unique():
            sub_df = df[df['subreddit'] == subreddit]
            
            if len(sub_df) >= min_mentions:
                trending.append({
                    'subreddit': subreddit,
                    'post_count': len(sub_df),
                    'avg_score': sub_df['score'].mean(),
                    'avg_comments': sub_df['num_comments'].mean(),
                    'engagement_score': sub_df['score'].sum() + sub_df['num_comments'].sum()
                })
        
        # Sort by engagement
        trending.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        return trending
    
    def analyze_price_sensitivity(self, df: pd.DataFrame) -> Dict:
        """Analyze what price points are discussed"""
        price_mentions = {
            'under_10': 0,
            '10_25': 0,
            '25_50': 0,
            '50_100': 0,
            'over_100': 0
        }
        
        price_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        
        for _, row in df.iterrows():
            text = f"{row['title']} {row['selftext']}"
            prices = re.findall(price_pattern, text)
            
            for price_str in prices:
                price = float(price_str.replace(',', ''))
                
                if price < 10:
                    price_mentions['under_10'] += 1
                elif price < 25:
                    price_mentions['10_25'] += 1
                elif price < 50:
                    price_mentions['25_50'] += 1
                elif price < 100:
                    price_mentions['50_100'] += 1
                else:
                    price_mentions['over_100'] += 1
        
        return price_mentions


class DeepSeekIntegration:
    """Integrates Reddit data with DeepSeek for deeper analysis"""
    
    def __init__(self, search_config: SearchConfig, model_name: str = "deepseek-r1:14b"):
        self.search_config = search_config
        self.model_name = model_name
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama not installed")
    
    def analyze_demand(self, reddit_df: pd.DataFrame, analysis_type: str = "demand_analysis") -> str:
        """Use DeepSeek to analyze Reddit demand data"""
        
        prompt_template = self.search_config.config['deepseek_prompts'].get(
            analysis_type,
            self.search_config.config['deepseek_prompts']['demand_analysis']
        )
        
        # Prepare data summary
        top_posts = reddit_df.nlargest(20, 'score')
        
        posts_summary = "\n\n".join([
            f"POST: {row['title']}\n"
            f"Subreddit: r/{row['subreddit']}\n"
            f"Score: {row['score']} | Comments: {row['num_comments']}\n"
            f"Content: {row['selftext'][:300]}...\n"
            f"Top Comment: {row['top_comments'][0]['body'][:200] if row['top_comments'] else 'N/A'}..."
            for _, row in top_posts.iterrows()
        ])
        
        full_prompt = f"{prompt_template}\n\nREDDIT DATA:\n{posts_summary}"
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error: {e}"
    
    def compare_with_etsy_data(self, reddit_df: pd.DataFrame, 
                              etsy_pricing: Dict, etsy_gaps: Dict) -> str:
        """Compare Reddit demand with Etsy supply data"""
        
        prompt = self.search_config.config['deepseek_prompts']['market_validation']
        
        reddit_summary = f"""
REDDIT DEMAND DATA:
- Total posts analyzed: {len(reddit_df)}
- Subreddits: {', '.join(reddit_df['subreddit'].unique()[:10])}
- Average engagement: {reddit_df['score'].mean():.1f} upvotes
- Top discussed topics: {', '.join(reddit_df.nlargest(10, 'score')['title'].tolist())}

ETSY SUPPLY DATA:
{json.dumps(etsy_pricing, indent=2)}

MARKET GAPS:
{json.dumps(etsy_gaps, indent=2)}
"""
        
        full_prompt = f"{prompt}\n\n{reddit_summary}\n\nProvide specific product recommendations with estimated demand and pricing."
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error: {e}"


class ReportGenerator:
    """Generates reports from Reddit analysis"""
    
    def __init__(self, output_dir: str = "reddit_demand_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, reddit_df: pd.DataFrame, 
                       character_mentions: Dict,
                       pain_points: Dict,
                       trending: List[Dict],
                       price_sensitivity: Dict,
                       ai_analysis: Optional[str] = None) -> str:
        """Generate comprehensive demand analysis report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_demand_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Reddit Cosplay Demand Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: Reddit Community Analysis\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Posts Analyzed:** {len(reddit_df)}\n")
            f.write(f"- **Subreddits Covered:** {len(reddit_df['subreddit'].unique())}\n")
            f.write(f"- **Total Engagement:** {reddit_df['score'].sum():,} upvotes\n")
            f.write(f"- **Average Comments per Post:** {reddit_df['num_comments'].mean():.1f}\n\n")
            
            # Character/Franchise Mentions
            f.write("## Most Mentioned Characters/Franchises\n\n")
            for char, count in list(character_mentions.items())[:20]:
                f.write(f"- **{char}**: {count} mentions\n")
            f.write("\n")
            
            # Pain Points
            f.write("## Community Pain Points\n\n")
            for category, examples in pain_points.items():
                f.write(f"### {category.title()}\n\n")
                for example in examples[:10]:
                    f.write(f"- {example}\n")
                f.write("\n")
            
            # Trending Topics
            f.write("## Trending Communities (by engagement)\n\n")
            for topic in trending[:15]:
                f.write(f"### r/{topic['subreddit']}\n")
                f.write(f"- Posts: {topic['post_count']}\n")
                f.write(f"- Avg Score: {topic['avg_score']:.1f}\n")
                f.write(f"- Avg Comments: {topic['avg_comments']:.1f}\n")
                f.write(f"- Engagement Score: {topic['engagement_score']:.0f}\n\n")
            
            # Price Sensitivity
            f.write("## Price Point Mentions\n\n")
            for range_name, count in price_sensitivity.items():
                f.write(f"- **{range_name.replace('_', ' ').title()}**: {count} mentions\n")
            f.write("\n")
            
            # AI Analysis
            if ai_analysis:
                f.write("## AI-Powered Demand Insights\n\n")
                f.write(ai_analysis)
                f.write("\n\n")
            
            # Top Posts
            f.write("## High-Engagement Posts\n\n")
            top_posts = reddit_df.nlargest(20, 'score')
            for _, post in top_posts.iterrows():
                f.write(f"### {post['title']}\n")
                f.write(f"- r/{post['subreddit']} | Score: {post['score']} | Comments: {post['num_comments']}\n")
                f.write(f"- [View Post]({post['url']})\n\n")
        
        print(f"\n✅ Report saved to: {filepath}")
        return filepath
    
    def export_data(self, reddit_df: pd.DataFrame) -> str:
        """Export raw Reddit data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_raw_data_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        reddit_df.to_csv(filepath, index=False)
        print(f"✅ Raw data exported to: {filepath}")
        return filepath


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Reddit Cosplay Demand Analyzer")
    print("=" * 60)
    
    # Load configurations
    try:
        reddit_config = RedditConfig()
        search_config = SearchConfig()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nTo get Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'create another app...'")
        print("3. Choose 'script' type")
        print("4. Copy client_id and client_secret to reddit_config.ini")
        return
    
    # Initialize scraper
    scraper = RedditScraper(reddit_config, search_config)
    analyzer = DemandAnalyzer(search_config)
    
    # Scrape Reddit for demand signals
    print("\n🔍 Phase 1: Scraping Reddit for demand signals...")
    reddit_df = scraper.get_demand_signals(
        time_filter='year',
        posts_per_sub=100
    )
    
    if reddit_df.empty:
        print("❌ No data collected. Check your Reddit credentials and subreddit access.")
        return
    
    # Analyze data
    print("\n📈 Phase 2: Analyzing demand patterns...")
    character_mentions = analyzer.extract_mentioned_characters(reddit_df)
    pain_points = analyzer.identify_pain_points(reddit_df)
    trending = analyzer.get_trending_topics(reddit_df)
    price_sensitivity = analyzer.analyze_price_sensitivity(reddit_df)
    
    # AI Analysis
    ai_analysis = None
    if OLLAMA_AVAILABLE:
        try:
            print("\n🤖 Phase 3: Running DeepSeek analysis...")
            deepseek = DeepSeekIntegration(search_config)
            ai_analysis = deepseek.analyze_demand(reddit_df)
        except Exception as e:
            print(f"⚠️ AI analysis skipped: {e}")
    
    # Generate report
    print("\n📝 Phase 4: Generating report...")
    report_gen = ReportGenerator()
    report_path = report_gen.generate_report(
        reddit_df,
        character_mentions,
        pain_points,
        trending,
        price_sensitivity,
        ai_analysis
    )
    data_path = report_gen.export_data(reddit_df)
    
    print("\n" + "=" * 60)
    print("✅ Reddit demand analysis complete!")
    print(f"📄 Report: {report_path}")
    print(f"💾 Data: {data_path}")
    print("\nTo analyze this with Etsy data:")
    print("1. Run the Etsy Market Research Tool")
    print("2. Use DeepSeekIntegration.compare_with_etsy_data()")
    print("=" * 60)


if __name__ == "__main__":
    main()




