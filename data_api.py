"""
Data API Module for cryptocurrency sentiment analysis.
This module provides functionality to fetch Twitter data for sentiment analysis.
"""

import json
import random
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwitterDataAPI:
    """
    A class to fetch Twitter data for sentiment analysis.
    This simulates Twitter API responses when actual API keys are not available or not working.
    """
    
    def __init__(self, api_key=None, api_secret=None, bearer_token=None):
        """
        Initialize the Twitter API client.
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            bearer_token: Twitter Bearer token
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.bearer_token = bearer_token
        self.has_credentials = all([api_key, api_secret, bearer_token])
        logger.info(f"TwitterDataAPI initialized with credentials: {self.has_credentials}")
    
    def search_recent_tweets(self, query, max_results=100):
        """
        Search for recent tweets matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of tweet objects
        """
        logger.info(f"Searching for tweets with query: {query}, max_results: {max_results}")
        
        # If we have actual credentials, we could use the real Twitter API here
        # For now, generate simulated data
        return self._generate_simulated_tweets(query, max_results)
    
    def _generate_simulated_tweets(self, query, count):
        """
        Generate simulated tweets for testing purposes.
        
        Args:
            query: The search query (used to customize tweets)
            count: Number of tweets to generate
            
        Returns:
            List of simulated tweet objects
        """
        # Extract the cryptocurrency name from the query
        crypto_name = query.split()[0]  # First word in query is usually the crypto name
        
        # Simulated sentiment content
        positive_content = [
            f"Just bought some {crypto_name}! To the moon! 🚀",
            f"{crypto_name} looking bullish today. Great technical indicators.",
            f"The future of {crypto_name} is bright. Holding long term.",
            f"Amazing news for {crypto_name} holders! New partnerships incoming.",
            f"{crypto_name} is the future of finance. So undervalued right now."
        ]
        
        negative_content = [
            f"Sold all my {crypto_name}. Too much volatility.",
            f"{crypto_name} looking bearish, might dump soon.",
            f"Not convinced about {crypto_name}'s long term prospects.",
            f"Regulations might hit {crypto_name} hard. Be careful.",
            f"{crypto_name} technology has serious flaws. Avoid for now."
        ]
        
        neutral_content = [
            f"Interesting developments in {crypto_name} lately.",
            f"Anyone following {crypto_name} price action?",
            f"What's your take on {crypto_name} for the next quarter?",
            f"New {crypto_name} update released. Reading the docs now.",
            f"How does {crypto_name} compare to other similar projects?"
        ]
        
        # Generate random tweets
        tweets = []
        now = datetime.now()
        
        for i in range(count):
            # Determine sentiment (weighted toward neutral)
            sentiment_type = random.choices(
                ["positive", "negative", "neutral"], 
                weights=[0.3, 0.2, 0.5], 
                k=1
            )[0]
            
            if sentiment_type == "positive":
                content = random.choice(positive_content)
            elif sentiment_type == "negative":
                content = random.choice(negative_content)
            else:
                content = random.choice(neutral_content)
                
            # Create tweet with random engagement metrics
            tweet = {
                "id": f"{i}_{hash(content)}",
                "text": content,
                "created_at": (now - timedelta(hours=random.randint(0, 72))).isoformat(),
                "public_metrics": {
                    "retweet_count": random.randint(0, 1000),
                    "reply_count": random.randint(0, 100),
                    "like_count": random.randint(0, 2000),
                    "quote_count": random.randint(0, 50)
                },
                "author": {
                    "username": f"crypto_user_{random.randint(1000, 9999)}",
                    "followers_count": random.randint(100, 100000)
                }
            }
            
            tweets.append(tweet)
            
        logger.info(f"Generated {len(tweets)} simulated tweets for query: {query}")
        return tweets

def get_twitter_client(api_key=None, api_secret=None, bearer_token=None):
    """
    Get a configured Twitter client.
    If credentials are provided, use them; otherwise use environment variables.
    
    Returns:
        Configured TwitterDataAPI instance
    """
    # Use provided credentials or try to get from environment
    api_key = api_key or os.environ.get('TWITTER_API_KEY')
    api_secret = api_secret or os.environ.get('TWITTER_API_SECRET')
    bearer_token = bearer_token or os.environ.get('TWITTER_BEARER_TOKEN')
    
    return TwitterDataAPI(api_key, api_secret, bearer_token)