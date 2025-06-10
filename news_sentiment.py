#!/usr/bin/env python3
"""
Enhanced StockNews: 7-napos átlagolt hír-szentiment
• Retry logic with exponential backoff
• Rate limiting with intelligent queueing  
• Comprehensive error handling
• Performance monitoring
"""
import os, json, requests, time, logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import aiohttp
from typing import Optional, Dict, List

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() 

BASE  = Path(__file__).resolve().parent
API   = os.getenv("STOCKNEWS_API_KEY")

# ── Enhanced configuration ──────────────────────────────────────────────────
RATE_LIMIT_CALLS = 5        # calls per second
RATE_LIMIT_PERIOD = 1.0     # seconds
RETRY_LIMIT = 3
RETRY_BACKOFF = 2.0
REQUEST_TIMEOUT = 30
NEG_SENTIMENT_THRESHOLD = -0.05
WEIGHT_CUT_PERCENTAGE = 0.30

class RateLimiter:
    """Smart rate limiter with queue management"""
    def __init__(self, calls_per_second: int = 5):
        self.calls_per_second = calls_per_second
        self.calls = []
    
    async def acquire(self):
        now = time.time()
        # Remove calls older than 1 second
        self.calls = [call_time for call_time in self.calls if now - call_time < 1.0]
        
        if len(self.calls) >= self.calls_per_second:
            sleep_time = 1.0 - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class StockNewsClient:
    """Enhanced StockNews API client with retry logic and rate limiting"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("STOCKNEWS_API_KEY is required")
        
        self.api_key = api_key
        self.rate_limiter = RateLimiter(RATE_LIMIT_CALLS)
        self.session = None
        
        # Statistics tracking
        self.stats = {
            "requests_made": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_retry_time": 0,
            "cache_hits": 0
        }
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, ticker: str) -> str:
        """Generate cache key for ticker sentiment"""
        return f"sentiment_{ticker}_{datetime.now().strftime('%Y%m%d')}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl
    
    async def get_sentiment(self, ticker: str) -> Optional[float]:
        """Get average sentiment for ticker with caching and retry logic"""
        
        # Check cache first
        cache_key = self._get_cache_key(ticker)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for {ticker}")
            return self.cache[cache_key]["sentiment"]
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Make API request with retry logic
        sentiment = await self._fetch_sentiment_with_retry(ticker)
        
        # Cache the result
        if sentiment is not None:
            self.cache[cache_key] = {
                "sentiment": sentiment,
                "timestamp": time.time()
            }
        
        return sentiment
    
    async def _fetch_sentiment_with_retry(self, ticker: str) -> Optional[float]:
        """Fetch sentiment with exponential backoff retry"""
        
        url = "https://stocknewsapi.com/api/v1"
        params = {
            "tickers": ticker,
            "items": 100,
            "date": "last7days",
            "token": self.api_key
        }
        
        for attempt in range(1, RETRY_LIMIT + 1):
            retry_start = time.time()
            
            try:
                self.stats["requests_made"] += 1
                logger.debug(f"Fetching sentiment for {ticker} (attempt {attempt}/{RETRY_LIMIT})")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        sentiment = self._calculate_sentiment(data, ticker)
                        
                        if sentiment is not None:
                            self.stats["requests_successful"] += 1
                            logger.debug(f"✓ {ticker} sentiment: {sentiment:.3f}")
                        else:
                            logger.warning(f"⚠️ {ticker} → No sentiment data available")
                            self.stats["requests_successful"] += 1  # Successful request, no data
                        
                        return sentiment
                    
                    elif response.status == 429:  # Rate limited
                        wait_time = RETRY_BACKOFF * (2 ** (attempt - 1))
                        logger.warning(f"Rate limited for {ticker}, waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        
                    else:
                        logger.warning(f"HTTP {response.status} for {ticker}: {await response.text()}")
                        if attempt == RETRY_LIMIT:
                            self.stats["requests_failed"] += 1
                            return None
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {ticker} (attempt {attempt})")
                if attempt < RETRY_LIMIT:
                    wait_time = RETRY_BACKOFF * attempt
                    await asyncio.sleep(wait_time)
                else:
                    self.stats["requests_failed"] += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                if attempt < RETRY_LIMIT:
                    wait_time = RETRY_BACKOFF * attempt
                    await asyncio.sleep(wait_time)
                else:
                    self.stats["requests_failed"] += 1
                    return None
            
            finally:
                retry_time = time.time() - retry_start
                self.stats["total_retry_time"] += retry_time
        
        self.stats["requests_failed"] += 1
        return None
    
    def _calculate_sentiment(self, data: Dict, ticker: str) -> Optional[float]:
        """Calculate average sentiment from API response"""
        
        articles = data.get("data", [])
        if not articles:
            return None
        
        sentiment_scores = []
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        
        for article in articles:
            sentiment = article.get("sentiment")
            if sentiment in sentiment_map:
                sentiment_scores.append(sentiment_map[sentiment])
        
        if not sentiment_scores:
            return None
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        logger.debug(f"{ticker}: {len(sentiment_scores)} articles, avg sentiment: {avg_sentiment:.3f}")
        
        return avg_sentiment
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        total_requests = self.stats["requests_made"]
        success_rate = (self.stats["requests_successful"] / max(total_requests, 1)) * 100
        
        return {
            **self.stats,
            "success_rate": round(success_rate, 1),
            "avg_retry_time": round(self.stats["total_retry_time"] / max(total_requests, 1), 2)
        }

async def process_portfolio_sentiment():
    """Main function to process portfolio sentiment with enhanced error handling"""
    
    logger.info("🔄 Starting enhanced news sentiment analysis...")
    start_time = time.time()
    
    # Load portfolio
    try:
        portfolio_file = BASE / "outputs/portfolio_latest.json"
        if not portfolio_file.exists():
            logger.error(f"❌ Portfolio file not found: {portfolio_file}")
            return
        
        portfolio_data = json.load(open(portfolio_file))
        portfolio = portfolio_data["table"]
        logger.info(f"📊 Loaded portfolio with {len(portfolio)} assets")
        
    except Exception as e:
        logger.error(f"❌ Failed to load portfolio: {e}")
        return
    
    # Process sentiment data
    results = []
    failed_tickers = []
    
    async with StockNewsClient(API) as client:
        
        for i, row in enumerate(portfolio, 1):
            ticker = row["Asset"].strip().upper()
            logger.info(f"→ Processing {i}/{len(portfolio)}: {ticker}")
            
            try:
                sentiment = await client.get_sentiment(ticker)
                
                if sentiment is not None:
                    results.append({
                        "ticker": ticker,
                        "avg_sent": round(sentiment, 3),
                        "is_negative": sentiment < NEG_SENTIMENT_THRESHOLD,
                        "weight_adjustment": WEIGHT_CUT_PERCENTAGE if sentiment < NEG_SENTIMENT_THRESHOLD else 0
                    })
                    
                    # Log negative sentiment warnings
                    if sentiment < NEG_SENTIMENT_THRESHOLD:
                        logger.warning(f"🔻 {ticker} has negative sentiment: {sentiment:.3f} (will reduce weight by {WEIGHT_CUT_PERCENTAGE*100}%)")
                
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"⚠️ No sentiment data for {ticker}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {ticker}: {e}")
                failed_tickers.append(ticker)
        
        # Get and log statistics
        stats = client.get_stats()
        
        logger.info("=" * 50)
        logger.info("📊 SENTIMENT ANALYSIS SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total processing time: {time.time() - start_time:.1f}s")
        logger.info(f"✅ Successfully processed: {len(results)}/{len(portfolio)}")
        logger.info(f"❌ Failed to process: {len(failed_tickers)}")
        logger.info(f"🔢 API requests made: {stats['requests_made']}")
        logger.info(f"📈 Success rate: {stats['success_rate']}%")
        logger.info(f"💾 Cache hits: {stats['cache_hits']}")
        logger.info(f"⏳ Average retry time: {stats['avg_retry_time']}s")
        
        # Sentiment distribution
        if results:
            negative_count = sum(1 for r in results if r['is_negative'])
            logger.info(f"📉 Negative sentiment tickers: {negative_count}/{len(results)}")
            
            if negative_count > 0:
                negative_tickers = [r['ticker'] for r in results if r['is_negative']]
                logger.info(f"🔻 Tickers with negative sentiment: {', '.join(negative_tickers)}")
        
        if failed_tickers:
            logger.warning(f"⚠️ Failed tickers: {', '.join(failed_tickers)}")
    
    # Save results
    try:
        output_dir = BASE / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "news_sentiment.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save enhanced results with metadata
        enhanced_output = {
            "generated_at": datetime.now().isoformat(),
            "processing_time_seconds": round(time.time() - start_time, 1),
            "statistics": stats,
            "sentiment_data": results,
            "failed_tickers": failed_tickers,
            "threshold_config": {
                "negative_threshold": NEG_SENTIMENT_THRESHOLD,
                "weight_cut_percentage": WEIGHT_CUT_PERCENTAGE
            }
        }
        
        enhanced_file = output_dir / "news_sentiment_detailed.json"
        with open(enhanced_file, "w") as f:
            json.dump(enhanced_output, f, indent=2)
        
        logger.info(f"✅ Sentiment analysis completed!")
        logger.info(f"📁 Results saved to: {output_file}")
        logger.info(f"📁 Detailed results saved to: {enhanced_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def main():
    """Synchronous wrapper for async main function"""
    try:
        asyncio.run(process_portfolio_sentiment())
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
    except Exception as e:
        logger.error(f"💥 Process failed: {e}")
        raise

if __name__ == "__main__":
    main()