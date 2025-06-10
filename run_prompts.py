#!/usr/bin/env python3
"""
Enhanced Aszinkron batch futtató:
• Sector-score  (11 rekord)    – async, timeout+retry, logging
• Firm-score + SHAP (33 rekord)– async, timeout+retry, max 8 párhuzamos
• Structured logging és performance monitoring
"""

import os, json, re, asyncio, time, logging
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from datetime import datetime

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gpt_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
load_dotenv(override=True)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL","gpt-4o")

# ── Enhanced configuration ──────────────────────────────────────────────────
MAX_CONCURRENCY = 8        # Increased from 2
REQ_TIMEOUT     = 90       # Increased from 60
RETRY_LIMIT     = 5        # Increased from 4
RETRY_BACKOFF   = 5        # Reduced from 8
MAX_RETRIES_PER_MINUTE = 20

# ── Performance tracking ────────────────────────────────────────────────────
class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.requests_made = 0
        self.requests_failed = 0
        self.total_retry_time = 0
        self.sector_times = []
        self.firm_times = []
    
    def log_request(self, success=True, retry_time=0):
        self.requests_made += 1
        if not success:
            self.requests_failed += 1
        self.total_retry_time += retry_time
    
    def log_sector_time(self, duration):
        self.sector_times.append(duration)
    
    def log_firm_time(self, duration):
        self.firm_times.append(duration)
    
    def get_stats(self):
        total_time = time.time() - self.start_time
        return {
            "total_time": round(total_time, 1),
            "requests_made": self.requests_made,
            "requests_failed": self.requests_failed,
            "success_rate": round((self.requests_made - self.requests_failed) / max(self.requests_made, 1) * 100, 1),
            "avg_sector_time": round(sum(self.sector_times) / max(len(self.sector_times), 1), 2) if self.sector_times else 0,
            "avg_firm_time": round(sum(self.firm_times) / max(len(self.firm_times), 1), 2) if self.firm_times else 0,
            "total_retry_time": round(self.total_retry_time, 1)
        }

tracker = PerformanceTracker()

# ── Enhanced GPT hívás retry-val, timeout-tal ──────────────────────────────
async def gpt_call(prompt, temperature=0, request_type="unknown"):
    """Enhanced GPT call with better error handling and logging"""
    start_time = time.time()
    
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            logger.debug(f"Making GPT call (attempt {attempt}/{RETRY_LIMIT}) for {request_type}")
            
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=700,
                timeout=REQ_TIMEOUT
            )
            
            duration = time.time() - start_time
            tracker.log_request(success=True)
            
            logger.debug(f"GPT call successful for {request_type} in {duration:.2f}s")
            return resp.choices[0].message.content
            
        except (APITimeoutError, RateLimitError) as e:
            wait = RETRY_BACKOFF * attempt
            retry_start = time.time()
            
            logger.warning(f"⚠️  Retry {attempt}/{RETRY_LIMIT} for {request_type} in {wait}s – {e}")
            await asyncio.sleep(wait)
            
            retry_time = time.time() - retry_start
            tracker.log_request(success=False, retry_time=retry_time)
            
        except Exception as e:
            logger.error(f"Unexpected error in GPT call for {request_type}: {e}")
            tracker.log_request(success=False)
            if attempt == RETRY_LIMIT:
                raise
            await asyncio.sleep(RETRY_BACKOFF)
    
    raise RuntimeError(f"GPT call failed after {RETRY_LIMIT} retries for {request_type}")

# ════════════════════════════════════════════════════════════════════════════
# 1) Enhanced Sector batch (async)                                           -
# ════════════════════════════════════════════════════════════════════════════
async def run_sectors_async():
    """Enhanced sector scoring with performance tracking"""
    logger.info("Starting sector scoring batch...")
    start_time = time.time()
    
    path = BASE/"inputs/sector_inputs.json"
    sectors = json.load(open(path))
    tpl = Template(open(BASE/"prompts/sector_prompt.j2").read())

    async def job(s):
        job_start = time.time()
        sector_name = s['name']
        
        try:
            logger.info(f"→ Processing sector: {sector_name}")
            out = await gpt_call(tpl.render(**s), request_type=f"Sector-{sector_name}")
            
            m = re.search(r"Score:\s*(\d+)", out)
            score = int(m.group(1)) if m else None
            s["sector_score"] = score
            
            job_duration = time.time() - job_start
            tracker.log_sector_time(job_duration)
            
            logger.info(f"✓ {sector_name} score = {score} (took {job_duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to process sector {sector_name}: {e}")
            s["sector_score"] = None

    # Process all sectors concurrently
    await asyncio.gather(*[job(s) for s in sectors], return_exceptions=True)
    
    # Save results
    json.dump(sectors, open(path,"w"), indent=2)
    
    duration = time.time() - start_time
    successful_scores = sum(1 for s in sectors if s.get("sector_score") is not None)
    
    logger.info(f"✅ Sector scoring completed: {successful_scores}/{len(sectors)} successful in {duration:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# 2) Enhanced Firm batch (async)                                             -
# ════════════════════════════════════════════════════════════════════════════
FIRM_W = {"P/E":0.2,"PEG":-0.1,"Beta":-0.1,"ROE":0.4,"Quick Ratio":0.3}
firm_tpl = Template(open(BASE/"prompts/firm_prompt.j2").read())

async def run_firms_async():
    """Enhanced firm scoring with performance tracking and smart skipping"""
    logger.info("Starting firm scoring batch...")
    start_time = time.time()
    
    path = BASE/"inputs/firm_inputs.json"
    firms = json.load(open(path))
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def job(f):
        job_start = time.time()
        ticker = f['ticker']
        
        # Smart skipping with logging
        if f.get("firm_score") and f.get("firm_shap"):
            logger.debug(f"⌛ Skipping {ticker} – already processed")
            return
        
        try:
            async with sem:
                logger.info(f"→ Processing firm: {ticker}")
                out = await gpt_call(firm_tpl.render(**f), request_type=f"Firm-{ticker}")
                
                # Extract score
                m = re.search(r"Score:\s*(\d+)", out)
                score = int(m.group(1)) if m else None
                f["firm_score"] = score
                
                # Calculate SHAP values
                fin = f["firm_financials_json"]
                f["firm_shap"] = {k: round(FIRM_W[k]*fin.get(k,0),2) for k in FIRM_W}
                
                job_duration = time.time() - job_start
                tracker.log_firm_time(job_duration)
                
                logger.info(f"✓ {ticker} score = {score} (took {job_duration:.2f}s)")
                
        except Exception as e:
            logger.error(f"Failed to process firm {ticker}: {e}")
            f["firm_score"] = None
            f["firm_shap"] = {}

    # Process all firms concurrently
    await asyncio.gather(*[job(f) for f in firms], return_exceptions=True)
    
    # Save results
    json.dump(firms, open(path,"w"), indent=2)
    
    duration = time.time() - start_time
    successful_scores = sum(1 for f in firms if f.get("firm_score") is not None)
    
    logger.info(f"✅ Firm scoring completed: {successful_scores}/{len(firms)} successful in {duration:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# Enhanced Main function with comprehensive logging                           -
# ════════════════════════════════════════════════════════════════════════════
async def main():
    """Enhanced main function with performance tracking and comprehensive logging"""
    logger.info("=" * 60)
    logger.info("🚀 Starting GPT Portfolio Scoring Pipeline")
    logger.info("=" * 60)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found in environment!")
        return
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   Model: {MODEL}")
    logger.info(f"   Max Concurrency: {MAX_CONCURRENCY}")
    logger.info(f"   Request Timeout: {REQ_TIMEOUT}s")
    logger.info(f"   Retry Limit: {RETRY_LIMIT}")
    
    try:
        # Run both scoring phases
        await run_sectors_async()
        await run_firms_async()
        
        # Final performance stats
        stats = tracker.get_stats()
        
        logger.info("=" * 60)
        logger.info("📊 PIPELINE PERFORMANCE SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total execution time: {stats['total_time']} seconds")
        logger.info(f"🔢 Total API requests: {stats['requests_made']}")
        logger.info(f"❌ Failed requests: {stats['requests_failed']}")
        logger.info(f"✅ Success rate: {stats['success_rate']}%")
        logger.info(f"📈 Average sector processing time: {stats['avg_sector_time']}s")
        logger.info(f"🏢 Average firm processing time: {stats['avg_firm_time']}s")
        logger.info(f"⏳ Total retry time: {stats['total_retry_time']}s")
        logger.info("=" * 60)
        
        # Save performance stats
        with open(f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
        raise

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())