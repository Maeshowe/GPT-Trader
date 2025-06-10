#!/usr/bin/env python3
"""
Enhanced Aszinkron batch futtató with Quantitative Pre-screening & Composite Scoring:
• STEP 0: Quantitative screening (4-factor filtering)
• STEP 1: Sector-score (11 rekord) – async, timeout+retry, logging
• STEP 2: Firm-score + SHAP (csak screened firms) – async, timeout+retry, max 8 párhuzamos
• STEP 3: Composite Scoring (Whitepaper V3 formula) – NEW!
• Structured logging és performance monitoring
"""

import os, json, re, asyncio, time, logging
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from datetime import datetime

# Import the new modules
from quantitative_screening import QuantitativeScreener, screen_portfolio_universe
from composite_scoring import run_composite_scoring

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

# ── Enhanced Performance tracking ───────────────────────────────────────────
class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.requests_made = 0
        self.requests_failed = 0
        self.total_retry_time = 0
        self.sector_times = []
        self.firm_times = []
        self.screening_time = 0
        self.composite_scoring_time = 0  # NEW
    
    def log_request(self, success=True, retry_time=0):
        self.requests_made += 1
        if not success:
            self.requests_failed += 1
        self.total_retry_time += retry_time
    
    def log_sector_time(self, duration):
        self.sector_times.append(duration)
    
    def log_firm_time(self, duration):
        self.firm_times.append(duration)
    
    def log_screening_time(self, duration):
        self.screening_time = duration
    
    def log_composite_scoring_time(self, duration):  # NEW
        self.composite_scoring_time = duration
    
    def get_stats(self):
        total_time = time.time() - self.start_time
        return {
            "total_time": round(total_time, 1),
            "screening_time": round(self.screening_time, 1),
            "composite_scoring_time": round(self.composite_scoring_time, 1),  # NEW
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
# 0) Quantitative Pre-screening Step                                         -
# ════════════════════════════════════════════════════════════════════════════
async def run_quantitative_screening():
    """Step 0: Apply quantitative pre-screening to universe"""
    logger.info("🔍 Step 0: Starting quantitative pre-screening...")
    start_time = time.time()
    
    try:
        # Run screening (this will also save results to outputs/)
        screening_results = screen_portfolio_universe()
        
        # Extract passed tickers
        passed_tickers = screening_results.get('passed_tickers', [])
        detailed_results = screening_results.get('detailed_results', {})
        
        duration = time.time() - start_time
        tracker.log_screening_time(duration)
        
        logger.info(f"✅ Quantitative screening completed in {duration:.1f}s")
        logger.info(f"🎯 Passed screening: {len(passed_tickers)} tickers")
        
        # Update firm_inputs.json to only include screened firms
        await update_firm_inputs_with_screening(passed_tickers, detailed_results)
        
        return screening_results
        
    except Exception as e:
        logger.error(f"❌ Quantitative screening failed: {e}")
        # Fall back to original behavior
        return None

async def update_firm_inputs_with_screening(passed_tickers: list, detailed_results: dict):
    """Update firm_inputs.json to only include firms that passed screening"""
    
    firm_path = BASE / "inputs/firm_inputs.json"
    
    if not firm_path.exists():
        logger.warning("firm_inputs.json not found, skipping screening integration")
        return
    
    # Load existing firm inputs
    with open(firm_path) as f:
        firm_data = json.load(f)
    
    # Filter to only include screened firms
    screened_firms = []
    for firm in firm_data:
        ticker = firm['ticker'].upper()
        if ticker in passed_tickers:
            # Add screening results to firm data
            screening_info = detailed_results.get(ticker, {})
            firm['quantitative_screening'] = {
                'passed': True,
                'overall_screening_score': screening_info.get('overall_screening_score', 0),
                'fundamental_score': screening_info.get('fundamental_score', {}).get('composite_score', 0),
                'technical_score': screening_info.get('technical_score', {}).get('composite_score', 0),
                'screening_timestamp': screening_info.get('screening_timestamp')
            }
            screened_firms.append(firm)
            logger.debug(f"✅ {ticker}: Added to screened firms (score: {screening_info.get('overall_screening_score', 0):.1f})")
        else:
            logger.debug(f"❌ {ticker}: Filtered out by quantitative screening")
    
    # Save updated firm inputs
    with open(firm_path, 'w') as f:
        json.dump(screened_firms, f, indent=2)
    
    logger.info(f"📊 Updated firm_inputs.json: {len(screened_firms)}/{len(firm_data)} firms passed screening")

# ════════════════════════════════════════════════════════════════════════════
# 1) Enhanced Sector batch (async)                                           -
# ════════════════════════════════════════════════════════════════════════════
async def run_sectors_async():
    """Step 1: Enhanced sector scoring with performance tracking"""
    logger.info("📊 Step 1: Starting sector scoring batch...")
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
# 2) Enhanced Firm batch (async) - now only processes screened firms         -
# ════════════════════════════════════════════════════════════════════════════
FIRM_W = {"P/E":0.2,"PEG":-0.1,"Beta":-0.1,"ROE":0.4,"Quick Ratio":0.3}
firm_tpl = Template(open(BASE/"prompts/firm_prompt.j2").read())

async def run_firms_async():
    """Step 2: Enhanced firm scoring with performance tracking and quantitative screening integration"""
    logger.info("🏢 Step 2: Starting firm scoring batch (screened firms only)...")
    start_time = time.time()
    
    path = BASE/"inputs/firm_inputs.json"
    firms = json.load(open(path))
    
    # Filter to only firms that passed quantitative screening
    screened_firms = [f for f in firms if f.get('quantitative_screening', {}).get('passed', False)]
    
    if not screened_firms:
        logger.warning("⚠️ No firms passed quantitative screening! Processing all firms as fallback.")
        screened_firms = firms
    else:
        logger.info(f"📊 Processing {len(screened_firms)}/{len(firms)} firms that passed screening")
    
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
                
                # Add screening info to prompt context if available
                screening_info = f.get('quantitative_screening', {})
                if screening_info:
                    f['screening_context'] = f"""
Quantitative Screening Results:
- Overall Score: {screening_info.get('overall_screening_score', 'N/A')}
- Fundamental Score: {screening_info.get('fundamental_score', 'N/A')}
- Technical Score: {screening_info.get('technical_score', 'N/A')}
- Status: PASSED all quantitative filters
"""
                
                out = await gpt_call(firm_tpl.render(**f), request_type=f"Firm-{ticker}")
                
                # Extract score
                m = re.search(r"Score:\s*(\d+)", out)
                score = int(m.group(1)) if m else None
                f["firm_score"] = score
                
                # Enhanced SHAP calculation (integrate screening scores)
                fin = f["firm_financials_json"]
                base_shap = {k: round(FIRM_W[k]*fin.get(k,0),2) for k in FIRM_W}
                
                # Add screening bonus to SHAP
                if screening_info.get('overall_screening_score'):
                    screening_bonus = (screening_info['overall_screening_score'] - 50) / 10  # Scale screening impact
                    base_shap['screening_bonus'] = round(screening_bonus, 2)
                
                f["firm_shap"] = base_shap
                
                job_duration = time.time() - job_start
                tracker.log_firm_time(job_duration)
                
                logger.info(f"✓ {ticker} score = {score} (took {job_duration:.2f}s)")
                
        except Exception as e:
            logger.error(f"Failed to process firm {ticker}: {e}")
            f["firm_score"] = None
            f["firm_shap"] = {}

    # Process only screened firms concurrently
    await asyncio.gather(*[job(f) for f in screened_firms], return_exceptions=True)
    
    # Update the full firms list with processed results
    ticker_to_updated = {f['ticker']: f for f in screened_firms}
    for i, firm in enumerate(firms):
        if firm['ticker'] in ticker_to_updated:
            firms[i] = ticker_to_updated[firm['ticker']]
    
    # Save results
    json.dump(firms, open(path,"w"), indent=2)
    
    duration = time.time() - start_time
    successful_scores = sum(1 for f in screened_firms if f.get("firm_score") is not None)
    
    logger.info(f"✅ Firm scoring completed: {successful_scores}/{len(screened_firms)} successful in {duration:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# 3) NEW: Composite Scoring Integration                                      -
# ════════════════════════════════════════════════════════════════════════════
async def run_composite_scoring_step():
    """Step 3: Apply Whitepaper V3 composite scoring model"""
    logger.info("🔢 Step 3: Starting composite scoring integration...")
    start_time = time.time()
    
    try:
        # Run composite scoring (this is synchronous)
        composite_results = run_composite_scoring()
        
        if 'error' not in composite_results:
            num_scores = len(composite_results.get('composite_scores', {}))
            logger.info(f"✅ Composite scoring completed: {num_scores} scores calculated")
            
            # Log some statistics
            scores = composite_results.get('composite_scores', {})
            if scores:
                avg_improvement = sum(
                    data['composite_score'] - data['original_llm_score'] 
                    for data in scores.values()
                ) / len(scores)
                
                logger.info(f"📊 Average score improvement: {avg_improvement:+.2f} points")
                
                # Show top performers
                sorted_scores = sorted(
                    scores.items(),
                    key=lambda x: x[1]['composite_score'],
                    reverse=True
                )
                
                logger.info("🏆 Top 3 composite scores:")
                for i, (ticker, data) in enumerate(sorted_scores[:3], 1):
                    composite = data['composite_score']
                    original = data['original_llm_score']
                    confidence = data['confidence_level']
                    logger.info(f"   {i}. {ticker}: {composite:.1f} (vs LLM: {original:.1f}, {confidence} confidence)")
                
                # Log component analysis
                missing_components = composite_results.get('scoring_statistics', {}).get('missing_components', {})
                if missing_components:
                    logger.info("⚠️ Missing components analysis:")
                    for component, count in missing_components.items():
                        logger.info(f"   {component}: missing for {count} assets")
        else:
            logger.error(f"❌ Composite scoring failed: {composite_results['error']}")
        
        duration = time.time() - start_time
        tracker.log_composite_scoring_time(duration)
        logger.info(f"🔢 Composite scoring step completed in {duration:.1f}s")
        
        return composite_results
        
    except Exception as e:
        logger.error(f"❌ Composite scoring step failed: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# Enhanced Main function with all V3 features                                -
# ════════════════════════════════════════════════════════════════════════════
async def main():
    """Enhanced main function with quantitative pre-screening, composite scoring and comprehensive logging"""
    logger.info("=" * 60)
    logger.info("🚀 Starting ENHANCED GPT Portfolio Scoring Pipeline V3 with Composite Scoring")
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
    logger.info(f"   🆕 Quantitative Screening: ENABLED")
    logger.info(f"   🆕 Composite Scoring: ENABLED")
    
    try:
        # Step 0: Quantitative Pre-screening
        screening_results = await run_quantitative_screening()
        
        # Step 1: Run sector scoring
        await run_sectors_async()
        
        # Step 2: Run firm scoring (only on screened firms)
        await run_firms_async()
        
        # Step 3: NEW - Composite Scoring Integration
        composite_results = await run_composite_scoring_step()
        
        # Final performance stats
        stats = tracker.get_stats()
        
        logger.info("=" * 60)
        logger.info("📊 ENHANCED PIPELINE V3 PERFORMANCE SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"🔍 Quantitative screening time: {stats['screening_time']} seconds")
        logger.info(f"🔢 Composite scoring time: {stats['composite_scoring_time']} seconds")
        logger.info(f"⏱️  Total execution time: {stats['total_time']} seconds")
        logger.info(f"🔢 Total API requests: {stats['requests_made']}")
        logger.info(f"❌ Failed requests: {stats['requests_failed']}")
        logger.info(f"✅ Success rate: {stats['success_rate']}%")
        logger.info(f"📈 Average sector processing time: {stats['avg_sector_time']}s")
        logger.info(f"🏢 Average firm processing time: {stats['avg_firm_time']}s")
        logger.info(f"⏳ Total retry time: {stats['total_retry_time']}s")
        
        # Screening summary
        if screening_results:
            screening_stats = screening_results.get('screening_statistics', {})
            logger.info(f"🎯 Screening results:")
            logger.info(f"   Total candidates: {screening_stats.get('total_candidates', 0)}")
            logger.info(f"   Final passed: {screening_stats.get('final_passed', 0)}")
            pass_rate = (screening_stats.get('final_passed', 0) / max(screening_stats.get('total_candidates', 1), 1)) * 100
            logger.info(f"   Pass rate: {pass_rate:.1f}%")
        
        # Composite scoring summary
        if composite_results and 'error' not in composite_results:
            composite_stats = composite_results.get('scoring_statistics', {})
            logger.info(f"🔢 Composite scoring results:")
            logger.info(f"   Assets processed: {composite_stats.get('total_processed', 0)}")
            logger.info(f"   Successful scores: {composite_stats.get('successful_scores', 0)}")
            if composite_stats.get('total_processed', 0) > 0:
                comp_success_rate = (composite_stats.get('successful_scores', 0) / composite_stats.get('total_processed', 1)) * 100
                logger.info(f"   Success rate: {comp_success_rate:.1f}%")
            
            # Show final composite scores summary
            composite_scores = composite_results.get('composite_scores', {})
            if composite_scores:
                avg_composite = sum(data['composite_score'] for data in composite_scores.values()) / len(composite_scores)
                avg_llm = sum(data['original_llm_score'] for data in composite_scores.values()) / len(composite_scores)
                overall_improvement = avg_composite - avg_llm
                logger.info(f"📊 Overall performance:")
                logger.info(f"   Average composite score: {avg_composite:.1f}")
                logger.info(f"   Average LLM score: {avg_llm:.1f}")
                logger.info(f"   Overall improvement: {overall_improvement:+.1f} points")
        
        logger.info("=" * 60)
        
        # Save performance stats
        enhanced_stats = stats.copy()
        enhanced_stats['screening_results'] = screening_results.get('screening_statistics', {}) if screening_results else {}
        enhanced_stats['composite_results'] = composite_results.get('scoring_statistics', {}) if composite_results and 'error' not in composite_results else {}
        
        with open(f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(enhanced_stats, f, indent=2)
        
        logger.info("🎉 Enhanced pipeline V3 completed successfully!")
        logger.info("🆕 V3 features active:")
        logger.info("   ✅ Quantitative pre-screening (4-factor filtering)")
        logger.info("   ✅ 🆕 Composite Scoring Model (Whitepaper V3 formula)")
        logger.info("   ✅ Enhanced SHAP with screening integration")
        logger.info("   ✅ Filtered firm processing (efficiency boost)")
        logger.info("   ✅ Comprehensive performance monitoring")
        
    except Exception as e:
        logger.error(f"💥 Enhanced pipeline V3 failed with error: {e}")
        raise

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())