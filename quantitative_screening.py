#!/usr/bin/env python3
"""
Quantitative Pre-screening Layer
Implements the 4-factor screening model from Whitepaper V3:
• Market cap: Minimum $10B
• Liquidity: Top 80% by avg daily volume
• Fundamentals: ROE, PEG, P/E, quick ratio (Z-scored)
• Technical: 50/200 DMA crossover, RSI 30–70 range
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Logging setup ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

class QuantitativeScreener:
    """Quantitative pre-screening implementation following Whitepaper V3"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize screener with configuration"""
        
        # Default thresholds from Whitepaper
        self.config = config or {
            'market_cap_min': 10e9,        # $10B minimum
            'liquidity_percentile': 80,     # Top 80% by volume
            'volume_min': 5e6,              # $5M daily minimum
            'rsi_min': 30,                  # RSI range
            'rsi_max': 70,
            'lookback_days': 252,           # 1 year for calculations
            'min_trading_days': 200         # Minimum trading history
        }
        
        # Fundamental metrics for Z-scoring
        self.fundamental_metrics = ['P/E', 'PEG', 'ROE', 'Quick Ratio', 'Beta']
        
        # Statistics tracking
        self.screening_stats = {
            'total_candidates': 0,
            'market_cap_passed': 0,
            'liquidity_passed': 0,
            'fundamentals_passed': 0,
            'technical_passed': 0,
            'final_passed': 0
        }
    
    def screen_universe(self, candidate_tickers: List[str]) -> Dict:
        """
        Apply complete 4-factor screening to ticker universe
        
        Args:
            candidate_tickers: List of ticker symbols to screen
            
        Returns:
            Dict with screening results and statistics
        """
        logger.info(f"🔍 Starting quantitative screening for {len(candidate_tickers)} candidates")
        
        self.screening_stats['total_candidates'] = len(candidate_tickers)
        
        # Step 1: Market Cap Filter
        logger.info("📊 Step 1: Applying market cap filter...")
        market_cap_passed = self._apply_market_cap_filter(candidate_tickers)
        
        # Step 2: Liquidity Filter  
        logger.info("💧 Step 2: Applying liquidity filter...")
        liquidity_passed = self._apply_liquidity_filter(market_cap_passed)
        
        # Step 3: Fundamentals Filter & Z-scoring
        logger.info("📈 Step 3: Applying fundamentals filter...")
        fundamentals_passed, fundamental_scores = self._apply_fundamentals_filter(liquidity_passed)
        
        # Step 4: Technical Filter
        logger.info("⚡ Step 4: Applying technical filter...")
        final_passed, technical_scores = self._apply_technical_filter(fundamentals_passed)
        
        # Compile results
        results = self._compile_screening_results(
            final_passed, fundamental_scores, technical_scores
        )
        
        # Log statistics
        self._log_screening_statistics()
        
        return results
    
    def _apply_market_cap_filter(self, tickers: List[str]) -> List[str]:
        """Filter by minimum market cap requirement"""
        
        passed_tickers = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                
                if market_cap >= self.config['market_cap_min']:
                    passed_tickers.append(ticker)
                    logger.debug(f"✅ {ticker}: Market cap ${market_cap/1e9:.1f}B - PASS")
                else:
                    failed_tickers.append(ticker)
                    logger.debug(f"❌ {ticker}: Market cap ${market_cap/1e9:.1f}B - FAIL")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error getting market cap for {ticker}: {e}")
                failed_tickers.append(ticker)
        
        self.screening_stats['market_cap_passed'] = len(passed_tickers)
        logger.info(f"📊 Market cap filter: {len(passed_tickers)}/{len(tickers)} passed")
        
        if failed_tickers:
            logger.info(f"❌ Failed market cap: {', '.join(failed_tickers[:10])}")
        
        return passed_tickers
    
    def _apply_liquidity_filter(self, tickers: List[str]) -> List[str]:
        """Filter by liquidity (top 80% by average daily volume)"""
        
        volume_data = {}
        
        # Collect volume data
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                avg_volume = info.get('averageVolume', 0)
                avg_volume_10day = info.get('averageVolume10days', avg_volume)
                
                # Use 10-day average if available, otherwise regular average
                volume_data[ticker] = max(avg_volume_10day, avg_volume)
                
            except Exception as e:
                logger.warning(f"⚠️ Error getting volume for {ticker}: {e}")
                volume_data[ticker] = 0
        
        # Calculate 80th percentile threshold
        volumes = list(volume_data.values())
        if not volumes:
            logger.warning("No volume data collected")
            return []
        
        percentile_80 = np.percentile(volumes, self.config['liquidity_percentile'])
        min_volume = max(percentile_80, self.config['volume_min'])
        
        # Filter tickers
        passed_tickers = []
        for ticker, volume in volume_data.items():
            if volume >= min_volume:
                passed_tickers.append(ticker)
                logger.debug(f"✅ {ticker}: Volume {volume:,.0f} - PASS")
            else:
                logger.debug(f"❌ {ticker}: Volume {volume:,.0f} - FAIL")
        
        self.screening_stats['liquidity_passed'] = len(passed_tickers)
        logger.info(f"💧 Liquidity filter: {len(passed_tickers)}/{len(tickers)} passed (threshold: {min_volume:,.0f})")
        
        return passed_tickers
    
    def _apply_fundamentals_filter(self, tickers: List[str]) -> Tuple[List[str], Dict]:
        """Apply fundamentals filter with Z-scoring"""
        
        fundamentals_data = {}
        
        # Collect fundamental data
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamentals = {
                    'P/E': info.get('trailingPE'),
                    'PEG': info.get('pegRatio'), 
                    'ROE': info.get('returnOnEquity'),
                    'Quick Ratio': info.get('quickRatio'),
                    'Beta': info.get('beta')
                }
                
                # Only include if we have at least 3 out of 5 metrics
                valid_metrics = sum(1 for v in fundamentals.values() if v is not None)
                if valid_metrics >= 3:
                    fundamentals_data[ticker] = fundamentals
                    logger.debug(f"✅ {ticker}: {valid_metrics}/5 fundamental metrics available")
                else:
                    logger.debug(f"❌ {ticker}: Only {valid_metrics}/5 fundamental metrics available")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error getting fundamentals for {ticker}: {e}")
        
        # Calculate Z-scores
        df = pd.DataFrame(fundamentals_data).T
        z_scores = {}
        
        for metric in self.fundamental_metrics:
            if metric in df.columns:
                # Calculate Z-scores (handle outliers)
                series = df[metric].dropna()
                if len(series) > 2:
                    # Remove extreme outliers (>3 std)
                    mean_val = series.mean()
                    std_val = series.std()
                    filtered_series = series[abs(series - mean_val) <= 3 * std_val]
                    
                    # Recalculate Z-scores
                    if len(filtered_series) > 2:
                        final_mean = filtered_series.mean()
                        final_std = filtered_series.std()
                        
                        for ticker in series.index:
                            if ticker not in z_scores:
                                z_scores[ticker] = {}
                            
                            raw_value = series[ticker]
                            z_score = (raw_value - final_mean) / final_std if final_std > 0 else 0
                            z_scores[ticker][metric] = round(z_score, 3)
        
        # Calculate composite fundamental scores
        fundamental_scores = {}
        passed_tickers = []
        
        for ticker in fundamentals_data:
            if ticker in z_scores:
                # Calculate weighted composite score
                ticker_z_scores = z_scores[ticker]
                
                # Weights for different metrics (can be adjusted)
                weights = {
                    'ROE': 0.3,      # Higher is better
                    'Quick Ratio': 0.2,  # Higher is better  
                    'P/E': -0.2,     # Lower is better (negative weight)
                    'PEG': -0.2,     # Lower is better (negative weight)
                    'Beta': -0.1     # Lower is better (less risky)
                }
                
                weighted_score = 0
                total_weight = 0
                
                for metric, weight in weights.items():
                    if metric in ticker_z_scores:
                        weighted_score += weight * ticker_z_scores[metric]
                        total_weight += abs(weight)
                
                # Normalize to 0-100 scale
                if total_weight > 0:
                    normalized_score = 50 + (weighted_score / total_weight) * 20  # ±20 points from 50
                    fundamental_scores[ticker] = {
                        'composite_score': round(max(0, min(100, normalized_score)), 2),
                        'z_scores': ticker_z_scores,
                        'raw_fundamentals': fundamentals_data[ticker]
                    }
                    
                    # Pass if composite score >= 40 (lenient threshold)
                    if fundamental_scores[ticker]['composite_score'] >= 40:
                        passed_tickers.append(ticker)
                        logger.debug(f"✅ {ticker}: Fundamental score {fundamental_scores[ticker]['composite_score']:.1f} - PASS")
                    else:
                        logger.debug(f"❌ {ticker}: Fundamental score {fundamental_scores[ticker]['composite_score']:.1f} - FAIL")
        
        self.screening_stats['fundamentals_passed'] = len(passed_tickers)
        logger.info(f"📈 Fundamentals filter: {len(passed_tickers)}/{len(tickers)} passed")
        
        return passed_tickers, fundamental_scores
    
    def _apply_technical_filter(self, tickers: List[str]) -> Tuple[List[str], Dict]:
        """Apply technical analysis filter (DMA crossover, RSI)"""
        
        technical_scores = {}
        passed_tickers = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['lookback_days'] + 50)
        
        for ticker in tickers:
            try:
                # Get price data
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) < self.config['min_trading_days']:
                    logger.debug(f"❌ {ticker}: Insufficient price history ({len(hist)} days)")
                    continue
                
                # Calculate technical indicators
                tech_score = self._calculate_technical_indicators(hist, ticker)
                
                if tech_score is not None:
                    technical_scores[ticker] = tech_score
                    
                    # Pass if technical score >= 50 (neutral threshold)
                    if tech_score['composite_score'] >= 50:
                        passed_tickers.append(ticker)
                        logger.debug(f"✅ {ticker}: Technical score {tech_score['composite_score']:.1f} - PASS")
                    else:
                        logger.debug(f"❌ {ticker}: Technical score {tech_score['composite_score']:.1f} - FAIL")
                
            except Exception as e:
                logger.warning(f"⚠️ Error calculating technical indicators for {ticker}: {e}")
        
        self.screening_stats['technical_passed'] = len(passed_tickers)
        logger.info(f"⚡ Technical filter: {len(passed_tickers)}/{len(tickers)} passed")
        
        return passed_tickers, technical_scores
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame, ticker: str) -> Optional[Dict]:
        """Calculate technical indicators for a single stock"""
        
        try:
            close_prices = price_data['Close']
            
            # Moving averages
            ma_50 = close_prices.rolling(50).mean()
            ma_200 = close_prices.rolling(200).mean()
            
            # RSI calculation
            rsi = self._calculate_rsi(close_prices)
            
            # Get latest values
            latest_price = close_prices.iloc[-1]
            latest_ma_50 = ma_50.iloc[-1]
            latest_ma_200 = ma_200.iloc[-1]
            latest_rsi = rsi.iloc[-1]
            
            # Technical scoring
            score = 50  # Start neutral
            
            # DMA crossover (50 > 200 is bullish)
            if pd.notna(latest_ma_50) and pd.notna(latest_ma_200):
                if latest_ma_50 > latest_ma_200:
                    score += 15  # Bullish signal
                    dma_signal = "BULLISH"
                else:
                    score -= 10  # Bearish signal
                    dma_signal = "BEARISH"
            else:
                dma_signal = "NEUTRAL"
            
            # Price vs MA50 
            if pd.notna(latest_ma_50):
                price_vs_ma50 = (latest_price - latest_ma_50) / latest_ma_50
                if price_vs_ma50 > 0.02:  # >2% above MA50
                    score += 10
                elif price_vs_ma50 < -0.02:  # >2% below MA50
                    score -= 10
            
            # RSI analysis
            if pd.notna(latest_rsi):
                if self.config['rsi_min'] <= latest_rsi <= self.config['rsi_max']:
                    score += 10  # Good RSI range
                    rsi_signal = "GOOD"
                elif latest_rsi < 30:
                    score += 5   # Oversold (potential buy)
                    rsi_signal = "OVERSOLD"
                elif latest_rsi > 70:
                    score -= 5   # Overbought (potential sell)
                    rsi_signal = "OVERBOUGHT"
                else:
                    rsi_signal = "NEUTRAL"
            else:
                rsi_signal = "NO_DATA"
            
            # Volatility check (penalize excessive volatility)
            returns = close_prices.pct_change().dropna()
            if len(returns) > 30:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                if volatility > 0.5:  # >50% annual volatility
                    score -= 10
            
            # Clamp score to 0-100
            final_score = max(0, min(100, score))
            
            return {
                'composite_score': round(final_score, 2),
                'ma_50': round(latest_ma_50, 2) if pd.notna(latest_ma_50) else None,
                'ma_200': round(latest_ma_200, 2) if pd.notna(latest_ma_200) else None,
                'rsi': round(latest_rsi, 2) if pd.notna(latest_rsi) else None,
                'dma_signal': dma_signal,
                'rsi_signal': rsi_signal,
                'current_price': round(latest_price, 2),
                'volatility': round(volatility, 3) if 'volatility' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compile_screening_results(self, final_tickers: List[str], 
                                 fundamental_scores: Dict, 
                                 technical_scores: Dict) -> Dict:
        """Compile final screening results"""
        
        self.screening_stats['final_passed'] = len(final_tickers)
        
        # Create detailed results for passed tickers
        detailed_results = {}
        
        for ticker in final_tickers:
            detailed_results[ticker] = {
                'passed_all_filters': True,
                'fundamental_score': fundamental_scores.get(ticker, {}),
                'technical_score': technical_scores.get(ticker, {}),
                'screening_timestamp': datetime.now().isoformat()
            }
        
        # Calculate overall screening score (average of fundamental + technical)
        for ticker in detailed_results:
            fund_score = detailed_results[ticker]['fundamental_score'].get('composite_score', 50)
            tech_score = detailed_results[ticker]['technical_score'].get('composite_score', 50)
            
            overall_score = (fund_score + tech_score) / 2
            detailed_results[ticker]['overall_screening_score'] = round(overall_score, 2)
        
        return {
            'passed_tickers': final_tickers,
            'detailed_results': detailed_results,
            'screening_statistics': self.screening_stats.copy(),
            'config_used': self.config.copy(),
            'screening_timestamp': datetime.now().isoformat()
        }
    
    def _log_screening_statistics(self):
        """Log comprehensive screening statistics"""
        
        stats = self.screening_stats
        
        logger.info("=" * 60)
        logger.info("📊 QUANTITATIVE SCREENING RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Total candidates: {stats['total_candidates']}")
        logger.info(f"💰 Market cap filter: {stats['market_cap_passed']} passed")
        logger.info(f"💧 Liquidity filter: {stats['liquidity_passed']} passed")
        logger.info(f"📊 Fundamentals filter: {stats['fundamentals_passed']} passed")
        logger.info(f"⚡ Technical filter: {stats['technical_passed']} passed")
        logger.info(f"✅ Final passed: {stats['final_passed']}")
        
        if stats['total_candidates'] > 0:
            pass_rate = (stats['final_passed'] / stats['total_candidates']) * 100
            logger.info(f"🎯 Overall pass rate: {pass_rate:.1f}%")
        
        logger.info("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# Integration Functions
# ═══════════════════════════════════════════════════════════════════════════

def screen_portfolio_universe(config_file: Path = None) -> Dict:
    """
    Main integration function for portfolio universe screening
    
    Args:
        config_file: Optional path to YAML config file
        
    Returns:
        Dict with screening results
    """
    
    # Load configuration
    if config_file and config_file.exists():
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        
        # Extract all tickers from sectors
        all_tickers = []
        for sector in cfg.get('sectors', []):
            tickers = sector.get('override_tickers', [])
            all_tickers.extend(tickers)
    else:
        # Fallback: use existing firm inputs
        base_path = Path(__file__).parent
        firm_file = base_path / "inputs" / "firm_inputs.json"
        
        if firm_file.exists():
            import json
            with open(firm_file) as f:
                firm_data = json.load(f)
            all_tickers = [f['ticker'] for f in firm_data]
        else:
            # Default universe for testing
            all_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'JPM', 'JNJ', 'PG', 'XOM', 'CVX', 'BA', 'CAT', 'WMT'
            ]
    
    # Remove duplicates and clean
    unique_tickers = list(set(ticker.strip().upper() for ticker in all_tickers))
    
    logger.info(f"🎯 Starting quantitative screening for {len(unique_tickers)} tickers")
    
    # Initialize screener
    screener = QuantitativeScreener()
    
    # Run screening
    results = screener.screen_universe(unique_tickers)
    
    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    screening_file = output_dir / "quantitative_screening_results.json"
    
    import json
    with open(screening_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Screening results saved to {screening_file}")
    
    return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Run screening
    results = screen_portfolio_universe()
    
    # Print summary
    print("\n🎯 SCREENING SUMMARY:")
    print(f"✅ Passed all filters: {len(results['passed_tickers'])}")
    print(f"📊 Success rate: {results['screening_statistics']['final_passed']}/{results['screening_statistics']['total_candidates']}")
    
    if results['passed_tickers']:
        print(f"\n🏆 Top performers:")
        # Sort by overall screening score
        sorted_results = sorted(
            results['detailed_results'].items(),
            key=lambda x: x[1]['overall_screening_score'],
            reverse=True
        )
        
        for ticker, data in sorted_results[:10]:
            score = data['overall_screening_score']
            print(f"  {ticker}: {score:.1f}")