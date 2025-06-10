#!/usr/bin/env python3
"""
Test script for quantitative screening functionality
"""

import logging
from pathlib import Path
from quantitative_screening import QuantitativeScreener, screen_portfolio_universe

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_screening():
    """Test basic screening functionality with a small universe"""
    
    print("🧪 Testing Quantitative Screening...")
    print("=" * 50)
    
    # Test universe - mix of large and small caps
    test_tickers = [
        # Large caps (should pass market cap filter)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'JNJ', 'PG', 'WMT', 'V', 'UNH', 'HD',
        
        # Some smaller/problematic tickers (may fail some filters)
        'GME', 'AMC', 'BBBY', 'NKLA', 'SPCE'
    ]
    
    # Initialize screener with slightly relaxed settings for testing
    test_config = {
        'market_cap_min': 5e9,          # $5B instead of $10B for testing
        'liquidity_percentile': 70,      # Top 70% instead of 80%
        'volume_min': 1e6,              # $1M instead of $5M
        'rsi_min': 25,                  # Slightly wider RSI range
        'rsi_max': 75,
        'lookback_days': 100,           # Shorter lookback for faster testing
        'min_trading_days': 50
    }
    
    screener = QuantitativeScreener(config=test_config)
    
    # Run screening
    results = screener.screen_universe(test_tickers)
    
    # Print results
    print(f"\n📊 SCREENING RESULTS:")
    print(f"Input tickers: {len(test_tickers)}")
    print(f"Passed all filters: {len(results['passed_tickers'])}")
    print(f"Success rate: {(len(results['passed_tickers'])/len(test_tickers)*100):.1f}%")
    
    print(f"\n✅ PASSED TICKERS:")
    passed_details = results['detailed_results']
    
    for ticker in results['passed_tickers']:
        details = passed_details[ticker]
        overall_score = details['overall_screening_score']
        fund_score = details['fundamental_score'].get('composite_score', 'N/A')
        tech_score = details['technical_score'].get('composite_score', 'N/A')
        
        print(f"  {ticker}: Overall={overall_score:.1f} (Fund={fund_score}, Tech={tech_score})")
    
    # Show some failed tickers for comparison
    failed_tickers = [t for t in test_tickers if t not in results['passed_tickers']]
    if failed_tickers:
        print(f"\n❌ FAILED TICKERS: {', '.join(failed_tickers[:10])}")
    
    print(f"\n📈 SCREENING STATISTICS:")
    stats = results['screening_statistics']
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return results

def test_integration():
    """Test integration with existing pipeline"""
    
    print("\n🔗 Testing Pipeline Integration...")
    print("=" * 50)
    
    # Test the main integration function
    try:
        results = screen_portfolio_universe()
        
        print(f"✅ Integration test successful!")
        print(f"Results saved to: outputs/quantitative_screening_results.json")
        print(f"Passed tickers: {len(results['passed_tickers'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Quantitative Screening Tests")
    print("=" * 60)
    
    # Test 1: Basic screening functionality
    test_results = test_basic_screening()
    
    # Test 2: Pipeline integration
    integration_success = test_integration()
    
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print(f"✅ Basic screening: {'PASS' if test_results['passed_tickers'] else 'FAIL'}")
    print(f"✅ Pipeline integration: {'PASS' if integration_success else 'FAIL'}")
    
    if test_results['passed_tickers'] and integration_success:
        print("\n🎉 All tests passed! Quantitative screening is ready to use.")
        print("\nNext steps:")
        print("1. Run: python quantitative_screening.py")
        print("2. Then: python run_prompts.py (to test full enhanced pipeline)")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")