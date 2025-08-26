#!/usr/bin/env python3
"""
Simple test of the stock screener engine without OpenAI dependency
"""

import asyncio
import sys
from stock_screener_engine import screener_engine, ScreenerRule, TemporalCondition

async def test_simple_filter():
    """Test simple price and market cap filtering"""
    print("\n" + "="*80)
    print("Test 1: Simple Price and Market Cap Filter")
    print("="*80)
    
    rules = [
        ScreenerRule(
            metric='price',
            operator='>',
            value=10,
            rule_type='simple'
        ),
        ScreenerRule(
            metric='marketCap',
            operator='>',
            value=10_000_000_000,  # 10 billion
            rule_type='simple'
        )
    ]
    
    result = await screener_engine.screen_stocks(
        rules=rules,
        sort_by='marketCap',
        sort_order='desc',
        limit=10
    )
    
    print(f"Found {result['total_matches']} stocks")
    print(f"\nTop 10 by Market Cap:")
    print(f"{'Symbol':<10} {'Name':<40} {'Price':<10} {'Market Cap':<15}")
    print("-" * 80)
    
    for stock in result['matched_stocks']:
        symbol = stock.get('symbol', 'N/A')
        name = stock.get('name', 'N/A')[:38]
        price = stock.get('price', 0)
        market_cap = stock.get('marketCap', 0)
        
        # Format market cap
        if market_cap > 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        else:
            cap_str = f"${market_cap/1e6:.2f}M"
        
        print(f"{symbol:<10} {name:<40} ${price:<9.2f} {cap_str:<15}")

async def test_temporal_filter():
    """Test temporal price movement detection"""
    print("\n" + "="*80)
    print("Test 2: Temporal Price Movement ($5 threshold)")
    print("="*80)
    
    # Create temporal condition for price movement
    temporal_condition = TemporalCondition(
        metric='price',
        start_condition={'operator': '<', 'value': 5},
        end_condition={'operator': '>', 'value': 5},
        time_period='past_year',
        duration_days=1
    )
    
    rules = [
        ScreenerRule(
            metric='price',
            operator='temporal',
            value=None,
            rule_type='temporal',
            temporal_condition=temporal_condition
        ),
        # Also require current price above $5
        ScreenerRule(
            metric='price',
            operator='>',
            value=5,
            rule_type='simple'
        )
    ]
    
    print("Searching for stocks that moved from below $5 to above $5 in the past year...")
    print("(This may take a moment as it checks historical data)")
    
    result = await screener_engine.screen_stocks(
        rules=rules,
        sort_by='price',
        sort_order='asc',
        limit=20
    )
    
    print(f"\nFound {result['total_matches']} stocks meeting the criteria")
    
    if result['matched_stocks']:
        print(f"\nTop Results:")
        print(f"{'Symbol':<10} {'Name':<40} {'Current Price':<15}")
        print("-" * 65)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:38]
            price = stock.get('price', 0)
            
            print(f"{symbol:<10} {name:<40} ${price:<14.2f}")

async def test_technical_indicators():
    """Test technical indicator filtering"""
    print("\n" + "="*80)
    print("Test 3: Technical Indicators (RSI > 70)")
    print("="*80)
    
    rules = [
        ScreenerRule(
            metric='rsi',
            operator='>',
            value=70,
            rule_type='simple'
        ),
        ScreenerRule(
            metric='avgVolume',
            operator='>',
            value=1_000_000,
            rule_type='simple'
        )
    ]
    
    result = await screener_engine.screen_stocks(
        rules=rules,
        sort_by='rsi',
        sort_order='desc',
        limit=10
    )
    
    print(f"Found {result['total_matches']} overbought stocks (RSI > 70)")
    
    if result['matched_stocks']:
        print(f"\nTop 10 by RSI:")
        print(f"{'Symbol':<10} {'Name':<30} {'Price':<10} {'RSI':<10} {'Volume':<15}")
        print("-" * 75)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:28]
            price = stock.get('price', 0)
            rsi = stock.get('rsi', 0)
            volume = stock.get('avgVolume', 0)
            
            # Format volume
            if volume > 1e6:
                vol_str = f"{volume/1e6:.1f}M"
            else:
                vol_str = f"{volume/1e3:.0f}K"
            
            print(f"{symbol:<10} {name:<30} ${price:<9.2f} {rsi:<10.2f} {vol_str:<15}")

async def run_all_tests():
    """Run all tests"""
    print("\nSTOCK SCREENER ENGINE TESTS")
    print("Testing core functionality without OpenAI dependency")
    
    await test_simple_filter()
    await test_technical_indicators()
    # Temporal test might be slow, so make it optional
    if '--include-temporal' in sys.argv:
        await test_temporal_filter()
    else:
        print("\n" + "="*80)
        print("Skipping temporal test (use --include-temporal to include)")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(run_all_tests())