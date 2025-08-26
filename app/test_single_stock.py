#!/usr/bin/env python3
"""
Test temporal condition on a single stock to verify functionality
"""

import asyncio
import json
from stock_screener_engine import screener_engine, TemporalCondition
from pathlib import Path

async def check_single_stock_history(symbol: str):
    """Check if a single stock meets the temporal condition"""
    print(f"\n{'='*60}")
    print(f"Checking {symbol} for $5 price crossing in past year")
    print(f"{'='*60}")
    
    # Load historical data directly
    file_path = Path("json/historical-price/one-year") / f"{symbol}.json"
    
    if not file_path.exists():
        print(f"❌ No historical data found for {symbol}")
        return False
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"❌ Empty historical data for {symbol}")
        return False
    
    print(f"✅ Found {len(data)} days of historical data")
    
    # Analyze the data
    below_5_dates = []
    above_5_dates = []
    
    for point in data:
        date = point.get('date', '')
        close = point.get('close', 0)
        
        if close < 5:
            below_5_dates.append((date, close))
        elif close > 5:
            above_5_dates.append((date, close))
    
    print(f"\n📊 Analysis:")
    print(f"  • Days below $5: {len(below_5_dates)}")
    print(f"  • Days above $5: {len(above_5_dates)}")
    
    # Check for crossing pattern
    if below_5_dates and above_5_dates:
        # Sort by date
        below_5_dates.sort()
        above_5_dates.sort()
        
        # Find if there's a below->above transition
        earliest_below = below_5_dates[0]
        
        # Find first date above $5 that comes after a below $5 date
        for above_date, above_price in above_5_dates:
            if above_date > earliest_below[0]:
                print(f"\n✅ FOUND CROSSING!")
                print(f"  • Was below $5 on {earliest_below[0]}: ${earliest_below[1]:.2f}")
                print(f"  • Went above $5 on {above_date}: ${above_price:.2f}")
                
                # Get current price
                if data:
                    current = data[-1]
                    print(f"  • Current price: ${current.get('close', 0):.2f}")
                
                return True
    
    print(f"\n❌ No crossing pattern found")
    return False

async def test_known_stocks():
    """Test with a few known stock symbols"""
    # Test with some popular stocks that might have crossed $5
    test_symbols = ['AAL', 'F', 'AMD', 'SNAP', 'NOK', 'BB', 'PLTR', 'SOFI']
    
    results = []
    for symbol in test_symbols:
        try:
            crossed = await check_single_stock_history(symbol)
            results.append((symbol, crossed))
        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            results.append((symbol, False))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    crossing_stocks = [s for s, c in results if c]
    
    print(f"\nStocks that crossed $5 threshold: {len(crossing_stocks)}/{len(test_symbols)}")
    if crossing_stocks:
        print("Crossing stocks:", ", ".join(crossing_stocks))

async def test_temporal_condition_directly():
    """Test the temporal condition check directly"""
    print(f"\n{'='*60}")
    print("Testing Temporal Condition Method")
    print(f"{'='*60}")
    
    condition = TemporalCondition(
        metric='price',
        start_condition={'operator': '<', 'value': 5},
        end_condition={'operator': '>', 'value': 5},
        time_period='past_year',
        duration_days=1
    )
    
    # Test on a specific symbol
    test_symbol = 'F'  # Ford often trades around $5
    
    print(f"\nChecking {test_symbol} with screener engine...")
    result = await screener_engine.check_temporal_condition(test_symbol, condition)
    
    if result:
        print(f"✅ {test_symbol} meets the temporal condition!")
    else:
        print(f"❌ {test_symbol} does not meet the temporal condition")

if __name__ == "__main__":
    asyncio.run(test_known_stocks())
    asyncio.run(test_temporal_condition_directly())