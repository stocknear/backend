#!/usr/bin/env python3
"""
Test the Python translation of the frontend stock screener
"""

import asyncio
import json
from stock_screener_python import python_screener, convert_unit_to_value

async def test_basic_filtering():
    """Test basic filtering functionality"""
    print("\n" + "="*80)
    print("Testing Basic Filtering (Price > $10, Market Cap > 10B)")
    print("="*80)
    
    rules = [
        {
            'name': 'price',
            'value': 10,
            'condition': 'over'
        },
        {
            'name': 'marketCap', 
            'value': '10B',
            'condition': 'over'
        }
    ]
    
    result = await python_screener.screen(rules, limit=10)
    
    print(f"✅ Found {result['total_matches']} stocks")
    print(f"Original dataset size: {result['original_data_length']}")
    
    if result['matched_stocks']:
        print(f"\nTop 10 Results:")
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

async def test_between_condition():
    """Test between condition"""
    print("\n" + "="*80)
    print("Testing Between Condition (Price between $50-$100)")
    print("="*80)
    
    rules = [
        {
            'name': 'price',
            'value': [50, 100],
            'condition': 'between'
        }
    ]
    
    result = await python_screener.screen(rules, limit=15)
    
    print(f"✅ Found {result['total_matches']} stocks with price between $50-$100")
    
    if result['matched_stocks']:
        prices = [s.get('price', 0) for s in result['matched_stocks']]
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        
        print(f"\nFirst 10 Results:")
        for i, stock in enumerate(result['matched_stocks'][:10], 1):
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:30]
            price = stock.get('price', 0)
            print(f"{i:2d}. {symbol:<8} {name:<30} ${price:<9.2f}")

async def test_sector_filtering():
    """Test sector filtering"""
    print("\n" + "="*80)
    print("Testing Sector Filtering (Technology)")
    print("="*80)
    
    rules = [
        {
            'name': 'sector',
            'value': 'Technology',
            'condition': 'exactly'
        },
        {
            'name': 'marketCap',
            'value': '1B',
            'condition': 'over'
        }
    ]
    
    result = await python_screener.screen(rules, limit=10)
    
    print(f"✅ Found {result['total_matches']} Technology stocks with market cap > $1B")
    
    if result['matched_stocks']:
        print(f"\nTop 10 Technology Stocks:")
        print(f"{'Symbol':<10} {'Name':<35} {'Price':<10} {'Market Cap':<15}")
        print("-" * 75)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:33]
            price = stock.get('price', 0)
            market_cap = stock.get('marketCap', 0)
            
            if market_cap > 1e9:
                cap_str = f"${market_cap/1e9:.2f}B"
            else:
                cap_str = f"${market_cap/1e6:.2f}M"
            
            print(f"{symbol:<10} {name:<35} ${price:<9.2f} {cap_str:<15}")

async def test_technical_indicators():
    """Test technical indicator filtering"""
    print("\n" + "="*80)
    print("Testing Technical Indicators (RSI > 70, Volume > 1M)")
    print("="*80)
    
    rules = [
        {
            'name': 'rsi',
            'value': 70,
            'condition': 'over'
        },
        {
            'name': 'avgVolume',
            'value': '1M',
            'condition': 'over'
        }
    ]
    
    result = await python_screener.screen(rules, limit=10)
    
    print(f"✅ Found {result['total_matches']} overbought stocks (RSI > 70)")
    
    if result['matched_stocks']:
        print(f"\nTop 10 Overbought Stocks:")
        print(f"{'Symbol':<10} {'Name':<30} {'Price':<10} {'RSI':<8} {'Avg Volume':<12}")
        print("-" * 75)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:28]
            price = stock.get('price', 0)
            rsi = stock.get('rsi', 0)
            volume = stock.get('avgVolume', 0)
            
            if volume > 1e6:
                vol_str = f"{volume/1e6:.1f}M"
            else:
                vol_str = f"{volume/1e3:.0f}K"
            
            print(f"{symbol:<10} {name:<30} ${price:<9.2f} {rsi:<8.2f} {vol_str:<12}")

async def test_unit_conversion():
    """Test unit conversion functionality"""
    print("\n" + "="*80)
    print("Testing Unit Conversion")
    print("="*80)
    
    test_cases = [
        ("10B", 10_000_000_000),
        ("5.5M", 5_500_000),
        ("100K", 100_000),
        ("25%", 25),
        ("15.5", 15.5),
        ("Technology", "Technology"),
        ("any", "any"),
        ([10, 20], [10, 20])
    ]
    
    print("Testing convert_unit_to_value function:")
    for input_val, expected in test_cases:
        result = convert_unit_to_value(input_val)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_val} -> {result} (expected: {expected})")

async def test_complex_query():
    """Test complex multi-condition query"""
    print("\n" + "="*80)
    print("Testing Complex Query (Tech stocks, PE < 30, Price > $20, Volume > 5M)")
    print("="*80)
    
    rules = [
        {
            'name': 'sector',
            'value': 'Technology',
            'condition': 'exactly'
        },
        {
            'name': 'pe',
            'value': 30,
            'condition': 'under'
        },
        {
            'name': 'price',
            'value': 20,
            'condition': 'over'
        },
        {
            'name': 'avgVolume',
            'value': '5M',
            'condition': 'over'
        }
    ]
    
    result = await python_screener.screen(rules, limit=10)
    
    print(f"✅ Found {result['total_matches']} stocks matching all conditions")
    
    if result['matched_stocks']:
        print(f"\nTop Results:")
        print(f"{'Symbol':<8} {'Name':<25} {'Price':<8} {'PE':<8} {'Volume':<10} {'Market Cap':<12}")
        print("-" * 80)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:23]
            price = stock.get('price', 0)
            pe = stock.get('pe', 0)
            volume = stock.get('avgVolume', 0)
            market_cap = stock.get('marketCap', 0)
            
            vol_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.0f}K"
            cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M"
            
            print(f"{symbol:<8} {name:<25} ${price:<7.2f} {pe:<7.1f} {vol_str:<10} {cap_str:<12}")

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("PYTHON STOCK SCREENER TEST SUITE")
    print("Direct translation of frontend filterWorker.ts logic")
    print("="*80)
    
    await test_unit_conversion()
    await test_basic_filtering()
    await test_between_condition()
    await test_sector_filtering()
    await test_technical_indicators()
    await test_complex_query()
    
    print("\n" + "="*80)
    print("All tests completed! ✅")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_all_tests())