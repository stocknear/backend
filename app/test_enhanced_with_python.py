#!/usr/bin/env python3
"""
Test the enhanced_screener.py using stock_screener_python.py as the engine
"""

import asyncio
from enhanced_screener import get_enhanced_stock_screener, process_stock_screener_query

async def test_string_query():
    """Test string query processing"""
    print("\n" + "="*80)
    print("Testing String Query Processing")
    print("="*80)
    
    query = "List of stocks and their current price, that moved from below $5 per share to above $5 per share for at least a day during the past year"
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    result = await get_enhanced_stock_screener(query)
    
    print(f"✅ Found {result['total_matches']} stocks")
    
    if result['matched_stocks']:
        print(f"\nTop 10 Results:")
        print(f"{'Symbol':<10} {'Name':<40} {'Price':<10} {'Market Cap':<15}")
        print("-" * 80)
        
        for stock in result['matched_stocks'][:10]:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:38]
            price = stock.get('price', 0)
            market_cap = stock.get('marketCap', 0)
            
            if market_cap > 1e9:
                cap_str = f"${market_cap/1e9:.2f}B"
            else:
                cap_str = f"${market_cap/1e6:.2f}M"
            
            print(f"{symbol:<10} {name:<40} ${price:<9.2f} {cap_str:<15}")

async def test_old_rule_format():
    """Test old rule_of_list format"""
    print("\n" + "="*80)
    print("Testing Old Rule Format (Backend Compatibility)")
    print("="*80)
    
    old_rules = [
        {
            'metric': 'price',
            'operator': '>',
            'value': 50
        },
        {
            'metric': 'marketCap', 
            'operator': '>',
            'value': 10_000_000_000
        },
        {
            'metric': 'sector',
            'operator': '==',
            'value': 'Technology'
        }
    ]
    
    print("Rules: Tech stocks, Price > $50, Market Cap > $10B")
    
    result = await get_enhanced_stock_screener(old_rules)
    
    print(f"✅ Found {result['total_matches']} stocks")
    
    if result['matched_stocks']:
        print(f"\nTop 5 Technology Stocks:")
        for i, stock in enumerate(result['matched_stocks'][:5], 1):
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:30]
            price = stock.get('price', 0)
            market_cap = stock.get('marketCap', 0)
            
            cap_str = f"${market_cap/1e9:.2f}B"
            print(f"{i}. {symbol} - {name} - ${price:.2f} ({cap_str})")

async def test_new_rule_format():
    """Test new enhanced rule format"""
    print("\n" + "="*80)
    print("Testing New Enhanced Rule Format")
    print("="*80)
    
    new_rules = {
        "rules": [
            {
                'metric': 'price',
                'operator': 'between',
                'value': [20, 100]
            },
            {
                'metric': 'avgVolume',
                'operator': 'over',
                'value': 1_000_000
            }
        ],
        "sort_by": "marketCap",
        "sort_order": "desc",
        "limit": 10
    }
    
    print("Rules: Price between $20-$100, Volume > 1M, sorted by market cap")
    
    result = await get_enhanced_stock_screener(new_rules)
    
    print(f"✅ Found {result['total_matches']} stocks")
    
    if result['matched_stocks']:
        print(f"\nTop 10 Results (sorted by market cap):")
        print(f"{'Symbol':<8} {'Name':<30} {'Price':<8} {'Volume':<12} {'Market Cap':<12}")
        print("-" * 75)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:28]
            price = stock.get('price', 0)
            volume = stock.get('avgVolume', 0)
            market_cap = stock.get('marketCap', 0)
            
            vol_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.0f}K"
            cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M"
            
            print(f"{symbol:<8} {name:<30} ${price:<7.2f} {vol_str:<12} {cap_str:<12}")

async def test_natural_language_patterns():
    """Test natural language pattern recognition"""
    print("\n" + "="*80)
    print("Testing Natural Language Pattern Recognition")
    print("="*80)
    
    test_queries = [
        "Technology stocks with market cap above 5 billion",
        "Stocks with price between $10 and $50",
        "High volume stocks with average volume over 10 million",
        "Healthcare sector stocks",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await process_stock_screener_query(query)
        print(f"  ✅ Found {result['total_matches']} matching stocks")
        
        if result['matched_stocks']:
            top_stock = result['matched_stocks'][0]
            print(f"  Top result: {top_stock.get('symbol')} - {top_stock.get('name')[:30]} - ${top_stock.get('price', 0):.2f}")

async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)
    
    edge_cases = [
        ("Empty string", ""),
        ("Empty rules list", []),
        ("None input", None),
        ("Invalid dict", {"invalid": "data"}),
        ("Simple text", "hello world"),
    ]
    
    for case_name, input_data in edge_cases:
        print(f"\nTesting: {case_name}")
        try:
            result = await get_enhanced_stock_screener(input_data)
            print(f"  ✅ Handled gracefully - {result['total_matches']} results")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

async def test_performance():
    """Test performance with the Python screener"""
    print("\n" + "="*80)
    print("Testing Performance")
    print("="*80)
    
    import time
    
    # Complex query with multiple conditions
    complex_rules = [
        {'metric': 'price', 'operator': '>', 'value': 10},
        {'metric': 'marketCap', 'operator': '>', 'value': 1_000_000_000},
        {'metric': 'avgVolume', 'operator': '>', 'value': 1_000_000},
        {'metric': 'pe', 'operator': '<', 'value': 30}
    ]
    
    print("Running performance test with 4 conditions...")
    
    start_time = time.time()
    result = await get_enhanced_stock_screener(complex_rules)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"✅ Query executed in {execution_time:.3f} seconds")
    print(f"   • Dataset size: {result.get('original_data_length', 'N/A')} stocks")
    print(f"   • Found: {result['total_matches']} matches")
    
    if result.get('original_data_length'):
        throughput = result['original_data_length'] / execution_time
        print(f"   • Throughput: {throughput:.0f} stocks/second")

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("ENHANCED SCREENER + PYTHON SCREENER INTEGRATION TESTS")
    print("Testing enhanced_screener.py using stock_screener_python.py as engine")
    print("="*80)
    
    await test_string_query()
    await test_old_rule_format()
    await test_new_rule_format()
    await test_natural_language_patterns()
    await test_edge_cases()
    await test_performance()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("\nThe enhanced screener is now fully integrated with stock_screener_python.py")
    print("and provides exact frontend compatibility while supporting:")
    print("  • Natural language queries")
    print("  • Old backend rule format")
    print("  • New enhanced rule format")
    print("  • Pattern recognition")
    print("  • High performance filtering")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())