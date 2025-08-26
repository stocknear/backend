#!/usr/bin/env python3
"""
Test the "most shorted stocks below price of 10" query
"""

import asyncio
from rule_extractor import extract_screener_rules, format_rules_for_screener
from stock_screener_python import python_screener

async def test_most_shorted_query():
    """Test the exact query that was failing"""
    print("\n" + "="*80)
    print("Testing: '@stockscreener most shorted stocks below price of 10'")
    print("="*80)
    
    query = "most shorted stocks below price of 10"
    
    try:
        # Step 1: Extract rules
        print("Step 1: Extracting rules...")
        extracted_rules = await extract_screener_rules(query)
        print(f"Extracted rules: {extracted_rules}")
        
        # Step 2: Format rules
        print("\nStep 2: Formatting rules for screener...")
        formatted_rules = await format_rules_for_screener(extracted_rules)
        print(f"Formatted rules: {formatted_rules}")
        
        # Step 3: Screen stocks
        print("\nStep 3: Screening stocks...")
        result = await python_screener.screen(formatted_rules, limit=20)
        
        print(f"\n✅ Success! Found {result['total_matches']} stocks")
        
        if result['matched_stocks']:
            print(f"\nTop 10 Most Shorted Stocks Under $10:")
            print(f"{'Symbol':<8} {'Name':<25} {'Price':<8} {'Short %':<10} {'Short Ratio':<12} {'Market Cap':<12}")
            print("-" * 85)
            
            for stock in result['matched_stocks'][:10]:
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')[:23]
                price = stock.get('price', 0)
                short_float = stock.get('shortFloatPercent', 0)
                short_ratio = stock.get('shortRatio', 0)
                market_cap = stock.get('marketCap', 0)
                
                # Format market cap
                if market_cap > 1e9:
                    cap_str = f"${market_cap/1e9:.1f}B"
                else:
                    cap_str = f"${market_cap/1e6:.0f}M"
                
                print(f"{symbol:<8} {name:<25} ${price:<7.2f} {short_float:<9.1f}% {short_ratio:<12.2f} {cap_str:<12}")
            
            # Analysis
            print(f"\n📊 Analysis:")
            prices = [s.get('price', 0) for s in result['matched_stocks'][:10]]
            short_floats = [s.get('shortFloatPercent', 0) for s in result['matched_stocks'][:10] if s.get('shortFloatPercent')]
            
            if prices:
                print(f"  • Average price: ${sum(prices)/len(prices):.2f}")
                print(f"  • Price range: ${min(prices):.2f} - ${max(prices):.2f}")
            
            if short_floats:
                print(f"  • Average short float: {sum(short_floats)/len(short_floats):.1f}%")
                print(f"  • Highest short float: {max(short_floats):.1f}%")
        
        else:
            print("No stocks found matching the criteria")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_other_queries():
    """Test other common screener queries"""
    print("\n" + "="*80)
    print("Testing Other Common Queries")
    print("="*80)
    
    test_queries = [
        "large cap tech stocks",
        "dividend stocks with yield over 5%",
        "penny stocks under $5",
        "high volume stocks over 10M",
        "stocks with RSI above 70"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            extracted_rules = await extract_screener_rules(query)
            formatted_rules = await format_rules_for_screener(extracted_rules)
            result = await python_screener.screen(formatted_rules, limit=5)
            
            print(f"  Rules: {len(formatted_rules)} extracted")
            print(f"  Results: {result['total_matches']} stocks found")
            
            if result['matched_stocks']:
                top = result['matched_stocks'][0]
                print(f"  Top result: {top.get('symbol')} - {top.get('name', '')[:20]} - ${top.get('price', 0):.2f}")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

async def main():
    """Run all tests"""
    print("STOCK SCREENER RULE EXTRACTION TESTS")
    print("Testing the new rule extraction system with complete frontend context")
    
    await test_most_shorted_query()
    await test_other_queries()
    
    print("\n" + "="*80)
    print("✅ Tests completed!")
    print("\nThe system should now properly handle:")
    print('  • "@stockscreener most shorted stocks below price of 10"')
    print("  • Other complex screening queries")
    print("  • Token limit issues (max 20 results, top 10 displayed)")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())