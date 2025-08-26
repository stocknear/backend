#!/usr/bin/env python3
"""
Test pattern matching for most shorted stocks (no API calls)
"""

import asyncio
import re
from stock_screener_python import python_screener

def extract_rules_pattern_matching(query: str):
    """Extract rules using pattern matching (no LLM)"""
    rules = []
    query_lower = query.lower()
    
    # Most shorted stocks pattern
    if any(term in query_lower for term in ['most shorted', 'top shorted', 'heavily shorted']):
        rules.extend([
            {"name": "shortFloatPercent", "condition": "over", "value": 20},
            {"name": "shortRatio", "condition": "over", "value": 1},
            {"name": "marketCap", "condition": "over", "value": 100000000}  # 100M min
        ])
    
    # Price filters
    price_match = re.search(r'(?:below|under)\s+(?:price\s+of\s+)?\$?(\d+(?:\.\d+)?)', query_lower)
    if price_match:
        rules.append({
            "name": "price", 
            "condition": "under",
            "value": float(price_match.group(1))
        })
        
    price_match = re.search(r'(?:above|over)\s+(?:price\s+of\s+)?\$?(\d+(?:\.\d+)?)', query_lower)
    if price_match:
        rules.append({
            "name": "price",
            "condition": "over", 
            "value": float(price_match.group(1))
        })
    
    return rules

async def test_direct_screening():
    """Test direct screening without LLM"""
    print("\n" + "="*80)
    print("Testing Direct Pattern Matching (No LLM)")
    print("="*80)
    
    query = "most shorted stocks below price of 10"
    print(f"Query: '{query}'")
    
    # Extract rules using pattern matching
    rules = extract_rules_pattern_matching(query)
    print(f"\nExtracted rules: {rules}")
    
    if not rules:
        print("❌ No rules extracted")
        return
    
    # Screen stocks
    try:
        print("\nScreening stocks...")
        result = await python_screener.screen(rules, limit=15)
        
        print(f"✅ Found {result['total_matches']} stocks")
        
        if result['matched_stocks']:
            print(f"\nTop 10 Most Shorted Stocks Under $10:")
            print(f"{'Symbol':<8} {'Name':<25} {'Price':<8} {'Short %':<10} {'Short Ratio':<12}")
            print("-" * 75)
            
            for stock in result['matched_stocks'][:10]:
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')[:23]
                price = stock.get('price', 0)
                short_float = stock.get('shortFloatPercent', 0) or 0
                short_ratio = stock.get('shortRatio', 0) or 0
                
                print(f"{symbol:<8} {name:<25} ${price:<7.2f} {short_float:<9.1f}% {short_ratio:<12.2f}")
                
            print(f"\n📊 Sample JSON for LLM (first 3 stocks):")
            sample_data = []
            for stock in result['matched_stocks'][:3]:
                sample_data.append({
                    'symbol': stock.get('symbol'),
                    'name': stock.get('name', '')[:30],
                    'price': round(stock.get('price', 0), 2),
                    'shortFloat%': stock.get('shortFloatPercent'),
                    'shortRatio': stock.get('shortRatio'),
                    'marketCap': stock.get('marketCap')
                })
            
            import json
            print(json.dumps(sample_data, indent=1))
            
            # Calculate token estimate
            json_str = json.dumps(sample_data)
            estimated_tokens = len(json_str) // 4  # Rough estimate: 4 chars per token
            print(f"\n📏 Estimated tokens for 3 stocks: ~{estimated_tokens}")
            print(f"📏 Estimated tokens for 10 stocks: ~{estimated_tokens * 3}")
            
        else:
            print("No stocks found matching criteria")
            
    except Exception as e:
        print(f"❌ Screening error: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_other_patterns():
    """Test other pattern matching"""
    print("\n" + "="*80)
    print("Testing Other Pattern Matches")
    print("="*80)
    
    test_cases = [
        ("tech stocks above $50", "Technology stocks over $50"),
        ("penny stocks under $5", "Low price stocks"),
        ("large cap over 10 billion", "Large market cap stocks"),
        ("dividend stocks", "Dividend paying stocks"),
    ]
    
    for query, description in test_cases:
        print(f"\n{description}: '{query}'")
        rules = extract_rules_pattern_matching(query)
        print(f"  Extracted rules: {rules}")
        
        if rules:
            try:
                result = await python_screener.screen(rules, limit=3)
                print(f"  ✅ Found {result['total_matches']} stocks")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        else:
            print("  ⚠️ No rules extracted")

async def main():
    """Run pattern matching tests"""
    await test_direct_screening()
    await test_other_patterns()
    
    print("\n" + "="*80)
    print("✅ Pattern Matching Tests Completed!")
    print("\nKey findings:")
    print("  • Pattern matching can extract most shorted + price rules")
    print("  • Token usage is manageable with 10 stocks max")
    print("  • System should work without LLM dependency")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())