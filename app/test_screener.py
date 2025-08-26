#!/usr/bin/env python3
"""
Test script for the enhanced stock screener
Tests complex temporal queries like the $5 price movement example
"""

import asyncio
import sys
import os
from datetime import datetime
from enhanced_screener import process_stock_screener_query

# Test queries
TEST_QUERIES = [
    {
        "name": "Price Movement Query",
        "query": "List of stocks and their current price, that moved from below $5 per share to above $5 per share for at least a day during the past year",
        "description": "Tests temporal price movement detection"
    },
    {
        "name": "Simple Price Filter",
        "query": "Stocks with current price between $10 and $50 with market cap over 1 billion",
        "description": "Tests simple filtering with multiple conditions"
    },
    {
        "name": "Tech Sector Gainers",
        "query": "Technology sector stocks with 1 year gain over 50% and PE ratio under 30",
        "description": "Tests sector filtering with performance metrics"
    },
    {
        "name": "Volume and Momentum",
        "query": "Stocks with average volume over 5 million and RSI above 70",
        "description": "Tests technical indicators and volume filters"
    },
    {
        "name": "Dividend Aristocrats",
        "query": "Stocks with dividend yield above 3% and payout ratio under 60%",
        "description": "Tests dividend-related metrics"
    }
]

async def test_single_query(query_info):
    """Test a single query and display results"""
    print(f"\n{'='*80}")
    print(f"Test: {query_info['name']}")
    print(f"Description: {query_info['description']}")
    print(f"Query: {query_info['query']}")
    print(f"{'='*80}")
    
    try:
        start_time = datetime.now()
        result = await process_stock_screener_query(query_info['query'])
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Query executed successfully in {execution_time:.2f} seconds")
        print(f"Total matches: {result['total_matches']}")
        
        if result['matched_stocks']:
            print(f"\nTop 10 Results:")
            print(f"{'Symbol':<10} {'Name':<40} {'Price':<10} {'Market Cap':<15}")
            print("-" * 80)
            
            for stock in result['matched_stocks'][:10]:
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')[:38]
                price = stock.get('price', 0)
                market_cap = stock.get('marketCap', 0)
                
                # Format market cap
                if market_cap > 1e12:
                    cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    cap_str = f"${market_cap/1e9:.2f}B"
                elif market_cap > 1e6:
                    cap_str = f"${market_cap/1e6:.2f}M"
                else:
                    cap_str = f"${market_cap:.0f}"
                
                print(f"{symbol:<10} {name:<40} ${price:<9.2f} {cap_str:<15}")
        else:
            print("No stocks matched the criteria")
            
    except Exception as e:
        print(f"\n❌ Error executing query: {str(e)}")
        import traceback
        traceback.print_exc()

async def run_all_tests():
    """Run all test queries"""
    print(f"\n{'='*80}")
    print("ENHANCED STOCK SCREENER TEST SUITE")
    print(f"Testing {len(TEST_QUERIES)} queries")
    print(f"{'='*80}")
    
    for query_info in TEST_QUERIES:
        await test_single_query(query_info)
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")

async def test_specific_query(custom_query):
    """Test a specific custom query"""
    query_info = {
        "name": "Custom Query",
        "query": custom_query,
        "description": "User-provided custom query"
    }
    await test_single_query(query_info)

if __name__ == "__main__":
    # Check if custom query provided
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        asyncio.run(test_specific_query(custom_query))
    else:
        # Run all test queries
        asyncio.run(run_all_tests())