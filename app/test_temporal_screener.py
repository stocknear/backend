#!/usr/bin/env python3
"""
Test temporal screening functionality specifically for the $5 price movement query
"""

import asyncio
from stock_screener_engine import screener_engine, ScreenerRule, TemporalCondition
import json

async def test_price_movement():
    """
    Test the specific query: 
    "List of stocks that moved from below $5 to above $5 for at least a day during the past year"
    """
    print("\n" + "="*80)
    print("Testing Temporal Price Movement Query")
    print("Query: Stocks that moved from below $5 to above $5 in the past year")
    print("="*80)
    
    # Create the temporal condition for price crossing $5
    temporal_condition = TemporalCondition(
        metric='price',
        start_condition={'operator': '<', 'value': 5},
        end_condition={'operator': '>', 'value': 5},
        time_period='past_year',
        duration_days=1  # At least 1 day above $5
    )
    
    # Create rules:
    # 1. Temporal rule for the price movement
    # 2. Optional: Current price filter (we might want stocks still above $5)
    rules = [
        ScreenerRule(
            metric='price',
            operator='temporal',
            value=None,
            rule_type='temporal',
            temporal_condition=temporal_condition
        )
    ]
    
    # You can add this to also require current price above a certain threshold
    # Uncomment if you want only stocks currently trading above $5
    # rules.append(ScreenerRule(
    #     metric='price',
    #     operator='>',
    #     value=5,
    #     rule_type='simple'
    # ))
    
    print("\nSearching for stocks... (this may take a moment)")
    print("Checking historical price data for each stock...")
    
    try:
        # Screen stocks with temporal condition
        result = await screener_engine.screen_stocks(
            rules=rules,
            sort_by='price',
            sort_order='asc',
            limit=50  # Get top 50 matches
        )
        
        print(f"\n✅ Search completed!")
        print(f"Found {result['total_matches']} stocks that moved from below $5 to above $5")
        
        if result['matched_stocks']:
            print(f"\nShowing results (sorted by current price):")
            print(f"{'#':<4} {'Symbol':<10} {'Name':<40} {'Current Price':<15} {'Sector':<20}")
            print("-" * 100)
            
            for i, stock in enumerate(result['matched_stocks'], 1):
                symbol = stock.get('symbol', 'N/A')
                name = stock.get('name', 'N/A')[:38]
                price = stock.get('price', 0)
                sector = stock.get('sector', 'N/A')[:18]
                
                # Highlight if currently above or below $5
                price_indicator = "📈" if price >= 5 else "📉"
                
                print(f"{i:<4} {symbol:<10} {name:<40} ${price:<14.2f} {sector:<20} {price_indicator}")
            
            # Summary statistics
            above_5 = sum(1 for s in result['matched_stocks'] if s.get('price', 0) >= 5)
            below_5 = result['total_matches'] - above_5
            
            print(f"\n📊 Summary Statistics:")
            print(f"  • Total stocks that crossed $5 threshold: {result['total_matches']}")
            print(f"  • Currently trading above $5: {above_5}")
            print(f"  • Currently trading below $5: {below_5}")
            
            # Price distribution
            if result['matched_stocks']:
                prices = [s.get('price', 0) for s in result['matched_stocks']]
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                
                print(f"\n💰 Price Distribution:")
                print(f"  • Average current price: ${avg_price:.2f}")
                print(f"  • Minimum current price: ${min_price:.2f}")
                print(f"  • Maximum current price: ${max_price:.2f}")
        else:
            print("\nNo stocks found matching the criteria.")
            print("This could mean:")
            print("  1. No stocks had the specific price movement pattern")
            print("  2. Historical data might not be available for all stocks")
            
    except Exception as e:
        print(f"\n❌ Error during screening: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_with_current_price_filter():
    """
    Test with additional constraint that current price must be above $10
    """
    print("\n" + "="*80)
    print("Testing with Current Price Filter")
    print("Query: Stocks that moved from below $5 to above $5 AND currently trade above $10")
    print("="*80)
    
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
        ScreenerRule(
            metric='price',
            operator='>',
            value=10,  # Current price must be above $10
            rule_type='simple'
        )
    ]
    
    print("\nSearching for stocks...")
    
    result = await screener_engine.screen_stocks(
        rules=rules,
        sort_by='marketCap',
        sort_order='desc',
        limit=20
    )
    
    print(f"\nFound {result['total_matches']} stocks")
    
    if result['matched_stocks']:
        print(f"\nTop results by market cap:")
        print(f"{'Symbol':<10} {'Name':<35} {'Price':<10} {'Market Cap':<15} {'Sector':<20}")
        print("-" * 90)
        
        for stock in result['matched_stocks']:
            symbol = stock.get('symbol', 'N/A')
            name = stock.get('name', 'N/A')[:33]
            price = stock.get('price', 0)
            market_cap = stock.get('marketCap', 0)
            sector = stock.get('sector', 'N/A')[:18]
            
            # Format market cap
            if market_cap > 1e9:
                cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap > 1e6:
                cap_str = f"${market_cap/1e6:.2f}M"
            else:
                cap_str = f"${market_cap:.0f}"
            
            print(f"{symbol:<10} {name:<35} ${price:<9.2f} {cap_str:<15} {sector:<20}")

async def main():
    """Run all temporal tests"""
    print("\n" + "="*80)
    print("TEMPORAL STOCK SCREENER TEST")
    print("Testing price movement detection capabilities")
    print("="*80)
    
    # Test 1: Basic temporal query
    await test_price_movement()
    
    # Test 2: Temporal with current price filter
    # Uncomment to run this test (it might be slower)
    # await test_with_current_price_filter()
    
    print("\n" + "="*80)
    print("Tests completed!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())