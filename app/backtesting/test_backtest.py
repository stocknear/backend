#!/usr/bin/env python3
"""
Command line testing script for backtesting tools
"""
import asyncio
import sys
from pathlib import Path

# Add the current directory to path so we can import modules
sys.path.append('.')

from backtesting_tools import run_single_strategy_backtest, run_comprehensive_backtest


async def test_single_strategy(ticker="AAPL", strategy="buy_and_hold", start_date="2024-01-01"):
    """Test a single strategy"""
    print(f"=== Testing {strategy} strategy for {ticker} since {start_date} ===")
    
    result = await run_single_strategy_backtest(
        ticker=ticker,
        strategy_name=strategy,
        start_date=start_date
    )
    
    if result.get('success'):
        print(f"✅ Success! Results:")
        print(f"   Ticker: {result['ticker']}")
        print(f"   Strategy: {result['strategy_name']}")
        print(f"   Period: {result['start_date']} to {result['end_date']}")
        print(f"   Total Return: {result['total_return']}%")
        print(f"   Annual Return: {result['annual_return']}%")
        print(f"   Max Drawdown: {result['max_drawdown']}%")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']}")
        print(f"   Final Value: ${result['final_portfolio_value']:,.2f}")
        print(f"   Total Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']}%")
        print(f"   Summary: {result['summary']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    return result


async def test_all_strategies(ticker="AAPL", start_date="2024-01-01"):
    """Test all strategies comparison"""
    print(f"\n=== Comparing all strategies for {ticker} since {start_date} ===")
    
    result = await run_comprehensive_backtest(
        ticker=ticker,
        start_date=start_date
    )
    
    if 'strategies' in result:
        print(f"✅ Success! Analyzed {len(result['strategies'])} strategies:")
        print(f"   Ticker: {result['ticker']}")
        print(f"   Period: {result['period']}")
        print(f"   Data Points: {result['data_points']}")
        print(f"   Initial Capital: ${result['initial_capital']:,.2f}")
        
        for i, strategy in enumerate(result['strategies']):
            print(f"{i+1}. {strategy['strategy_name']}")
            print(f"   Total Return: {strategy['total_return']}%")
            print(f"   Annual Return: {strategy['annual_return']}%")
            print(f"   Max Drawdown: {strategy['max_drawdown']}%")
            print(f"   Sharpe Ratio: {strategy['sharpe_ratio']}")
            print(f"   Total Trades: {strategy['total_trades']}")
            print(f"   Win Rate: {strategy['win_rate']}%")
            print()
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    return result


async def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_backtest.py single [TICKER] [STRATEGY] [START_DATE]")
        print("  python test_backtest.py all [TICKER] [START_DATE]")
        print()
        print("Examples:")
        print("  python test_backtest.py single AAPL buy_and_hold 2024-01-01")
        print("  python test_backtest.py single TSLA rsi 2023-01-01")
        print("  python test_backtest.py all AAPL 2024-01-01")
        print()
        print("Available strategies: buy_and_hold, rsi, ma_crossover, bollinger, macd")
        return
    
    command = sys.argv[1].lower()
    
    if command == "single":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        strategy = sys.argv[3] if len(sys.argv) > 3 else "buy_and_hold"
        start_date = sys.argv[4] if len(sys.argv) > 4 else "2024-01-01"
        
        await test_single_strategy(ticker, strategy, start_date)
        
    elif command == "all":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        start_date = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
        
        await test_all_strategies(ticker, start_date)
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'single' or 'all'")


if __name__ == "__main__":
    asyncio.run(main())