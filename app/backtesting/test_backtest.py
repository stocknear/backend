import asyncio
import sys
from pathlib import Path

sys.path.append('.')

from backtest_engine import BacktestingEngine


async def testing_strategy(ticker="AAPL", strategy="rsi", start_date="2024-01-01"):    
    # Create engine instance
    engine = BacktestingEngine(initial_capital=100000)
    
    # Run backtest with clean interface
    result = await engine.run(
        ticker=ticker,
        strategy_name=strategy,
        start_date=start_date,
        rsi_buy=30,  # Custom RSI parameters
        rsi_sell=70
    )
    
    if result.get('success'):
        print(f"Success: {result['strategy_name']}")
        print(f"Total Return: {result['total_return']}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']}")
        print(f"Final Value: ${result['final_portfolio_value']:,.2f}")
        print(f"Win Rate: {result['win_rate']}%")
    
    if result.get('spy_benchmark'):
        spy_data = result['spy_benchmark']
        print(f"SPY Benchmark Data:")
        print(f"SPY Return: {spy_data['spy_return']}%")
        print(f"SPY Annual Return: {spy_data['spy_annual_return']}%")
        print(f"SPY Period: {spy_data['spy_period']}")
        print(f"SPY Data Points: {spy_data['spy_data_points']}")
        print(f"Strategy vs SPY: {spy_data['vs_spy']}%")

    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result



async def main():
    """Main test function - demonstrates the improved approach"""
    ticker = 'AAPL'
    start_date = '2020-01-01'
    
    # Run the comprehensive demonstration
    await testing_strategy(ticker, strategy="rsi", start_date=start_date)


if __name__ == "__main__":
    asyncio.run(main())