import asyncio
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


sys.path.append('.')

from backtest_engine import BacktestingEngine


async def testing_strategy(ticker="AAPL", strategy="rsi", start_date="2020-01-01"):    
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


    if result.get("plot_data"):
        plot_data = result['plot_data']
        print("\nPlot data available:")
        print(f"- Strategy data points: {len(plot_data.get('strategy', []))}")
        print(f"- Stock buy-hold data points: {len(plot_data.get('stock_buy_hold', []))}")
        print(f"- SPY benchmark data points: {len(plot_data.get('spy_benchmark', []))}")
        
        # Create visualization
        create_performance_plot(plot_data, ticker, result.get('strategy_name', 'Strategy'))


    return result




def create_performance_plot(plot_data, ticker, strategy_name):
    """Create enhanced performance plot comparing strategy, stock buy-hold, and SPY"""
    
    try:
        plt.figure(figsize=(14, 7))
        
        # Plot strategy performance
        if 'strategy' in plot_data and plot_data['strategy']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['strategy']]
            returns = [item['return_pct'] for item in plot_data['strategy']]
            plt.plot(dates, returns, label=f'{strategy_name} Strategy', linewidth=2.5, color='royalblue')

        # Plot stock buy-and-hold
        if 'stock_buy_hold' in plot_data and plot_data['stock_buy_hold']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['stock_buy_hold']]
            returns = [item['return_pct'] for item in plot_data['stock_buy_hold']]
            plt.plot(dates, returns, label=f'{ticker} Buy & Hold', linewidth=2.5, color='forestgreen')

        # Plot SPY benchmark
        if 'spy_benchmark' in plot_data and plot_data['spy_benchmark']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['spy_benchmark']]
            returns = [item['return_pct'] for item in plot_data['spy_benchmark']]
            plt.plot(dates, returns, label='SPY Buy & Hold', linewidth=2.5, color='firebrick')

        # Chart aesthetics
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.title(f'{ticker} Backtesting Results — {strategy_name}', fontsize=14, weight='bold')
        plt.legend(frameon=True, loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Format x-axis with year locator for better spacing
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Save plot
        filename = f'backtest_plot_{ticker}_{strategy_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300)
        print(f"\n✅ Plot saved as {filename}")
        
    except Exception as e:
        print(f"⚠️ Error creating plot: {e}")



async def main():
    """Main test function - demonstrates the improved approach"""
    ticker = 'AAPL'
    start_date = '2020-01-01'
    
    # Run the comprehensive demonstration
    await testing_strategy(ticker, strategy="rsi", start_date=start_date)


if __name__ == "__main__":
    asyncio.run(main())