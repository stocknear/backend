import asyncio
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


sys.path.append('.')

from backtesting.backtest_engine import BacktestingEngine


async def testing_strategy(tickers, start_date="2020-01-01", end_date=None, buy_conditions=None, sell_conditions=None, stop_loss=None, profit_taker=None):    
    # Create engine instance
    engine = BacktestingEngine(initial_capital=100000)
    
    # Validate that both conditions are provided
    if not buy_conditions or not sell_conditions:
        print("Error: Both buy_conditions and sell_conditions are required")
        return
    
    # Run custom rules engine
    result = await engine.run(
        tickers=tickers,
        buy_conditions=buy_conditions,
        sell_conditions=sell_conditions,
        start_date=start_date,
        end_date=end_date,
        stop_loss=stop_loss,
        profit_taker=profit_taker
    )
    
    if result.get('success'):
        
        # Display risk management parameters if set
        if stop_loss or profit_taker:
            print(f"\n--- Risk Management ---")
            if stop_loss:
                print(f"Stop Loss: {stop_loss}%")
            if profit_taker:
                print(f"Profit Taker: {profit_taker}%")
            print("")
        
        # Display appropriate results based on single vs multi-ticker
        if 'tickers' in result:
            # Multi-ticker results
            print(f"Tickers: {', '.join(result['tickers'])}")
            print(f"Total Return: {result['total_return']}%")
            print(f"Multi-Ticker Buy & Hold Return: {result.get('multi_ticker_buy_hold_return', 'N/A')}%")
        else:
            # Single ticker results
            print(f"Ticker: {result.get('ticker', 'N/A')}")
            print(f"Total Return: {result['total_return']}%")
            print(f"Buy & Hold Return: {result.get('buy_hold_return', 'N/A')}%")
        
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {result.get('sortino_ratio', 'N/A'):.3f}" if isinstance(result.get('sortino_ratio'), (int, float)) else f"Sortino Ratio: {result.get('sortino_ratio', 'N/A')}")
        print(f"Beta: {result.get('beta', 'N/A'):.3f}" if isinstance(result.get('beta'), (int, float)) else f"Beta: {result.get('beta', 'N/A')}")
        print(f"Alpha: {result.get('alpha', 'N/A'):.3f}" if isinstance(result.get('alpha'), (int, float)) else f"Alpha: {result.get('alpha', 'N/A')}")
        print(f"Final Value: ${result['final_portfolio_value']:,.2f}")
        print(f"Win Rate: {result['win_rate']}%")
        print(f"Total Trades: {result['total_trades']}")
        
        # Show trade history summary
        if 'trade_history' in result:
            trades = result['trade_history']
            print(f"Trade History: {len(trades)} total trades")
            if trades and 'tickers' in result:
                # Multi-ticker trade breakdown
                ticker_trades = {}
                for trade in trades:
                    ticker = trade.get('ticker', 'N/A')
                    if ticker not in ticker_trades:
                        ticker_trades[ticker] = 0
                    ticker_trades[ticker] += 1
                print("Trades per ticker:", ", ".join([f"{k}: {v}" for k, v in ticker_trades.items()]))
    
    if result.get('spy_benchmark'):
        spy_data = result['spy_benchmark']
        print(f"SPY Benchmark Data:")
        print(f"SPY Return: {spy_data['spy_return']}%")
        print(f"SPY Annual Return: {spy_data['spy_annual_return']}%")


    if result.get("plot_data"):
        plot_data = result['plot_data']
        # Create visualization
        create_performance_plot(plot_data, tickers, result.get('strategy_name', 'Strategy'))

    # Display live recommendations if available
    if result.get("live_recommendations"):
        print("\n" + "="*50)
        print("LIVE TRADING RECOMMENDATIONS")
        print("="*50)
        
        for rec in result["live_recommendations"]:
            print(f"\nTicker: {rec['ticker']}")
            print(f"Recommendation: {rec['recommendation']}")
            print(f"Reason: {rec['reason']}")
            if rec['last_price']:
                print(f"Current Price: ${rec['last_price']}")
            print(f"Analysis Date: {rec['date']}")
            if rec.get('signal_date'):
                print(f"Signal Date: {rec['signal_date']}")

    return result




def create_performance_plot(plot_data, tickers, strategy_name):
    """Create enhanced performance plot supporting both single and multi-ticker backtesting"""
    
    try:
        plt.figure(figsize=(15, 8))
        
        # Determine if this is multi-ticker or single ticker
        is_multi_ticker = 'multi_ticker_buy_hold' in plot_data
        ticker_str = ', '.join(tickers) if isinstance(tickers, list) else str(tickers)
        
        # Plot strategy performance
        if 'strategy' in plot_data and plot_data['strategy']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['strategy']]
            returns = [item['return_pct'] for item in plot_data['strategy']]
            plt.plot(dates, returns, label=f'{strategy_name}', linewidth=2.5, color='royalblue')

        # Plot buy-and-hold benchmark (multi-ticker or single ticker)
        if is_multi_ticker and 'multi_ticker_buy_hold' in plot_data and plot_data['multi_ticker_buy_hold']:
            # Multi-ticker buy-and-hold
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['multi_ticker_buy_hold']]
            returns = [item['return_pct'] for item in plot_data['multi_ticker_buy_hold']]
            tickers_list = plot_data.get('tickers', tickers)
            ticker_label = f"Equal-Weight Buy & Hold ({', '.join(tickers_list)})" if len(tickers_list) <= 3 else f"Equal-Weight Buy & Hold ({len(tickers_list)} stocks)"
            plt.plot(dates, returns, label=ticker_label, linewidth=2.5, color='forestgreen')
        elif 'stock_buy_hold' in plot_data and plot_data['stock_buy_hold']:
            # Single ticker buy-and-hold
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['stock_buy_hold']]
            returns = [item['return_pct'] for item in plot_data['stock_buy_hold']]
            plt.plot(dates, returns, label=f'{ticker_str} Buy & Hold', linewidth=2.5, color='forestgreen')

        # Plot SPY benchmark
        if 'spy_benchmark' in plot_data and plot_data['spy_benchmark']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['spy_benchmark']]
            returns = [item['return_pct'] for item in plot_data['spy_benchmark']]
            plt.plot(dates, returns, label='SPY Buy & Hold', linewidth=2.5, color='firebrick')

        # Add horizontal line at 0% return
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Chart aesthetics
        plt.xlabel('Date', fontsize=12, weight='bold')
        plt.ylabel('Return (%)', fontsize=12, weight='bold')
        
        title = f'Backtesting Results'
        
        plt.title(title, fontsize=14, weight='bold', pad=20)
        
        # Enhanced legend with better positioning
        plt.legend(frameon=True, loc='upper left', fontsize=11, 
                  fancybox=True, shadow=True, framealpha=0.9)
        
        # Grid styling
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Format axes
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Enhanced x-axis formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # Y-axis formatting with percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # Add summary statistics as text box
        if 'strategy' in plot_data and plot_data['strategy']:
            final_return = plot_data['strategy'][-1]['return_pct']
            max_return = max(item['return_pct'] for item in plot_data['strategy'])
            min_return = min(item['return_pct'] for item in plot_data['strategy'])
            
            stats_text = f'Final Return: {final_return:.1f}%\nMax Return: {max_return:.1f}%\nMin Return: {min_return:.1f}%'
            plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Save plot with descriptive filename
        filename="plot.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    except Exception as e:
        print(e)


async def run():
    
    # Test ROC (Rate of Change) Strategy
    print("=" * 60)
    print("ROC (RATE OF CHANGE) MOMENTUM STRATEGY TEST")
    print("=" * 60)
    
    roc_data = {
        "tickers": ["AAPL"],
        "start_date": "2023-01-01", 
        "end_date": "2024-12-31",
        "buy_condition": [
            {"name": "roc", "value": 5, "operator": "above"}  # Strong positive momentum
        ],
        "sell_condition": [
            {"name": "roc", "value": -2, "operator": "below"}  # Negative momentum
        ],
        "stop_loss": 3,
        "profit_taker": 8
    }
  
    await testing_strategy(
        roc_data["tickers"],
        start_date=roc_data["start_date"],
        end_date=roc_data["end_date"],
        buy_conditions=roc_data.get("buy_condition", []),
        sell_conditions=roc_data.get("sell_condition", []),
        stop_loss=roc_data.get("stop_loss"),
        profit_taker=roc_data.get("profit_taker")
    )
    
    

async def main():
   await run()

if __name__ == "__main__":
    asyncio.run(main())