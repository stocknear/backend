import asyncio
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


sys.path.append('.')

from backtest_engine import BacktestingEngine


async def testing_strategy(tickers, strategy="rsi", start_date="2020-01-01"):    
    # Create engine instance
    engine = BacktestingEngine(initial_capital=100000)
    
    # Run backtest with clean interface
    result = await engine.run(
        tickers=tickers,
        strategy_name=strategy,
        start_date=start_date,
        rsi_buy=30,  # Custom RSI parameters
        rsi_sell=70
    )
    
    if result.get('success'):
        print(f"Success: {result['strategy_name']}")
        
        # Display appropriate results based on single vs multi-ticker
        if 'tickers' in result:
            # Multi-ticker results
            print(f"Tickers: {', '.join(result['tickers'])}")
            print(f"Total Return: {result['total_return']}%")
            print(f"Multi-Ticker B&H Return: {result.get('multi_ticker_buy_hold_return', 'N/A')}%")
            if 'multi_ticker_buy_hold_return' in result:
                excess = result['total_return'] - result['multi_ticker_buy_hold_return']
                print(f"Excess Return vs Equal-Weight B&H: {excess:+.2f}%")
        else:
            # Single ticker results
            print(f"Ticker: {result.get('ticker', 'N/A')}")
            print(f"Total Return: {result['total_return']}%")
            print(f"Buy & Hold Return: {result.get('buy_hold_return', 'N/A')}%")
            if 'buy_hold_return' in result:
                excess = result['total_return'] - result['buy_hold_return']
                print(f"Excess Return vs B&H: {excess:+.2f}%")
        
        print(f"Sharpe Ratio: {result['sharpe_ratio']}")
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
        print(f"SPY Period: {spy_data['spy_period']}")
        print(f"SPY Data Points: {spy_data['spy_data_points']}")
        print(f"Strategy vs SPY: {spy_data['vs_spy']}%")


    if result.get("plot_data"):
        plot_data = result['plot_data']
        print("\nPlot data available:")
        print(f"- Strategy data points: {len(plot_data.get('strategy', []))}")
        
        # Check if it's multi-ticker or single ticker
        if 'multi_ticker_buy_hold' in plot_data:
            print(f"- Multi-ticker buy-hold data points: {len(plot_data.get('multi_ticker_buy_hold', []))}")
            print(f"- Tickers: {', '.join(plot_data.get('tickers', []))}")
        elif 'stock_buy_hold' in plot_data:
            print(f"- Stock buy-hold data points: {len(plot_data.get('stock_buy_hold', []))}")
        
        print(f"- SPY benchmark data points: {len(plot_data.get('spy_benchmark', []))}")
        
        # Create visualization
        create_performance_plot(plot_data, tickers, result.get('strategy_name', 'Strategy'))


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
            ticker_label = f"Equal-Weight B&H ({', '.join(tickers_list)})" if len(tickers_list) <= 3 else f"Equal-Weight B&H ({len(tickers_list)} stocks)"
            plt.plot(dates, returns, label=ticker_label, linewidth=2.5, color='forestgreen', linestyle='--')
        elif 'stock_buy_hold' in plot_data and plot_data['stock_buy_hold']:
            # Single ticker buy-and-hold
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['stock_buy_hold']]
            returns = [item['return_pct'] for item in plot_data['stock_buy_hold']]
            plt.plot(dates, returns, label=f'{ticker_str} Buy & Hold', linewidth=2.5, color='forestgreen', linestyle='--')

        # Plot SPY benchmark
        if 'spy_benchmark' in plot_data and plot_data['spy_benchmark']:
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in plot_data['spy_benchmark']]
            returns = [item['return_pct'] for item in plot_data['spy_benchmark']]
            plt.plot(dates, returns, label='SPY Buy & Hold', linewidth=2.5, color='firebrick', linestyle=':')

        # Add horizontal line at 0% return
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

        # Chart aesthetics
        plt.xlabel('Date', fontsize=12, weight='bold')
        plt.ylabel('Return (%)', fontsize=12, weight='bold')
        
        # Dynamic title based on ticker type
        if is_multi_ticker:
            title = f'Multi-Ticker Backtesting Results — {strategy_name}'
            subtitle = f'Tickers: {ticker_str}' if len(ticker_str) < 50 else f'{len(plot_data.get("tickers", []))} Tickers'
        else:
            title = f'{ticker_str} Backtesting Results — {strategy_name}'
            subtitle = 'Single Ticker Analysis'
        
        plt.title(title, fontsize=14, weight='bold', pad=20)
        plt.text(0.5, 0.98, subtitle, transform=plt.gca().transAxes, 
                ha='center', va='top', fontsize=10, style='italic', alpha=0.7)
        
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
        if is_multi_ticker:
            filename = f'multi_ticker_backtest_{len(plot_data.get("tickers", []))}stocks_{strategy_name.replace(" ", "_")}.png'
        else:
            clean_ticker = ticker_str.replace(' ', '').replace(',', '_')
            filename = f'backtest_plot_{clean_ticker}_{strategy_name.replace(" ", "_")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Also display performance summary
        print(f"Performance Summary:")
        if 'strategy' in plot_data and plot_data['strategy']:
            strategy_final = plot_data['strategy'][-1]['return_pct']
            print(f"   Strategy Final Return: {strategy_final:.2f}%")
        
        if is_multi_ticker and 'multi_ticker_buy_hold' in plot_data and plot_data['multi_ticker_buy_hold']:
            bh_final = plot_data['multi_ticker_buy_hold'][-1]['return_pct']
            print(f"   Multi-Ticker B&H Return: {bh_final:.2f}%")
            if 'strategy' in plot_data:
                excess = strategy_final - bh_final
                print(f"   Excess Return: {excess:+.2f}%")
        elif 'stock_buy_hold' in plot_data and plot_data['stock_buy_hold']:
            bh_final = plot_data['stock_buy_hold'][-1]['return_pct']
            print(f"   Stock B&H Return: {bh_final:.2f}%")
            if 'strategy' in plot_data:
                excess = strategy_final - bh_final
                print(f"   Excess Return: {excess:+.2f}%")
        
        if 'spy_benchmark' in plot_data and plot_data['spy_benchmark']:
            spy_final = plot_data['spy_benchmark'][-1]['return_pct']
            print(f"   SPY Benchmark Return: {spy_final:.2f}%")
            if 'strategy' in plot_data:
                vs_spy = strategy_final - spy_final
                print(f"   Strategy vs SPY: {vs_spy:+.2f}%")
        
    except Exception as e:
        print(e)


async def main():

    multi_tickers = ['AAPL']
    start_date = '2020-01-01'
    
    await testing_strategy(multi_tickers, strategy="rsi", start_date=start_date)


if __name__ == "__main__":
    asyncio.run(main())