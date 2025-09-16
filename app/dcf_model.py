import os
import json
import traceback
from decimal import Decimal
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_context('paper')


def visualize(dcf_prices, current_share_prices, regress = True):
    """
    2d plot comparing dcf-forecasted per share price with
    where a list of stocks is currently trading

    args:
        dcf_prices: dict of {'ticker': price, ...} for dcf-values
        current_share_prices: dict of {'ticker': price, ...} for (guess)
        regress: regress a line of best fit, because why not

    returns:
        nada
    """
    # TODO: implement
    return NotImplementedError


def visualize_bulk_historicals(dcfs, ticker, condition):
    """
    multiple 2d plot comparing historical DCFS of different growth
    assumption conditions

    args:
        dcfs: list of dcfs of format {'value1', {'year1': dcf}, ...}
        condition: dict of format {'condition': [value1, value2, value3]}

    """
    dcf_share_prices = {}
    variable = list(condition.keys())[0]
    
    #TODO: make this more eloquent for handling the plotting of multiple condition formats
    try:
        conditions = [str(cond) for cond in list(condition.values())[0]]
    except IndexError:
        print(condition)
        conditions = [condition['Ticker']]

    for cond in conditions:
        dcf_share_prices[cond] = {}
        years = dcfs[cond].keys()
        for year in years:
            dcf_share_prices[cond][year] = dcfs[cond][year]['share_price']

    for cond in conditions:
        plt.plot(list(dcf_share_prices[cond].keys())[::-1], 
                 list(dcf_share_prices[cond].values())[::-1], label = cond)

    # Add current stock price as reference
    try:
        quote_data = get_quote_data(ticker)
        current_price = quote_data.get('price', 0)
        if current_price:
            dates = list(dcf_share_prices[list(dcf_share_prices.keys())[0]].keys())[::-1]
            plt.axhline(y=current_price, color='r', linestyle='--', label=f'Current ${ticker} Price: ${current_price:.2f}')
    except:
        pass

    plt.xlabel('Date')
    plt.ylabel('Share price ($)')
    plt.legend(loc = 'upper right')
    plt.title('$' + ticker + '  ')
    
    # Create imgs directory if it doesn't exist
    os.makedirs('imgs', exist_ok=True)
    plt.savefig('imgs/{}_{}.png'.format(ticker, list(condition.keys())[0]))
    print(f"Plot saved to imgs/{ticker}_{list(condition.keys())[0]}.png")
    # plt.show()  # Comment out show for server environments


def visualize_historicals(dcfs):
    """
    2d plot comparing dcf history to share price history
    """
    pass

    dcf_share_prices = {}
    for k, v in dcfs.items():
        dcf_share_prices[dcfs[k]['date']] = dcfs[k]['share_price']

    xs = list(dcf_share_prices.keys())[::-1]
    ys = list(dcf_share_prices.values())[::-1]

    plt.scatter(xs, ys)
    plt.show()
def prettyprint(dcfs, years):
    '''
    Pretty print-out results of a DCF query.
    Handles formatting for all output variatisons.
    '''
    if years > 1:
        for k, v in dcfs.items():
            print('ticker: {}'.format(k))
            if len(dcfs[k].keys()) > 1:
                for yr, dcf in v.items():
                    print('date: {} \
                        \nvalue: {}'.format(yr, dcf))
    else:
        for k, v in dcfs.items():
            print('ticker: {}  \
                  \nvalue: {}'.format(k, v))


def safe_load_json(path):
    """Load JSON data from file safely."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and 'date' in data[0]:
            data = sorted(data, key=lambda x: x["date"], reverse=True)
        return data

def get_quote_data(symbol):
    """Get real-time stock quote data."""
    base_path = Path(__file__).parent
    path = base_path / f"json/quote/{symbol}.json"
    if not path.exists():
        raise FileNotFoundError(f"Quote data not found for {symbol}")
    with open(path, "r") as f:
        return json.load(f)

def get_financial_data(statement_type, symbol, period='annual'):
    """Get financial statement data.
    
    Args:
        statement_type: 'income-statement', 'balance-sheet-statement', or 'cash-flow-statement'
        symbol: Stock ticker symbol
        period: 'annual' or 'quarter'
    """
    base_path = Path(__file__).parent
    path = base_path / f"json/financial-statements/{statement_type}/{period}/{symbol}.json"
    if not path.exists():
        raise FileNotFoundError(f"{statement_type} not found for {symbol}")
    with open(path, "r") as f:
        data = json.load(f)
        return sorted(data, key=lambda x: x["date"], reverse=True)


def DCF(ticker, quote_data, income_statement, balance_statement, cashflow_statement, discount_rate, forecast, earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate):
    """
    a very basic 2-stage DCF implemented for learning purposes.
    see enterprise_value() for details on arguments. 

    args:
        see enterprise value for more info...

    returns:
        dict: {'share price': __, 'enterprise_value': __, 'equity_value': __, 'date': __}
        CURRENT DCF VALUATION. See historical_dcf to fetch a history. 

    """
    enterprise_val = enterprise_value(income_statement,
                                        cashflow_statement,
                                        balance_statement,
                                        forecast, 
                                        discount_rate,
                                        earnings_growth_rate, 
                                        cap_ex_growth_rate, 
                                        perpetual_growth_rate)

    equity_val, share_price = equity_value(enterprise_val,
                                           balance_statement[0],
                                           quote_data)

    print('\nEnterprise Value for {}: ${}.'.format(ticker, '%.2E' % Decimal(str(enterprise_val))), 
              '\nEquity Value for {}: ${}.'.format(ticker, '%.2E' % Decimal(str(equity_val))),
           '\nPer share value for {}: ${}.\n'.format(ticker, '%.2E' % Decimal(str(share_price))),
            )

    return {
        'date': income_statement[0]['date'],       # statement date used
        'enterprise_value': enterprise_val,
        'equity_value': equity_val,
        'share_price': share_price
    }


def historical_DCF(ticker, years, forecast, discount_rate, earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate, interval='annual'):
    """
    Wrap DCF to fetch DCF values over a historical timeframe, denoted period. 

    args:
        same as DCF, except for
        period: number of years to fetch DCF for

    returns:
        {'date': dcf, ..., 'date', dcf}
    """
    dcfs = {}

    try:
        income_statement = get_financial_data('income-statement', ticker, interval)
        balance_statement = get_financial_data('balance-sheet-statement', ticker, interval)
        cashflow_statement = get_financial_data('cash-flow-statement', ticker, interval)
        quote_data = get_quote_data(ticker)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return {}

    if interval == 'quarter':
        intervals = years * 4
    else:
        intervals = years

    for i in range(0, min(intervals, len(income_statement)-1)):
        try:
            dcf = DCF(ticker, 
                    quote_data,
                    income_statement[i:i+2],        # pass year + 1 bc we need change in working capital
                    balance_statement[i:i+2],
                    cashflow_statement[i:i+2],
                    discount_rate,
                    forecast, 
                    earnings_growth_rate,  
                    cap_ex_growth_rate, 
                    perpetual_growth_rate)
        except (Exception, IndexError) as e:
            print(traceback.format_exc())
            print('Interval {} unavailable, no historical statement.'.format(i)) # catch
        else: 
            dcfs[dcf['date']] = dcf 
        print('-'*60)
    
    return dcfs


def ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex):
    """
    Formula to derive unlevered free cash flow to firm. Used in forecasting.

    args:
        ebit: Earnings before interest payments and taxes.
        tax_rate: The tax rate a firm is expected to pay. Usually a company's historical effective rate.
        non_cash_charges: Depreciation and amortization costs. 
        cwc: Annual change in net working capital.
        cap_ex: capital expenditures, or what is spent to maintain zgrowth rate.

    returns:
        unlevered free cash flow
    """
    return ebit * (1-tax_rate) + non_cash_charges + cwc + cap_ex


def get_discount_rate():
    """
    Calculate the Weighted Average Cost of Capital (WACC) for our company.
    Used for consideration of existing capital structure.

    args:
    
    returns:
        W.A.C.C.
    """
    return .1 # TODO: implement 


def equity_value(enterprise_value, balance_sheet, quote_data):
    """
    Given an enterprise value, return the equity value by adjusting for cash/cash equivs. and total debt.

    args:
        enterprise_value: (EV = market cap + total debt - cash), or total value
        balance_sheet: Most recent balance sheet data
        quote_data: Real-time quote data with shares outstanding
    
    returns:
        equity_value: (enterprise value - debt + cash)
        share_price: equity value/shares outstanding
    """
    total_debt = balance_sheet.get('totalDebt', 0) or 0
    cash = balance_sheet.get('cashAndCashEquivalents', 0) or 0
    
    equity_val = enterprise_value - total_debt + cash
    
    shares_outstanding = quote_data.get('sharesOutstanding', 0)
    if shares_outstanding == 0:
        # Use balance sheet data if available
        shares_outstanding = balance_sheet.get('commonStock', 1) / 10  # Rough approximation
    
    share_price = equity_val / shares_outstanding if shares_outstanding > 0 else 0

    return equity_val, share_price


def enterprise_value(income_statement, cashflow_statement, balance_statement, period, discount_rate, earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate):
    """
    Calculate enterprise value by NPV of explicit _period_ free cash flows + NPV of terminal value,
    both discounted by W.A.C.C.

    args:
        ticker: company for forecasting
        period: years into the future
        earnings growth rate: assumed growth rate in earnings, YoY
        cap_ex_growth_rate: assumed growth rate in cap_ex, YoY
        perpetual_growth_rate: assumed growth rate in perpetuity for terminal value, YoY

    returns:
        enterprise value
    """
    # XXX: statements are returned as historical list, 0 most recent
    ebit_value = income_statement[0].get('ebit') or income_statement[0].get('operatingIncome', 0)
    if ebit_value:
        ebit = float(ebit_value)
    else:
        ebit = float(income_statement[0].get('revenue', 0)) * 0.1  # Default to 10% margin if missing
    income_tax = float(income_statement[0].get('incomeTaxExpense', 0) or 0)
    income_before_tax = float(income_statement[0].get('incomeBeforeTax', 1) or 1)
    tax_rate = income_tax / income_before_tax if income_before_tax != 0 else 0.21
    
    non_cash_charges = float(cashflow_statement[0].get('depreciationAndAmortization', 0) or 0)
    
    # Calculate change in working capital
    current_assets_0 = float(balance_statement[0].get('totalCurrentAssets', 0) or 0)
    current_assets_1 = float(balance_statement[1].get('totalCurrentAssets', 0) or 0) if len(balance_statement) > 1 else current_assets_0
    current_liab_0 = float(balance_statement[0].get('totalCurrentLiabilities', 0) or 0)
    current_liab_1 = float(balance_statement[1].get('totalCurrentLiabilities', 0) or 0) if len(balance_statement) > 1 else current_liab_0
    
    cwc = (current_assets_0 - current_liab_0) - (current_assets_1 - current_liab_1)
    
    cap_ex = abs(float(cashflow_statement[0].get('capitalExpenditure', 0) or 
                      cashflow_statement[0].get('investmentsInPropertyPlantAndEquipment', 0) or 0))
    discount = discount_rate

    flows = []

    # Now let's iterate through years to calculate FCF, starting with most recent year
    print('Forecasting flows for {} years out, starting at {}.'.format(period, income_statement[0]['date']),
         ('\n         DFCF   |    EBIT   |    D&A    |    CWC     |   CAP_EX   | '))
    for yr in range(1, period+1):    

        # increment each value by growth rate
        ebit = ebit * (1 + (yr * earnings_growth_rate))
        non_cash_charges = non_cash_charges * (1 + (yr * earnings_growth_rate))
        cwc = cwc * 0.7                             # TODO: evaluate this cwc rate? 0.1 annually?
        cap_ex = cap_ex * (1 + (yr * cap_ex_growth_rate))         

        # discount by WACC
        flow = ulFCF(ebit, tax_rate, non_cash_charges, cwc, cap_ex)
        PV_flow = flow/((1 + discount)**yr)
        flows.append(PV_flow)

        print(str(int(income_statement[0]['date'][0:4]) + yr) + '  ',
              '%.2E' % Decimal(PV_flow) + ' | ',
              '%.2E' % Decimal(ebit) + ' | ',
              '%.2E' % Decimal(non_cash_charges) + ' | ',
              '%.2E' % Decimal(cwc) + ' | ',
              '%.2E' % Decimal(cap_ex) + ' | ')

    NPV_FCF = sum(flows)
    
    # now calculate terminal value using perpetual growth rate
    final_cashflow = flows[-1] * (1 + perpetual_growth_rate)
    TV = final_cashflow/(discount - perpetual_growth_rate)
    NPV_TV = TV/(1+discount)**(1+period)

    return NPV_TV+NPV_FCF



def main(args):
    """
    although the if statements are less than desirable, it allows rapid exploration of 
    historical or present DCF values for either a single or list of tickers.
    """

    if args.s > 0:
        if args.v is not None:
            if args.v in ['eg', 'earnings_growth_rate']:
                cond, dcfs = run_setup(args, variable='eg')
            elif args.v in ['cg', 'cap_ex_growth_rate']:
                cond, dcfs = run_setup(args, variable='cg')
            elif args.v in ['pg', 'perpetual_growth_rate']:
                cond, dcfs = run_setup(args, variable='pg')
            elif args.v in ['discount_rate', 'discount', 'd']:
                cond, dcfs = run_setup(args, variable='d')
            else:
                raise ValueError('args.variable is invalid, must choose from: [earnings_growth_rate, cap_ex_growth_rate, perpetual_growth_rate, discount_rate]')
        else:
            raise ValueError('If step (--s) is > 0, you must specify the variable via --v.')
    else:
        cond, dcfs = {'Ticker': [args.t]}, {}
        dcfs[args.t] = historical_DCF(args.t, args.y, args.p, args.d, args.eg, args.cg, args.pg, args.i)

    if args.y > 1 and dcfs and len(dcfs) > 0: # can't graph single timepoint very well....
        visualize_bulk_historicals(dcfs, args.t, cond)
    else:
        prettyprint(dcfs, args.y)


def run_setup(args, variable):
    dcfs, cond = {}, {args.v: []}
    
    for increment in range(1, int(args.steps) + 1):
        # Calculate the new variable value
        var = vars(args)[variable] * (1 + (args.s * increment))
        step = '{}: {}'.format(args.v, str(var)[0:4])

        cond[args.v].append(step)
        # Temporarily set the variable
        original = vars(args)[variable]
        vars(args)[variable] = var
        dcfs[step] = historical_DCF(args.t, args.y, args.p, args.d, args.eg, args.cg, args.pg, args.i)
        # Restore original value for next iteration
        vars(args)[variable] = original

    return cond, dcfs


def load_historical_prices(symbol):
    """Load historical price data from JSON file."""
    base_path = Path(__file__).parent
    price_file = base_path / f"json/historical-price/adj/{symbol}.json"
    
    if not price_file.exists():
        raise FileNotFoundError(f"Historical price data not found for {symbol}")
    
    with open(price_file, 'r') as f:
        price_data = json.load(f)
    
    # Convert to dict with date as key for easy lookup
    prices_by_date = {}
    for entry in price_data:
        date_str = entry['date']
        prices_by_date[date_str] = entry['adjClose']
    
    return prices_by_date

def get_fiscal_year_end_price(prices_by_date, fiscal_year_end):
    """Get the stock price closest to fiscal year end date."""
    # Try exact match first
    if fiscal_year_end in prices_by_date:
        return prices_by_date[fiscal_year_end]
    
    # Find closest date (within a few days)
    from datetime import datetime
    target_date = datetime.strptime(fiscal_year_end, '%Y-%m-%d')
    best_date = None
    best_diff = float('inf')
    
    for date_str in prices_by_date.keys():
        try:
            price_date = datetime.strptime(date_str, '%Y-%m-%d')
            diff = abs((price_date - target_date).days)
            if diff < best_diff and diff <= 7:  # Within a week
                best_diff = diff
                best_date = date_str
        except ValueError:
            continue
    
    return prices_by_date[best_date] if best_date else None

def plot_historical_dcf_vs_prices(symbol='AAPL', years_to_analyze=10):
    """
    Generate historical DCF analysis and plot against actual stock prices.
    
    Args:
        symbol: Stock ticker symbol
        years_to_analyze: Number of years to analyze
    
    Returns:
        str: Path to the saved plot file
    """
    print(f"Analyzing {symbol} for the past {years_to_analyze} years...")
    
    # Load historical prices
    try:
        prices_by_date = load_historical_prices(symbol)
    except FileNotFoundError as e:
        print(f"Error loading price data: {e}")
        return None
    
    # Get financial data
    try:
        income_statements = get_financial_data('income-statement', symbol, 'annual')
        balance_statements = get_financial_data('balance-sheet-statement', symbol, 'annual')
        cashflow_statements = get_financial_data('cash-flow-statement', symbol, 'annual')
    except FileNotFoundError as e:
        print(f"Error loading financial data: {e}")
        return None
    
    # Prepare data storage
    results = {
        'dates': [],
        'dcf_values': [],
        'actual_prices': [],
        'years': []
    }
    
    # DCF model parameters
    discount_rate = 0.1
    forecast_period = 5
    earnings_growth_rate = 0.05
    cap_ex_growth_rate = 0.045
    perpetual_growth_rate = 0.025
    
    # Analyze each year
    analyzed_years = 0
    for i in range(min(years_to_analyze, len(income_statements) - 1)):
        try:
            # Get financial statements for this year and previous year
            year_income = income_statements[i:i+2]
            year_balance = balance_statements[i:i+2]
            year_cashflow = cashflow_statements[i:i+2]
            
            fiscal_year_end = year_income[0]['date']
            fiscal_year = year_income[0]['fiscalYear']
            
            print(f"Processing {fiscal_year} (ended {fiscal_year_end})...")
            
            # Get actual stock price at fiscal year end
            actual_price = get_fiscal_year_end_price(prices_by_date, fiscal_year_end)
            if actual_price is None:
                print(f"  No price data found for {fiscal_year_end}, skipping...")
                continue
            
            # Get quote data (we'll use current shares outstanding as approximation)
            quote_data = get_quote_data(symbol)
            
            dcf_result = DCF(symbol, quote_data, year_income, year_balance, year_cashflow,
                           discount_rate, forecast_period, earnings_growth_rate,
                           cap_ex_growth_rate, perpetual_growth_rate)
            
            # Store results
            results['dates'].append(fiscal_year_end)
            results['dcf_values'].append(dcf_result['share_price'])
            results['actual_prices'].append(actual_price)
            results['years'].append(fiscal_year)
            
            print(f"  DCF Value: ${dcf_result['share_price']:.2f}, Actual Price: ${actual_price:.2f}")
            analyzed_years += 1
            
        except Exception as e:
            print(f"  Error processing {fiscal_year}: {e}")
            continue
    
    print(f"Successfully analyzed {analyzed_years} years")
    
    if len(results['dates']) == 0:
        print("No data available for plotting")
        return None
    
    # Create the plot
    from datetime import datetime
    plt.figure(figsize=(14, 8))
    
    # Convert dates to datetime for plotting
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in results['dates']]
    
    # Plot DCF values and actual prices
    plt.plot(dates, results['dcf_values'], 'bo-', linewidth=2, markersize=8, 
             label='DCF Intrinsic Value', alpha=0.8)
    plt.plot(dates, results['actual_prices'], 'ro-', linewidth=2, markersize=8,
             label='Actual Stock Price', alpha=0.8)
    
    # Fill area between lines to show over/undervaluation
    plt.fill_between(dates, results['dcf_values'], results['actual_prices'], 
                     where=np.array(results['dcf_values']) > np.array(results['actual_prices']),
                     color='green', alpha=0.2, label='Undervalued')
    plt.fill_between(dates, results['dcf_values'], results['actual_prices'],
                     where=np.array(results['dcf_values']) <= np.array(results['actual_prices']),
                     color='red', alpha=0.2, label='Overvalued')
    
    # Calculate and display accuracy metrics
    dcf_array = np.array(results['dcf_values'])
    actual_array = np.array(results['actual_prices'])
    
    # Percentage difference
    pct_diff = ((dcf_array - actual_array) / actual_array) * 100
    mean_abs_error = np.mean(np.abs(pct_diff))
    
    # Correlation
    correlation = np.corrcoef(dcf_array, actual_array)[0, 1]
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title(f'{symbol} - DCF Valuation vs Actual Price\n'
              f'Mean Absolute Error: {mean_abs_error:.1f}% | Correlation: {correlation:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Add annotations for latest values
    if len(results['dates']) > 0:
        latest_dcf = results['dcf_values'][-1]
        latest_actual = results['actual_prices'][-1]
        latest_date = dates[-1]
        
        plt.annotate(f'DCF: ${latest_dcf:.2f}', 
                     xy=(latest_date, latest_dcf), 
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                     fontsize=9, color='white')
        
        plt.annotate(f'Actual: ${latest_actual:.2f}', 
                     xy=(latest_date, latest_actual), 
                     xytext=(10, -25), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                     fontsize=9, color='white')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('imgs', exist_ok=True)
    
    # Save the plot
    output_file = f'imgs/{symbol}_DCF_Historical_Analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Print summary statistics
    print(f"\n{symbol} DCF Analysis Summary:")
    print("=" * 80)
    
    for i in range(len(results['years'])):
        dcf_val = results['dcf_values'][i]
        actual_val = results['actual_prices'][i]
        diff = dcf_val - actual_val
        diff_pct = (diff / actual_val) * 100
        print(f"{results['years'][i]} {results['dates'][i]}: DCF ${dcf_val:.2f} vs Actual ${actual_val:.2f} ({diff_pct:+.1f}%)")
    
    # Calculate statistics
    print(f"\nStatistics:")
    print(f"Mean Absolute Error: {mean_abs_error:.1f}%")
    print(f"Root Mean Square Error: {np.sqrt(np.mean(pct_diff**2)):.1f}%")
    print(f"Correlation Coefficient: {correlation:.3f}")
    
    # Count over/undervaluation instances
    undervalued = sum(1 for d, a in zip(dcf_array, actual_array) if d >= a)
    overvalued = sum(1 for d, a in zip(dcf_array, actual_array) if d < a)
    
    print(f"Undervalued instances: {undervalued} ({undervalued/len(dcf_array)*100:.1f}%)")
    print(f"Overvalued instances: {overvalued} ({overvalued/len(dcf_array)*100:.1f}%)")
    
    return output_file

if __name__ == '__main__':
    # Create a simple namespace object with hardcoded values
    args = type('Args', (), {})()
    
    # Hardcoded configuration values
    args.t = 'NVDA'  # ticker symbol
    args.p = 5  # years to forecast
    args.y = 5  # number of years to compute DCF analysis for
    args.i = 'annual'  # interval period: "annual" or "quarter"
    args.d = 0.1  # discount rate (WACC)
    args.eg = 0.05  # earnings growth rate
    args.cg = 0.045  # capital expenditure growth rate
    args.pg = 0.025  # perpetual growth rate
    args.s = 0  # step increase for sensitivity analysis (0 = disabled)
    args.steps = 5  # number of steps for sensitivity
    args.v = None  # variable to test: 'eg', 'cg', 'pg', or 'd'
    
    # Example configurations (uncomment to use):
    
    # For sensitivity analysis on earnings growth:
    # args.y = 2
    # args.s = 0.1
    # args.v = 'eg'
    # args.steps = 3
    
    # For historical analysis (3 years):
    # args.y = 3
    
    # For different growth assumptions:
    # args.eg = 0.08
    # args.cg = 0.05
    # args.pg = 0.03
    
    # Run historical DCF analysis and plotting
    plot_output = plot_historical_dcf_vs_prices(args.t, 10)
    
    try:
        main(args)
    except Exception as e:
        print(f"Error running DCF model: {e}")
        traceback.print_exc()
