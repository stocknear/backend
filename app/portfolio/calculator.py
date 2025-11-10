"""
Portfolio Health Score Calculations

Calculates health scores across 5 dimensions:
- Moat: Competitive advantage (ROIC, ROE, margins, consistency)
- Trend: Momentum (MA positioning, returns, strength)
- Growth: Expansion (revenue, EPS, operating income growth)
- Fundamentals: Quality (debt, liquidity, profitability, Piotroski)
- Volatility: Risk - inverse (beta, std dev, max drawdown)

Optimized with:
- LRU caching to avoid redundant file reads
- Async file I/O for better performance
- Batch file loading to reduce I/O operations
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from functools import lru_cache
import aiofiles
import asyncio

# Base path for data files
BASE_PATH = Path(__file__).parent.parent / "json"

# Load US Market benchmarks (computed from market cap weighted averages)
def load_us_market_benchmarks() -> Dict[str, float]:
    """Load US Market benchmark data from spy-average"""
    try:
        spy_avg_path = BASE_PATH / "spy-average/data.json"
        with open(spy_avg_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# Cache the benchmarks
US_MARKET_BENCHMARKS = load_us_market_benchmarks()


@lru_cache(maxsize=512)
def load_json_sync(file_path_str: str) -> Optional[Any]:
    """Load JSON file synchronously with LRU caching"""
    try:
        with open(file_path_str, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path_str}: {e}")
        return None


async def load_json_async(file_path: Path) -> Optional[Any]:
    """Load JSON file asynchronously"""
    try:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


async def batch_load_files(file_paths: List[Path]) -> Dict[str, Any]:
    """Load multiple JSON files concurrently"""
    tasks = []
    for path in file_paths:
        tasks.append(load_json_async(path))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map results back to file paths
    loaded_data = {}
    for path, result in zip(file_paths, results):
        if not isinstance(result, Exception) and result is not None:
            loaded_data[str(path)] = result

    return loaded_data


async def load_portfolio_data(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Batch load all required data files for a list of tickers
    Returns nested dict: {ticker: {data_type: data}}
    """
    file_paths = []
    file_metadata = []  # Track what each file is for

    # Collect all file paths needed
    for ticker in tickers:
        # Ratios (for moat and fundamentals)
        ratios_path = BASE_PATH / "financial-statements/ratios/annual" / f"{ticker}.json"
        file_paths.append(ratios_path)
        file_metadata.append((ticker, 'ratios'))

        # Income statements (for growth)
        income_path = BASE_PATH / "financial-statements/income-statement/annual" / f"{ticker}.json"
        file_paths.append(income_path)
        file_metadata.append((ticker, 'income'))

        # Historical prices (for trend and volatility)
        price_path = BASE_PATH / "historical-price/one-year" / f"{ticker}.json"
        file_paths.append(price_path)
        file_metadata.append((ticker, 'prices'))

        # Financial score (for fundamentals)
        score_path = BASE_PATH / "financial-score" / f"{ticker}.json"
        file_paths.append(score_path)
        file_metadata.append((ticker, 'score'))

    # Add SPY for beta calculation (load once for all tickers)
    spy_path = BASE_PATH / "historical-price/one-year/SPY.json"
    file_paths.append(spy_path)
    file_metadata.append(('SPY', 'prices'))

    # Batch load all files
    loaded_files = await batch_load_files(file_paths)

    # Organize by ticker
    portfolio_data = {}
    for (ticker, data_type), path in zip(file_metadata, file_paths):
        if ticker not in portfolio_data:
            portfolio_data[ticker] = {}

        path_str = str(path)
        if path_str in loaded_files:
            portfolio_data[ticker][data_type] = loaded_files[path_str]

    return portfolio_data


def calculate_moat_score_from_data(ratios: Optional[List[Dict]]) -> float:
    """
    Calculate Moat score (0-100) based on competitive advantage
    Uses: ROIC, ROE, Profit Margins, Revenue Consistency
    """
    if not ratios or len(ratios) < 3:
        return 50.0  # Default

    # Get recent data (last 3 years)
    recent_ratios = ratios[:3]

    # 1. ROIC Score (0-30 points)
    roic_values = [r.get('returnOnInvestedCapital', 0) * 100 for r in recent_ratios]
    avg_roic = np.mean(roic_values)
    if avg_roic > 20:
        roic_score = 30
    elif avg_roic > 15:
        roic_score = 25
    elif avg_roic > 10:
        roic_score = 18
    else:
        roic_score = 10

    # 2. ROE Score (0-25 points)
    roe_values = [r.get('returnOnEquity', 0) * 100 for r in recent_ratios]
    avg_roe = np.mean(roe_values)
    if avg_roe > 20:
        roe_score = 25
    elif avg_roe > 15:
        roe_score = 20
    elif avg_roe > 10:
        roe_score = 15
    else:
        roe_score = 8

    # 3. Profit Margin Score (0-25 points)
    margins = [r.get('netProfitMargin', 0) * 100 for r in recent_ratios]
    avg_margin = np.mean(margins)
    if avg_margin > 20:
        margin_score = 25
    elif avg_margin > 15:
        margin_score = 20
    elif avg_margin > 10:
        margin_score = 15
    elif avg_margin > 5:
        margin_score = 10
    else:
        margin_score = 5

    # 4. Consistency Score (0-20 points)
    # Low variance = more consistent
    roic_std = np.std(roic_values)
    if roic_std < 3:
        consistency_score = 20
    elif roic_std < 5:
        consistency_score = 15
    elif roic_std < 10:
        consistency_score = 10
    else:
        consistency_score = 5

    total = roic_score + roe_score + margin_score + consistency_score
    return min(total, 100.0)


def calculate_growth_score_from_data(income: Optional[List[Dict]]) -> float:
    """
    Calculate Growth score (0-100)
    Uses: Revenue growth, Earnings growth, Operating income growth
    """
    if not income or len(income) < 5:
        return 50.0

    # Get recent 5 years
    recent = income[:5]

    # 1. Revenue Growth CAGR (0-40 points)
    revenues = [r.get('revenue', 0) for r in recent]
    revenue_cagr = ((revenues[0] / revenues[-1]) ** (1/4) - 1) * 100 if revenues[-1] > 0 else 0

    if revenue_cagr > 20:
        revenue_score = 40
    elif revenue_cagr > 15:
        revenue_score = 32
    elif revenue_cagr > 10:
        revenue_score = 25
    elif revenue_cagr > 5:
        revenue_score = 18
    else:
        revenue_score = 10

    # 2. EPS Growth (0-30 points)
    eps_values = [r.get('eps', 0) for r in recent]
    eps_growth = ((eps_values[0] / eps_values[1]) - 1) * 100 if eps_values[1] > 0 else 0

    if eps_growth > 25:
        eps_score = 30
    elif eps_growth > 15:
        eps_score = 22
    elif eps_growth > 10:
        eps_score = 18
    elif eps_growth > 0:
        eps_score = 12
    else:
        eps_score = 5

    # 3. Operating Income Growth (0-30 points)
    op_income = [r.get('operatingIncome', 0) for r in recent[:3]]
    op_cagr = ((op_income[0] / op_income[-1]) ** (1/2) - 1) * 100 if op_income[-1] > 0 else 0

    if op_cagr > 20:
        op_score = 30
    elif op_cagr > 15:
        op_score = 22
    elif op_cagr > 10:
        op_score = 16
    elif op_cagr > 0:
        op_score = 10
    else:
        op_score = 5

    total = revenue_score + eps_score + op_score
    return min(total, 100.0)


def calculate_fundamentals_score_from_data(ratios: Optional[List[Dict]], fin_score: Optional[Dict]) -> float:
    """
    Calculate Fundamentals score (0-100)
    Uses: Debt ratios, liquidity, profitability, Piotroski score
    """
    if not ratios:
        return 50.0

    latest = ratios[0]

    # 1. Debt Score (0-20 points) - Lower is better
    debt_to_equity = latest.get('debtToEquityRatio', 1)
    if debt_to_equity < 0.3:
        debt_score = 20
    elif debt_to_equity < 0.5:
        debt_score = 17
    elif debt_to_equity < 1.0:
        debt_score = 13
    elif debt_to_equity < 2.0:
        debt_score = 8
    else:
        debt_score = 3

    # 2. Liquidity Score (0-20 points)
    current_ratio = latest.get('currentRatio', 1)
    if current_ratio > 2.5:
        liquidity_score = 20
    elif current_ratio > 2.0:
        liquidity_score = 17
    elif current_ratio > 1.5:
        liquidity_score = 14
    elif current_ratio > 1.0:
        liquidity_score = 10
    else:
        liquidity_score = 5

    # 3. Profitability Score (0-30 points)
    net_margin = latest.get('netProfitMargin', 0) * 100
    if net_margin > 20:
        profit_score = 30
    elif net_margin > 15:
        profit_score = 24
    elif net_margin > 10:
        profit_score = 18
    elif net_margin > 5:
        profit_score = 12
    else:
        profit_score = 6

    # 4. Piotroski Score (0-30 points)
    piotroski = fin_score.get('piotroskiScore', 5) if fin_score else 5
    piotroski_score = (piotroski / 9) * 30

    total = debt_score + liquidity_score + profit_score + piotroski_score
    return min(total, 100.0)


def calculate_trend_score_from_data(prices: Optional[List[Dict]]) -> float:
    """
    Calculate Trend score (0-100)
    Uses: Price vs MA, momentum, relative strength
    """
    if not prices or len(prices) < 200:
        return 50.0

    # Get recent prices
    closes = [p.get('close', 0) for p in prices[:250]]
    current_price = closes[0]

    # 1. MA Score (0-40 points)
    ma_50 = np.mean(closes[:50])
    ma_200 = np.mean(closes[:200])

    ma_score = 0
    if current_price > ma_200:
        ma_score += 20
    if current_price > ma_50:
        ma_score += 20

    # 2. Momentum Score (0-30 points) - 3 month return
    price_3m = closes[min(60, len(closes)-1)]
    return_3m = ((current_price / price_3m) - 1) * 100

    if return_3m > 20:
        momentum_score = 30
    elif return_3m > 10:
        momentum_score = 24
    elif return_3m > 5:
        momentum_score = 18
    elif return_3m > 0:
        momentum_score = 12
    else:
        momentum_score = 5

    # 3. Trend Strength (0-30 points) - Higher highs
    recent_50 = closes[:50]
    max_price = max(recent_50)
    strength = (current_price / max_price) * 100

    if strength > 95:
        strength_score = 30
    elif strength > 90:
        strength_score = 24
    elif strength > 85:
        strength_score = 18
    elif strength > 80:
        strength_score = 12
    else:
        strength_score = 6

    total = ma_score + momentum_score + strength_score
    return min(total, 100.0)


def calculate_volatility_score_from_data(prices: Optional[List[Dict]], spy_prices: Optional[List[Dict]]) -> float:
    """
    Calculate Volatility score (0-100) - INVERSE, lower volatility = higher score
    Uses: Beta, standard deviation, max drawdown
    """
    if not prices or not spy_prices or len(prices) < 200:
        return 50.0

    # Calculate returns
    closes = [p.get('close', 0) for p in prices[:250]]
    spy_closes = [p.get('close', 0) for p in spy_prices[:250]]

    returns = [(closes[i] / closes[i+1] - 1) for i in range(len(closes)-1)]
    spy_returns = [(spy_closes[i] / spy_closes[i+1] - 1) for i in range(min(len(spy_closes)-1, len(closes)-1))]

    # 1. Beta Score (0-35 points) - Lower is better
    if len(returns) > 1 and len(spy_returns) > 1:
        covariance = np.cov(returns[:len(spy_returns)], spy_returns)[0][1]
        variance = np.var(spy_returns)
        beta = covariance / variance if variance > 0 else 1
    else:
        beta = 1

    if beta < 0.7:
        beta_score = 35
    elif beta < 0.9:
        beta_score = 28
    elif beta < 1.1:
        beta_score = 20
    elif beta < 1.3:
        beta_score = 12
    else:
        beta_score = 5

    # 2. Standard Deviation Score (0-35 points) - Annualized volatility
    std_dev = np.std(returns) * np.sqrt(252) * 100  # Annualized %

    if std_dev < 20:
        vol_score = 35
    elif std_dev < 30:
        vol_score = 28
    elif std_dev < 40:
        vol_score = 20
    elif std_dev < 50:
        vol_score = 12
    else:
        vol_score = 5

    # 3. Max Drawdown Score (0-30 points)
    peak = closes[0]
    max_dd = 0
    for price in closes:
        if price > peak:
            peak = price
        dd = ((price - peak) / peak) * 100
        if dd < max_dd:
            max_dd = dd

    if max_dd > -10:
        dd_score = 30
    elif max_dd > -20:
        dd_score = 23
    elif max_dd > -30:
        dd_score = 16
    elif max_dd > -40:
        dd_score = 10
    else:
        dd_score = 5

    total = beta_score + vol_score + dd_score
    return min(total, 100.0)


async def calculate_portfolio_health(holdings: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate overall portfolio health across 5 dimensions
    Optimized with batch file loading and async I/O

    Args:
        holdings: List of dicts with keys: symbol, shares, avgPrice
                  Example: [{"symbol": "AMD", "shares": "50", "avgPrice": "210"}, ...]

    Returns:
        dict with health scores for each dimension:
        {
            "Moat": 58.0,
            "Trend": 52.0,
            "Growth": 71.0,
            "Fundamentals": 69.0,
            "Volatility": 15.0
        }
    """
    if not holdings:
        return {
            'Moat': 50.0,
            'Trend': 50.0,
            'Growth': 50.0,
            'Fundamentals': 50.0,
            'Volatility': 50.0
        }

    # Calculate position values and weights
    position_values = []
    tickers = []

    for holding in holdings:
        symbol = holding.get('symbol')
        shares = float(holding.get('shares', 0) or 0)
        avg_price = float(holding.get('avgPrice', 0) or 0)

        if symbol and shares > 0 and avg_price > 0:
            position_value = shares * avg_price
            position_values.append(position_value)
            tickers.append(symbol)

    if not tickers:
        return {
            'Moat': 50.0,
            'Trend': 50.0,
            'Growth': 50.0,
            'Fundamentals': 50.0,
            'Volatility': 50.0
        }

    # Calculate weights
    total_value = sum(position_values)
    weights = [value / total_value for value in position_values]

    # Batch load all required data
    portfolio_data = await load_portfolio_data(tickers)

    # Initialize scores
    portfolio_scores = {
        'Moat': 0.0,
        'Trend': 0.0,
        'Growth': 0.0,
        'Fundamentals': 0.0,
        'Volatility': 0.0
    }

    # Get SPY data for beta calculations
    spy_prices = portfolio_data.get('SPY', {}).get('prices')

    # Calculate for each ticker using pre-loaded data
    for ticker, weight in zip(tickers, weights):
        try:
            ticker_data = portfolio_data.get(ticker, {})

            moat = calculate_moat_score_from_data(ticker_data.get('ratios'))
            trend = calculate_trend_score_from_data(ticker_data.get('prices'))
            growth = calculate_growth_score_from_data(ticker_data.get('income'))
            fundamentals = calculate_fundamentals_score_from_data(
                ticker_data.get('ratios'),
                ticker_data.get('score')
            )
            volatility = calculate_volatility_score_from_data(
                ticker_data.get('prices'),
                spy_prices
            )

            # Add weighted scores
            portfolio_scores['Moat'] += moat * weight
            portfolio_scores['Trend'] += trend * weight
            portfolio_scores['Growth'] += growth * weight
            portfolio_scores['Fundamentals'] += fundamentals * weight
            portfolio_scores['Volatility'] += volatility * weight
        except Exception as e:
            print(f"Error calculating scores for {ticker}: {e}")
            # Use default score of 50 for failed tickers
            portfolio_scores['Moat'] += 50.0 * weight
            portfolio_scores['Trend'] += 50.0 * weight
            portfolio_scores['Growth'] += 50.0 * weight
            portfolio_scores['Fundamentals'] += 50.0 * weight
            portfolio_scores['Volatility'] += 50.0 * weight

    # Round final scores to 1 decimal place
    for key in portfolio_scores:
        portfolio_scores[key] = round(portfolio_scores[key], 1)

    return portfolio_scores


# Portfolio Fundamental Analysis Functions

def safe_float(value, default=None):
    """Safely convert value to float with default fallback"""
    try:
        if value is None or value == '' or value == 'None':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_portfolio_fundamentals(holdings: List[Dict[str, Any]], stock_screener_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate weighted-average fundamental metrics for portfolio

    Args:
        holdings: List of dicts with keys: symbol, shares, avgPrice
        stock_screener_data_dict: Dictionary mapping symbol to screener data

    Returns:
        Dict with valuation, growth, efficiency, margins, and diversification data
    """
    if not holdings or not stock_screener_data_dict:
        return {
            "valuation": None,
            "growth": None,
            "efficiency": None,
            "margins": None,
            "diversification": []
        }

    # Calculate portfolio weights
    total_value = 0
    portfolio_holdings = []

    for holding in holdings:
        symbol = holding.get('symbol', '')
        shares = safe_float(holding.get('shares'), 0)
        avg_price = safe_float(holding.get('avgPrice'), 0)

        if symbol and shares > 0 and avg_price > 0:
            value = shares * avg_price
            total_value += value
            portfolio_holdings.append({
                'symbol': symbol,
                'value': value,
                'weight': 0  # Will calculate after total is known
            })

    if total_value == 0:
        return {
            "valuation": None,
            "growth": None,
            "efficiency": None,
            "margins": None,
            "diversification": []
        }

    # Calculate weights and add sector/industry data
    diversification_data = []
    for holding in portfolio_holdings:
        holding['weight'] = (holding['value'] / total_value) * 100  # Convert to percentage

        # Get sector and industry from stock screener data
        symbol = holding['symbol']
        stock_data = stock_screener_data_dict.get(symbol, {})

        diversification_data.append({
            'symbol': symbol,
            'sector': stock_data.get('sector', 'ETF'),
            'industry': stock_data.get('industry', 'ETF'),
            'weight': round(holding['weight'], 1)
        })

    # Normalize weights for calculation (0-1 range)
    for holding in portfolio_holdings:
        holding['weight'] = holding['weight'] / 100

    # Calculate each category
    valuation = _calculate_valuation_metrics(portfolio_holdings, stock_screener_data_dict)
    growth = _calculate_growth_metrics(portfolio_holdings, stock_screener_data_dict)
    efficiency = _calculate_efficiency_metrics(portfolio_holdings, stock_screener_data_dict)
    margins = _calculate_margins_metrics(portfolio_holdings, stock_screener_data_dict)

    return {
        "valuation": valuation,
        "growth": growth,
        "efficiency": efficiency,
        "margins": margins,
        "diversification": diversification_data
    }


def _calculate_valuation_metrics(holdings: List[Dict[str, Any]], screener_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weighted valuation metrics"""
    # Initialize
    weighted_pe = weighted_pb = weighted_ps = weighted_pcf = 0
    weighted_forward_pe = weighted_ev_ebitda = weighted_dividend_yield = weighted_peg = 0
    valid_count_pe = valid_count_pb = 0

    # Loop through holdings
    for holding in holdings:
        symbol = holding['symbol']
        weight = holding['weight']
        stock_data = screener_dict.get(symbol, {})

        # PE Ratio
        pe = safe_float(stock_data.get('priceToEarningsRatio'))
        if pe:
            weighted_pe += pe * weight
            valid_count_pe += weight

        # PB Ratio
        pb = safe_float(stock_data.get('priceToBookRatio'))
        if pb:
            weighted_pb += pb * weight
            valid_count_pb += weight

        # Other metrics
        ps = safe_float(stock_data.get('priceToSalesRatio'))
        if ps:
            weighted_ps += ps * weight

        pcf = safe_float(stock_data.get('priceToOperatingCashFlowRatio'))
        if pcf:
            weighted_pcf += pcf * weight

        forward_pe = safe_float(stock_data.get('forwardPE'))
        if forward_pe:
            weighted_forward_pe += forward_pe * weight

        ev_ebitda = safe_float(stock_data.get('evToEBITDA'))
        if ev_ebitda:
            weighted_ev_ebitda += ev_ebitda * weight

        dividend_yield = safe_float(stock_data.get('dividendYield'))
        if dividend_yield:
            weighted_dividend_yield += dividend_yield * weight

        peg = safe_float(stock_data.get('priceToEarningsGrowthRatio'))
        if peg is not None:
            weighted_peg += peg * weight


    if valid_count_pe > 0:
        weighted_pe /= valid_count_pe
    if valid_count_pb > 0:
        weighted_pb /= valid_count_pb

    # US Market benchmarks (from computed weighted averages)
    us_market_pe = US_MARKET_BENCHMARKS.get('priceToEarningsRatio', 0)
    us_market_pb = US_MARKET_BENCHMARKS.get('priceToBookRatio', 0)

    return {
        "gauge1": {
            "value": round(weighted_pe, 1) if weighted_pe != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_pe,
            "compareLabel": "US Market"
        },
        "gauge2": {
            "value": round(weighted_pb, 1) if weighted_pb != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_pb,
            "compareLabel": "US Market"
        },
        "stats": [
            {"label": "Dividend Yield", "value": f"{weighted_dividend_yield:.1f}%"},
            {"label": "PEG Ratio", "value": f"{weighted_peg:.1f}"},
            {"label": "Price/Sales Ratio", "value": f"{weighted_ps:.1f}"},
            {"label": "Price/Cash Flow", "value": f"{weighted_pcf:.1f}"},
            {"label": "Forward P/E", "value": f"{weighted_forward_pe:.1f}"},
            {"label": "EV/EBITDA", "value": f"{weighted_ev_ebitda:.1f}"}
        ]
    }


def _calculate_growth_metrics(holdings: List[Dict[str, Any]], screener_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weighted growth metrics based on actual portfolio growth"""
    # Initialize weighted current and previous values
    weighted_current_revenue = weighted_previous_revenue = 0
    weighted_current_eps = weighted_previous_eps = 0
    weighted_current_operating_income = weighted_previous_operating_income = 0
    weighted_current_fcf = weighted_previous_fcf = 0
    weighted_current_net_income = weighted_previous_net_income = 0

    # For CAGR, we still use weighted average
    weighted_revenue_cagr_5y = 0

    valid_count_revenue = valid_count_eps = 0
    valid_count_operating_income = valid_count_fcf = valid_count_net_income = 0

    # Loop through holdings
    for holding in holdings:
        symbol = holding['symbol']
        weight = holding['weight']
        stock_data = screener_dict.get(symbol, {})

        # Revenue growth calculation
        current_revenue = safe_float(stock_data.get('revenue'))
        revenue_growth = safe_float(stock_data.get('growthRevenue'))
        if current_revenue and revenue_growth is not None:
            # Convert percentage to decimal (10.8% -> 0.108)
            revenue_growth_decimal = revenue_growth / 100
            previous_revenue = current_revenue / (1 + revenue_growth_decimal) if revenue_growth_decimal != -1 else 0
            weighted_current_revenue += current_revenue * weight
            weighted_previous_revenue += previous_revenue * weight
            valid_count_revenue += weight

        # EPS growth calculation
        eps = safe_float(stock_data.get('eps'))
        eps_growth = safe_float(stock_data.get('growthEPS'))
        if eps and eps_growth is not None:
            # Convert percentage to decimal
            eps_growth_decimal = eps_growth / 100
            previous_eps = eps / (1 + eps_growth_decimal) if eps_growth_decimal != -1 else 0
            weighted_current_eps += eps * weight
            weighted_previous_eps += previous_eps * weight
            valid_count_eps += weight

        # Operating income growth calculation
        operating_income = safe_float(stock_data.get('operatingIncome'))
        operating_income_growth = safe_float(stock_data.get('growthOperatingIncome'))
        if operating_income and operating_income_growth is not None:
            # Convert percentage to decimal
            operating_income_growth_decimal = operating_income_growth / 100
            previous_operating_income = operating_income / (1 + operating_income_growth_decimal) if operating_income_growth_decimal != -1 else 0
            weighted_current_operating_income += operating_income * weight
            weighted_previous_operating_income += previous_operating_income * weight
            valid_count_operating_income += weight

        # FCF growth calculation
        fcf = safe_float(stock_data.get('freeCashFlow'))
        fcf_growth = safe_float(stock_data.get('growthFreeCashFlow'))
        if fcf and fcf_growth is not None:
            # Convert percentage to decimal
            fcf_growth_decimal = fcf_growth / 100
            previous_fcf = fcf / (1 + fcf_growth_decimal) if fcf_growth_decimal != -1 else 0
            weighted_current_fcf += fcf * weight
            weighted_previous_fcf += previous_fcf * weight
            valid_count_fcf += weight

        # Revenue CAGR is already in percentage format (e.g., 10.8 for 10.8%)
        revenue_cagr_5y = safe_float(stock_data.get('cagr5YearRevenue'))
        if revenue_cagr_5y is not None:
            weighted_revenue_cagr_5y += revenue_cagr_5y * weight

        # Net income growth calculation
        net_income = safe_float(stock_data.get('netIncome'))
        net_income_growth = safe_float(stock_data.get('growthNetIncome'))
        if net_income and net_income_growth is not None:
            # Convert percentage to decimal
            net_income_growth_decimal = net_income_growth / 100
            previous_net_income = net_income / (1 + net_income_growth_decimal) if net_income_growth_decimal != -1 else 0
            weighted_current_net_income += net_income * weight
            weighted_previous_net_income += previous_net_income * weight
            valid_count_net_income += weight

    # Calculate actual portfolio growth rates
    weighted_revenue_growth = 0
    if valid_count_revenue > 0 and weighted_previous_revenue > 0:
        weighted_revenue_growth = ((weighted_current_revenue - weighted_previous_revenue) / weighted_previous_revenue) * 100

    weighted_eps_growth = 0
    if valid_count_eps > 0 and weighted_previous_eps > 0:
        weighted_eps_growth = ((weighted_current_eps - weighted_previous_eps) / weighted_previous_eps) * 100

    weighted_operating_income_growth = 0
    if valid_count_operating_income > 0 and weighted_previous_operating_income > 0:
        weighted_operating_income_growth = ((weighted_current_operating_income - weighted_previous_operating_income) / weighted_previous_operating_income) * 100

    weighted_fcf_growth = 0
    if valid_count_fcf > 0 and weighted_previous_fcf > 0:
        weighted_fcf_growth = ((weighted_current_fcf - weighted_previous_fcf) / weighted_previous_fcf) * 100

    weighted_net_income_growth = 0
    if valid_count_net_income > 0 and weighted_previous_net_income > 0:
        weighted_net_income_growth = ((weighted_current_net_income - weighted_previous_net_income) / weighted_previous_net_income) * 100

    # US Market benchmarks (from computed weighted averages)
    us_market_revenue = US_MARKET_BENCHMARKS.get('growthRevenue', 0)
    us_market_eps = US_MARKET_BENCHMARKS.get('growthEPS', 0)

    return {
        "gauge1": {
            "value": round(weighted_revenue_growth, 1) if weighted_revenue_growth != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_revenue,
            "compareLabel": "US Market"
        },
        "gauge2": {
            "value": round(weighted_eps_growth, 1) if weighted_eps_growth != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_eps,
            "compareLabel": "US Market"
        },
        "stats": [
            {"label": "Revenue Growth (YoY)", "value": f"{weighted_revenue_growth:.1f}%"},
            {"label": "EPS Growth (YoY)", "value": f"{weighted_eps_growth:.1f}%"},
            {"label": "Revenue Growth (5Y CAGR)", "value": f"{weighted_revenue_cagr_5y:.1f}%"},
            {"label": "Operating Income Growth", "value": f"{weighted_operating_income_growth:.1f}%"},
            {"label": "Net Income Growth", "value": f"{weighted_net_income_growth:.1f}%"},
            {"label": "FCF Growth", "value": f"{weighted_fcf_growth:.1f}%"}
        ]
    }


def _calculate_efficiency_metrics(holdings: List[Dict[str, Any]], screener_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weighted efficiency metrics"""
    # Initialize
    weighted_roe = weighted_roa = 0
    weighted_asset_turnover = weighted_roic = weighted_current_ratio = weighted_inventory_turnover = 0
    weighted_rota = weighted_roce = 0
    valid_count_roe = valid_count_roa = 0

    # Loop through holdings
    for holding in holdings:
        symbol = holding['symbol']
        weight = holding['weight']
        stock_data = screener_dict.get(symbol, {})

        # ROE is already in percentage format (e.g., 18.2 for 18.2%)
        roe = safe_float(stock_data.get('returnOnEquity'))
        if roe is not None:
            weighted_roe += roe * weight
            valid_count_roe += weight

        # ROA is already in percentage format
        roa = safe_float(stock_data.get('returnOnAssets'))
        if roa is not None:
            weighted_roa += roa * weight
            valid_count_roa += weight

        # Asset turnover is a ratio, not percentage
        asset_turnover = safe_float(stock_data.get('assetTurnover'))
        if asset_turnover is not None:
            weighted_asset_turnover += asset_turnover * weight

        # ROIC is already in percentage format
        roic = safe_float(stock_data.get('returnOnInvestedCapital'))
        if roic is not None:
            weighted_roic += roic * weight

        # Current ratio is a ratio, not percentage
        current_ratio = safe_float(stock_data.get('currentRatio'))
        if current_ratio is not None:
            weighted_current_ratio += current_ratio * weight

        # Inventory turnover is a ratio, not percentage
        inventory_turnover = safe_float(stock_data.get('inventoryTurnover'))
        if inventory_turnover is not None:
            weighted_inventory_turnover += inventory_turnover * weight

        # Return on Tangible Assets is already in percentage format
        rota = safe_float(stock_data.get('returnOnTangibleAssets'))
        if rota is not None:
            weighted_rota += rota * weight

        # Return On Capital Employed is already in percentage format
        roce = safe_float(stock_data.get('returnOnCapitalEmployed'))
        if roce is not None:
            weighted_roce += roce * weight

    # Normalize
    if valid_count_roe > 0:
        weighted_roe /= valid_count_roe
    if valid_count_roa > 0:
        weighted_roa /= valid_count_roa

    # US Market benchmarks (from computed weighted averages)
    us_market_roe = US_MARKET_BENCHMARKS.get('returnOnEquity', 0)
    us_market_roa = US_MARKET_BENCHMARKS.get('returnOnAssets', 0)

    return {
        "gauge1": {
            "value": round(weighted_roe, 1) if weighted_roe != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_roe,
            "compareLabel": "US Market"
        },
        "gauge2": {
            "value": round(weighted_roa, 1) if weighted_roa != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_roa,
            "compareLabel": "US Market"
        },
        "stats": [
            {"label": "Return on Equity (ROE)", "value": f"{weighted_roe:.1f}%"},
            {"label": "Return on Assets (ROA)", "value": f"{weighted_roa:.1f}%"},
            {"label": "Asset Turnover", "value": f"{weighted_asset_turnover:.2f}"},
            {"label": "Return on Invested Capital", "value": f"{weighted_roic:.1f}%"},
            {"label": "Inventory Turnover", "value": f"{weighted_inventory_turnover:.2f}"},
            {"label": "Working Capital Ratio", "value": f"{weighted_current_ratio:.2f}"},
            {"label": "Return on Tangible Assets", "value": f"{weighted_rota:.1f}%"},
            {"label": "Return On Capital Employed", "value": f"{weighted_roce:.1f}%"}
        ]
    }


def _calculate_margins_metrics(holdings: List[Dict[str, Any]], screener_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weighted margin metrics"""
    # Initialize
    weighted_gross_margin = weighted_operating_margin = 0
    weighted_net_margin = weighted_ebitda_margin = weighted_fcf_margin = weighted_pretax_margin = 0
    valid_count_gross = valid_count_operating = valid_count_net = 0

    # Loop through holdings
    for holding in holdings:
        symbol = holding['symbol']
        weight = holding['weight']
        stock_data = screener_dict.get(symbol, {})

        # All margins are already in percentage format (e.g., 35.2 for 35.2%)
        gross_margin = safe_float(stock_data.get('grossProfitMargin'))
        if gross_margin is not None:
            weighted_gross_margin += gross_margin * weight
            valid_count_gross += weight

        operating_margin = safe_float(stock_data.get('operatingMargin'))
        if operating_margin is not None:
            weighted_operating_margin += operating_margin * weight
            valid_count_operating += weight

        net_margin = safe_float(stock_data.get('netProfitMargin'))
        if net_margin is not None:
            weighted_net_margin += net_margin * weight
            valid_count_net += weight

        ebitda_margin = safe_float(stock_data.get('ebitdaMargin'))
        if ebitda_margin is not None:
            weighted_ebitda_margin += ebitda_margin * weight

        fcf_margin = safe_float(stock_data.get('freeCashFlowMargin'))
        if fcf_margin is not None:
            weighted_fcf_margin += fcf_margin * weight

        pretax_margin = safe_float(stock_data.get('pretaxProfitMargin'))
        if pretax_margin is not None:
            weighted_pretax_margin += pretax_margin * weight

    # Normalize
    if valid_count_gross > 0:
        weighted_gross_margin /= valid_count_gross
    if valid_count_operating > 0:
        weighted_operating_margin /= valid_count_operating
    if valid_count_net > 0:
        weighted_net_margin /= valid_count_net

    # US Market benchmarks (from computed weighted averages)
    us_market_gross = US_MARKET_BENCHMARKS.get('grossProfitMargin', 0)
    us_market_net = US_MARKET_BENCHMARKS.get('netProfitMargin', 0)

    return {
        "gauge1": {
            "value": round(weighted_gross_margin, 1) if weighted_gross_margin != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_gross,
            "compareLabel": "US Market"
        },
        "gauge2": {
            "value": round(weighted_net_margin, 1) if weighted_net_margin != 0 else None,
            "label": "Portfolio",
            "compareValue": us_market_net,
            "compareLabel": "US Market"
        },
        "stats": [
            {"label": "Gross Margin", "value": f"{weighted_gross_margin:.1f}%"},
            {"label": "Operating Margin", "value": f"{weighted_operating_margin:.1f}%"},
            {"label": "Profit Margin", "value": f"{weighted_net_margin:.1f}%"},
            {"label": "EBITDA Margin", "value": f"{weighted_ebitda_margin:.1f}%"},
            {"label": "FCF Margin", "value": f"{weighted_fcf_margin:.1f}%"},
            {"label": "Pretax Margin", "value": f"{weighted_pretax_margin:.1f}%"}
        ]
    }
