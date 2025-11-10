import os
import sqlite3
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import islice
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import orjson


BASE_DIR = os.path.dirname(__file__)


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        try:
            return orjson.loads(file.read())
        except orjson.JSONDecodeError:
            return None


def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None


def as_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg_pair(current: Dict[str, float], previous: Optional[Dict[str, float]], key: str) -> Optional[float]:
    curr_val = as_float(current.get(key))
    prev_val = as_float(previous.get(key)) if previous else None
    if curr_val is None or prev_val is None:
        return None
    return (curr_val + prev_val) / 2.0


def growth_rate(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    current = as_float(current)
    previous = as_float(previous)
    if current is None or previous in (None, 0):
        return None
    try:
        return (current - previous) / abs(previous)
    except ZeroDivisionError:
        return None


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(symbol: str, data: Sequence[Dict], file_type: str) -> None:
    if file_type == "ratios":
        relative_parts = ("json", "financial-statements", "ratios", "ttm-updated")
    elif file_type == "income-statement-growth":
        relative_parts = ("json", "financial-statements", "income-statement-growth", "ttm-updated")
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    output_dir = os.path.join(BASE_DIR, *relative_parts)
    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, f"{symbol}.json")
    with open(output_path, "wb") as file:
        file.write(orjson.dumps(data, default=convert_types))


def load_json(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        try:
            data = orjson.loads(file.read())
            if isinstance(data, list):
                return data
        except orjson.JSONDecodeError:
            return None
    return None


def load_ttm_statement(symbol: str, statement: str) -> Optional[List[Dict]]:
    path = os.path.join(BASE_DIR, "json", "financial-statements", statement, "ttm", f"{symbol}.json")
    data = load_json(path)
    if not data:
        return None
    # Ensure chronological order for sliding calculations
    return sorted(data, key=lambda row: row.get("date", ""))


def load_price_series(symbol: str) -> Optional["PriceLookup"]:
    ranges = [
        "max",
        "five-years",
        "one-year",
        "six-months",
        "ytd",
        "one-month",
        "one-week",
    ]
    base = "json/historical-price"
    for history_range in ranges:
        path = os.path.join(BASE_DIR, base, history_range, f"{symbol}.json")
        data = load_json(path)
        if data:
            return PriceLookup(data)
    return None


def load_quote_price(symbol: str) -> Optional[float]:
    path = os.path.join(BASE_DIR, "json", "quote", f"{symbol}.json")
    data = read_json(path)
    if isinstance(data, dict):
        return as_float(data.get("price"))
    return None


def load_forward_pe_reference(symbol: str) -> Optional[float]:
    path = os.path.join(BASE_DIR, "json", "forward-pe", f"{symbol}.json")
    data = read_json(path)
    if isinstance(data, dict):
        return as_float(data.get("forwardPE"))
    return None


def load_estimated_eps(symbol: str, next_year: int) -> Optional[float]:
    path = os.path.join(BASE_DIR, "json", "analyst-estimate", f"{symbol}.json")
    data = read_json(path)
    if isinstance(data, list):
        entry = next((row for row in data if row.get("date") == next_year), None)
        if entry:
            return as_float(entry.get("estimatedEpsAvg"))
    return None


def build_forward_inputs(symbol: str, income_rows: Optional[List[Dict]]) -> Dict[str, Optional[float]]:
    next_year = datetime.now().year + 1
    estimated_eps = load_estimated_eps(symbol, next_year)
    if estimated_eps is None:
        return {"forward_eps": None, "eps_fx_factor": 1.0, "forward_pe_reference": load_forward_pe_reference(symbol)}

    quote_price = load_quote_price(symbol)
    forward_pe_reference = load_forward_pe_reference(symbol)
    factor = 1.0

    implied_eps_from_reference = None
    if forward_pe_reference and quote_price:
        implied_eps_from_reference = safe_div(quote_price, forward_pe_reference)

    if implied_eps_from_reference and estimated_eps not in (None, 0):
        factor_candidate = safe_div(implied_eps_from_reference, estimated_eps)
        if factor_candidate:
            factor = factor_candidate

    currency = None
    if income_rows:
        currency = income_rows[0].get("reportedCurrency")
    if currency and currency.upper() == "USD":
        factor = factor or 1.0

    forward_eps = estimated_eps * (factor or 1.0)
    return {
        "forward_eps": forward_eps,
        "eps_fx_factor": factor or 1.0,
        "forward_pe_reference": forward_pe_reference,
    }


class PriceLookup:
    def __init__(self, rows: Sequence[Dict]):
        ordered = sorted(rows, key=lambda row: row["time"])
        self.dates = [datetime.fromisoformat(row["time"]).date() for row in ordered]
        self.closes = [as_float(row.get("close")) for row in ordered]

    def close_on_or_before(self, iso_date: str) -> Optional[float]:
        if not self.dates:
            return None
        target = datetime.fromisoformat(iso_date).date()
        index = bisect_right(self.dates, target) - 1
        if index < 0:
            return None
        return self.closes[index]


def market_cap(price: Optional[float], shares: Optional[float]) -> Optional[float]:
    price = as_float(price)
    shares = as_float(shares)
    if price is None or shares is None:
        return None
    return price * shares


def enterprise_value(cap: Optional[float], total_debt: Optional[float], cash: Optional[float]) -> Optional[float]:
    cap = as_float(cap)
    total_debt = as_float(total_debt) or 0.0
    cash = as_float(cash) or 0.0
    if cap is None:
        return None
    return cap + total_debt - cash


def tangible_equity(balance: Dict[str, float]) -> Optional[float]:
    total_equity = as_float(balance.get("totalEquity"))
    if total_equity is None:
        return None
    goodwill_and_intangibles = as_float(balance.get("goodwillAndIntangibleAssets"))
    if goodwill_and_intangibles is None:
        goodwill = as_float(balance.get("goodwill")) or 0.0
        intangible_assets = as_float(balance.get("intangibleAssets")) or 0.0
        goodwill_and_intangibles = goodwill + intangible_assets
    return total_equity - goodwill_and_intangibles


def get_shares(income_row: Dict[str, float]) -> Optional[float]:
    shares = income_row.get("weightedAverageShsOutDil")
    if shares in (None, 0):
        shares = income_row.get("weightedAverageShsOut")
    return as_float(shares)


def compute_price_metrics(price: Optional[float], shares: Optional[float], balance: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    cap = market_cap(price, shares)
    ev = enterprise_value(cap, balance.get("totalDebt"), balance.get("cashAndCashEquivalents"))
    return cap, ev


def calculate_ttm_ratios(
    symbol: str,
    income_rows: Optional[List[Dict]],
    balance_rows: Optional[List[Dict]],
    cash_rows: Optional[List[Dict]],
    forward_inputs: Optional[Dict[str, Optional[float]]] = None,
) -> List[Dict]:
    if not income_rows or not balance_rows or not cash_rows:
        return []

    income_by_date = {row["date"]: row for row in income_rows if "date" in row}
    balance_by_date = {row["date"]: row for row in balance_rows if "date" in row}
    cash_by_date = {row["date"]: row for row in cash_rows if "date" in row}
    common_dates = sorted(set(income_by_date) & set(balance_by_date) & set(cash_by_date))
    if not common_dates:
        return []

    price_lookup = load_price_series(symbol)
    results: List[Dict] = []

    prev_balance: Optional[Dict] = None
    prev_income: Optional[Dict] = None

    forward_eps_target = None
    eps_fx_factor = 1.0
    forward_pe_reference = None
    if forward_inputs:
        forward_eps_target = as_float(forward_inputs.get("forward_eps"))
        forward_pe_reference = as_float(forward_inputs.get("forward_pe_reference"))
        candidate_factor = as_float(forward_inputs.get("eps_fx_factor"))
        if candidate_factor not in (None, 0):
            eps_fx_factor = candidate_factor

    for date in common_dates:
        income = income_by_date[date]
        balance = balance_by_date[date]
        cash = cash_by_date[date]

        shares = get_shares(income)
        price = price_lookup.close_on_or_before(date) if price_lookup else None
        cap, ev = compute_price_metrics(price, shares, balance)

        revenue = as_float(income.get("revenue"))
        gross_profit = as_float(income.get("grossProfit"))
        cost_of_revenue = as_float(income.get("costOfRevenue"))
        ebit = as_float(income.get("ebit"))
        ebitda = as_float(income.get("ebitda"))
        operating_income = as_float(income.get("operatingIncome"))
        income_before_tax = as_float(income.get("incomeBeforeTax"))
        net_income = as_float(income.get("netIncome"))
        bottom_line_income = as_float(income.get("bottomLineNetIncome")) or net_income
        cont_income = as_float(income.get("netIncomeFromContinuingOperations")) or net_income
        depreciation = as_float(income.get("depreciationAndAmortization"))
        interest_expense = as_float(income.get("interestExpense"))
        income_tax_expense = as_float(income.get("incomeTaxExpense"))

        total_assets = as_float(balance.get("totalAssets"))
        total_equity = as_float(balance.get("totalEquity"))
        total_liabilities = as_float(balance.get("totalLiabilities"))
        total_debt = as_float(balance.get("totalDebt"))
        long_term_debt = as_float(balance.get("longTermDebt"))
        short_term_debt = as_float(balance.get("shortTermDebt"))
        total_current_assets = as_float(balance.get("totalCurrentAssets"))
        total_current_liabilities = as_float(balance.get("totalCurrentLiabilities"))
        inventory = as_float(balance.get("inventory"))
        net_receivables = as_float(balance.get("netReceivables"))
        accounts_payable = as_float(balance.get("accountPayables"))
        ppe = as_float(balance.get("propertyPlantEquipmentNet"))
        cash_equivalents = as_float(balance.get("cashAndCashEquivalents"))

        working_cap_current = None
        if total_current_assets is not None and total_current_liabilities is not None:
            working_cap_current = total_current_assets - total_current_liabilities

        avg_assets = avg_pair(balance, prev_balance, "totalAssets")
        avg_equity = avg_pair(balance, prev_balance, "totalEquity")
        avg_ppe = avg_pair(balance, prev_balance, "propertyPlantEquipmentNet")
        avg_inventory = avg_pair(balance, prev_balance, "inventory")
        avg_receivables = avg_pair(balance, prev_balance, "netReceivables")
        avg_payables = avg_pair(balance, prev_balance, "accountPayables")

        avg_working_capital = None
        if prev_balance:
            prev_current_assets = as_float(prev_balance.get("totalCurrentAssets"))
            prev_current_liabilities = as_float(prev_balance.get("totalCurrentLiabilities"))
            if prev_current_assets is not None and prev_current_liabilities is not None and working_cap_current is not None:
                working_cap_prev = prev_current_assets - prev_current_liabilities
                avg_working_capital = (working_cap_current + working_cap_prev) / 2.0

        operating_cash_flow = as_float(cash.get("netCashProvidedByOperatingActivities")) or as_float(
            cash.get("operatingCashFlow")
        )
        free_cash_flow = as_float(cash.get("freeCashFlow"))
        capital_expenditure = as_float(cash.get("capitalExpenditure"))
        dividends_paid = abs(as_float(cash.get("commonDividendsPaid")) or 0.0)

        eps = safe_div(net_income, shares)
        eps_converted = eps * eps_fx_factor if eps is not None else None

        forward_pe_value = None
        peg_ratio = None
        if forward_eps_target not in (None, 0):
            forward_pe_value = safe_div(price, forward_eps_target)
            if forward_pe_value is None and forward_pe_reference:
                forward_pe_value = forward_pe_reference
            if forward_pe_value not in (None, 0) and eps_converted not in (None, 0):
                growth_decimal = growth_rate(forward_eps_target, eps_converted)
                if growth_decimal and growth_decimal > 0:
                    growth_percent = growth_decimal * 100.0
                    peg_ratio = safe_div(forward_pe_value, growth_percent)

        dividend_per_share = safe_div(dividends_paid, shares)
        capex_required = abs(capital_expenditure) if capital_expenditure not in (None, 0) else None
        dividend_capex_denom = None
        if (capex_required not in (None, 0)) or dividends_paid:
            dividend_capex_denom = (capex_required or 0.0) + dividends_paid

        invested_capital = None
        if total_equity is not None and total_debt is not None:
            invested_capital = total_equity + total_debt - (cash_equivalents or 0.0)
        avg_invested_capital = None
        if prev_balance:
            prev_total_equity = as_float(prev_balance.get("totalEquity"))
            prev_total_debt = as_float(prev_balance.get("totalDebt"))
            prev_cash = as_float(prev_balance.get("cashAndCashEquivalents")) or 0.0
            if prev_total_equity is not None and prev_total_debt is not None and invested_capital is not None:
                prev_invested_capital = prev_total_equity + prev_total_debt - prev_cash
                avg_invested_capital = (invested_capital + prev_invested_capital) / 2.0

        tax_rate = None
        if income_before_tax not in (None, 0) and income_tax_expense is not None:
            tax_rate = max(0.0, min(1.0, income_tax_expense / income_before_tax))

        nopat = None
        if operating_income is not None and tax_rate is not None:
            nopat = operating_income * (1 - tax_rate)

        ratio_row: Dict[str, Optional[float]] = {
            "symbol": symbol,
            "date": income.get("date"),
            "period": income.get("period"),
            "fiscalYear": income.get("fiscalYear"),
            "reportedCurrency": income.get("reportedCurrency"),
            "grossProfitMargin": safe_div(gross_profit, revenue),
            "ebitMargin": safe_div(ebit, revenue),
            "ebitdaMargin": safe_div(ebitda, revenue),
            "operatingProfitMargin": safe_div(operating_income, revenue),
            "pretaxProfitMargin": safe_div(income_before_tax, revenue),
            "continuousOperationsProfitMargin": safe_div(cont_income, revenue),
            "netProfitMargin": safe_div(net_income, revenue),
            "bottomLineProfitMargin": safe_div(bottom_line_income, revenue),
            "receivablesTurnover": safe_div(revenue, avg_receivables),
            "payablesTurnover": safe_div(cost_of_revenue, avg_payables),
            "inventoryTurnover": safe_div(cost_of_revenue, avg_inventory),
            "fixedAssetTurnover": safe_div(revenue, avg_ppe),
            "assetTurnover": safe_div(revenue, avg_assets),
            "currentRatio": safe_div(total_current_assets, total_current_liabilities),
            "quickRatio": safe_div(
                (total_current_assets - inventory) if (total_current_assets is not None and inventory is not None) else None,
                total_current_liabilities,
            ),
            "cashRatio": safe_div(cash_equivalents, total_current_liabilities),
            "solvencyRatio": safe_div((net_income or 0) + (depreciation or 0), total_liabilities),
            "priceToEarningsRatio": safe_div(price, eps),
            "priceToBookRatio": safe_div(price, safe_div(total_equity, shares)),
            "priceToSalesRatio": safe_div(cap, revenue),
            "priceToFreeCashFlowRatio": safe_div(cap, free_cash_flow),
            "priceToOperatingCashFlowRatio": safe_div(cap, operating_cash_flow),
            "debtToAssetsRatio": safe_div(total_debt, total_assets),
            "debtToEquityRatio": safe_div(total_debt, total_equity),
            "debtToCapitalRatio": safe_div(total_debt, (total_debt or 0) + (total_equity or 0)),
            "longTermDebtToCapitalRatio": safe_div(long_term_debt, (long_term_debt or 0) + (total_equity or 0)),
            "financialLeverageRatio": safe_div(total_assets, total_equity),
            "workingCapitalTurnoverRatio": safe_div(revenue, avg_working_capital),
            "operatingCashFlowRatio": safe_div(operating_cash_flow, total_current_liabilities),
            "operatingCashFlowSalesRatio": safe_div(operating_cash_flow, revenue),
            "freeCashFlowOperatingCashFlowRatio": safe_div(free_cash_flow, operating_cash_flow),
            "debtServiceCoverageRatio": safe_div(
                (operating_cash_flow or 0.0) + (interest_expense or 0.0),
                (interest_expense or 0.0) + (short_term_debt or 0.0),
            ),
            "interestCoverageRatio": safe_div(operating_income, interest_expense),
            "shortTermOperatingCashFlowCoverageRatio": safe_div(operating_cash_flow, short_term_debt),
            "operatingCashFlowCoverageRatio": safe_div(operating_cash_flow, total_debt),
            "capitalExpenditureCoverageRatio": safe_div(operating_cash_flow, capex_required),
            "dividendPaidAndCapexCoverageRatio": safe_div(
                operating_cash_flow,
                dividend_capex_denom,
            ),
            "dividendPayoutRatio": safe_div(dividends_paid, net_income) if net_income else None,
            "dividendPerShare": dividend_per_share,
            "dividendYield": safe_div(dividend_per_share, price),
            "dividendYieldPercentage": None,
            "revenuePerShare": safe_div(revenue, shares),
            "netIncomePerShare": eps,
            "interestDebtPerShare": safe_div((total_debt or 0.0) + (interest_expense or 0.0), shares),
            "cashPerShare": safe_div(cash_equivalents, shares),
            "bookValuePerShare": safe_div(total_equity, shares),
            "tangibleBookValuePerShare": safe_div(tangible_equity(balance), shares),
            "shareholdersEquityPerShare": safe_div(total_equity, shares),
            "operatingCashFlowPerShare": safe_div(operating_cash_flow, shares),
            "capexPerShare": safe_div(capex_required, shares),
            "freeCashFlowPerShare": safe_div(free_cash_flow, shares),
            "netIncomePerEBT": safe_div(net_income, income_before_tax),
            "ebtPerEbit": safe_div(income_before_tax, ebit),
            "priceToFairValue": None,
            "debtToMarketCap": safe_div(total_debt, cap),
            "effectiveTaxRate": safe_div(income_tax_expense, income_before_tax),
            "enterpriseValueMultiple": safe_div(ev, ebitda),
            "enterpriseValue": ev,
            "evToSales": safe_div(ev, revenue),
            "evToEBITDA": safe_div(ev, ebitda),
            "evToFreeCashFlow": safe_div(ev, free_cash_flow),
            "earningsYield": safe_div(net_income, cap),
            "freeCashFlowYield": safe_div(free_cash_flow, cap),
            "freeCashFlowMargin": safe_div(free_cash_flow, revenue),
            "forwardPE": forward_pe_value,
            "forwardPriceToEarningsGrowthRatio": peg_ratio,
            "priceToEarningsGrowthRatio": peg_ratio,
            "returnOnAssets": safe_div(net_income, avg_assets),
            "returnOnEquity": safe_div(net_income, avg_equity),
            "returnOnInvestedCapital": safe_div(nopat, avg_invested_capital),
        }

        if ratio_row["priceToFairValue"] is None:
            ratio_row["priceToFairValue"] = ratio_row["priceToBookRatio"]

        if ratio_row["dividendYield"] is not None:
            ratio_row["dividendYieldPercentage"] = ratio_row["dividendYield"] * 100.0

        results.append(ratio_row)
        prev_balance = balance
        prev_income = income

    return results


GROWTH_FIELD_MAP: Tuple[Tuple[str, str], ...] = (
    ("growthRevenue", "revenue"),
    ("growthCostOfRevenue", "costOfRevenue"),
    ("growthGrossProfit", "grossProfit"),
    ("growthGrossProfitRatio", "grossProfitRatio"),
    ("growthResearchAndDevelopmentExpenses", "researchAndDevelopmentExpenses"),
    ("growthGeneralAndAdministrativeExpenses", "generalAndAdministrativeExpenses"),
    ("growthSellingAndMarketingExpenses", "sellingAndMarketingExpenses"),
    ("growthOtherExpenses", "otherExpenses"),
    ("growthOperatingExpenses", "operatingExpenses"),
    ("growthCostAndExpenses", "costAndExpenses"),
    ("growthInterestIncome", "interestIncome"),
    ("growthInterestExpense", "interestExpense"),
    ("growthDepreciationAndAmortization", "depreciationAndAmortization"),
    ("growthEBITDA", "ebitda"),
    ("growthOperatingIncome", "operatingIncome"),
    ("growthIncomeBeforeTax", "incomeBeforeTax"),
    ("growthIncomeTaxExpense", "incomeTaxExpense"),
    ("growthNetIncome", "netIncome"),
    ("growthEPS", "eps"),
    ("growthEPSDiluted", "epsDiluted"),
    ("growthWeightedAverageShsOut", "weightedAverageShsOut"),
    ("growthWeightedAverageShsOutDil", "weightedAverageShsOutDil"),
    ("growthEBIT", "ebit"),
    ("growthNonOperatingIncomeExcludingInterest", "nonOperatingIncomeExcludingInterest"),
    ("growthNetInterestIncome", "netInterestIncome"),
    ("growthTotalOtherIncomeExpensesNet", "totalOtherIncomeExpensesNet"),
    ("growthNetIncomeFromContinuingOperations", "netIncomeFromContinuingOperations"),
    ("growthOtherAdjustmentsToNetIncome", "otherAdjustmentsToNetIncome"),
    ("growthNetIncomeDeductions", "netIncomeDeductions"),
)


def calculate_ttm_income_growth(symbol: str, income_rows: Optional[List[Dict]]) -> List[Dict]:
    if not income_rows or len(income_rows) < 2:
        return []

    results: List[Dict] = []
    prev_row: Optional[Dict] = None

    def gross_profit_ratio(row: Dict[str, float]) -> Optional[float]:
        return safe_div(row.get("grossProfit"), row.get("revenue"))

    for row in income_rows:
        if not prev_row:
            prev_row = row
            continue

        growth_entry: Dict[str, Optional[float]] = {
            "symbol": symbol,
            "date": row.get("date"),
            "period": row.get("period"),
            "fiscalYear": row.get("fiscalYear"),
            "reportedCurrency": row.get("reportedCurrency"),
        }

        for growth_field, source_field in GROWTH_FIELD_MAP:
            if source_field == "grossProfitRatio":
                current_value = gross_profit_ratio(row)
                previous_value = gross_profit_ratio(prev_row)
            else:
                current_value = row.get(source_field)
                previous_value = prev_row.get(source_field)
            growth_entry[growth_field] = growth_rate(current_value, previous_value)

        results.append(growth_entry)
        prev_row = row

    return results


def get_symbols() -> List[str]:
    with sqlite3.connect("stocks.db") as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA cache_size = -50000")
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute("PRAGMA synchronous = NORMAL")
        rows = con.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'").fetchall()
    return [row[0] for row in rows]


def batch(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    iterator = iter(iterable)
    while True:
        batch_iter = list(islice(iterator, size))
        if not batch_iter:
            break
        yield batch_iter


def process_symbol(symbol: str) -> List[Tuple[str, str, bool, str]]:
    income_rows = load_ttm_statement(symbol, "income-statement")
    balance_rows = load_ttm_statement(symbol, "balance-sheet-statement")
    cash_rows = load_ttm_statement(symbol, "cash-flow-statement")
    forward_inputs = build_forward_inputs(symbol, income_rows)

    results: List[Tuple[str, str, bool, str]] = []

    ratios = calculate_ttm_ratios(symbol, income_rows, balance_rows, cash_rows, forward_inputs)
    if ratios:
        save_json(symbol, ratios, "ratios")
        results.append((symbol, "ratios", True, "Success"))
    else:
        results.append((symbol, "ratios", False, "Insufficient data"))

    growth = calculate_ttm_income_growth(symbol, income_rows)
    if growth:
        save_json(symbol, growth, "income-statement-growth")
        results.append((symbol, "income-statement-growth", True, "Success"))
    else:
        results.append((symbol, "income-statement-growth", False, "Insufficient data"))

    return results


def main():
    symbols = get_symbols()
    #testing
    #symbols = ['SPT']
    
    batch_size = 100
    stats = {
        "ratios": {"success": 0, "error": 0},
        "income-statement-growth": {"success": 0, "error": 0},
    }
    max_workers = max(os.cpu_count() - 1, 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for symbol_batch in batch(symbols, batch_size):
            for result in executor.map(process_symbol, symbol_batch):
                for symbol, file_type, success, message in result:
                    if success:
                        stats[file_type]["success"] += 1
                    else:
                        stats[file_type]["error"] += 1
                        print(f"Error processing {file_type} for {symbol}: {message}")

    for file_type, counts in stats.items():
        print(f"{file_type}: {counts['success']} successful, {counts['error']} failed")


if __name__ == "__main__":
    main()
