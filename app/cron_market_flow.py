import os
import numpy as np
import pandas as pd
import orjson
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
import aiohttp
from GetStartEndDate import GetStartEndDate
from typing import Any, Dict, List


load_dotenv()
fmp_api_key = os.getenv('FMP_API_KEY')


OPTIONS_FLOW_PATH = "json/options-flow/feed/data.json"
SP500_LIST_PATH = "json/stocks-list/list/sp500.json"


def load_options_flow_dataframe(path: str = OPTIONS_FLOW_PATH) -> pd.DataFrame:
    try:
        with open(path, "rb") as file:
            raw = orjson.loads(file.read())
    except FileNotFoundError:
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    if df.empty:
        return df

    required_string_columns = ["ticker", "date", "time", "sentiment", "put_call"]
    for column in required_string_columns:
        if column not in df.columns:
            df[column] = ""

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = df["date"].astype(str)
    df["time"] = df["time"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str)
    df["put_call"] = df["put_call"].astype(str)

    df["datetime"] = pd.to_datetime(
        df["date"].str.strip() + " " + df["time"].str.strip(),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.floor("T")

    df["cost_value"] = pd.to_numeric(df.get("cost_basis", 0), errors="coerce").fillna(0.0)
    df["volume_value"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df["open_interest_value"] = pd.to_numeric(df.get("open_interest", 0), errors="coerce").fillna(0.0)

    return df[
        [
            "ticker",
            "date",
            "time",
            "datetime",
            "sentiment",
            "put_call",
            "cost_value",
            "volume_value",
            "open_interest_value",
        ]
    ].copy()


def load_holdings_weights(sector_ticker: str) -> Dict[str, float]:
    path = f"json/etf/holding/{sector_ticker}.json"
    try:
        with open(path, "rb") as file:
            holdings_data = orjson.loads(file.read())
    except FileNotFoundError:
        return {}

    holdings = holdings_data.get("holdings", [])
    weights = {}
    for item in holdings:
        symbol = item.get("symbol")
        if not symbol:
            continue
        weights[symbol.upper()] = float(item.get("weightPercentage", 0) or 0)
    return weights


def load_sp500_tickers(path: str = SP500_LIST_PATH) -> List[str]:
    try:
        with open(path, "rb") as file:
            data = orjson.loads(file.read())
    except FileNotFoundError:
        return []
    return [item.get("symbol", "").upper() for item in data if item.get("symbol")]


def load_sector_map(tickers: List[str]) -> Dict[str, str]:
    sector_map: Dict[str, str] = {}
    for ticker in tickers:
        if not ticker:
            continue
        path = f"json/stockdeck/{ticker}.json"
        try:
            with open(path, "rb") as file:
                info = orjson.loads(file.read())
        except FileNotFoundError:
            continue
        sector = info.get("sector")
        if sector:
            sector_map[ticker] = sector
    return sector_map


def save_json(data, filename):
    directory = "json/market-flow"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    with open(f"{directory}/{filename}.json", 'wb') as file:  # Use binary mode for orjson
        file.write(orjson.dumps(data))


def safe_round(value):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value

def add_close_to_data(price_list, data):
    if not data:
        return data

    price_map = {}
    for price in price_list or []:
        price_time = price.get("time") or price.get("date")
        if not price_time:
            continue
        price_map[price_time] = price.get("close")

    for entry in data:
        entry["close"] = price_map.get(entry["time"])
    return data


async def get_stock_chart_data(ticker):
    start_date_1d, end_date_1d = GetStartEndDate().run()
    start_date = start_date_1d.strftime("%Y-%m-%d")
    end_date = end_date_1d.strftime("%Y-%m-%d")

    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?from={start_date}&to={end_date}&apikey={fmp_api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                data = sorted(data, key=lambda x: x['date'])
                return data
            else:
                return []


def get_sector_data(
    sector_ticker: str,
    options_df: pd.DataFrame,
    holdings_weights: Dict[str, float],
    trading_date: str,
) -> List[Dict[str, Any]]:
    tickers = [ticker.upper() for ticker in holdings_weights.keys()]
    if not tickers:
        return []

    df = options_df[options_df["ticker"].isin(tickers)].copy()
    if df.empty:
        return []

    if trading_date:
        df = df[df["date"] == trading_date].copy()
        if df.empty:
            return []

    df.sort_values("datetime", inplace=True)
    df["weight"] = 1.0  # maintain previous behaviour of equal weighting

    weighted_cost = df["cost_value"] * df["weight"]
    weighted_volume = df["volume_value"] * df["weight"]

    call_mask = df["put_call"].str.upper() == "CALLS"
    put_mask = df["put_call"].str.upper() == "PUTS"
    bullish_mask = df["sentiment"].str.upper() == "BULLISH"
    bearish_mask = df["sentiment"].str.upper() == "BEARISH"

    call_prem_delta = np.where(
        call_mask & bullish_mask,
        weighted_cost,
        np.where(call_mask & bearish_mask, -weighted_cost, 0.0),
    )
    put_prem_delta = np.where(
        put_mask & bullish_mask,
        weighted_cost,
        np.where(put_mask & bearish_mask, -weighted_cost, 0.0),
    )
    call_ask = np.where(call_mask & bullish_mask, weighted_volume, 0.0)
    call_bid = np.where(call_mask & bearish_mask, weighted_volume, 0.0)
    put_ask = np.where(put_mask & bullish_mask, weighted_volume, 0.0)
    put_bid = np.where(put_mask & bearish_mask, weighted_volume, 0.0)

    agg_df = pd.DataFrame(
        {
            "datetime": df["datetime"],
            "call_prem_delta": call_prem_delta,
            "put_prem_delta": put_prem_delta,
            "call_ask": call_ask,
            "call_bid": call_bid,
            "put_ask": put_ask,
            "put_bid": put_bid,
        }
    )

    grouped = agg_df.groupby("datetime", sort=True).sum()
    if grouped.empty:
        return []

    grouped["net_call_premium"] = grouped["call_prem_delta"].cumsum()
    grouped["net_put_premium"] = grouped["put_prem_delta"].cumsum()
    grouped["cum_call_ask"] = grouped["call_ask"].cumsum()
    grouped["cum_call_bid"] = grouped["call_bid"].cumsum()
    grouped["cum_put_ask"] = grouped["put_ask"].cumsum()
    grouped["cum_put_bid"] = grouped["put_bid"].cumsum()

    grouped["call_volume"] = grouped["cum_call_ask"] + grouped["cum_call_bid"]
    grouped["put_volume"] = grouped["cum_put_ask"] + grouped["cum_put_bid"]
    grouped["net_volume"] = (grouped["cum_call_ask"] - grouped["cum_call_bid"]) - (
        grouped["cum_put_ask"] - grouped["cum_put_bid"]
    )

    result_df = grouped[
        ["net_call_premium", "net_put_premium", "call_volume", "put_volume", "net_volume"]
    ].apply(np.rint).astype(int)
    result_df.reset_index(inplace=True)
    result_df["time"] = result_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result_df.drop(columns=["datetime"], inplace=True)

    records = result_df[
        ["time", "net_call_premium", "net_put_premium", "call_volume", "put_volume", "net_volume"]
    ].to_dict(orient="records")

    if not records:
        return []

    price_list = asyncio.run(get_stock_chart_data(sector_ticker))
    if not price_list:
        try:
            with open(f"json/one-day-price/{sector_ticker}.json", "rb") as file:
                price_list = orjson.loads(file.read())
        except FileNotFoundError:
            price_list = []

    data = add_close_to_data(price_list, records)

    if data:
        try:
            last_time = datetime.strptime(data[-1]["time"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            return data

        end_time = last_time.replace(hour=16, minute=1, second=0)
        fields = ["net_call_premium", "net_put_premium", "call_volume", "put_volume", "net_volume", "close"]
        while last_time < end_time:
            last_time += timedelta(minutes=1)
            data.append(
                {
                    "time": last_time.strftime("%Y-%m-%d %H:%M:%S"),
                    **{field: None for field in fields},
                }
            )

    return data


def get_overview_data(
    sector_ticker: str,
    options_df: pd.DataFrame,
    holdings_weights: Dict[str, float],
    trading_date: str,
) -> Dict[str, Any]:
    tickers = [ticker.upper() for ticker in holdings_weights.keys()]
    if not tickers:
        return {
            "putVol": 0,
            "callVol": 0,
            "putOI": 0,
            "callOI": 0,
            "pcVol": 0,
            "pcOI": 0,
            "date": trading_date or datetime.today().strftime("%Y-%m-%d"),
        }

    df = options_df[options_df["ticker"].isin(tickers)].copy()
    if df.empty:
        return {
            "putVol": 0,
            "callVol": 0,
            "putOI": 0,
            "callOI": 0,
            "pcVol": 0,
            "pcOI": 0,
            "date": trading_date or datetime.today().strftime("%Y-%m-%d"),
        }

    if trading_date:
        df = df[df["date"] == trading_date].copy()
        if df.empty:
            return {
                "putVol": 0,
                "callVol": 0,
                "putOI": 0,
                "callOI": 0,
                "pcVol": 0,
                "pcOI": 0,
                "date": trading_date,
            }

    call_mask = df["put_call"].str.upper() == "CALLS"
    put_mask = df["put_call"].str.upper() == "PUTS"

    total_call_size = float(df.loc[call_mask, "volume_value"].sum())
    total_put_size = float(df.loc[put_mask, "volume_value"].sum())
    total_call_oi = float(df.loc[call_mask, "open_interest_value"].sum())
    total_put_oi = float(df.loc[put_mask, "open_interest_value"].sum())

    put_call_size_ratio = total_put_size / total_call_size if total_call_size > 0 else 0
    put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    return {
        "putVol": int(round(total_put_size)),
        "callVol": int(round(total_call_size)),
        "putOI": int(round(total_put_oi)),
        "callOI": int(round(total_call_oi)),
        "pcVol": round(put_call_size_ratio, 2),
        "pcOI": round(put_call_oi_ratio, 2),
        "date": trading_date or datetime.today().strftime("%Y-%m-%d"),
    }


def get_30_day_average_data(
    sector_ticker: str,
    holdings_tickers: List[str],
    days: int = 30,
) -> Dict[str, int]:
    if not holdings_tickers:
        return {"avg30Vol": 0, "avg30OI": 0}

    end_date = datetime.now()
    date_list = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    daily_totals: List[Dict[str, float]] = []

    for date_str in date_list:
        file_path = f"json/options-historical-data/flow-data/{date_str}.json"
        try:
            with open(file_path, "rb") as file:
                daily_data = orjson.loads(file.read())
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"Error reading file {file_path}: {exc}")
            continue

        if not daily_data:
            continue

        daily_df = pd.DataFrame(daily_data)
        if daily_df.empty or "ticker" not in daily_df.columns:
            continue

        daily_df["ticker"] = daily_df["ticker"].astype(str).str.upper()
        daily_df = daily_df[daily_df["ticker"].isin(holdings_tickers)]
        if daily_df.empty:
            continue

        volume_series = pd.to_numeric(daily_df.get("volume", 0), errors="coerce").fillna(0.0)
        oi_series = pd.to_numeric(daily_df.get("open_interest", 0), errors="coerce").fillna(0.0)

        daily_totals.append(
            {
                "total_size": float(volume_series.sum()),
                "total_oi": float(oi_series.sum()),
            }
        )

    if not daily_totals:
        return {"avg30Vol": 0, "avg30OI": 0}

    totals_df = pd.DataFrame(daily_totals)
    return {
        "avg30Vol": int(round(totals_df["total_size"].mean())),
        "avg30OI": int(round(totals_df["total_oi"].mean())),
    }


def get_sector_flow_analysis(
    options_df: pd.DataFrame,
    sector_map: Dict[str, str],
    trading_date: str,
) -> List[Dict[str, int]]:
    sector_list = [
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ]

    if options_df.empty or not sector_map:
        return []

    df = options_df.copy()
    if trading_date:
        df = df[df["date"] == trading_date].copy()
    df["sector"] = df["ticker"].map(sector_map)
    df = df[df["sector"].isin(sector_list)]
    if df.empty:
        return []

    grouped = (
        df.groupby(["sector", df["put_call"].str.upper()])["cost_value"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(sector_list, fill_value=0.0)
    )

    results: List[Dict[str, int]] = []
    for sector in sector_list:
        row = grouped.loc[sector]
        call_premium = float(row.get("CALLS", 0.0))
        put_premium = float(row.get("PUTS", 0.0))
        results.append(
            {
                "sector": sector,
                "callPrem": int(round(call_premium)),
                "putPrem": int(round(put_premium)),
                "totalPremium": int(round(call_premium + put_premium)),
            }
        )

    results.sort(key=lambda item: item["totalPremium"], reverse=True)
    return results


def get_market_flow():
    options_df = load_options_flow_dataframe()
    if options_df.empty:
        print("No options flow data available to generate market flow.")
        return

    trading_date = str(options_df["date"].max())

    holdings_weights = load_holdings_weights("SPY")
    market_tide = get_sector_data("SPY", options_df, holdings_weights, trading_date)
    overview = get_overview_data("SPY", options_df, holdings_weights, trading_date)
    avg_30_day = get_30_day_average_data("SPY", list(holdings_weights.keys()))

    sp500_tickers = load_sp500_tickers()
    sector_map = load_sector_map(sp500_tickers)
    sector_flow = get_sector_flow_analysis(options_df, sector_map, trading_date)

    if market_tide:
        try:
            date_obj = datetime.strptime(market_tide[0]["time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            date_obj = datetime.strptime(trading_date, "%Y-%m-%d")
    else:
        date_obj = datetime.strptime(trading_date, "%Y-%m-%d")

    data = {
        "date": date_obj.strftime("%b %d, %Y"),
        "marketTide": market_tide,
        "overview": {**overview, **avg_30_day},
        "sectorFlow": sector_flow,
    }
    save_json(data, "data")

    

if __name__ == '__main__':
    get_market_flow()
