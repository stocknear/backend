#!/usr/bin/env python3
import aiofiles
import orjson
import sqlite3
import asyncio
import os
from datetime import datetime, timezone, timedelta
import email.utils  # For parsing RFC 2822 dates
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional, List, Dict, Any


def parse_date_to_dt(date_str: Optional[str]) -> Optional[datetime]:
    """
    Try multiple ways to parse a date string into an aware UTC datetime.
    Returns None if it cannot be parsed.
    """
    if not date_str:
        return None

    # Try RFC-2822 / email.utils first
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        if dt is not None:
            # Ensure timezone aware and convert to UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Try ISO formats (including trailing 'Z')
    try:
        iso = date_str
        if iso.endswith("Z"):
            iso = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Try common strptime fallbacks
    fmts = [
        "%Y-%m-%d %H:%M:%S",    # "2025-07-15 18:25:51"
        "%Y-%m-%d",             # "2025-07-15"
        "%a, %d %b %Y %H:%M:%S %z",  # RFC style explicitly
        "%d %b %Y %H:%M:%S %z",      # maybe missing weekday
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(date_str, fmt)
            # assume naive datetimes are UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue

    # Could not parse
    return None


def add_time_ago(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    for item in news_items:
        try:
            dt = parse_date_to_dt(item.get("date"))
            if dt is None:
                item["timeAgo"] = "N/A"
                continue

            diff = now_utc - dt
            minutes = int(diff.total_seconds() / 60)

            if minutes < 1:
                item["timeAgo"] = "1m"
            elif minutes < 60:
                item["timeAgo"] = f"{minutes}m"
            elif minutes < 1440:
                item["timeAgo"] = f"{minutes // 60}h"
            else:
                item["timeAgo"] = f"{minutes // 1440}D"
        except Exception:
            item["timeAgo"] = "N/A"
    return news_items


async def save_json(data: List[Dict[str, Any]]):
    path = "json/wiim/flow"
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "data.json")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


async def get_data(
    stock_symbols: List[str],
    etf_symbols: List[str],
    total_symbols: List[str],
    limit: int = 500,
) -> List[Dict[str, Any]]:
    res_list: List[Dict[str, Any]] = []

    for symbol in tqdm(total_symbols, desc="Processing symbols"):
        file_path = f"json/wiim/company/{symbol}.json"
        if not os.path.exists(file_path):
            continue

        # Read company news / items
        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
            data = orjson.loads(content)
            # Expecting a list of items; if single dict, wrap it
            if isinstance(data, dict):
                data = [data]
        except Exception as e:
            print(f"Error loading JSON for {symbol}: {e}")
            continue

        # Compute timeAgo for items (works with multiple formats)
        data = add_time_ago(data)

        for item in data:
            try:
                # Read quote data (optional)
                quote_path = f"json/quote/{symbol}.json"
                marketCap = None
                changesPercentage = None
                name = None
                if os.path.exists(quote_path):
                    try:
                        async with aiofiles.open(quote_path, "rb") as qf:
                            qcontent = await qf.read()
                        quote_data = orjson.loads(qcontent)
                        marketCap = quote_data.get("marketCap")
                        name = quote_data.get("name")
                        cp = quote_data.get("changesPercentage")
                        if cp is not None:
                            try:
                                changesPercentage = round(float(cp), 2)
                            except Exception:
                                changesPercentage = cp
                    except Exception:
                        # fail silently, keep None
                        pass

                assetType = "stocks" if symbol in stock_symbols else "etf"

                # Handle '-' or missing changesPercentage in item
                item_changes = item.get("changesPercentage")
                if item_changes in (None, "-", ""):
                    item_changes = changesPercentage
                else:
                    # try to normalize numeric values
                    try:
                        item_changes = round(float(item.get("changesPercentage")), 2)
                    except Exception:
                        pass

                # parse date into datetime for sorting
                dt = parse_date_to_dt(item.get("date"))
                dt_for_sort = dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)

                res_list.append({
                    "original_date": item.get("date"),
                    "date": item.get("date"),  # we'll reformat after sorting
                    "text": item.get("title", "") or item.get("text", "") or "",
                    "marketCap": marketCap,
                    "changesPercentage": item_changes,
                    "symbol": symbol,
                    "name": name,
                    "assetType": assetType,
                    "timeAgo": item.get("timeAgo", "N/A"),
                    "_dt": dt_for_sort,
                })

            except Exception as e:
                print(f"Error processing item for {symbol}: {e}")

    # Sort by datetime (newest first)
    res_list.sort(key=lambda x: x.get("_dt", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    # Convert date to YYYY-MM-DD (use parsed _dt where possible), remove internal _dt
    for item in res_list:
        dt_obj = item.get("_dt")
        if isinstance(dt_obj, datetime) and dt_obj != datetime.min.replace(tzinfo=timezone.utc):
            # format using UTC date
            item["date"] = dt_obj.astimezone(timezone.utc).strftime("%Y-%m-%d")
        else:
            # fallback: try to parse original_date and format, otherwise N/A
            parsed = parse_date_to_dt(item.get("original_date"))
            if parsed:
                item["date"] = parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")
            else:
                item["date"] = "N/A"
        # remove helper keys (keep original_date if you want - here we drop it)
        item.pop("_dt", None)
        item.pop("original_date", None)

    return res_list[:limit]


async def run(stock_symbols: List[str], etf_symbols: List[str], total_symbols: List[str]):
    data = await get_data(stock_symbols, etf_symbols, total_symbols)
    if data:
        print(f"Processed {len(data)} news items")
        await save_json(data)
    else:
        print("No news items processed.")


def get_symbols():
    stock_symbols, etf_symbols = [], []
    try:
        with sqlite3.connect("stocks.db") as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol FROM stocks")
            stock_symbols = [row[0] for row in cursor.fetchall()]

        with sqlite3.connect("etf.db") as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol FROM etfs")
            etf_symbols = [row[0] for row in cursor.fetchall()]

    except Exception as e:
        print("DB error:", e)

    return stock_symbols, etf_symbols


if __name__ == "__main__":
    stock_symbols, etf_symbols = get_symbols()
    total_symbols = stock_symbols + etf_symbols

    # run the main async flow
    asyncio.run(run(stock_symbols, etf_symbols, total_symbols))
