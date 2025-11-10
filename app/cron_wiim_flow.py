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
    temp_list: List[Dict[str, Any]] = []

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

    
                temp_list.append({
                    "date": item.get("date"),
                    "text": item.get("title", "") or item.get("text", "") or "",
                    "marketCap": marketCap,
                    "changesPercentage": item_changes,
                    "symbol": symbol,
                    "name": name,
                    "assetType": assetType,
                })

            except Exception as e:
                print(f"Error processing item for {symbol}: {e}")

    # Group by text and date to combine symbols with identical text
    grouped_data: Dict[tuple, Dict[str, Any]] = {}
    for item in temp_list:
        key = (item["text"], item["date"])
        if key in grouped_data:
            grouped_data[key]["symbolList"].append(item["symbol"])
        else:
            grouped_data[key] = {
                "date": item["date"],
                "text": item["text"],
                "marketCap": item["marketCap"],
                "changesPercentage": item["changesPercentage"],
                "symbolList": [item["symbol"]],
                "name": item["name"],
                "assetType": item["assetType"],
            }
    
    res_list = list(grouped_data.values())
    res_list = sorted(res_list,key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M:%S"),reverse=True)

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
