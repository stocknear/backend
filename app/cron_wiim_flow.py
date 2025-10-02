import aiohttp
import aiofiles
import orjson
import sqlite3
import pandas as pd
import asyncio
import pytz
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, date, timezone
import email.utils  # for parsing RFC 2822 dates like "Fri, 30 May 2025 16:01:09 -0400"

headers = {"accept": "application/json"}

load_dotenv()
API_KEY = os.getenv("BENZINGA_API_KEY")


def add_time_ago(news_items):
    now_utc = datetime.now(timezone.utc)
    for item in news_items:
        created_dt = email.utils.parsedate_to_datetime(item["created"]).astimezone(timezone.utc)
        diff = now_utc - created_dt
        minutes = int(diff.total_seconds() / 60)

        if minutes < 1:
            item["timeAgo"] = "1m"
        elif minutes < 60:
            item["timeAgo"] = f"{minutes}m"
        elif minutes < 1440:
            hours = minutes // 60
            item["timeAgo"] = f"{hours}h"
        else:
            days = minutes // 1440
            item["timeAgo"] = f"{days}D"

    return news_items


async def save_json(data):
    path = "json/wiim/flow"
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "data.json")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


async def get_data():
    url = "https://api.benzinga.com/api/v2/news"
    querystring = {
        "token": API_KEY,
        "pageSize": "1000",
        "displayOutput": "headline",
        "sort": "updated:desc",
        "channels": "wiim",
    }

    res_list = []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=querystring, headers=headers) as response:
                if response.status != 200:
                    return []

                data = orjson.loads(await response.read())  # use .read() → bytes
                data = add_time_ago(data)

                for item in data:
                    try:
                        item["ticker"] = item["stocks"][0].get("name", None).replace("/", "-")

                        # read quote data
                        quote_path = f"json/quote/{item['ticker']}.json"
                        if os.path.exists(quote_path):
                            async with aiofiles.open(quote_path, "rb") as f:
                                quote_data = orjson.loads(await f.read())
                                item["marketCap"] = quote_data.get("marketCap", None)
                                #item["name"] = quote_data.get("name", None)
                                item["changesPercentage"] = round(quote_data.get("changesPercentage", None),2)

                        item["assetType"] = "stocks" if item["ticker"] in stock_symbols else "etf"

                        res_list.append(
                            {
                                "date": item["created"],
                                "text": item["title"],
                                "marketCap": item.get("marketCap"),
                                "changesPercentage": item.get("changesPercentage"),
                                "ticker": item["ticker"],
                                "assetType": item["assetType"],
                                "timeAgo": item["timeAgo"],
                            }
                        )
                    except Exception:
                        pass
    except Exception as e:
        print("Error in get_data:", e)
        return []

    # sort results
    res_list = sorted(
        res_list,
        key=lambda item: datetime.strptime(item["date"], "%a, %d %b %Y %H:%M:%S %z"),
        reverse=True,
    )

    # convert date
    for item in res_list:
        dt = datetime.strptime(item["date"], "%a, %d %b %Y %H:%M:%S %z")
        item["date"] = dt.strftime("%Y-%m-%d")

    return res_list[:150]


async def run():
    data = await get_data()
    if data:
        await save_json(data)


try:
    con = sqlite3.connect("stocks.db")
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    asyncio.run(run())

except Exception as e:
    print("DB/Run error:", e)
