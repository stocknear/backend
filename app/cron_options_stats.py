from __future__ import print_function
import asyncio
import time
from datetime import datetime, timedelta
import orjson
from tqdm import tqdm
import sqlite3
from dotenv import load_dotenv
import os
import re
from statistics import mean


# Database connection and symbol retrieval
def get_total_symbols():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        stocks_symbols = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect('etf.db') as etf_con:
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

    return stocks_symbols + etf_symbols


def save_json(data, symbol):
    directory = "json/options-stats/companies"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))


def safe_round(value):
    try:
        return round(float(value), 2)
    except (ValueError, TypeError):
        return value


async def main():
    
    with open(f"json/options-flow/feed/data.json", "r") as file:
        data = orjson.loads(file.read())

    total_symbols = get_total_symbols()
    
    for symbol in tqdm(total_symbols):
        try:
            call_premium = 0
            put_premium = 0
            call_open_interest = 0
            put_open_interest = 0
            call_volume = 0
            put_volume = 0
            bearish_premium = 0
            bullish_premium = 0
            neutral_premium = 0

            net_call_premium = 0
            net_put_premium = 0
            net_premium = 0
            
            for item in data:
                if item['ticker'] == symbol:
                    if item['put_call'] == 'Calls':
                        call_premium += item['cost_basis']
                        call_open_interest += int(item['open_interest'])
                        call_volume += int(item['volume'])
                    elif item['put_call'] == 'Puts':
                        put_premium += item['cost_basis']
                        put_open_interest += int(item['open_interest'])
                        put_volume += int(item['volume'])

                    if item['sentiment'] == 'Bullish':
                        bullish_premium +=item['cost_basis']
                        if item['put_call'] == 'Calls':
                            net_call_premium +=item['cost_basis']
                        elif item['put_call'] == 'Puts':
                            net_put_premium +=item['cost_basis']
                    
                    if item['sentiment'] == 'Bearish':
                        bearish_premium +=item['cost_basis']
                        if item['put_call'] == 'Calls':
                            net_call_premium -=item['cost_basis']
                        elif item['put_call'] == 'Puts':
                            net_put_premium -=item['cost_basis']

                    if item['sentiment'] == 'Neutral':
                        neutral_premium +=item['cost_basis']

            with open(f"json/options-historical-data/companies/{symbol}.json", "r") as file:
                past_data = orjson.loads(file.read())[0]
                previous_open_interest = past_data['total_open_interest']
                iv_rank = past_data['iv_rank']
                iv = past_data['iv']

            total_open_interest = call_open_interest+put_open_interest

            changesPercentageOI = round((total_open_interest/previous_open_interest-1)*100, 2) if previous_open_interest > 0 else 0
            changeOI = total_open_interest - previous_open_interest
            put_call_ratio = round(put_volume/call_volume,2) if call_volume > 0 else 0

            net_premium = net_call_premium + net_put_premium
            premium_ratio = [
                safe_round(bearish_premium),
                safe_round(neutral_premium),
                safe_round(bullish_premium)
            ]
            aggregate = {
                "call_premium": round(call_premium,0),
                "call_open_interest": round(call_open_interest,0),
                "call_volume": round(call_volume,0),
                "put_premium": round(put_premium,0),
                "put_open_interest": round(put_open_interest,0),
                "put_volume": round(put_volume,0),
                "putCallRatio": round(put_volume/call_volume,0),
                "total_open_interest": round(total_open_interest,0),
                "changeOI": round(changeOI,0),
                "changesPercentageOI": changesPercentageOI,
                "iv": round(iv,2),
                "iv_rank": round(iv_rank,2),
                "putCallRatio": put_call_ratio,
                "premium_ratio": premium_ratio,
                "net_premium": round(net_premium),
            }

            if aggregate:
                save_json(aggregate, symbol)

        except:
            pass
if __name__ == "__main__":
    asyncio.run(main())
