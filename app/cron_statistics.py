from datetime import datetime, timedelta
import orjson
import time
import sqlite3
import asyncio
import aiohttp
import random
from tqdm import tqdm
from dotenv import load_dotenv
import os


stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}




async def run():
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    con.close()
    
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())