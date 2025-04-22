import orjson
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
import pandas as pd
from tqdm import tqdm

async def save_json(symbol, data):
    with open(f"json/similar-stocks/{symbol}.json", 'wb') as file:
        file.write(orjson.dumps(data))

# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

query_template = """
    SELECT 
        stock_peers
    FROM 
        stocks 
    WHERE
        symbol = ?
"""

async def get_data(symbol):
    """Extract specified columns data for a given symbol."""
    columns = ['dividendYield', 'employees', 'marketCap','relativeFTD','name','revenue','shortFloatPercent']
    
    if symbol in stock_screener_data_dict:
        result = {}
        for column in columns:
            try:
                result[column] = stock_screener_data_dict[symbol][column]
            except:
                pass
        return result
    return {}

async def run():
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    total_symbols = [row[0] for row in cursor.fetchall()]
    #total_symbols = ['MRIN']  # For testing purposes
    
    for ticker in tqdm(total_symbols):
        # Get peers for the current ticker
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        try:
            # Get the list of peer stocks
            peers = orjson.loads(df['stock_peers'].iloc[0])
            # Create a list to store peer data
            peer_data_list = []
            
            # Process each peer
            for peer_symbol in peers:
                # Get additional data for this peer
                data = await get_data(peer_symbol)
                
                # Combine symbol with additional data
                peer_info = {
                    'symbol': peer_symbol,
                    **data
                }
                peer_data_list.append(peer_info)
            
            # Sort by marketCap if available
            sorted_peers = sorted(
                peer_data_list,
                key=lambda x: x.get('marketCap', 0) or 0,
                reverse=True
            )
            
            # Save the results
            if sorted_peers:
                await save_json(ticker, sorted_peers)
            
        except:
            pass

if __name__ == "__main__":
    try:
        con = sqlite3.connect('stocks.db')
        asyncio.run(run())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'con' in locals():
            con.close()