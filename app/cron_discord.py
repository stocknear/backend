import discord
import re
import sqlite3
from contextlib import contextmanager

STOCK_DB = 'stocks'
ETF_DB = 'etf'
CRYPTO_DB = 'crypto'


@contextmanager
def db_connection(db_name):
    conn = sqlite3.connect(f'{db_name}.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    try:
        yield cursor
    finally:
        conn.commit()
        cursor.close()
        conn.close()

#------Start Stocks DB------------#
with db_connection(STOCK_DB) as cursor:
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    symbols = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT symbol FROM stocks")
    raw_data = cursor.fetchall()
    stock_list_data = [{
        'symbol': row[0],
        'type': 'stocks',
    } for row in raw_data]
#------End Stocks DB------------#

#------Start ETF DB------------#
with db_connection(ETF_DB) as cursor:
    cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT symbol FROM etfs")
    raw_data = cursor.fetchall()
    etf_list_data = [{
        'symbol': row[0],
        'type': 'etf',
    } for row in raw_data]
#------End ETF DB------------#

#------Start Crypto DB------------#
with db_connection(CRYPTO_DB) as cursor:
    cursor.execute("SELECT DISTINCT symbol FROM cryptos")
    crypto_symbols = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT symbol FROM cryptos")
    raw_data = cursor.fetchall()
    crypto_list_data = [{
        'symbol': row[0],
        'type': 'crypto',
    } for row in raw_data]
#------End Crypto DB------------#

#------Init Searchbar Data------------#
searchbar_data = stock_list_data + etf_list_data + crypto_list_data

# Replace with your bot token
TOKEN = 'token'

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Pattern to match $stockname format (e.g., $AAPL, $TSLA)
ticker_pattern = re.compile(r"\$(\w+)")


@client.event
async def on_ready():
    print(f'Logged in as {client.user}!')


@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Copy of original message content to modify
    modified_content = message.content

    # Find all tickers in the message
    tickers = ticker_pattern.findall(message.content)

    # If no tickers found, exit
    if not tickers:
        return

    for ticker in tickers:
        # Find the corresponding symbol in the searchbar_data
        matched = next((item for item in searchbar_data if item['symbol'].upper() == ticker.upper()), None)
        
        if matched:
            symbol_type = matched['type']
            # Construct the URL based on the symbol type (stock, etf, crypto)
            if symbol_type == 'stocks':
                stock_url = f"https://stocknear.com/stocks/{ticker.upper()}"
            elif symbol_type == 'etf':
                stock_url = f"https://stocknear.com/etf/{ticker.upper()}"
            elif symbol_type == 'crypto':
                stock_url = f"https://stocknear.com/crypto/{ticker.upper()}"
            
            # Replace the ticker in the content with a markdown hyperlink
            modified_content = modified_content.replace(f"${ticker}", f"[${ticker}]({stock_url})")

    # Edit the original message with the modified content
    await message.edit(content=modified_content)

# Run the bot
client.run(TOKEN)
