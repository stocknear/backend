from datetime import date, datetime, timedelta, time
import ujson
import sqlite3
import pandas as pd
import asyncio
import aiohttp
import pytz

from GetStartEndDate import GetStartEndDate

#Update Market Movers Price, ChangesPercentage, Volume and MarketCap regularly
berlin_tz = pytz.timezone('Europe/Berlin')

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('FMP_API_KEY')





async def get_todays_data(ticker):

    current_weekday = datetime.today().weekday()
    current_time_berlin = datetime.now(berlin_tz)
    is_afternoon = current_time_berlin.hour > 15 or (current_time_berlin.hour == 15 and current_time_berlin.minute >= 30)

    start_date_1d, end_date_1d = GetStartEndDate().run()

    print(start_date_1d, end_date_1d)
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?from={start_date_1d}&to={end_date_1d}&apikey={api_key}"

    df_1d = pd.DataFrame()

    current_date = start_date_1d
    target_time = time(15,30)
    extract_date = current_date.strftime('%Y-%m-%d')

    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(session.get(url))

        for response in responses:
            try:
                json_data = await response.json()
                df_1d = pd.DataFrame(json_data).iloc[::-1].reset_index(drop=True)
                opening_price = df_1d['open'].iloc[0]
                df_1d = df_1d.drop(['open', 'high', 'low', 'volume'], axis=1)
                df_1d = df_1d.round(2).rename(columns={"date": "time", "close": "value"})

                if current_weekday == 5 or current_weekday == 6:
                    pass
                else:
                    if current_date.time() < target_time:
                        pass                    
                    else:
                        end_time = pd.to_datetime(f'{extract_date} 16:00:00')
                        new_index = pd.date_range(start=df_1d['time'].iloc[-1], end=end_time, freq='1min')
                        
                        remaining_df = pd.DataFrame(index=new_index, columns=['value'])
                        remaining_df = remaining_df.reset_index().rename(columns={"index": "time"})
                        remaining_df['time'] = remaining_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        remainind_df = remaining_df.set_index('time')

                        df_1d = pd.concat([df_1d, remaining_df[1:: ]])
                        #To-do FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
    
                df_1d = ujson.loads(df_1d.to_json(orient="records"))
            except:
                df_1d = []
    return df_1d

async def get_jsonparsed_data(session, url):
    async with session.get(url) as response:
        data = await response.json()
        return data

async def get_quote_of_stocks(ticker_list):
    ticker_str = ','.join(ticker_list)
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/quote/{ticker_str}?apikey={api_key}" 
        async with session.get(url) as response:
            df = await response.json()
    return df


async def get_gainer_loser_active_stocks():

    #Database read 1y and 3y data
    query_fundamental_template = """
        SELECT 
            marketCap
        FROM 
            stocks 
        WHERE
            symbol = ?
    """

    query_template = """
        SELECT
            volume
        FROM
            "{ticker}"
        ORDER BY
            rowid DESC
        LIMIT 1
    """

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        gainer_url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={api_key}"
        loser_url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={api_key}"
        active_url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={api_key}"

        # Gather all the HTTP requests concurrently
        tasks = [
            get_jsonparsed_data(session, gainer_url),
            get_jsonparsed_data(session, loser_url),
            get_jsonparsed_data(session, active_url)
        ]

        gainer_json, loser_json, active_json = await asyncio.gather(*tasks)



        gainer_json = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in gainer_json]
        gainer_json = [entry for entry in gainer_json if entry]

        loser_json = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in loser_json]
        loser_json = [entry for entry in loser_json if entry]

        active_json = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in active_json]
        active_json = [entry for entry in active_json if entry]
        for entry in active_json:
            try:
                symbol = entry['symbol']
                query = query_template.format(ticker=symbol)
                fundamental_data = pd.read_sql_query(query_fundamental_template, con, params=(symbol,))
                volume = pd.read_sql_query(query, con)
                entry['marketCap'] = int(fundamental_data['marketCap'].iloc[0])
                entry['volume'] = int(volume['volume'].iloc[0])
            except:
                entry['marketCap'] = None
                entry['volume'] = None

        active_json = sorted(active_json, key=lambda x: (x['marketCap'] >= 10**9, x['volume']), reverse=True)


        stocks = gainer_json[:20] + loser_json[:20] + active_json[:20]

        #remove change key element
        stocks = [{k: v for k, v in stock.items() if k != "change"} for stock in stocks]


        for entry in stocks:
            try:
                symbol = entry['symbol']
                query = query_template.format(ticker=symbol)
                fundamental_data = pd.read_sql_query(query_fundamental_template, con, params=(symbol,))
                volume = pd.read_sql_query(query, con)
                entry['marketCap'] = int(fundamental_data['marketCap'].iloc[0])
                entry['volume'] = int(volume['volume'].iloc[0])
            except:
                entry['marketCap'] = None
                entry['volume'] = None

      
        day_gainer_json = stocks[:20]
        day_loser_json = stocks[20:40]
        day_active_json = stocks[40:60]

        query_market_movers = """
            SELECT 
                gainer,loser,most_active
            FROM 
                market_movers 
        """
        past_gainer = pd.read_sql_query(query_market_movers, con)

        gainer_json = eval(past_gainer['gainer'].iloc[0])
        loser_json = eval(past_gainer['loser'].iloc[0])
        active_json = eval(past_gainer['most_active'].iloc[0])

        gainer_json['1D'] = day_gainer_json
        loser_json['1D'] = day_loser_json
        active_json['1D'] = day_active_json #sorted(day_active_json, key=lambda x: x.get('volume', 0) if x.get('volume') is not None else 0, reverse=True)
        
    
    data = {'gainers': gainer_json, 'losers': loser_json, 'active': active_json}
    #Extract all unique symbols from gainer,loser, active
    unique_symbols = set()

    # Iterate through time periods, categories, and symbols
    for time_period in data.keys():
        for category in data[time_period].keys():
            for stock_data in data[time_period][category]:
                symbol = stock_data["symbol"]
                unique_symbols.add(symbol)

    # Convert the set to a list if needed
    unique_symbols_list = list(unique_symbols)

    #Get the latest quote of all unique symbol and map it back to the original data list to update all values

    latest_quote = await get_quote_of_stocks(unique_symbols_list)
    # Updating values in the data list based on matching symbols from the quote list
    for time_period in data.keys():
        for category in data[time_period].keys():
            for stock_data in data[time_period][category]:
                symbol = stock_data["symbol"]
                quote_stock = next((item for item in latest_quote if item["symbol"] == symbol), None)
                if quote_stock:
                    stock_data['price'] = quote_stock['price']
                    stock_data['changesPercentage'] = quote_stock['changesPercentage']
                    stock_data['marketCap'] = quote_stock['marketCap']
                    stock_data['volume'] = quote_stock['volume']


    return data 



async def get_historical_data():
    res_list = []
    ticker_list = ['SPY', 'QQQ', 'DIA', 'IWM', 'IVV']
    latest_quote = await get_quote_of_stocks(ticker_list)

    for quote in latest_quote:
        ticker = quote['symbol']
        df = await get_todays_data(ticker)
        res_list.append({'symbol': ticker, 'priceData': df, 'changesPercentage': round(quote['changesPercentage'],2), 'previousClose': round(quote['previousClose'],2)})

    return res_list


try:
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    symbols = [row[0] for row in cursor.fetchall()]

    data = asyncio.run(get_historical_data())
    with open(f"json/mini-plots-index/data.json", 'w') as file:
        ujson.dump(data, file)

    data = asyncio.run(get_gainer_loser_active_stocks())
    with open(f"json/market-movers/data.json", 'w') as file:
        ujson.dump(data, file)
    con.close()
except Exception as e:
    print(e)