import orjson
import sqlite3
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FMP_API_KEY')


# Load stock screener data
with open(f"json/stock-screener/data.json", 'rb') as file:
    stock_screener_data = orjson.loads(file.read())
stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}


query_etf_holding = f"SELECT holding from etfs WHERE symbol = ?"
quote_cache = {}

async def save_json(category, data, category_type='market-cap'):
    with open(f"json/{category_type}/list/{category}.json", 'wb') as file:
        file.write(orjson.dumps(data))

async def get_quote_data(symbol):
    """Get quote data for a symbol from JSON file"""
    if symbol in quote_cache:
        return quote_cache[symbol]
    else:
        try:
            with open(f"json/quote/{symbol}.json") as file:
                quote_data = orjson.loads(file.read())
                quote_cache[symbol] = quote_data  # Cache the loaded data
                return quote_data
        except:
            return None

async def process_category(cursor, category, condition, category_type='market-cap'):
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.sector
        FROM stocks s 
        WHERE {}
    """
    
    full_query = base_query.format(condition)
    cursor.execute(full_query)
    raw_data = cursor.fetchall()
    
    res_list = []
    for row in raw_data:
        try:
            symbol = row[0]
            quote_data = await get_quote_data(symbol)
            if quote_data:
                item = {
                    'symbol': symbol,
                    'name': row[1],
                    'price': round(quote_data.get('price'), 2) if quote_data.get('price') is not None else None,
                    'changesPercentage': round(quote_data.get('changesPercentage'), 2) if quote_data.get('changesPercentage') is not None else None,
                    'marketCap': quote_data.get('marketCap', 0),
                    'revenue': 0,
                }
                
                # Add screener data if available
                if symbol in stock_screener_data_dict:
                    item['revenue'] = stock_screener_data_dict[symbol].get('revenue',0)
                
                if item['marketCap'] > 0 and item['revenue'] > 0:
                    res_list.append(item)
        except:
            pass
    
    # Sort by market cap and save
    sorted_result = sorted(res_list, key=lambda x: x['marketCap'] if x['marketCap'] else 0, reverse=True)
    # Add rank to each item
    for rank, item in enumerate(sorted_result, 1):
        item['rank'] = rank

    await save_json(category, sorted_result, category_type)
    print(f"Processed and saved {len(sorted_result)} stocks for {category}")
    return sorted_result



async def get_etf_holding():
    # Create a connection to the ETF database
    etf_con = sqlite3.connect('etf.db')
    etf_cursor = etf_con.cursor()
    etf_cursor.execute("PRAGMA journal_mode = wal")
    
    # Fetch distinct ETF symbols
    etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
    etf_symbols = [row[0] for row in etf_cursor.fetchall()]
    
    for ticker in etf_symbols:
        res = []
        df = pd.read_sql_query(query_etf_holding, etf_con, params=(ticker,))
        try:
            # Load holdings data from the SQL query result
            data = orjson.loads(df['holding'].iloc[0])
            last_update = data[0]['updated'][0:10]
            
            # Prepare initial holdings
            raw_res = [
                {
                    'symbol': item.get('asset', None),
                    'name': item.get('name', None).capitalize() if item.get('name') else None,
                    'weightPercentage': item.get('weightPercentage', None),
                    'sharesNumber': (
                        item.get('marketValue', None)
                        if not item.get('asset') and item.get('sharesNumber') == 0
                        else item.get('sharesNumber', None)
                    )
                }
                for item in data
                if item.get('marketValue', 0) >= 0 and item.get('weightPercentage', 0) > 0
            ]

            for item in raw_res:
                try:
                    symbol = item['symbol']

                    # Adjustments for ticker = 'IBIT'
                    if ticker == 'IBIT' and symbol == 'BTC':
                        item['symbol'] = 'BTCUSD'
                        item['name'] = 'Bitcoin'

                    quote_data = await get_quote_data(item['symbol'])

                    if quote_data:
                        price = quote_data.get('price')
                        changes = quote_data.get('changesPercentage')

                        # Skip the whole element if price is None
                        if price is None or changes is None:
                            continue  

                        # Otherwise add valid fields
                        item['price'] = round(price, 2)
                        if changes is not None:
                            item['changesPercentage'] = round(changes, 2)

                        if quote_data.get('name'):
                            item['name'] = quote_data['name']

                        # Round weightPercentage if available
                        item['weightPercentage'] = (
                            round(item.get('weightPercentage'), 2)
                            if item['weightPercentage']
                            else None
                        )

                        res.append(item)

                except Exception:
                    pass

        except Exception:
            last_update = None
            res = []

        # Save results to a file if there's data to write
        if res:
            for rank, item in enumerate(res, 1):
                item['rank'] = rank
            with open(f"json/etf/holding/{ticker}.json", 'wb') as file:
                final_res = {'lastUpdate': last_update, 'holdings': res}
                file.write(orjson.dumps(final_res))
    
    # Close the database connection
    etf_con.close()




async def get_etf_provider():
    # Create a connection to the ETF database
    etf_con = sqlite3.connect('etf.db')
    cursor = etf_con.cursor()
    cursor.execute("SELECT DISTINCT etfProvider FROM etfs")
    etf_provider = [row[0] for row in cursor.fetchall()]
    
    query = "SELECT symbol, name, expenseRatio, totalAssets, numberOfHoldings FROM etfs WHERE etfProvider = ?"
    
    for provider in etf_provider:
        try:
            cursor.execute(query, (provider,))
            raw_data = cursor.fetchall()
            # Extract only relevant data and filter only integer or float totalAssets
            res = [
                {'symbol': row[0], 'name': row[1], 'expenseRatio': row[2], 'totalAssets': row[3], 'numberOfHoldings': row[4]}
                for row in raw_data if isinstance(row[3], float) or isinstance(row[3], int)
            ]
            for item in res:
                try:
                    symbol = item['symbol']
                    with open(f"json/quote/{symbol}.json") as file:
                        quote_data = orjson.loads(file.read())
                    # Assign price and changesPercentage if available, otherwise set to None
                    item['price'] = round(quote_data.get('price'), 2) if quote_data else None
                    item['changesPercentage'] = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                    item['name'] = quote_data.get('name') if quote_data else None
                except Exception:
                    pass

            sorted_res = sorted(res, key=lambda x: x['totalAssets'], reverse=True)

            # Save results to a file if there's data to write
            if sorted_res:
                with open(f"json/etf/provider/{provider}.json", 'wb') as file:
                    file.write(orjson.dumps(sorted_res))
        except Exception as e:
            print(e)
            pass
    
    # Close the cursor and connection
    cursor.close()
    etf_con.close()



async def generate_stock_list(symbol_list, output_file):
    """
    Generate a stock list for the given symbols and save it as a JSON file.
    
    :param symbol_list: List of stock symbols.
    :param output_file: Path to save the resulting JSON file.
    """
    res_list = []
    for symbol in symbol_list:
        try:
            # Get revenue data
            revenue = stock_screener_data_dict.get(symbol, {}).get('revenue', None)

            # Load quote data from file
            try:
                with open(f"json/quote/{symbol}.json") as file:
                    quote_data = orjson.loads(file.read())
            except FileNotFoundError:
                quote_data = None

            # Extract data from quote_data
            price = round(quote_data.get('price', None), 2) if quote_data else None
            changesPercentage = round(quote_data.get('changesPercentage', None), 2) if quote_data else None
            marketCap = quote_data.get('marketCap', None) if quote_data else None
            name = quote_data.get('name', None) if quote_data else None

            # Append to result list
            res_list.append({
                'symbol': symbol,
                'name': name,
                'price': price,
                'changesPercentage': changesPercentage,
                'marketCap': marketCap,
                'revenue': revenue
            })

        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")

    if res_list:
        # Sort by market cap and assign ranks
        res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Save the resulting list to the output file
        with open(output_file, 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_ai_stocks():
    symbol_list = [
        "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSM", "ORCL", "SAP",
        "ASML", "ACN", "NOW", "ISRG", "IBM", "AMD", "ADBE", "PLTR", "ARM", "ANET",
        "PANW", "MRVL", "KLAC", "CRWD", "SNPS", "WDAY", "TEAM", "TTD", "SNOW",
        "NXPI", "IRM", "ROK", "BIDU", "SPLK", "TER", "ALAB", "SYM", "TWLO", "EPAM",
        "TTEK", "PATH", "CGNX", "UPST", "TEM", "SOUN", "AVAV", "AI", "AMBA", "SPT",
        "RXRX", "HOLI", "SSTK", "BBAI", "EXAI", "PDYN", "IRBT", "AISP", "REKR",
        "VICR", "OSS", "KSCP", "MDAI", "NTC", "GFAI", "KITT", "OTRK"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/ai-stocks.json")

async def get_clean_energy():
    symbol_list = [
        "NEE", "ED", "FSLR", "BEP", "ENPH", "SMR", "BE", "CWEN.A", "BEPC", "CWEN",
        "FLNC", "RNW", "PLUG", "RUN", "ENLT", "DQ", "AMRC", "JKS", "ARRY", "SEDG",
        "SHLS", "REX", "GPRE", "OPAL", "GEVO", "NOVA", "ELLO", "AMTX", "MAXN",
        "SOL", "SMXT", "GWH", "TURB", "CETY", "ADN", "DFLI", "VVPR"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/clean-energy.json")

async def get_esports():
    symbol_list = ['MSFT','SE','EA','TTWO','SKLZ','AGAE','SLE','VS']
    await generate_stock_list(symbol_list, "json/stocks-list/list/esports.json")

async def get_car_company_stocks():
    symbol_list = ["TSLA", "TM", "RACE", "GM", "HMC", "F", "STLA", "LI", "RIVN", "XPEV",
    "VFS", "LCID", "NIO", "PII", "PSNY", "NKLA", "FFIE", "CENN", "EVTV",
    "GOEV", "HYZN", "MULN"]
    await generate_stock_list(symbol_list, "json/stocks-list/list/car-company-stocks.json")

async def get_electric_vehicles():
    symbol_list = [
        "TSLA", "LI", "RIVN", "XPEV", "VFS", "LCID", "NIO", "ZK", "PSNY", "FFIE",
        "CENN", "EVTV", "LOBO", "GOEV", "MULN"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/electric-vehicles.json")

async def get_augmented_reality():
    symbol_list = [
    "AAPL", "NVDA", "GOOGL", "AMD", "QCOM", "SONY", "KLAC", "ADSK",
    "RBLX", "ANSS", "SPLK", "PTC", "SNAP", "U", "OLED", "ETSY",
    "HIMX", "MVIS", "IMMR", "VUZI", "KOPN", "EMAN", "WIMI", "VRAR", "BHAT"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/augmented-reality.json")

async def get_gaming_stocks():
    symbol_list = [
        "NVDA", "MSFT", "SONY", "SE", "NTES", "RBLX", "EA", "TTWO", 
        "DKNG", "GME", "LOGI", "U", "PLTK", "CRSR", "HUYA", "DDI", 
        "GRVY", "SOHU", "GDEV", "INSE", "MYPS", "NCTY", "CMCM", "SKLZ", 
        "SNAL", "AGAE", "GIGM", "SLE", "VS", "BHAT"
    ]

    await generate_stock_list(symbol_list, "json/stocks-list/list/gaming-stocks.json")

async def get_pharmaceutical_stocks():
    symbol_list = [
        "LLY", "NVO", "JNJ", "ABBV", "MRK", "AZN", "NVS", "PFE", "AMGN", "SNY",
        "BMY", "GILD", "ZTS", "GSK", "TAK", "HLN", "TEVA", "BIIB", "NBIX", "VTRS",
        "ITCI", "RDY", "CTLT", "LNTH", "ELAN", "GRFS", "ALKS", "OGN", "ALVO", "PBH",
        "PRGO", "BHC", "HCM", "AMRX", "SUPN", "AMPH", "DVAX", "TARO", "EVO", "INDV",
        "KNSA", "HROW", "TLRY", "COLL", "ANIP", "BGM", "PCRX", "PETQ", "PAHC", "AVDL",
        "CRON", "EOLS", "IRWD", "EBS", "ESPR", "SIGA", "TKNO", "KMDA", "AKBA", "ORGO",
        "ETON", "AQST", "CGC", "LFCR", "ANIK", "ACB", "AMRN", "ZYBT", "OGI", "PROC",
        "BIOA", "CRDL", "DERM", "CTOR", "ASRT", "INCR", "RGC", "RMTI", "SCLX", "OPTN",
        "SCYX", "CPIX", "IXHL", "DRRX", "MIRA", "GELS", "CYTH", "FLGC", "TXMD", "AGRX",
        "AYTU", "TLPH", "BFRI", "EVOK", "RDHL", "IMCC", "QNTM", "SBFM", "CPHI", "PTPI",
        "SNOA", "UPC", "SHPH", "YCBD", "AKAN", "PRFX", "SXTC", "ACORQ"
    ]

    await generate_stock_list(symbol_list, "json/stocks-list/list/pharmaceutical-stocks.json")

async def get_online_gambling():
    symbol_list = [
        "DKNG", "LNW", "BYD", "SRAD", "IGT", "RSI", "PENN", "PLTK", "GENI", "EVRI",
        "DDI", "GAMB", "AGS", "INSE", "GAN", "GIGM"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/online-gambling.json")

async def get_online_dating():
    symbol_list = [
        "META","MTCH","GRND","MOMO","BMBL"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/online-dating.json")

async def get_virtual_reality():
    symbol_list = [
        "AAPL", "NVDA", "META", "AMD", "QCOM", "SONY", "KLAC", "ADSK", 
        "ANSS", "U", "OLED", "MTTR", "HIMX", "IMMR", "KOPN", "EMAN", 
        "RBOT", "VRAR"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/virtual-reality.json")
async def get_mobile_games():
    symbol_list = [
        "SE", "RBLX", "MAT", "PLTK", "DDI", "GRVY", 
        "SOHU", "GDEV", "MYPS", "NCTY", "CMCM", "SKLZ", "GIGM"
    ]

    await generate_stock_list(symbol_list, "json/stocks-list/list/mobile-games.json")

async def get_social_media_stocks():
    symbol_list = [
        "META", "NTES", "RDDT", "PINS", "SNAP", "DJT", 
        "MTCH", "WB", "YY", "SPT", "MOMO", "YALA"
    ]

    await generate_stock_list(symbol_list, "json/stocks-list/list/social-media-stocks.json")

async def get_sports_betting():
    symbol_list = [
        "DKNG", "CHDN", "LNW", "SRAD", "PENN", "GAMB", "GAN"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/sports-betting.json")

async def get_metaverse():
    symbol_list = [
        "AAPL", "NVDA", "META", "AMD", "ADBE", "QCOM", "SHOP", 
        "ADSK", "RBLX", "U", "MTTR", "GMM"
    ]
    await generate_stock_list(symbol_list, "json/stocks-list/list/metaverse.json")

async def get_magnificent_seven():
    symbol_list = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    await generate_stock_list(symbol_list, "json/stocks-list/list/magnificent-seven.json")

async def get_faang():
    symbol_list = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX']
    await generate_stock_list(symbol_list, "json/stocks-list/list/faang.json")



async def get_penny_stocks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:

            # Load quote data from JSON file
            quote_data = await get_quote_data(symbol)

            # Assign price and volume, and check if they meet the penny stock criteria
            if quote_data:
                price = round(quote_data.get('price',None), 2)
                volume = quote_data.get('volume',None)
                
                if price < 5 and volume > 10000:
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    res_list.append({
                        'symbol': symbol,
                        'name': name,
                        'price': price,
                        'changesPercentage': changesPercentage,
                        'marketCap': marketCap,
                        'volume': volume
                    })

        except Exception as e:
            print(e)

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['volume'], reverse=True)
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/penny-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_oversold_stocks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:

            # Load quote data from JSON file
            rsi = stock_screener_data_dict[symbol].get('rsi',None)

            if rsi < 30 and rsi > 0:
                quote_data = await get_quote_data(symbol)

                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')
                    volume = quote_data.get('volume')
                    if marketCap > 100_000 and changesPercentage != 0 and volume > 10_000:
                        # Append stock data to res_list if it meets the criteria
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                            'rsi': rsi
                        })

        except Exception as e:
            print(e)

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['rsi'])
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/oversold-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_overbought_stocks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:

            # Load quote data from JSON file
            rsi = stock_screener_data_dict[symbol].get('rsi',None)

            if rsi > 70 and rsi < 100:
                quote_data = await get_quote_data(symbol)

                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')
                    volume = quote_data.get('volume')

                    if marketCap > 100_000 and changesPercentage != 0 and volume > 10_000:
                        # Append stock data to res_list if it meets the criteria
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                            'rsi': rsi
                        })

        except Exception as e:
            print(e)

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['rsi'], reverse=True)
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/overbought-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_top_dividend_stocks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:

            # Load quote data from JSON file
            analyst_rating = stock_screener_data_dict[symbol].get('analystRating',None)
            analyst_counter = stock_screener_data_dict[symbol].get('analystCounter',0)
            dividend_yield = stock_screener_data_dict[symbol].get('dividendYield',0)
            payout_ratio = stock_screener_data_dict[symbol].get('payoutRatio',100)
            country = stock_screener_data_dict[symbol].get('country',None)
            if country == 'United States' and analyst_rating in ['Buy','Strong Buy'] and analyst_counter >= 10 and dividend_yield >=2 and payout_ratio < 60:
                quote_data = await get_quote_data(symbol)

                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    res_list.append({
                        'symbol': symbol,
                        'name': name,
                        'price': price,
                        'changesPercentage': changesPercentage,
                        'marketCap': marketCap,
                        'dividendYield': dividend_yield
                    })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/top-rated-dividend-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_monthly_dividends():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            payout_frequency = stock_screener_data_dict[symbol].get('payoutFrequency',None)
            dividend_yield = stock_screener_data_dict[symbol].get('dividendYield',None)
            exchange = stock_screener_data_dict[symbol].get('exchange',None)
            if dividend_yield > 0  and exchange in ['NASDAQ','AMEX','NYSE'] and payout_frequency == 'Monthly':
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    res_list.append({
                        'symbol': symbol,
                        'name': name,
                        'dividendYield': dividend_yield,
                        'price': price,
                        'changesPercentage': changesPercentage,
                        'marketCap': marketCap,
                    })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/monthly-dividend-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))



async def get_highest_revenue():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            revenue = stock_screener_data_dict[symbol].get('revenue',None)
            country = stock_screener_data_dict[symbol].get('country',None)
            if revenue > 1E9 and revenue < 1E12 and country == 'United States': #bug where some companies have wrong revenue
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    if marketCap >= 1E9:
                        # Append stock data to res_list if it meets the criteria
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                            'revenue': revenue
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['revenue'], reverse=True)[:500]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/highest-revenue.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_highest_income_tax():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            income_tax = stock_screener_data_dict[symbol].get('incomeTaxExpense',0)
            country = stock_screener_data_dict[symbol].get('country',None)
            if income_tax > 10E6 and country == 'United States':
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    if marketCap >= 1E9:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                            'incomeTaxExpense': income_tax
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['incomeTaxExpense'], reverse=True)[:100]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/highest-income-tax.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_most_employees():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            employees = stock_screener_data_dict[symbol].get('employees',None)
            country = stock_screener_data_dict[symbol].get('country',None)
            if employees > 10_000 and country == 'United States':
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    res_list.append({
                        'symbol': symbol,
                        'name': name,
                        'price': price,
                        'changesPercentage': changesPercentage,
                        'marketCap': marketCap,
                        'employees': employees
                    })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['employees'], reverse=True)[:100]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/most-employees.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_most_ftd_shares():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            relative_ftd = stock_screener_data_dict[symbol].get('relativeFTD',None)
            ftd_shares = stock_screener_data_dict[symbol].get('failToDeliver',None)
            country = stock_screener_data_dict[symbol].get('country',None)
            if relative_ftd > 10 and ftd_shares > 10_000 and country == 'United States':
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    volume = round(quote_data.get('volume',None), 2)
                    market_cap = round(quote_data.get('marketCap',None), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0 and volume > 10_000 and market_cap > 50E6:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'relativeFTD': relative_ftd,
                            'failToDeliver': ftd_shares
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['relativeFTD'], reverse=True)[:50]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/most-ftd-shares.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_most_shorted_stocks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            short_percent_float = stock_screener_data_dict[symbol].get('shortFloatPercent',None)
            if short_percent_float > 10 and short_percent_float < 100:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    market_cap = round(quote_data.get('marketCap',None), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'shortFloatPercent': short_percent_float,
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['shortFloatPercent'], reverse=True)[:100]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/most-shorted-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_most_buybacks():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            stock_buybacks = stock_screener_data_dict[symbol].get('commonStockRepurchased',None)
            country = stock_screener_data_dict[symbol].get('country',None)
            if country == 'United States' and stock_buybacks < -1E6:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')

                    if abs(stock_buybacks/marketCap) < 1:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                            'commonStockRepurchased': stock_buybacks
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['commonStockRepurchased'])[:100]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/most-buybacks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_highest_oi_change():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            iv_rank = stock_screener_data_dict[symbol].get('ivRank',0)
            total_prem = stock_screener_data_dict[symbol].get('totalPrem',0)
            total_oi = stock_screener_data_dict[symbol].get('totalOI',0)
            change_oi = stock_screener_data_dict[symbol].get('changeOI',0)

            if change_oi > 0 and total_oi > 0:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    market_cap = round(quote_data.get('marketCap',None), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'ivRank': iv_rank,
                            'totalPrem': total_prem,
                            'totalOI': total_oi,
                            'changeOI': change_oi,
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        highest_oi_change = sorted(res_list, key=lambda x: x['changeOI'], reverse=True)[:100]
        highest_oi = sorted(res_list, key=lambda x: x['totalOI'], reverse=True)[:100]
        
        # Create independent lists with ranks
        highest_oi_change_ranks = [
            {**item, "rank": rank} for rank, item in enumerate(highest_oi_change, start=1)
        ]
        highest_oi_ranks = [
            {**item, "rank": rank} for rank, item in enumerate(highest_oi, start=1)
        ]

        # Write the filtered and ranked stocks to JSON files
        with open("json/stocks-list/list/highest-open-interest-change.json", 'wb') as file:
            file.write(orjson.dumps(highest_oi_change_ranks))

        with open("json/stocks-list/list/highest-open-interest.json", 'wb') as file:
            file.write(orjson.dumps(highest_oi_ranks))


async def get_highest_option_iv_rank():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            iv_rank = stock_screener_data_dict[symbol].get('ivRank',0)
            total_prem = stock_screener_data_dict[symbol].get('totalPrem',0)
            total_oi = stock_screener_data_dict[symbol].get('totalOI',0)
            change_oi = stock_screener_data_dict[symbol].get('changeOI',0)

            if iv_rank > 0 and iv_rank < 100 and change_oi > 100 and total_prem > 1E4:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    market_cap = round(quote_data.get('marketCap',None), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'ivRank': iv_rank,
                            'totalPrem': total_prem,
                            'totalOI': total_oi,
                            'changeOI': change_oi,
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['ivRank'], reverse=True)[:50]
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/highest-option-iv-rank.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

async def get_highest_option_premium():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            iv_rank = stock_screener_data_dict[symbol].get('ivRank',0)
            total_prem = stock_screener_data_dict[symbol].get('totalPrem',0)
            total_oi = stock_screener_data_dict[symbol].get('totalOI',0)
            change_oi = stock_screener_data_dict[symbol].get('changeOI',0)

            if total_prem > 0 and iv_rank > 0:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    market_cap = round(quote_data.get('marketCap',None), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'ivRank': iv_rank,
                            'totalPrem': total_prem,
                            'totalOI': total_oi,
                            'changeOI': change_oi,
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['totalPrem'], reverse=True)[:50]

        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/highest-option-premium.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_highest_option_volume(volume_type):
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            call_volume = stock_screener_data_dict[symbol].get('callVolume',0)
            put_volume = stock_screener_data_dict[symbol].get('putVolume',0)

            if call_volume > 0 and put_volume > 0:
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    name = quote_data.get('name')

                    # Append stock data to res_list if it meets the criteria
                    if changesPercentage != 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'callVolume': call_volume,
                            'putVolume': put_volume,
                        })
        except:
            pass

    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x[volume_type], reverse=True)[:50]

        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open(f"json/stocks-list/list/highest-{volume_type.replace('V', '-v')}.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def etf_bitcoin_list():
    try:
        with sqlite3.connect('etf.db') as etf_con:
            etf_cursor = etf_con.cursor()
            etf_cursor.execute("PRAGMA journal_mode = wal")
            etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
            etf_symbols = [row[0] for row in etf_cursor.fetchall()]

            res_list = []
            query_template = """
                SELECT 
                    symbol, name, expenseRatio, totalAssets
                FROM 
                    etfs
                WHERE
                    symbol = ?
            """
            
            for symbol in etf_symbols:
                try:
                    data = pd.read_sql_query(query_template, etf_con, params=(symbol,))
                    name = data['name'].iloc[0]
                    
                    if 'bitcoin' in name.lower():
                        expense_ratio = round(float(data['expenseRatio'].iloc[0]), 2)
                        total_assets = int(data['totalAssets'].iloc[0])
                        
                        try:
                            with open(f"json/quote/{symbol}.json", "rb") as file:
                                quote_data = orjson.loads(file.read())
                        except (FileNotFoundError, orjson.JSONDecodeError):
                            quote_data = None

                        price = round(quote_data.get('price'), 2) if quote_data else None
                        changesPercentage = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                        if total_assets > 0:
                            res_list.append({
                                'symbol': symbol,
                                'name': name,
                                'expenseRatio': expense_ratio,
                                'totalAssets': total_assets,
                                'price': price,
                                'changesPercentage': changesPercentage
                            })
                except:
                    pass
            
            if res_list:
                res_list = sorted(res_list, key=lambda x: x['totalAssets'], reverse=True)
                for rank, item in enumerate(res_list, start=1):
                    item['rank'] = rank
                    
                with open("json/etf-bitcoin-list/data.json", 'wb') as file:
                    file.write(orjson.dumps(res_list))

    except:
        pass

async def ethereum_bitcoin_list():
    try:
        with sqlite3.connect('etf.db') as etf_con:
            etf_cursor = etf_con.cursor()
            etf_cursor.execute("PRAGMA journal_mode = wal")
            etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
            etf_symbols = [row[0] for row in etf_cursor.fetchall()]

            res_list = []
            query_template = """
                SELECT 
                    symbol, name, expenseRatio, totalAssets
                FROM 
                    etfs
                WHERE
                    symbol = ?
            """
            
            for symbol in etf_symbols:
                try:
                    data = pd.read_sql_query(query_template, etf_con, params=(symbol,))
                    name = data['name'].iloc[0]
                    
                    if 'ether' in name.lower() or 'ethereum' in name.lower():
                        expense_ratio = round(float(data['expenseRatio'].iloc[0]), 2)
                        total_assets = int(data['totalAssets'].iloc[0])
                        
                        try:
                            with open(f"json/quote/{symbol}.json", "rb") as file:
                                quote_data = orjson.loads(file.read())
                        except (FileNotFoundError, orjson.JSONDecodeError):
                            quote_data = None

                        price = round(quote_data.get('price'), 2) if quote_data else None
                        changesPercentage = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                        if total_assets > 0:
                            res_list.append({
                                'symbol': symbol,
                                'name': name,
                                'expenseRatio': expense_ratio,
                                'totalAssets': total_assets,
                                'price': price,
                                'changesPercentage': changesPercentage
                            })
                except:
                    pass
            
            if res_list:
                res_list = sorted(res_list, key=lambda x: x['totalAssets'], reverse=True)
                for rank, item in enumerate(res_list, start=1):
                    item['rank'] = rank
                    
                with open("json/stocks-list/list/ethereum-etfs.json", 'wb') as file:
                    file.write(orjson.dumps(res_list))

    except:
        pass


async def get_covered_call_etfs():
    try:
        etf_symbols = [
            "JEPQ", "QYLD", "MSTY", "XYLD", "NVDY", "RYLD", "TSLY", "TLTW", "EIPI", "YMAX",
            "QDTE", "BUYW", "FTQI", "ULTY", "XDTE", "YMAG", "AIPI", "BTCI", "AMZY", "OMAH",
            "LQDW", "YBTC", "HYGW", "SMCY", "IWMI", "FBY", "IQQQ", "NFLY", "YBIT", "AMDY",
            "PBP", "RDTE", "DJIA", "LFGY", "APLY", "MSFO", "GOOY", "QYLG", "MDST", "KLIP",
            "AIYY", "SNOY", "GDXY", "MRNY", "IMST", "OARK", "QDVO", "MARO", "JPMO", "PYPY",
            "XOMO", "XYLG", "SRHR", "ITWO", "CVRD", "CEPI", "IVVW", "DISO", "YETH", "GPTY",
            "EGGY", "IWMW", "DIVP", "FEAT", "SDTY", "MAGY", "FYEE", "MLPD", "HOOY", "TYLG",
            "TLTP", "FIVY", "BIGY", "QDTY", "RYLG", "BTCC", "SOXY", "BITY", "RDTY", "BPI",
            "NVII", "ICOI", "BAGY", "RNTY", "DYLG", "QDCC", "BCCC", "BRKC", "COII", "IAUI",
            "IGME", "MSII", "TSII"
        ]

        res_list = []            
        for symbol in etf_symbols:
            try:
                with open(f"json/dividends/companies/{symbol}.json","rb") as file:
                    data = orjson.loads(file.read())
                    dividend_yield = data.get('dividendYield',0)
                try:
                    with open(f"json/quote/{symbol}.json", "rb") as file:
                        quote_data = orjson.loads(file.read())

                except (FileNotFoundError, orjson.JSONDecodeError):
                    quote_data = None

                price = round(quote_data.get('price'), 2) if quote_data else None
                changesPercentage = round(quote_data.get('changesPercentage'), 2) if quote_data else None
                market_cap = quote_data.get('marketCap',0)
                name = quote_data.get('name')
                
                if dividend_yield > 0 and market_cap > 0:
                    res_list.append({
                        'symbol': symbol,
                        'name': name,
                        'dividendYield': dividend_yield,
                        'marketCap': market_cap,
                        'price': price,
                        'changesPercentage': changesPercentage
                    })
            except Exception as e:
                print(e)
        
        if res_list:
            res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
            for rank, item in enumerate(res_list, start=1):
                item['rank'] = rank
                
            with open("json/stocks-list/list/covered-call-etfs.json", 'wb') as file:
                file.write(orjson.dumps(res_list))

    except:
        pass

async def get_monthly_dividends_etfs():
    with sqlite3.connect('etf.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM etfs")
        symbols = [row[0] for row in cursor.fetchall()]

    res_list = []
    for symbol in symbols:
        try:
            # Load quote data from JSON file
            with open(f"json/dividends/companies/{symbol}.json","rb") as file:
                data = orjson.loads(file.read())

            payout_frequency = data.get('payoutFrequency',None)
            dividend_yield = data.get('dividendYield',None)
            if dividend_yield > 0 and payout_frequency == 'Monthly':
                quote_data = await get_quote_data(symbol)
                # Assign price and volume, and check if they meet the penny stock criteria
                if quote_data:
                    price = round(quote_data.get('price',None), 2)
                    changesPercentage = round(quote_data.get('changesPercentage'), 2)
                    marketCap = quote_data.get('marketCap')
                    name = quote_data.get('name')
                    if marketCap > 0:
                        res_list.append({
                            'symbol': symbol,
                            'name': name,
                            'dividendYield': dividend_yield,
                            'price': price,
                            'changesPercentage': changesPercentage,
                            'marketCap': marketCap,
                        })
        except:
            pass
    if res_list:
        # Sort by market cap in descending order
        res_list = sorted(res_list, key=lambda x: x['dividendYield'], reverse=True)
        
        # Assign rank to each stock
        for rank, item in enumerate(res_list, start=1):
            item['rank'] = rank

        # Write the filtered and ranked penny stocks to a JSON file
        with open("json/stocks-list/list/monthly-dividend-etfs.json", 'wb') as file:
            file.write(orjson.dumps(res_list))


async def get_all_reits_list(cursor):
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.sector
        FROM stocks s 
        WHERE {}
    """
    
    condition = "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND industry LIKE '%REIT%' AND symbol NOT LIKE '%-%'"
    full_query = base_query.format(condition)
    
    cursor.execute(full_query)
    raw_data = cursor.fetchall()
    
    res_list = []
    for row in raw_data:
        symbol = row[0]
        
        try:
            quote_data = await get_quote_data(symbol)
            if not quote_data:
                continue
                
            price = quote_data.get('price')
            changes_percentage = quote_data.get('changesPercentage')
            
            item = {
                'symbol': symbol,
                'name': row[1],
                'price': round(float(price) if price is not None else 0, 2),
                'changesPercentage': round(float(changes_percentage) if changes_percentage is not None else 0, 2),
                'marketCap': quote_data.get('marketCap', 0),
            }
            
            dividend_yield = stock_screener_data_dict.get(symbol, {}).get('dividendYield')
            item['dividendYield'] = dividend_yield
            
            if item['marketCap'] > 0 and dividend_yield is not None:
                res_list.append(item)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if res_list:
        res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
        
        for rank, item in enumerate(res_list, 1):
            item['rank'] = rank

        
        with open("json/industry/list/reits.json", 'wb') as file:
            file.write(orjson.dumps(res_list))
    
    return res_list


async def get_all_spacs_list(cursor):
    base_query = """
        SELECT DISTINCT s.symbol, s.name, s.exchangeShortName, s.marketCap, s.industry
        FROM stocks s 
        WHERE {}
    """
    
    condition = "industry LIKE '%Shell Companies%' AND symbol NOT LIKE '%-%'"
    full_query = base_query.format(condition)
    
    cursor.execute(full_query)
    raw_data = cursor.fetchall()
    
    res_list = []
    for row in raw_data:
        symbol = row[0]
        
        try:
            quote_data = await get_quote_data(symbol)
            if not quote_data:
                continue
                
            price = quote_data.get('price')
            changes_percentage = quote_data.get('changesPercentage')
            
            item = {
                'symbol': symbol,
                'name': row[1],
                'price': round(float(price) if price is not None else 0, 2),
                'changesPercentage': round(float(changes_percentage) if changes_percentage is not None else 0, 2),
                'marketCap': quote_data.get('marketCap', 0),
            }
            
            #dividend_yield = stock_screener_data_dict.get(symbol, {}).get('dividendYield')
            #item['dividendYield'] = dividend_yield

            if item['marketCap'] > 0:
                res_list.append(item)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if res_list:
        res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
        
        for rank, item in enumerate(res_list, 1):
            item['rank'] = rank

        
        with open("json/stocks-list/list/spacs-stocks.json", 'wb') as file:
            file.write(orjson.dumps(res_list))

    return res_list



async def get_index_list():
    with sqlite3.connect('stocks.db') as con:
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]

    async with aiohttp.ClientSession() as session:

        for index_list in ['nasdaq','dowjones','sp500']:
            url = f"https://financialmodelingprep.com/api/v3/{index_list}_constituent?apikey={api_key}"
            async with session.get(url) as response:
                data = await response.json()
                data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in data]
                data = [entry for entry in data if entry]

                res_list = []
                for item in data:
                    try:
                        symbol = item['symbol']
                        quote_data = await get_quote_data(symbol)

                        if quote_data:
                            item = {
                                'symbol': symbol,
                                'name': quote_data.get('name',None),
                                'price': round(quote_data.get('price', 0), 2),
                                'changesPercentage': round(quote_data.get('changesPercentage', 0), 2),
                                'marketCap': quote_data.get('marketCap', 0),
                                'revenue': None,
                            }
                            item['revenue'] = stock_screener_data_dict[symbol].get('revenue')

                        if item['marketCap'] > 0:
                            res_list.append(item)
                    except Exception as e:
                        print(e)

            if res_list:
                res_list = sorted(res_list, key=lambda x: x['marketCap'] or 0, reverse=True)
                
                # Add rank to each item
                for rank, item in enumerate(res_list, 1):
                    item['rank'] = rank

                if index_list == 'nasdaq':
                    extension = '100'
                else:
                    extension = ''
                with open(f"json/stocks-list/list/{index_list+extension}.json", 'wb') as file:
                    file.write(orjson.dumps(res_list))

async def get_all_stock_tickers():
    try:
        with sqlite3.connect('stocks.db') as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
            stock_symbols = [row[0] for row in cursor.fetchall()]

        res_list = []
        for symbol in stock_symbols:
            try:
                
                try:
                    with open(f"json/quote/{symbol}.json", "rb") as file:
                        quote_data = orjson.loads(file.read())
                except (FileNotFoundError, orjson.JSONDecodeError):
                    quote_data = None

                if quote_data:
                    item = {
                        'symbol': symbol,
                        'name': quote_data.get('name',None),
                        #'price': round(quote_data.get('price'), 2) if quote_data.get('price') is not None else None,
                        #'changesPercentage': round(quote_data.get('changesPercentage'), 2) if quote_data.get('changesPercentage') is not None else None,
                        'marketCap': quote_data.get('marketCap', None),
                        'revenue': None,
                    }
                    
                    exchange = None
                    # Add screener data if available
                    if symbol in stock_screener_data_dict:
                        item['revenue'] = stock_screener_data_dict[symbol].get('revenue')
                        item['industry'] = stock_screener_data_dict[symbol].get('industry')
                        exchange = stock_screener_data_dict[symbol].get('exchange')
                    
                        if item['marketCap'] > 0 and item['revenue'] > 0 and exchange in ['NYSE','NASDAQ','AMEX']:
                            res_list.append(item)
            except:
                pass
        
        if res_list:
            res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
                
            with open("json/stocks-list/list/all-stock-tickers.json", 'wb') as file:
                file.write(orjson.dumps(res_list))

    except:
        pass

async def get_all_etf_tickers():
    try:
        # 1) load symbols + raw profile JSON from SQLite
        with sqlite3.connect('etf.db') as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA journal_mode = wal")
            cursor.execute("SELECT DISTINCT symbol, profile FROM etfs")
            rows = cursor.fetchall()

        # 2) parse profiles into a list of dicts
        etf_meta = []
        for symbol, raw_profile in rows:
            asset_class = aum = None
            if raw_profile:
                try:
                    profiles = orjson.loads(raw_profile)
                    first = profiles[0] if isinstance(profiles, list) and profiles else {}
                    asset_class = first.get('assetClass')
                    aum = first.get('aum')
                    expense_ratio = first.get('expenseRatio')

                except:
                    pass
            etf_meta.append({'symbol': symbol, 'assetClass': asset_class, 'aum': aum, 'expenseRatio': expense_ratio})

        # 3) read each quote JSON, build output only if aum > 0
        result = []
        for meta in etf_meta:
            try:
                symbol, aum = meta['symbol'], meta['aum'] or 0
                asset_class = meta['assetClass']
                expense_ratio = meta['expenseRatio']
                if aum > 0 and aum < 1E12 and asset_class in ['Equity', 'Fixed Income', 'Commodity','Currency','Asset Allocation']:
                    with open(f"json/quote/{symbol}.json", 'rb') as file:
                        quote = orjson.loads(file.read())
                    result.append({
                        'symbol': symbol,
                        'name': quote.get('name'),
                        'assetClass': asset_class,
                        'aum': aum,
                        'expenseRatio': expense_ratio
                    })
            except:
                pass

        if result:
            result.sort(key=lambda x: x['aum'], reverse=True)
            with open("json/stocks-list/list/all-etf-tickers.json", 'wb') as file:
                file.write(orjson.dumps(result))

    except Exception as e:
        print(e)

async def run():
    
    await asyncio.gather(
        get_ai_stocks(),
        get_clean_energy(),
        get_esports(),
        get_car_company_stocks(),
        get_electric_vehicles(),
        get_augmented_reality(),
        get_gaming_stocks(),
        get_pharmaceutical_stocks(),
        get_online_gambling(),
        get_online_dating(),
        get_social_media_stocks(),
        get_mobile_games(),
        get_virtual_reality(),
        get_sports_betting(),
        get_metaverse(),
        get_all_stock_tickers(),
        get_all_etf_tickers(),
        get_index_list(),
        etf_bitcoin_list(),
        ethereum_bitcoin_list(),
        get_covered_call_etfs(),
        get_monthly_dividends_etfs(),
        get_magnificent_seven(),
        get_faang(),
        get_penny_stocks(),
        get_oversold_stocks(),
        get_overbought_stocks(),
        get_top_dividend_stocks(),
        get_highest_revenue(),
        get_highest_income_tax(),
        get_most_employees(),
        get_most_ftd_shares(),
        get_most_shorted_stocks(),
        get_highest_oi_change(),
        get_highest_option_iv_rank(),
        get_highest_option_premium(),
        get_highest_option_volume('callVolume'),
        get_highest_option_volume('putVolume'),
        get_etf_holding(),
        get_etf_provider(),
        get_most_buybacks(),
        get_monthly_dividends(),
    )
    


    """Main function to run the analysis for all categories"""
    market_cap_conditions = {
        'mega-cap-stocks': "marketCap >= 200e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'large-cap-stocks': "marketCap < 200e9 AND marketCap >= 10e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'mid-cap-stocks': "marketCap < 10e9 AND marketCap >= 2e9 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'small-cap-stocks': "marketCap < 2e9 AND marketCap >= 300e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'micro-cap-stocks': "marketCap < 300e6 AND marketCap >= 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')",
        'nano-cap-stocks': "marketCap < 50e6 AND (exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX')"
    }

    sector_conditions = {
        'financial': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Financials' OR sector = 'Financial Services')",
        'healthcare': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Healthcare')",
        'technology': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Technology')",
        'industrials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Industrials')",
        'consumer-cyclical': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Consumer Cyclical')",
        'real-estate': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Real Estate')",
        'basic-materials': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Basic Materials')",
        'communication-services': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Communication Services')",
        'energy': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Energy')",
        'consumer-defensive': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Consumer Defensive')",
        'utilities': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' OR exchangeShortName = 'AMEX') AND (sector = 'Utilities')"
    }

    country_conditions = {
        'de': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'DE'",
        'ca': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'CA'",
        'cn': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'CN'",
        'in': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'IN'",
        'il': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'IL'",
        'gb': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'GB'",
        'jp': "(exchangeShortName = 'NYSE' OR exchangeShortName = 'NASDAQ' or exchangeShortName = 'AMEX') AND country = 'JP'",
    }

    exchange_conditions = {
        'nasdaq': "exchangeShortName = 'NASDAQ'",
        'nyse': "exchangeShortName = 'NYSE'",
        'amex': "exchangeShortName = 'AMEX'",
    }

    try:
        con = sqlite3.connect('stocks.db')
        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")

        etf_con = sqlite3.connect('etf.db')
        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]


        await get_all_spacs_list(cursor)

        await get_all_reits_list(cursor)
        
        for category, condition in exchange_conditions.items():
            await process_category(cursor, category, condition, 'stocks-list')
            #await asyncio.sleep(1)  # Small delay between categories

        for category, condition in country_conditions.items():
            await process_category(cursor, category, condition, 'stocks-list')
            #await asyncio.sleep(1)  # Small delay between categories

        for category, condition in market_cap_conditions.items():
            await process_category(cursor, category, condition, 'market-cap')
            #await asyncio.sleep(1)  # Small delay between categories

        # Process sector categories
        
        for category, condition in sector_conditions.items():
            await process_category(cursor, category, condition, 'sector')
            #await asyncio.sleep(1)  # Small delay between categories
        
        

    except Exception as e:
        print(e)
        raise
    finally:
        con.close()
        etf_con.close()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run())
    except Exception as e:
        print(e)
    finally:
        loop.close()