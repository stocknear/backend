import pytz
from datetime import datetime, timedelta
from urllib.request import urlopen
import certifi
import json
import ujson
import schedule
import time
import subprocess
import asyncio
import aiohttp
import pytz
import sqlite3
import pandas as pd
import numpy as np
from pocketbase import PocketBase
from collections import Counter
import re
import hashlib

from dotenv import load_dotenv
import os

pb = PocketBase('http://127.0.0.1:8090')
load_dotenv()
api_key = os.getenv('FMP_API_KEY')

berlin_tz = pytz.timezone('Europe/Berlin')

# Replace NaN values with None in the resulting JSON object
def replace_nan_inf_with_none(obj):
    if isinstance(obj, list):
        return [replace_nan_inf_with_none(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: replace_nan_inf_with_none(value) for key, value in obj.items()}
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj

def custom_symbol_sort(item):
    symbol = item['symbol']
    # Use regular expression to check if the symbol matches the typical stock ticker format (e.g., AAPL)
    if re.match(r'^[A-Z]+$', symbol):
        return symbol  # Sort uppercase symbols first
    else:
        return 'ZZZZZZZZZZZZZZ'  # Place non-standard symbols at the bottom

def generate_id(name):
    hashed = hashlib.sha256(name.encode()).hexdigest()
    return hashed[:10]

async def get_stock_screener(con,symbols):
    
    #Stock Screener Data
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
   
    #Stock Screener Data
    cursor.execute("SELECT symbol, name, avgVolume, change_1W, change_1M, change_3M, change_6M, change_1Y, change_3Y, sma_50, sma_200, ema_50, ema_200, rsi, atr, stoch_rsi, mfi, cci, priceToSalesRatio, priceToBookRatio, eps, pe, ESGScore, marketCap, revenue, netIncome, grossProfit, costOfRevenue, costAndExpenses, interestIncome, interestExpense, researchAndDevelopmentExpenses, ebitda, operatingExpenses, operatingIncome, growthRevenue, growthNetIncome, growthGrossProfit, growthCostOfRevenue, growthCostAndExpenses, growthInterestExpense, growthResearchAndDevelopmentExpenses, growthEBITDA, growthEPS, growthOperatingExpenses, growthOperatingIncome, beta FROM stocks WHERE eps IS NOT NULL AND revenue IS NOT NULL AND marketCap IS NOT NULL AND beta IS NOT NULL")
    raw_data = cursor.fetchall()
    stock_screener_data = [{
            'symbol': symbol,
            'name': name,
            'avgVolume': avgVolume,
            'change1W': change_1W,
            'change1M': change_1M,
            'change3M': change_3M,
            'change6M': change_6M,
            'change1Y': change_1Y,
            'change3Y': change_3Y,
            'sma50': sma_50,
            'sma200': sma_200,
            'ema50': ema_50,
            'ema200': ema_200,
            'rsi': rsi,
            'atr': atr,
            'stochRSI': stoch_rsi,
            'mfi': mfi,
            'cci': cci,
            'priceToSalesRatio': priceToSalesRatio,
            'priceToBookRatio': priceToBookRatio,
            'eps': eps,
            'pe': pe,
            'esgScore': ESGScore,
            'marketCap': marketCap,
            'revenue': revenue,
            'netIncome': netIncome,
            'grossProfit': grossProfit,
            'costOfRevenue': costOfRevenue,
            'costAndExpenses': costAndExpenses,
            'interestIncome': interestIncome,
            'interestExpense': interestExpense,
            'researchAndDevelopmentExpenses': researchAndDevelopmentExpenses,
            'ebitda': ebitda,
            'operatingExpenses': operatingExpenses,
            'operatingIncome': operatingIncome,
            'growthRevenue': growthRevenue,
            'growthNetIncome': growthNetIncome,
            'growthGrossProfit': growthGrossProfit,
            'growthCostOfRevenue': growthCostOfRevenue,
            'growthCostAndExpenses': growthCostAndExpenses,
            'growthInterestExpense': growthInterestExpense,
            'growthResearchAndDevelopmentExpenses': growthResearchAndDevelopmentExpenses,
            'growthEBITDA': growthEBITDA,
            'growthEPS': growthEPS,
            'growthOperatingExpenses': growthOperatingExpenses,
            'growthOperatingIncome': growthOperatingIncome,
            'beta': beta,
        } for (symbol, name, avgVolume, change_1W, change_1M, change_3M, change_6M, change_1Y, change_3Y, sma_50, sma_200, ema_50, ema_200, rsi, atr, stoch_rsi, mfi, cci, priceToSalesRatio, priceToBookRatio, eps, pe, ESGScore, marketCap, revenue, netIncome, grossProfit,costOfRevenue, costAndExpenses, interestIncome, interestExpense, researchAndDevelopmentExpenses, ebitda, operatingExpenses, operatingIncome, growthRevenue, growthNetIncome, growthGrossProfit, growthCostOfRevenue, growthCostAndExpenses, growthInterestExpense, growthResearchAndDevelopmentExpenses, growthEBITDA, growthEPS, growthOperatingExpenses, growthOperatingIncome, beta) in raw_data if name != 'SP 500']

    stock_screener_data = [{k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in entry.items()} for entry in stock_screener_data]

    cursor.execute("SELECT symbol, name, price, changesPercentage FROM stocks WHERE price IS NOT NULL AND changesPercentage IS NOT NULL")
    raw_data = cursor.fetchall()
    stocks_data = [{
        'symbol': row[0],
        'name': row[1],
        'price': row[2],
        'changesPercentage': row[3],
    } for row in raw_data]


    # Create a dictionary to map symbols to 'price' and 'changesPercentage' from stocks_data
    stocks_data_map = {entry['symbol']: (entry['price'], entry['changesPercentage']) for entry in stocks_data}

    # Iterate through stock_screener_data and update 'price' and 'changesPercentage' if symbols match
    # Add VaR value to stock screener
    for item in stock_screener_data:
        symbol = item['symbol']
        if symbol in stocks_data_map:
            item['price'], item['changesPercentage'] = stocks_data_map[symbol]
        try:
            with open(f"json/var/{symbol}.json", 'r') as file:
                item['var'] = ujson.load(file)['var']
        except:
            item['var'] = None

        try:
            with open(f"json/analyst/summary/{symbol}.json", 'r') as file:
                rating = ujson.load(file)['consensusRating']
                if rating == 'Sell':
                    item['ratingRecommendation'] = 0
                elif rating == 'Hold':
                    item['ratingRecommendation'] = 1
                elif rating == 'Buy':
                    item['ratingRecommendation'] = 2
                else:
                    item['ratingRecommendation'] = None
        except:
            item['ratingRecommendation'] = None


    return stock_screener_data



async def get_dividends_calendar(con,symbols):

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
        
    async with aiohttp.ClientSession() as session:

        #Database read 1y and 3y data
        query_template = """
            SELECT 
                name, marketCap, revenue
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/stock_dividend_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if '.' not in stock['symbol'] and stock['symbol'] in symbols} for stock in data]
            filtered_data = [entry for entry in filtered_data if entry]

            for entry in filtered_data:
                try:
                    symbol = entry['symbol']
                    data = pd.read_sql_query(query_template, con, params=(symbol,))
                    entry['name'] = data['name'].iloc[0]
                    entry['marketCap'] = int(data['marketCap'].iloc[0])
                    entry['revenue'] = int(data['revenue'].iloc[0])
                except:
                    entry['name'] = 'n/a'
                    entry['marketCap'] = None
                    entry['revenue'] = None

    filtered_data = [d for d in filtered_data if d['symbol'] in symbols]

    return filtered_data


async def get_earnings_calendar(con, symbols):

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    async with aiohttp.ClientSession() as session:

        query_template = """
            SELECT 
                name,marketCap,revenue,eps
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols and '.' not in stock['symbol']} for stock in data]
            #filtered_data = [entry for entry in filtered_data if entry]

            for entry in filtered_data:
                try:
                    symbol = entry['symbol']
                    fundamental_data = pd.read_sql_query(query_template, con, params=(symbol,))
                    entry['name'] = fundamental_data['name'].iloc[0]
                    entry['marketCap'] = int(fundamental_data['marketCap'].iloc[0])
                    entry['revenue'] = int(fundamental_data['revenue'].iloc[0])
                    entry['eps'] = float(fundamental_data['eps'].iloc[0])
                except:
                    entry['marketCap'] = 'n/a'
                    entry['marketCap'] = None
                    entry['revenue'] = None
                    entry['eps'] = None

            filtered_data = [item for item in filtered_data if 'date' in item]

    seen_symbols = set()
    unique_data = []

    for item in filtered_data:
        symbol = item.get('symbol')
        try:
            with open(f"json/quote/{symbol}.json", 'r') as file:
                quote = ujson.load(file)
                try:
                    earnings_date = datetime.strptime(quote['earningsAnnouncement'].split('T')[0], '%Y-%m-%d').strftime('%Y-%m-%d')
                except:
                    earnings_date = '-'
        except Exception as e:
            earnings_date = '-'
            print(e)

        if symbol is None or symbol not in seen_symbols:
            #bug in fmp endpoint. Double check that earnigns date is the same as in quote endpoint
            if item['date'] == earnings_date:
                #print(symbol, item['date'], earnings_date)
                unique_data.append(item)
            seen_symbols.add(symbol)

    return unique_data


async def get_stock_splits_calendar(con,symbols):

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
        
    async with aiohttp.ClientSession() as session:

        #Database read 1y and 3y data
        query_template = """
            SELECT 
                name, marketCap,eps, revenue, netIncome
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/stock_split_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in data]
            filtered_data = [entry for entry in filtered_data if entry]

            for entry in filtered_data:
                try:
                    symbol = entry['symbol']
                    data = pd.read_sql_query(query_template, con, params=(symbol,))
                    entry['name'] = data['name'].iloc[0]
                    entry['marketCap'] = int(data['marketCap'].iloc[0])
                    entry['revenue'] = int(data['revenue'].iloc[0])
                    entry['netIncome'] = int(data['netIncome'].iloc[0])
                    entry['eps'] = float(data['eps'].iloc[0])
                except:
                    entry['name'] = 'n/a'
                    entry['marketCap'] = None
                    entry['revenue'] = None
                    entry['netIncome'] = None
                    entry['eps'] = None

    filtered_data = [d for d in filtered_data if d['symbol'] in symbols]

    return filtered_data


async def get_economic_calendar():

    berlin_tz = pytz.timezone('Europe/Berlin')
    today = datetime.now(berlin_tz)

    # Calculate the start date (Monday) 4 weeks before
    start_date = today - timedelta(weeks=4)
    start_date = start_date - timedelta(days=(start_date.weekday() - 0) % 7)

    # Calculate the end date (Friday) 4 weeks after
    end_date = today + timedelta(weeks=4)
    end_date = end_date + timedelta(days=(4 - end_date.weekday()) % 7)

    # Format dates as strings in 'YYYY-MM-DD' format
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    country_list = [{'short': 'AW', 'long': 'Aruba'}, {'short': 'AF', 'long': 'Afghanistan'}, {'short': 'AO', 'long': 'Angola'}, {'short': 'AI', 'long': 'Anguilla'}, {'short': 'AX', 'long': 'Åland Islands'}, {'short': 'AL', 'long': 'Albania'}, {'short': 'AD', 'long': 'Andorra'}, {'short': 'AE', 'long': 'United Arab Emirates'}, {'short': 'AR', 'long': 'Argentina'}, {'short': 'AM', 'long': 'Armenia'}, {'short': 'AS', 'long': 'American Samoa'}, {'short': 'AQ', 'long': 'Antarctica'}, {'short': 'TF', 'long': 'French Southern Territories'}, {'short': 'AG', 'long': 'Antigua and Barbuda'}, {'short': 'AU', 'long': 'Australia'}, {'short': 'AT', 'long': 'Austria'}, {'short': 'AZ', 'long': 'Azerbaijan'}, {'short': 'BI', 'long': 'Burundi'}, {'short': 'BE', 'long': 'Belgium'}, {'short': 'BJ', 'long': 'Benin'}, {'short': 'BQ', 'long': 'Bonaire, Sint Eustatius and Saba'}, {'short': 'BF', 'long': 'Burkina Faso'}, {'short': 'BD', 'long': 'Bangladesh'}, {'short': 'BG', 'long': 'Bulgaria'}, {'short': 'BH', 'long': 'Bahrain'}, {'short': 'BS', 'long': 'Bahamas'}, {'short': 'BA', 'long': 'Bosnia and Herzegovina'}, {'short': 'BL', 'long': 'Saint Barthélemy'}, {'short': 'BY', 'long': 'Belarus'}, {'short': 'BZ', 'long': 'Belize'}, {'short': 'BM', 'long': 'Bermuda'}, {'short': 'BO', 'long': 'Bolivia, Plurinational State of'}, {'short': 'BR', 'long': 'Brazil'}, {'short': 'BB', 'long': 'Barbados'}, {'short': 'BN', 'long': 'Brunei Darussalam'}, {'short': 'BT', 'long': 'Bhutan'}, {'short': 'BV', 'long': 'Bouvet Island'}, {'short': 'BW', 'long': 'Botswana'}, {'short': 'CF', 'long': 'Central African Republic'}, {'short': 'CA', 'long': 'Canada'}, {'short': 'CC', 'long': 'Cocos (Keeling) Islands'}, {'short': 'CH', 'long': 'Switzerland'}, {'short': 'CL', 'long': 'Chile'}, {'short': 'CN', 'long': 'China'}, {'short': 'CI', 'long': "Côte d'Ivoire"}, {'short': 'CM', 'long': 'Cameroon'}, {'short': 'CD', 'long': 'Congo, The Democratic Republic of the'}, {'short': 'CG', 'long': 'Congo'}, {'short': 'CK', 'long': 'Cook Islands'}, {'short': 'CO', 'long': 'Colombia'}, {'short': 'KM', 'long': 'Comoros'}, {'short': 'CV', 'long': 'Cabo Verde'}, {'short': 'CR', 'long': 'Costa Rica'}, {'short': 'CU', 'long': 'Cuba'}, {'short': 'CW', 'long': 'Curaçao'}, {'short': 'CX', 'long': 'Christmas Island'}, {'short': 'KY', 'long': 'Cayman Islands'}, {'short': 'CY', 'long': 'Cyprus'}, {'short': 'CZ', 'long': 'Czechia'}, {'short': 'DE', 'long': 'Germany'}, {'short': 'DJ', 'long': 'Djibouti'}, {'short': 'DM', 'long': 'Dominica'}, {'short': 'DK', 'long': 'Denmark'}, {'short': 'DO', 'long': 'Dominican Republic'}, {'short': 'DZ', 'long': 'Algeria'}, {'short': 'EC', 'long': 'Ecuador'}, {'short': 'EG', 'long': 'Egypt'}, {'short': 'ER', 'long': 'Eritrea'}, {'short': 'EH', 'long': 'Western Sahara'}, {'short': 'ES', 'long': 'Spain'}, {'short': 'EE', 'long': 'Estonia'}, {'short': 'ET', 'long': 'Ethiopia'}, {'short': 'FI', 'long': 'Finland'}, {'short': 'FJ', 'long': 'Fiji'}, {'short': 'FK', 'long': 'Falkland Islands (Malvinas)'}, {'short': 'FR', 'long': 'France'}, {'short': 'FO', 'long': 'Faroe Islands'}, {'short': 'FM', 'long': 'Micronesia, Federated States of'}, {'short': 'GA', 'long': 'Gabon'}, {'short': 'GB', 'long': 'United Kingdom'}, {'short': 'GE', 'long': 'Georgia'}, {'short': 'GG', 'long': 'Guernsey'}, {'short': 'GH', 'long': 'Ghana'}, {'short': 'GI', 'long': 'Gibraltar'}, {'short': 'GN', 'long': 'Guinea'}, {'short': 'GP', 'long': 'Guadeloupe'}, {'short': 'GM', 'long': 'Gambia'}, {'short': 'GW', 'long': 'Guinea-Bissau'}, {'short': 'GQ', 'long': 'Equatorial Guinea'}, {'short': 'GR', 'long': 'Greece'}, {'short': 'GD', 'long': 'Grenada'}, {'short': 'GL', 'long': 'Greenland'}, {'short': 'GT', 'long': 'Guatemala'}, {'short': 'GF', 'long': 'French Guiana'}, {'short': 'GU', 'long': 'Guam'}, {'short': 'GY', 'long': 'Guyana'}, {'short': 'HK', 'long': 'Hong Kong'}, {'short': 'HM', 'long': 'Heard Island and McDonald Islands'}, {'short': 'HN', 'long': 'Honduras'}, {'short': 'HR', 'long': 'Croatia'}, {'short': 'HT', 'long': 'Haiti'}, {'short': 'HU', 'long': 'Hungary'}, {'short': 'ID', 'long': 'Indonesia'}, {'short': 'IM', 'long': 'Isle of Man'}, {'short': 'IN', 'long': 'India'}, {'short': 'IO', 'long': 'British Indian Ocean Territory'}, {'short': 'IE', 'long': 'Ireland'}, {'short': 'IR', 'long': 'Iran, Islamic Republic of'}, {'short': 'IQ', 'long': 'Iraq'}, {'short': 'IS', 'long': 'Iceland'}, {'short': 'IL', 'long': 'Israel'}, {'short': 'IT', 'long': 'Italy'}, {'short': 'JM', 'long': 'Jamaica'}, {'short': 'JE', 'long': 'Jersey'}, {'short': 'JO', 'long': 'Jordan'}, {'short': 'JP', 'long': 'Japan'}, {'short': 'KZ', 'long': 'Kazakhstan'}, {'short': 'KE', 'long': 'Kenya'}, {'short': 'KG', 'long': 'Kyrgyzstan'}, {'short': 'KH', 'long': 'Cambodia'}, {'short': 'KI', 'long': 'Kiribati'}, {'short': 'KN', 'long': 'Saint Kitts and Nevis'}, {'short': 'KR', 'long': 'Korea, Republic of'}, {'short': 'KW', 'long': 'Kuwait'}, {'short': 'LA', 'long': "Lao People's Democratic Republic"}, {'short': 'LB', 'long': 'Lebanon'}, {'short': 'LR', 'long': 'Liberia'}, {'short': 'LY', 'long': 'Libya'}, {'short': 'LC', 'long': 'Saint Lucia'}, {'short': 'LI', 'long': 'Liechtenstein'}, {'short': 'LK', 'long': 'Sri Lanka'}, {'short': 'LS', 'long': 'Lesotho'}, {'short': 'LT', 'long': 'Lithuania'}, {'short': 'LU', 'long': 'Luxembourg'}, {'short': 'LV', 'long': 'Latvia'}, {'short': 'MO', 'long': 'Macao'}, {'short': 'MF', 'long': 'Saint Martin (French part)'}, {'short': 'MA', 'long': 'Morocco'}, {'short': 'MC', 'long': 'Monaco'}, {'short': 'MD', 'long': 'Moldova, Republic of'}, {'short': 'MG', 'long': 'Madagascar'}, {'short': 'MV', 'long': 'Maldives'}, {'short': 'MX', 'long': 'Mexico'}, {'short': 'MH', 'long': 'Marshall Islands'}, {'short': 'MK', 'long': 'North Macedonia'}, {'short': 'ML', 'long': 'Mali'}, {'short': 'MT', 'long': 'Malta'}, {'short': 'MM', 'long': 'Myanmar'}, {'short': 'ME', 'long': 'Montenegro'}, {'short': 'MN', 'long': 'Mongolia'}, {'short': 'MP', 'long': 'Northern Mariana Islands'}, {'short': 'MZ', 'long': 'Mozambique'}, {'short': 'MR', 'long': 'Mauritania'}, {'short': 'MS', 'long': 'Montserrat'}, {'short': 'MQ', 'long': 'Martinique'}, {'short': 'MU', 'long': 'Mauritius'}, {'short': 'MW', 'long': 'Malawi'}, {'short': 'MY', 'long': 'Malaysia'}, {'short': 'YT', 'long': 'Mayotte'}, {'short': 'NA', 'long': 'Namibia'}, {'short': 'NC', 'long': 'New Caledonia'}, {'short': 'NE', 'long': 'Niger'}, {'short': 'NF', 'long': 'Norfolk Island'}, {'short': 'NG', 'long': 'Nigeria'}, {'short': 'NI', 'long': 'Nicaragua'}, {'short': 'NU', 'long': 'Niue'}, {'short': 'NL', 'long': 'Netherlands'}, {'short': 'NO', 'long': 'Norway'}, {'short': 'NP', 'long': 'Nepal'}, {'short': 'NR', 'long': 'Nauru'}, {'short': 'NZ', 'long': 'New Zealand'}, {'short': 'OM', 'long': 'Oman'}, {'short': 'PK', 'long': 'Pakistan'}, {'short': 'PA', 'long': 'Panama'}, {'short': 'PN', 'long': 'Pitcairn'}, {'short': 'PE', 'long': 'Peru'}, {'short': 'PH', 'long': 'Philippines'}, {'short': 'PW', 'long': 'Palau'}, {'short': 'PG', 'long': 'Papua New Guinea'}, {'short': 'PL', 'long': 'Poland'}, {'short': 'PR', 'long': 'Puerto Rico'}, {'short': 'KP', 'long': "Korea, Democratic People's Republic of"}, {'short': 'PT', 'long': 'Portugal'}, {'short': 'PY', 'long': 'Paraguay'}, {'short': 'PS', 'long': 'Palestine, State of'}, {'short': 'PF', 'long': 'French Polynesia'}, {'short': 'QA', 'long': 'Qatar'}, {'short': 'RE', 'long': 'Réunion'}, {'short': 'RO', 'long': 'Romania'}, {'short': 'RU', 'long': 'Russian Federation'}, {'short': 'RW', 'long': 'Rwanda'}, {'short': 'SA', 'long': 'Saudi Arabia'}, {'short': 'SD', 'long': 'Sudan'}, {'short': 'SN', 'long': 'Senegal'}, {'short': 'SG', 'long': 'Singapore'}, {'short': 'GS', 'long': 'South Georgia and the South Sandwich Islands'}, {'short': 'SH', 'long': 'Saint Helena, Ascension and Tristan da Cunha'}, {'short': 'SJ', 'long': 'Svalbard and Jan Mayen'}, {'short': 'SB', 'long': 'Solomon Islands'}, {'short': 'SL', 'long': 'Sierra Leone'}, {'short': 'SV', 'long': 'El Salvador'}, {'short': 'SM', 'long': 'San Marino'}, {'short': 'SO', 'long': 'Somalia'}, {'short': 'PM', 'long': 'Saint Pierre and Miquelon'}, {'short': 'RS', 'long': 'Serbia'}, {'short': 'SS', 'long': 'South Sudan'}, {'short': 'ST', 'long': 'Sao Tome and Principe'}, {'short': 'SR', 'long': 'Suriname'}, {'short': 'SK', 'long': 'Slovakia'}, {'short': 'SI', 'long': 'Slovenia'}, {'short': 'SE', 'long': 'Sweden'}, {'short': 'SZ', 'long': 'Eswatini'}, {'short': 'SX', 'long': 'Sint Maarten (Dutch part)'}, {'short': 'SC', 'long': 'Seychelles'}, {'short': 'SY', 'long': 'Syrian Arab Republic'}, {'short': 'TC', 'long': 'Turks and Caicos Islands'}, {'short': 'TD', 'long': 'Chad'}, {'short': 'TG', 'long': 'Togo'}, {'short': 'TH', 'long': 'Thailand'}, {'short': 'TJ', 'long': 'Tajikistan'}, {'short': 'TK', 'long': 'Tokelau'}, {'short': 'TM', 'long': 'Turkmenistan'}, {'short': 'TL', 'long': 'Timor-Leste'}, {'short': 'TO', 'long': 'Tonga'}, {'short': 'TT', 'long': 'Trinidad and Tobago'}, {'short': 'TN', 'long': 'Tunisia'}, {'short': 'TR', 'long': 'Türkiye'}, {'short': 'TV', 'long': 'Tuvalu'}, {'short': 'TW', 'long': 'Taiwan, Province of China'}, {'short': 'TZ', 'long': 'Tanzania, United Republic of'}, {'short': 'UG', 'long': 'Uganda'}, {'short': 'UA', 'long': 'Ukraine'}, {'short': 'UM', 'long': 'United States Minor Outlying Islands'}, {'short': 'UY', 'long': 'Uruguay'}, {'short': 'US', 'long': 'United States'}, {'short': 'UZ', 'long': 'Uzbekistan'}, {'short': 'VA', 'long': 'Holy See (Vatican City State)'}, {'short': 'VC', 'long': 'Saint Vincent and the Grenadines'}, {'short': 'VE', 'long': 'Venezuela, Bolivarian Republic of'}, {'short': 'VG', 'long': 'Virgin Islands, British'}, {'short': 'VI', 'long': 'Virgin Islands, U.S.'}, {'short': 'VN', 'long': 'Viet Nam'}, {'short': 'VU', 'long': 'Vanuatu'}, {'short': 'WF', 'long': 'Wallis and Futuna'}, {'short': 'WS', 'long': 'Samoa'}, {'short': 'YE', 'long': 'Yemen'}, {'short': 'ZA', 'long': 'South Africa'}, {'short': 'ZM', 'long': 'Zambia'}, {'short': 'ZW', 'long': 'Zimbabwe'}]


    async with aiohttp.ClientSession() as session:

        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={start_date}&to={end_date}&apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            for item in data:
                date_obj = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
                item['date'] = date_obj.strftime('%Y-%m-%d')
                item['time'] = date_obj.strftime('%H:%M')


                for country in country_list:
                    if country['short'] == item['country']:
                        if country['long'] == 'Korea, Republic of':
                            item['country'] = 'Korea'
                        elif country['long'] == 'Russian Federation':
                            item['country'] = 'Russia'
                        elif country['long'] == 'Taiwan, Province of China':
                            item['country'] = 'Taiwan'
                        else:
                            item['country'] = country['long']
                        item['countryCode'] = country['short'].lower()


    return data

async def get_ai_signals(con,symbols):
        
    query = f"""
        SELECT
            symbol,
            name,
            marketCap,
            avgVolume,
            tradingSignals
        FROM
            stocks
        WHERE
            symbol = ?
    """
    res_list = []
    selected_data = []


    for ticker in symbols:
        try:
            # Execute the query and read the result into a DataFrame
            query_result = pd.read_sql_query(query, con, params=(ticker,))
            
            # Convert the DataFrame to a JSON object
            if not query_result.empty:
                res = query_result['tradingSignals'][0]
                res = ujson.loads(res)[0]  # Assuming 'tradingSignals' column contains JSON strings        
                res = replace_nan_inf_with_none(res)
                avgVolume = int(query_result['avgVolume'].iloc[0])

                # Check if "Win Rate [%]" is a number before adding it to the list
                if "Win Rate [%]" in res and isinstance(res["Win Rate [%]"], (int, float)) and "# Trades" in res and res["# Trades"] >= 10 and avgVolume >= 20000:
                    res['symbol'] = query_result['symbol'][0]
                    res['name'] = query_result['name'][0]
                    try:
                        res['marketCap'] = int(query_result['marketCap'].iloc[0])
                    except:
                        res['marketCap'] = None
                    res_list.append(res)
            else:
                pass
        except Exception as e:
            print("Error fetching data from the database:", e)
            pass

    sorted_res = sorted(res_list, key=lambda x: x.get("Win Rate [%]", 0), reverse=True)

    for item in sorted_res[0:50]:
        selected_item = {
            "symbol": item.get("symbol", ""),
            "name": item.get("name", ""),
            "marketCap": item.get("marketCap", 0),
            "winRate": round(item.get("Win Rate [%]", 0),2),
            "maxDrawdown": round(item.get("Max. Drawdown [%]", 0),2),
            "return": round(item.get("Return [%]", 0),2),
            "nextSignal": item.get("nextSignal", ""),
        }
        selected_data.append(selected_item)

    return selected_data


async def get_index_list(con,symbols, index_list):
    
    async with aiohttp.ClientSession() as session:

        query_template = """
            SELECT 
                price, changesPercentage, marketCap, revenue, netIncome
            FROM 
                stocks 
            WHERE
                symbol = ?
        """
        
        url = f"https://financialmodelingprep.com/api/v3/{index_list}?apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
            filtered_data = [{k: v for k, v in stock.items() if stock['symbol'] in symbols} for stock in data]
            filtered_data = [entry for entry in filtered_data if entry]

            res_list = []
            for entry in filtered_data:
                query_data = pd.read_sql_query(query_template, con, params=(entry['symbol'],))

                if query_data['marketCap'].iloc[0] != None and query_data['revenue'].iloc[0] !=None and query_data['price'].iloc[0] != None and query_data['changesPercentage'].iloc[0] != None:
                    entry['marketCap'] = int(query_data['marketCap'].iloc[0])
                    entry['revenue'] = int(query_data['revenue'].iloc[0])
                    entry['netIncome'] = int(query_data['netIncome'].iloc[0])
                    entry['price'] = round(float(query_data['price'].iloc[0]),2)
                    entry['changesPercentage'] = round(float(query_data['changesPercentage'].iloc[0]),2)
                    res_list.append(entry)

    sorted_res_list = sorted(res_list, key=lambda x: x['marketCap'], reverse=True)
    return sorted_res_list


async def get_delisted_list():
    
    async with aiohttp.ClientSession() as session:
        url = f"https://financialmodelingprep.com/api/v3/delisted-companies?apikey={api_key}"
        async with session.get(url) as response:
            data = await response.json()
    return data




def replace_representative(office):
    replacements = {
        'Carper, Thomas R. (Senator)': 'Tom Carper',
        'Thomas R. Carper': 'Tom Carper',
        'Tuberville, Tommy (Senator)': 'Tommy Tuberville',
        'Ricketts, Pete (Senator)': 'John Ricketts',
        'Pete Ricketts': 'John Ricketts',
        'Moran, Jerry (Senator)': 'Jerry Moran',
        'Fischer, Deb (Senator)': 'Deb Fischer',
        'Mullin, Markwayne (Senator)': 'Markwayne Mullin',
        'Whitehouse, Sheldon (Senator)': 'Sheldon Whitehouse',
        'Toomey, Pat (Senator)': 'Pat Toomey',
        'Sullivan, Dan (Senator)': 'Dan Sullivan',
        'Capito, Shelley Moore (Senator)': 'Shelley Moore Capito',
        'Roberts, Pat (Senator)': 'Pat Roberts',
        'King, Angus (Senator)': 'Angus King',
        'Hoeven, John (Senator)': 'John Hoeven',
        'Duckworth, Tammy (Senator)': 'Tammy Duckworth',
        'Perdue, David (Senator)': 'David Perdue',
        'Inhofe, James M. (Senator)': 'James M. Inhofe',
        'Murray, Patty (Senator)': 'Patty Murray',
        'Boozman, John (Senator)': 'John Boozman',
        'Loeffler, Kelly (Senator)': 'Kelly Loeffler',
        'Reed, John F. (Senator)': 'John F. Reed',
        'Collins, Susan M. (Senator)': 'Susan M. Collins',
        'Cassidy, Bill (Senator)': 'Bill Cassidy',
        'Wyden, Ron (Senator)': 'Ron Wyden',
        'Hickenlooper, John (Senator)': 'John Hickenlooper',
        'Booker, Cory (Senator)': 'Cory Booker',
        'Donald Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Peters, Gary (Senator)': 'Gary Peters',
        'Donald Sternoff Beyer, (Senator).': 'Donald Sternoff Beyer',
        'Donald S. Beyer, Jr.': 'Donald Sternoff Beyer',
        'Donald Sternoff Honorable Beyer': 'Donald Sternoff Beyer',
        'K. Michael Conaway': 'Michael Conaway',
        'C. Scott Franklin': 'Scott Franklin',
        'Robert C. "Bobby" Scott': 'Bobby Scott',
        'Madison Cawthorn': 'David Madison Cawthorn',
        'Cruz, Ted (Senator)': 'Ted Cruz',
        'Smith, Tina (Senator)': 'Tina Smith',
        'Graham, Lindsey (Senator)': 'Lindsey Graham',
        'Hagerty, Bill (Senator)': 'Bill Hagerty',
        'Scott, Rick (Senator)': 'Rick Scott',
        'Warner, Mark (Senator)': 'Mark Warner',
        'McConnell, A. Mitchell Jr. (Senator)': 'Mitch McConnell',
        'Mitchell McConnell': 'Mitch McConnell',
        'Charles J. "Chuck" Fleischmann': 'Chuck Fleischmann',
        'Vance, J.D. (Senator)': 'James Vance',
        'Neal Patrick MD, Facs Dunn': 'Neal Dunn',
        'Neal Patrick MD, Facs Dunn (Senator)': 'Neal Dunn',
        'Neal Patrick Dunn, MD, FACS': 'Neal Dunn',
        'Neal P. Dunn': 'Neal Dunn',
        'Tillis, Thom (Senator)': 'Thom Tillis',
        'W. Gregory Steube': 'Greg Steube',
        'W. Grego Steube': 'Greg Steube',
        'W. Greg Steube': 'Greg Steube',
        'David David Madison Cawthorn': 'David Madison Cawthorn',
        'Blunt, Roy (Senator)': 'Roy Blunt',
        'Thune, John (Senator)': 'John Thune',
        'Rosen, Jacky (Senator)': 'Jacky Rosen',
        'James Costa': 'Jim Costa',
        'Lummis, Cynthia (Senator)': 'Cynthia Lummis',
        'Coons, Chris (Senator)': 'Chris Coons',
        'Udall, Tom (Senator)': 'Tom Udall',
        'Kennedy, John (Senator)': 'John Kennedy',
        'Bennet, Michael (Senator)': 'Michael Bennet',
        'Casey, Robert P. Jr. (Senator)': 'Robert Casey',
        'Van Hollen, Chris (Senator)': 'Chris Van Hollen',
        'Manchin, Joe (Senator)': 'Joe Manchin',
        'Cornyn, John (Senator)': 'John Cornyn',
        'Enzy, Michael (Senator)': 'Michael Enzy',
        'Cardin, Benjamin (Senator)': 'Benjamin Cardin',
        'Kaine, Tim (Senator)': 'Tim Kaine',
        'Joseph P. Kennedy III': 'Joe Kennedy',
        'James E Hon Banks': 'Jim Banks',
        'Michael F. Q. San Nicolas': 'Michael San Nicolas',
        'Barbara J Honorable Comstock': 'Barbara Comstock',
        'Mr ': '',
        'Mr. ': '',
        'Dr ': '',
        'Dr. ': '',
        'Mrs ': '',
        'Mrs. ': '',
        '(Senator)': '',
    }

    for old, new in replacements.items():
        office = office.replace(old, new)
        office = ' '.join(office.split())
    return office

async def get_congress_rss_feed(symbols, etf_symbols, crypto_symbols):

    amount_mapping = {
    '$1,001 -': '$1K-$15K',
    '$1,001 - $15,000': '$1K-$15K',
    '$15,001 - $50,000': '$15K-$50K',
    '$15,001 -': '$15K-$50K',
    '$50,001 - $100,000': '$50K-$100K',
    '$100,001 - $250,000': '$100K-$250K',
    '$100,001 - $500,000': '$100K-$500K',
    '$250,001 - $500,000': '$250K-$500K',
    '$500,001 - $1,000,000': '$500K-$1M',
    '$1,000,001 - $5,000,000': '$1M-$5M',
    'Spouse/DC Over $1,000,000': 'Over $1M'
    }

    urls = [f"https://financialmodelingprep.com/api/v4/senate-disclosure-rss-feed?page=0&apikey={api_key}",
            f"https://financialmodelingprep.com/api/v4/senate-disclosure-rss-feed?page=1&apikey={api_key}"]

    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        data = [await response.json() for response in responses]

    data = data[0] +data[1]
    congressional_districts = {"UT": "Utah","CA": "California","NY": "New York","TX": "Texas","FL": "Florida","IL": "Illinois","PA": "Pennsylvania","OH": "Ohio","GA": "Georgia","MI": "Michigan","NC": "North Carolina","AZ": "Arizona","WA": "Washington","CO": "Colorado","OR": "Oregon","VA": "Virginia","NJ": "New Jersey","TN": "Tennessee","MA": "Massachusetts","WI": "Wisconsin","SC": "South Carolina","KY": "Kentucky","LA": "Louisiana","AR": "Arkansas","AL": "Alabama","MS": "Mississippi","NDAL": "North Dakota","SDAL": "South Dakota","MN": "Minnesota","IA": "Iowa","OK": "Oklahoma","ID": "Idaho","NH": "New Hampshire","NE": "Nebraska","MTAL": "Montana","WYAL": "Wyoming","WV": "West Virginia","VTAL": "Vermont","DEAL": "Delaware","RI": "Rhode Island","ME": "Maine","HI": "Hawaii","AKAL": "Alaska","NM": "New Mexico","KS": "Kansas","MS": "Mississippi","CT": "Connecticut","MD": "Maryland","NV": "Nevada",}
    
    for item in data:
        ticker = item.get("ticker")
        ticker = ticker.replace('BRK.A','BRK-A')
        ticker = ticker.replace('BRK.B','BRK-B')
        
        if item['assetDescription'] == 'Bitcoin':
            item['ticker'] = 'BTCUSD'
            ticker = item.get("ticker")
        
        item['assetDescription'] = item['assetDescription'].replace('U.S','US')


        if 'Sale' in item['type']:
            item['type'] = 'Sold'
        if 'Purchase' in item['type']:
            item['type'] = 'Bought'

        item['amount'] = amount_mapping.get(item['amount'], item['amount'])


        item['ticker'] = ticker
        if ticker in symbols:
           item["assetType"] = "stock"
        elif ticker in etf_symbols:
            item["assetType"] = "etf"
        elif ticker in crypto_symbols:
            item['assetType'] = "crypto"
        else:
            item['assetType'] = ''

        if 'representative' in item:
            item['representative'] = replace_representative(item['representative'])

        item['id'] = generate_id(item['representative'])

        # Check if 'district' key exists in item
        if 'district' in item:
            # Extract state code from the 'district' value
            state_code = item['district'][:2]
            
            # Replace 'district' value with the corresponding value from congressional_districts
            item['district'] = f"{congressional_districts.get(state_code, state_code)}"

    return data


async def get_analysts_rss_feed(con, symbols, etf_symbols):
    urls = [
        f"https://financialmodelingprep.com/api/v4/price-target-rss-feed?page=0&apikey={api_key}",
        f"https://financialmodelingprep.com/api/v4/upgrades-downgrades-rss-feed?page=0&apikey={api_key}",
    ]

    query_template = """
        SELECT 
            name, quote
        FROM 
            stocks 
        WHERE
            symbol = ?
    """

    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)

        data = [await response.json() for response in responses]
        price_targets_list = [
        {
            "symbol": entry["symbol"],
            "publishedDate": entry["publishedDate"],
            "analystName": entry["analystName"],
            "adjPriceTarget": entry["adjPriceTarget"],
            "priceWhenPosted": entry["priceWhenPosted"],
            "analystCompany": entry["analystCompany"],
        }
        for entry in data[0]
        ]
        #Add ticker name
        for entry in price_targets_list:
            try:
                symbol = entry['symbol']
                df = pd.read_sql_query(query_template, con, params=(symbol,))
                entry['name'] = df['name'].iloc[0]
            except:
                entry['name'] = 'n/a'
        #Add ticker assetType
        for item in price_targets_list:
            symbol = item.get("symbol")
            symbol = symbol.replace('BRK.A','BRK-A')
            symbol = symbol.replace('BRK.B','BRK-B')
            item['symbol'] = symbol
            if symbol in symbols:
               item["assetType"] = "Stock"
            elif symbol in etf_symbols:
                item["assetType"] = "ETF"
            else:
                item['assetType'] = ''

        #Remove elements who have assetType = '' or priceWhenPosted = 0
        #price_targets_list = [item for item in price_targets_list if item.get("assetType") != ""]
        price_targets_list = [item for item in price_targets_list if item.get("assetType") != ""]
        price_targets_list = [item for item in price_targets_list if item.get("priceWhenPosted") != 0]



        upgrades_downgrades_list = [
        {
            "symbol": entry["symbol"],
            "publishedDate": entry["publishedDate"],
            "newGrade": entry["newGrade"],
            "previousGrade": entry["previousGrade"],
            "priceWhenPosted": entry["priceWhenPosted"],
            "gradingCompany": entry["gradingCompany"],
            "action": entry["action"],
        }
        for entry in data[1]
        ]

        #Add ticker name
        new_upgrades_downgrades_list = []
        for entry in upgrades_downgrades_list:
            try:
                symbol = entry['symbol']
                df = pd.read_sql_query(query_template, con, params=(symbol,))
                entry['name'] = df['name'].iloc[0]
                entry['currentPrice'] =  (ujson.loads(df['quote'].iloc[0])[0]).get('price')

                new_upgrades_downgrades_list.append(entry)
            except:
                #Remove all elements that don't have a name and currentPrice in the db for better UX with new_upgrades_downgrades_list
                pass

        #Add ticker assetType
        for item in new_upgrades_downgrades_list:
            symbol = item.get("symbol")
            symbol = symbol.replace('BRK.A','BRK-A')
            symbol = symbol.replace('BRK.B','BRK-B')
            item['symbol'] = symbol
            if symbol in symbols:
               item["assetType"] = "Stock"
            elif symbol in etf_symbols:
                item["assetType"] = "ETF"
            else:
                item['assetType'] = ''

        #Remove elements who have assetType = ''
        new_upgrades_downgrades_list = [item for item in new_upgrades_downgrades_list if item.get("assetType") != ""]
        new_upgrades_downgrades_list = [item for item in new_upgrades_downgrades_list if item.get("priceWhenPosted") != 0]

    return price_targets_list, new_upgrades_downgrades_list


async def ticker_mentioning(con):
    results = pb.collection("posts").get_full_list()

    symbol_list = []
    
    query_template = """
        SELECT 
            name, marketCap
        FROM 
            stocks 
        WHERE
            symbol = ?
    """
    

    for x in results:
        if len(x.tagline) != 0:
            symbol_list.append(x.tagline)

    symbol_counts = Counter(symbol_list)
    symbol_counts_list = [{'symbol': symbol, 'count': count} for symbol, count in symbol_counts.items()]
    sorted_symbol_list = sorted(symbol_counts_list, key=lambda x: x['count'], reverse=True)
    
    for entry in sorted_symbol_list:
        try:
            symbol = entry['symbol']
            data = pd.read_sql_query(query_template, con, params=(symbol,))
            entry['name'] = data['name'].iloc[0]
            entry['marketCap'] = int(data['marketCap'].iloc[0])
        except:
            entry['name'] = 'n/a'
            entry['marketCap'] = None

    return sorted_symbol_list


async def get_all_stock_tickers(con):
    cursor = con.cursor()
    cursor.execute("SELECT symbol, name, marketCap, sector FROM stocks WHERE symbol != ? AND marketCap IS NOT NULL", ('%5EGSPC',))
    raw_data = cursor.fetchall()

    # Extract only relevant data and sort it
    stock_list_data = sorted([{'symbol': row[0], 'name': row[1], 'marketCap': row[2], 'sector': row[3]} for row in raw_data], key=custom_symbol_sort)
    return stock_list_data

async def get_all_etf_tickers(etf_con):
    cursor = etf_con.cursor()
    cursor.execute("SELECT symbol, name, totalAssets, numberOfHoldings FROM etfs WHERE totalAssets IS NOT NULL")
    raw_data = cursor.fetchall()

    # Extract only relevant data and sort it
    etf_list_data = sorted([{'symbol': row[0], 'name': row[1], 'totalAssets': row[2], 'numberOfHoldings': row[3]} for row in raw_data], key=custom_symbol_sort)
    return etf_list_data

async def get_all_crypto_tickers(crypto_con):
    cursor = crypto_con.cursor()
    cursor.execute("SELECT symbol, name, marketCap, circulatingSupply, maxSupply FROM cryptos")
    raw_data = cursor.fetchall()

    # Extract only relevant data and sort it
    crypto_list_data = sorted([{'symbol': row[0], 'name': row[1], 'marketCap': row[2], 'circulatingSupply': row[3], 'maxSupply': row[4]} for row in raw_data], key=custom_symbol_sort)
    return crypto_list_data


async def get_magnificent_seven(con):
  
    symbol_list = ['MSFT','AAPL','GOOGL','AMZN','NVDA','META','TSLA']
    
    query_template = """
        SELECT 
            symbol, name, price, changesPercentage, revenue, netIncome, marketCap,pe
        FROM 
            stocks 
        WHERE
            symbol = ?
    """
    res_list = []
    for symbol in symbol_list:
        try:
            data = pd.read_sql_query(query_template, con, params=(symbol,))
            
            name = data['name'].iloc[0]

            price = round(float(data['price'].iloc[0]),2)
            changesPercentage = round(float(data['changesPercentage'].iloc[0]),2)
            marketCap = int(data['marketCap'].iloc[0])
            revenue = int(data['revenue'].iloc[0])
            netIncome = int(data['netIncome'].iloc[0])
            pe = round(float(data['pe'].iloc[0]),2)

            res_list.append({'symbol': symbol, 'name': name, 'price': price, \
                    'changesPercentage': changesPercentage, 'marketCap': marketCap, \
                    'revenue': revenue, 'netIncome': netIncome, 'pe': pe})

        except Exception as e:
            print(e)

    return res_list

async def etf_providers(etf_con, etf_symbols):

    etf_provider_list = []

    query_template = """
        SELECT 
            symbol, etfProvider, expenseRatio, totalAssets, numberOfHoldings
        FROM 
            etfs
        WHERE
            symbol = ?
    """
    
    for symbol in etf_symbols:
        try:
            data = pd.read_sql_query(query_template, etf_con, params=(symbol,))
            etf_provider = data['etfProvider'].iloc[0]
            expense_ratio = float(data['expenseRatio'].iloc[0])
            total_assets = int(data['totalAssets'].iloc[0])
            number_of_holdings = int(data['numberOfHoldings'].iloc[0])

            etf_provider_list.append(
                {'symbol': symbol,
                'etfProvider': etf_provider,
                'expenseRatio': expense_ratio,
                'totalAssets': total_assets,
                'numberOfHoldings': number_of_holdings
                }
            )

        except:
            pass
    # Dictionary to store counts and total expense ratios for each etfProvider
    etf_provider_stats = {}

    # Iterate through the list and update the dictionary
    for etf in etf_provider_list:
        etf_provider = etf['etfProvider']
        expense_ratio = etf['expenseRatio']
        number_of_holdings = etf['numberOfHoldings']
        total_assets = etf['totalAssets']

        if etf_provider in etf_provider_stats:
            etf_provider_stats[etf_provider]['count'] += 1
            etf_provider_stats[etf_provider]['totalExpenseRatio'] += expense_ratio
            etf_provider_stats[etf_provider]['totalNumberOfHoldings'] += number_of_holdings
            etf_provider_stats[etf_provider]['totalAssets'] += total_assets
        else:
            etf_provider_stats[etf_provider] = {'count': 1, 'totalExpenseRatio': expense_ratio, 'totalAssets': total_assets, 'totalNumberOfHoldings': number_of_holdings}

    # Create the new list with average expense ratio
    result_list = [
        {'etfProvider': provider, 'funds': stats['count'], 'totalAssets': stats['totalAssets'] ,'avgExpenseRatio': round(stats['totalExpenseRatio'] / stats['count'],2), 'avgHoldings': int(stats['totalNumberOfHoldings'] / stats['count'])}
        for provider, stats in etf_provider_stats.items()
    ]
    result_list = sorted(result_list, key=lambda x: x['totalAssets'], reverse=True)

    return result_list

async def etf_bitcoin_list(etf_con, etf_symbols):


    result_list = []

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
            if ('Bitcoin' or 'bitcoin') in name:
                expense_ratio = round(float(data['expenseRatio'].iloc[0]),2)
                total_assets = int(data['totalAssets'].iloc[0])

                result_list.append(
                    {'symbol': symbol,
                    'name': name,
                    'expenseRatio': expense_ratio,
                    'totalAssets': total_assets
                    }
                )
            else:
                pass

        except Exception as e:
            print(e)
    
    result_list = sorted(result_list, key=lambda x: x['totalAssets'], reverse=True)

    return result_list



async def get_ipo_calendar(con, symbols):
    # Define function to get end date of each quarter
    import datetime
    def get_end_of_quarter(year, quarter):
        month = quarter * 3
        return datetime.date(year, month, 1) + datetime.timedelta(days=30)

    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date.today()
    urls = []
    combined_data = []
    query_quote = """
        SELECT 
            quote
        FROM 
            stocks 
        WHERE
            symbol = ?
    """
    query_open_price = """
        SELECT open
        FROM "{ticker}"
        LIMIT 1
    """

    # Iterate through quarters
    current_date = start_date
    while current_date < end_date:
        # Get end date of current quarter
        end_of_quarter = get_end_of_quarter(current_date.year, (current_date.month - 1) // 3 + 1)
        
        # Ensure end date does not exceed end_date
        if end_of_quarter > end_date:
            end_of_quarter = end_date
        
        # Construct URL with current quarter's start and end dates
        url = f"https://financialmodelingprep.com/api/v3/ipo_calendar?from={current_date}&to={end_of_quarter}&apikey={api_key}"
        
        # Append URL to list
        urls.append(url)
        
        # Move to next quarter
        current_date = end_of_quarter + datetime.timedelta(days=1)

    #print(urls)
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        data = [await response.json() for response in responses]
    
    for sublist in data:
        for item in sublist:
            if item not in combined_data and item['symbol'] in symbols and item['exchange'] in ['NASDAQ Global','NASDAQ Capital','NASDAQ Global Select','NYSE','NASDAQ','Nasdaq','Nyse','Amex']:
                if item['priceRange'] != None:
                    item['priceRange'] = round(float(item['priceRange'].split('-')[0]),2)

                combined_data.append(item)

    res = []
    for entry in combined_data:
        df = pd.read_sql_query(query_quote, con, params=(entry['symbol'],))
        try:
            entry['currentPrice'] = round((ujson.loads(df['quote'].iloc[0])[0]).get('price'),2)
        except:
            entry['currentPrice'] = None
        try:
            entry['marketCap'] = (ujson.loads(df['quote'].iloc[0])[0]).get('marketCap')
        except:
            entry['marketCap'] = None
        try:
            df =  pd.read_sql_query(query_open_price.format(ticker = entry['symbol']), con)
            entry['ipoPrice'] = round(df['open'].iloc[0], 2) if df['open'].iloc[0] != 0 else None
        except:
            entry['ipoPrice'] = entry['priceRange']

        entry['return'] = None if (entry['ipoPrice'] in (0, None) or entry['currentPrice'] in (0, None)) else round(((entry['currentPrice'] / entry['ipoPrice'] - 1) * 100), 2)
        
        res.append({
            "symbol": entry["symbol"],
            "name": entry["company"],
            "date": entry["date"],
            "marketCap": entry["marketCap"],
            "ipoPrice": entry["ipoPrice"],
            "currentPrice": entry["currentPrice"],
            "return": entry["return"],
        })
    
    res_sorted = sorted(res, key=lambda x: x['date'], reverse=True)

    return res_sorted


async def save_json_files():
    week = datetime.today().weekday()
    if week <= 7:
        con = sqlite3.connect('stocks.db')
        etf_con = sqlite3.connect('etf.db')
        crypto_con = sqlite3.connect('crypto.db')

        cursor = con.cursor()
        cursor.execute("PRAGMA journal_mode = wal")
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]

        etf_cursor = etf_con.cursor()
        etf_cursor.execute("PRAGMA journal_mode = wal")
        etf_cursor.execute("SELECT DISTINCT symbol FROM etfs")
        etf_symbols = [row[0] for row in etf_cursor.fetchall()]

        crypto_cursor = crypto_con.cursor()
        crypto_cursor.execute("PRAGMA journal_mode = wal")
        crypto_cursor.execute("SELECT DISTINCT symbol FROM cryptos")
        crypto_symbols = [row[0] for row in crypto_cursor.fetchall()]


        data = await get_congress_rss_feed(symbols, etf_symbols, crypto_symbols)
        with open(f"json/congress-trading/rss-feed/data.json", 'w') as file:
            ujson.dump(data, file)

        
        data = await get_magnificent_seven(con)
        with open(f"json/magnificent-seven/data.json", 'w') as file:
            ujson.dump(data, file)

        earnings_list = await get_earnings_calendar(con,symbols)
        with open(f"json/earnings-calendar/calendar.json", 'w') as file:
            ujson.dump(earnings_list, file)
        
        data = await get_ipo_calendar(con, symbols)
        with open(f"json/ipo-calendar/data.json", 'w') as file:
            ujson.dump(data, file)

        data = await get_all_stock_tickers(con)
        with open(f"json/all-symbols/stocks.json", 'w') as file:
            ujson.dump(data, file)

        data = await get_all_etf_tickers(etf_con)
        with open(f"json/all-symbols/etfs.json", 'w') as file:
            ujson.dump(data, file)

        data = await get_all_crypto_tickers(crypto_con)
        with open(f"json/all-symbols/cryptos.json", 'w') as file:
            ujson.dump(data, file)

        
        data = await etf_bitcoin_list(etf_con, etf_symbols)
        with open(f"json/etf-bitcoin-list/data.json", 'w') as file:
            ujson.dump(data, file)
        
        data = await etf_providers(etf_con, etf_symbols)
        with open(f"json/all-etf-providers/data.json", 'w') as file:
            ujson.dump(data, file)

        data = await ticker_mentioning(con)
        with open(f"json/ticker-mentioning/data.json", 'w') as file:
            ujson.dump(data, file)

        delisted_data = await get_delisted_list()
        with open(f"json/delisted-companies/data.json", 'w') as file:
            ujson.dump(delisted_data, file)

        economic_list = await get_economic_calendar()
        with open(f"json/economic-calendar/calendar.json", 'w') as file:
            ujson.dump(economic_list, file)

        dividends_list = await get_dividends_calendar(con,symbols)
        with open(f"json/dividends-calendar/calendar.json", 'w') as file:
            ujson.dump(dividends_list, file)
                
        stock_splits_data = await get_stock_splits_calendar(con,symbols)
        with open(f"json/stock-splits-calendar/calendar.json", 'w') as file:
            ujson.dump(stock_splits_data, file)

        #Stocks Lists
        data = await get_index_list(con,symbols,'nasdaq_constituent')
        with open(f"json/stocks-list/nasdaq_constituent.json", 'w') as file:
            ujson.dump(data, file)

        data = await get_index_list(con,symbols,'dowjones_constituent')
        with open(f"json/stocks-list/dowjones_constituent.json", 'w') as file:
            ujson.dump(data, file)

        data = await get_index_list(con,symbols,'sp500_constituent')
        with open(f"json/stocks-list/sp500_constituent.json", 'w') as file:
            ujson.dump(data, file)
        

        stock_screener_data = await get_stock_screener(con,symbols)
        with open(f"json/stock-screener/data.json", 'w') as file:
            ujson.dump(stock_screener_data, file)
    
    

        con.close()
        etf_con.close()
        crypto_con.close()
    
        
try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_json_files())
except Exception as e:
    print(e)
