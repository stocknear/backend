import ujson
import sqlite3
import asyncio
import pandas as pd
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup
import re

class Short_Data:
    def __init__(self, data):
        self.short_interest_ratio_days_to_cover = data.get('shortInterestRatioDaysToCover')
        self.short_percent_of_float = data.get('shortPercentOfFloat')
        self.short_percent_increase_decrease = data.get('shortPercentIncreaseDecrease')
        self.short_interest_current_shares_short = data.get('shortInterestCurrentSharesShort')
        self.shares_float = data.get('sharesFloat')
        self.short_interest_prior_shares_short = data.get('shortInterestPriorSharesShort')
        self.percent_from_52_wk_high = data.get('percentFrom52WkHigh')
        self.percent_from_50_day_ma = data.get('percentFrom50DayMa')
        self.percent_from_200_day_ma = data.get('percentFrom200DayMa')
        self.percent_from_52_wk_low = data.get('percentFrom52WkLow')
        self.n_52_week_performance = data.get('n52WeekPerformance')
        self.trading_volume_today_vs_avg = data.get('tradingVolumeTodayVsAvg')
        self.trading_volume_today = data.get('tradingVolumeToday')
        self.trading_volume_average = data.get('tradingVolumeAverage')
        self.market_cap = data.get('marketCap')
        self.percent_owned_by_insiders = data.get('percentOwnedByInsiders')
        self.percent_owned_by_institutions = data.get('percentOwnedByInstitutions')
        self.price = data.get('price')
        self.name = data.get('name')
        self.ticker = data.get('ticker')

def camel_case(s):
    s = re.sub(r'[^A-Za-z0-9 ]+', '', s)
    s = s.replace('%', 'Percent')
    s = re.sub(r'(\d)', r'n\1', s)
    s = re.sub(r'(\d+)', '', s)
    parts = s.split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

def parse_stock_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    table_rows = soup.select('div.inner_box_2 > table > tr')
    parsed_data = {}
    
    for row in table_rows:
        try:
            key_element = row.select_one('td:nth-child(1)')
            value_element = row.select_one('td:nth-child(2)')
            if key_element and value_element:
                key = camel_case(key_element.get_text().strip())
                value = value_element.get_text().strip()

                # Clean and convert value
                if 'view' in value.lower():
                    value = None
                else:
                    value = re.sub(r'[\s%,\$]', '', value)
                    value = float(value) if value and value.replace('.', '', 1).isdigit() else value

                if key:
                    parsed_data[key] = value
        except:
            pass

    # Add price, name, and ticker separately
    price = float(table_rows[0].select_one('td:nth-child(2)').get_text().strip().replace('$', '') or 'NaN')
    name = table_rows[0].select_one('td').get_text().strip()
    ticker = table_rows[1].select_one('td').get_text().strip()

    parsed_data.update({
        'price': price,
        'name': name,
        'ticker': ticker
    })

    return Short_Data(parsed_data) if name.lower() != 'not available - try again' else None

def shortsqueeze(ticker=''):
    try:
        url = f'https://shortsqueeze.com/?symbol={ticker}'
        response = requests.get(url, allow_redirects=False)
        if response.status_code == 200:
            return parse_stock_data(response.text)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


async def save_as_json(symbol, data):
    with open(f"json/shareholders/{symbol}.json", 'w') as file:
        ujson.dump(data, file)


query_template = f"""
    SELECT 
        shareholders
    FROM 
        stocks
    WHERE
        symbol = ?
"""

async def get_data(ticker, con):

    try:
        df = pd.read_sql_query(query_template, con, params=(ticker,))
        shareholders_list = ujson.loads(df.to_dict()['shareholders'][0])
        # Keys to keep
        keys_to_keep = ["cik","ownership", "investorName", "weight", "sharesNumber", "marketValue"]

        # Create new list with only the specified keys
        shareholders_list = [
            {key: d[key] for key in keys_to_keep}
            for d in shareholders_list
        ]
    except Exception as e:
        #print(e)
        shareholders_list = []

    return shareholders_list


async def run():

    con = sqlite3.connect('stocks.db')

    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol FROM stocks")
    stock_symbols = [row[0] for row in cursor.fetchall()]

    for ticker in tqdm(stock_symbols):
        shareholders_list = await get_data(ticker, con)
        if len(shareholders_list) > 0:
            await save_as_json(ticker, shareholders_list)

    con.close()

try:
    asyncio.run(run())
except Exception as e:
    print(e)
