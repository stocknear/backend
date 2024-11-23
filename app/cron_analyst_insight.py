import os
import ujson
import sqlite3
import aiohttp
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
import aiofiles

# Load environment variables
load_dotenv()

# Initialize OpenAI client
benzinga_api_key = os.getenv('BENZINGA_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
org_id = os.getenv('OPENAI_ORG')
client = OpenAI(
    api_key=openai_api_key,
    organization=org_id,
)

headers = {"accept": "application/json"}
url = "https://api.benzinga.com/api/v1/analyst/insights"

# Save JSON asynchronously
async def save_json(symbol, data):
    async with aiofiles.open(f"json/analyst/insight/{symbol}.json", 'w') as file:
        await file.write(ujson.dumps(data))

async def get_analyst_insight(session, ticker):
    #there is a bug in benzinga api where the latest data can be at the second element instead the first
    #solution check which date is the latest by comparing these two
    res_dict = {}
    try:
        querystring = {"token": benzinga_api_key, "symbols": ticker}
        async with session.get(url, params=querystring) as response:
            output = await response.json()
            if 'analyst-insights' in output and len(output['analyst-insights']) >= 2:
                insight_1, insight_2 = output['analyst-insights'][:2]
                
                # Parse dates for comparison
                date_1 = datetime.strptime(insight_1['date'], "%Y-%m-%d")
                date_2 = datetime.strptime(insight_2['date'], "%Y-%m-%d")
                
                # Choose the insight with the latest date
                latest_insight = insight_1 if date_1 >= date_2 else insight_2
            elif 'analyst-insights' in output and output['analyst-insights']:
                latest_insight = output['analyst-insights'][0]  # Only one insight available
            else:
                return res_dict  # No insights available
            
            # Populate res_dict with the latest insight data
            res_dict = {
                'insight': latest_insight['analyst_insights'],
                'id': latest_insight['id'],
                'date': datetime.strptime(latest_insight['date'], "%Y-%m-%d").strftime("%b %d, %Y")
            }
    except Exception as e:
        print(f"Error fetching analyst insight: {e}")
    return res_dict

# Summarize insights using OpenAI
async def get_summary(data):
    try:
        data_string = f"Insights: {data['insight']}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize analyst insights clearly and concisely in under 400 characters. Ensure the summary is professional and easy to understand. Conclude with whether the report is bullish or bearish."},
                {"role": "user", "content": data_string}
            ],
            max_tokens=150,
            temperature=0.7
        )
        summary = response.choices[0].message.content
        data['insight'] = summary
        return data
    except Exception as e:
        print(e)

# Process individual symbol
async def process_symbol(session, symbol):
    try:
        data = await get_analyst_insight(session, symbol)
        if data:
            new_report_id = data.get('id', '')
            try:
                async with aiofiles.open(f"json/analyst/insight/{symbol}.json", 'r') as file:
                    old_report_id = ujson.loads(await file.read()).get('id', '')
            except:
                old_report_id = ''

            if new_report_id != old_report_id and data['insight']:
                res = await get_summary(data)
                if res:
                    await save_json(symbol, res)
            else:
                print(f'Skipped: {symbol}')
    except:
        pass

# Function to split list into batches
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Main function with batch processing
async def main():
    # Fetch stock symbols from SQLite database
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in cursor.fetchall()]
    con.close()

    async with aiohttp.ClientSession(headers=headers) as session:
        # Process in batches of 100
        for batch in chunk_list(stock_symbols, 100):
            print(f"Processing batch of {len(batch)} tickers")
            await asyncio.gather(*[process_symbol(session, symbol) for symbol in tqdm(batch)])

if __name__ == "__main__":
    asyncio.run(main())
