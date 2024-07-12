from openai import OpenAI
import time
import ujson
import sqlite3
import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

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


def save_json(symbol, data):
    with open(f"json/analyst/insight/{symbol}.json", 'w') as file:
        ujson.dump(data, file)

def get_analyst_insight(ticker):
    
    res_dict = {}
    
    try:
        querystring = {"token": benzinga_api_key,"symbols": ticker}
        response = requests.request("GET", url, params=querystring)
        output = ujson.loads(response.text)['analyst-insights'][0] #get the latest insight only
        # Extracting required fields
        res_dict = {
            'insight': output['analyst_insights'],
            'id': output['id'],
            'date': datetime.strptime(output['date'], "%Y-%m-%d").strftime("%b %d, %Y")
        }
    except:
        pass

    return res_dict


# Function to summarize the text using GPT-3.5-turbo
def get_summary(data):
    # Define the data to be summarized
    
    # Format the data as a string
    data_string = (
        f"Insights: {data['insight']}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Summarize analyst insights clearly and concisely in under 400 characters. Ensure the summary is professional and easy to understand. Conclude with whether the report is bullish or bearish."},
            {"role": "user", "content": data_string}
        ],
        max_tokens=150,
        temperature=0.7
    )


    summary = response.choices[0].message.content
    data = {
            'insight': summary,
            'id': data['id'],
            'date': data['date']
    }

    return data


try:
    stock_con = sqlite3.connect('stocks.db')
    stock_cursor = stock_con.cursor()
    stock_cursor.execute("SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'")
    stock_symbols = [row[0] for row in stock_cursor.fetchall()]

    stock_con.close()
    
    for symbol in tqdm(stock_symbols):
        try:
            data = get_analyst_insight(symbol)
            new_report_id = data.get('id', '')
            try:
                with open(f"json/analyst/insight/{symbol}.json", 'r') as file:
                    old_report_id = ujson.load(file).get('id', '')
            except:
                old_report_id = ''
            #check first if new report id exist already to save money before sending it to closedai company
            if new_report_id != old_report_id and len(data['insight']) > 0:
                res = get_summary(data)
                save_json(symbol, res)
            else:
                print('skipped')
        except:
            pass


except Exception as e:
    print(e)
