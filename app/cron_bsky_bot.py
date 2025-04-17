from atproto import Client
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone, date
import requests
import hashlib
import orjson
import pytz

load_dotenv()

API_KEY = os.getenv('BENZINGA_API_KEY')
BSKY_USERNAME = os.getenv('BSYK_USERNAME')
BSKY_SECRET = os.getenv('BSKY_SECRET')

client = Client()
client.login(BSKY_USERNAME, BSKY_SECRET)

ny_tz = pytz.timezone("America/New_York")

today = datetime.utcnow().date()
now = datetime.now(timezone.utc)
now_ny = datetime.now(ny_tz)
N_minutes_ago = now - timedelta(minutes=30)


def save_json(data, file):
    directory = "json/bsky"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{file}.json", 'wb') as f:
        f.write(orjson.dumps(data))


def wiim():
    try:
        with open("json/bsyk/wiim.json", "r") as f:
            seen_list = orjson.loads(f.read())
            seen_list = [item for item in seen_list if datetime.fromisoformat(item['date']).date() == today]
    except Exception:
        seen_list = []

    with open("json/dashboard/data.json", 'rb') as f:
        data = orjson.loads(f.read())['wiim']
        data = [item for item in data if datetime.fromisoformat(item['date']).date() == today]

    res_list = []
    for item in data:
        try:
            unique_str = f"{item['date']}-{item['ticker']}-{item.get('text', '')}"
            item['id'] = hashlib.md5(unique_str.encode()).hexdigest()
            res_list.append(item)
        except Exception:
            pass

    if res_list:
        if seen_list:
            seen_ids = {item['id'] for item in seen_list}
        else:
            seen_ids = set()

        for item in res_list:
            try:
                if item is not None and item['id'] not in seen_ids:
                    symbol = item['ticker']
                    description = item.get('text', '')
                    message = f"#{symbol} {description}"
                    post = client.send_post(message)

                    seen_list.append({'date': item['date'], 'id': item['id'], 'symbol': symbol})
            except Exception as e:
                print(e)

        try:
            save_json(seen_list, "wiim")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    wiim()
