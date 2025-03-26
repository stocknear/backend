import orjson
from datetime import datetime, timedelta
import os
from tqdm import tqdm

today = datetime.today().date()
threshold_date = today - timedelta(days=14)

directory_path = "json/analyst"

def save_json(data):
    os.makedirs(directory_path, exist_ok=True)
    with open(f"{directory_path}/flow-data.json", 'wb') as file:
        file.write(orjson.dumps(data))

def get_analyst_from_directory():
    directory = "json/analyst/analyst-db/"
    res = []
    try:
        data = [file for file in os.listdir(directory) if file.endswith(".json")]

        for file_name in data:
            try:
                with open(f"{directory}{file_name}", "r") as file:
                    analyst_data = orjson.loads(file.read())
                    if analyst_data['analystScore'] >= 3:
                        ratings = [item for item in analyst_data['ratingsList'] 
                                   if datetime.strptime(item["date"], "%Y-%m-%d").date() >= threshold_date]
                        if ratings:
                            for item_ratings in ratings:
                                try:
                                    res.append({
                                        'analystName': analyst_data['analystName'],
                                        'analystId': analyst_data['analystId'],
                                        'analystScore': analyst_data['analystScore'],
                                        'date': item_ratings['date'],
                                        'name': item_ratings['name'],
                                        'symbol': item_ratings['ticker'],
                                        'adjusted_pt_current': item_ratings['adjusted_pt_current'],
                                        'adjusted_pt_prior': item_ratings['adjusted_pt_prior'],
                                        'upside': item_ratings['upside'],
                                        'action': item_ratings['action_company'],
                                        'rating_current': item_ratings['rating_current']
                                    })
                                except Exception as e:
                                    print(e)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    except Exception as e:
        print("Error reading directory:", e)
        return []
    return res


if __name__ == "__main__":
    data = get_analyst_from_directory()
    sorted_data = sorted(data, key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"), reverse=True)
    if sorted_data:
        save_json(sorted_data)
