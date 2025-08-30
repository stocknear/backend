import fear_and_greed
import requests
from datetime import datetime, date
import json
import orjson
import os

current = fear_and_greed.get()

print(f"Current value: {current.value}")
print(f"Category: {current.description}")
print(f"Last updated: {current.last_update}")

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
start_date = f"{date.today().year - 2}-01-01"


def save_json(data, path="json/fear-and-greed/data.json"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save JSON as bytes
    with open(path, "wb") as file:
        file.write(orjson.dumps(data))

# Add headers to avoid bot detection
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.cnn.com/markets/fear-and-greed',
    'Origin': 'https://www.cnn.com'
}

try:
    response = requests.get(BASE_URL + start_date, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Extract historical data
    history = data.get("fear_and_greed_historical", {}).get("data", [])
    
    # Helper function to categorize values
    def get_category(value):
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    # Parse historical records
    records = []
    for item in history:
        date = datetime.fromtimestamp(item["x"] / 1000)
        value = round(item["y"], 2)
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "datetime": date,
            "timestamp": item["x"],
            "value": value,
            "category": get_category(value)
        })
    
    if records:
        print(f"Total data points: {len(records)}")
        print(f"Date range: {records[0]['date']} to {records[-1]['date']}")

        for record in records[-15:]:
            print(f"{record['date']:<12} {record['value']:<8.2f} {record['category']:<15}")
        
        # Calculate statistics
        values = [r["value"] for r in records]
        avg_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
        
        # Find dates for extremes
        max_record = next(r for r in records if r["value"] == max_value)
        min_record = next(r for r in records if r["value"] == min_value)
    
        # Category distribution
        category_counts = {}
        for record in records:
            cat = record["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\nDistribution by category:")
        for category in ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]:
            if category in category_counts:
                count = category_counts[category]
                percentage = (count / len(records)) * 100
                print(f"  {category:<15}: {count:4} days ({percentage:5.1f}%)")
        
     
        monthly_data = {}
        for record in records:
            month_key = record["date"][:7]  # YYYY-MM format
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(record["value"])
        
        # Get last 6 months
        sorted_months = sorted(monthly_data.keys())[-6:]
        for month in sorted_months:
            values = monthly_data[month]
            avg = sum(values) / len(values)
            print(f"  {month}: {avg:.2f} ({get_category(avg)})")
        
        # Calculate monthly averages
        monthly_averages = {}
        for month in sorted_months:
            values = monthly_data[month]
            avg = round(sum(values) / len(values), 2)
            monthly_averages[month] = {
                "value": avg,
                "category": get_category(avg)
            }
        
        # Add percentages to category distribution
        category_distribution = {}
        for category, count in category_counts.items():
            percentage = round((count / len(records)) * 100, 2)
            category_distribution[category] = {
                "count": count,
                "percentage": percentage
            }

        data = {
            "fetched_at": datetime.now().isoformat(),
            "start_date": start_date,
            "current": {
                "value": current.value,
                "category": current.description,
                "last_update": current.last_update.isoformat() if hasattr(current.last_update, 'isoformat') else str(current.last_update)
            },
            "statistics": {
                "total_days": len(records),
                "average": round(avg_value, 2),
                "max": {"value": max_value, "date": max_record['date']},
                "min": {"value": min_value, "date": min_record['date']},
                "category_distribution": category_distribution,
                "monthly_averages": monthly_averages
            },
            "historical": records
        }

        save_json(data)
        
        # Extract data for plotting
        dates = [r["datetime"] for r in records]
        values = [r["value"] for r in records]
        
        
except Exception as e:
    print(f"Error fetching historical data: {e}")
