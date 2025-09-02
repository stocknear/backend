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
    
    # Load SPY data
    spy_data = {}
    try:
        with open("json/historical-price/adj/SPY.json", "r") as f:
            spy_json = json.load(f)
            for item in spy_json:
                spy_data[item["date"]] = item["adjClose"]
    except Exception as e:
        print(f"Warning: Could not load SPY data: {e}")
    
    # Parse historical records
    records = []
    for item in history:
        date = datetime.fromtimestamp(item["x"] / 1000)
        value = round(item["y"], 2)
        date_str = date.strftime("%Y-%m-%d")
        record = {
            "date": date_str,
            "datetime": date,
            "timestamp": item["x"],
            "value": value,
            "category": get_category(value)
        }
        
        # Add SPY close price if available
        if date_str in spy_data:
            record["spy_close"] = spy_data[date_str]
        
        records.append(record)
    
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
        
        # Calculate correlation and insights
        insights = {}
        records_with_spy = [r for r in records if "spy_close" in r]
        
        if len(records_with_spy) > 10:
            # Calculate correlation
            fear_values = [r["value"] for r in records_with_spy]
            spy_values = [r["spy_close"] for r in records_with_spy]
            
            # Calculate Pearson correlation
            n = len(fear_values)
            sum_fear = sum(fear_values)
            sum_spy = sum(spy_values)
            sum_fear_sq = sum(x**2 for x in fear_values)
            sum_spy_sq = sum(x**2 for x in spy_values)
            sum_fear_spy = sum(fear_values[i] * spy_values[i] for i in range(n))
            
            numerator = n * sum_fear_spy - sum_fear * sum_spy
            denominator = ((n * sum_fear_sq - sum_fear**2) * (n * sum_spy_sq - sum_spy**2))**0.5
            correlation = numerator / denominator if denominator != 0 else 0
            
            # Calculate 30-day returns for extreme periods
            extreme_fear_returns = []
            extreme_greed_returns = []
            
            for i, record in enumerate(records_with_spy):
                if i < len(records_with_spy) - 30:
                    current_fear = record["value"]
                    current_spy = record["spy_close"]
                    
                    # Find SPY price 30 days later
                    future_record = None
                    for j in range(i+20, min(i+40, len(records_with_spy))):
                        future_record = records_with_spy[j]
                        break
                    
                    if future_record and "spy_close" in future_record:
                        future_spy = future_record["spy_close"]
                        return_pct = ((future_spy - current_spy) / current_spy) * 100
                        
                        if current_fear <= 10:
                            extreme_fear_returns.append(return_pct)
                        elif current_fear >= 80:
                            extreme_greed_returns.append(return_pct)
            
            # Find extreme examples
            sorted_by_fear = sorted(records_with_spy, key=lambda x: x["value"])
            sorted_by_greed = sorted(records_with_spy, key=lambda x: x["value"], reverse=True)
            
            most_fear = sorted_by_fear[0] if sorted_by_fear else None
            most_greed = sorted_by_greed[0] if sorted_by_greed else None
            
            insights = {
                "correlation": round(correlation, 4),
                "correlation_percent": round(correlation * 100, 0),
                "extreme_fear_avg_return": round(sum(extreme_fear_returns) / len(extreme_fear_returns), 2) if extreme_fear_returns else 0,
                "extreme_greed_avg_return": round(sum(extreme_greed_returns) / len(extreme_greed_returns), 2) if extreme_greed_returns else 0,
                "most_fear_example": {
                    "value": most_fear["value"],
                    "spy_price": most_fear["spy_close"],
                    "date": most_fear["date"]
                } if most_fear else None,
                "most_greed_example": {
                    "value": most_greed["value"], 
                    "spy_price": most_greed["spy_close"],
                    "date": most_greed["date"]
                } if most_greed else None
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
                "value": round(current.value,0),
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
            "insights": insights,
            "historical": records
        }

        save_json(data)
        
        # Extract data for plotting
        dates = [r["datetime"] for r in records]
        values = [r["value"] for r in records]
        
        
except Exception as e:
    print(f"Error fetching historical data: {e}")
