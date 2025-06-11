import os
import orjson
from collections import defaultdict
from statistics import median
from tqdm import tqdm

# Ensure output directories exist
os.makedirs("json/average/industry", exist_ok=True)
os.makedirs("json/average/sector", exist_ok=True)

# Load stock screener data
with open("json/stock-screener/data.json", 'rb') as file:
    stock_data = orjson.loads(file.read())

# Initialize group containers
industry_data = defaultdict(lambda: defaultdict(list))
sector_data = defaultdict(lambda: defaultdict(list))

# Group values
for item in tqdm(stock_data, desc="Grouping by industry and sector"):
    industry = item.get('industry')
    sector = item.get('sector')
    if not industry and not sector:
        continue

    for field, value in item.items():
        # Skip sector/industry fields themselves
        if field in {'industry', 'sector'}:
            continue
        if isinstance(value, (int, float)):
            if industry:
                industry_data[industry][field].append(value)
            if sector:
                sector_data[sector][field].append(value)

# Compute medians
def compute_median(data_group):
    result = {}
    for group, field_dict in data_group.items():
        result[group] = {}
        for field, values in field_dict.items():
            if values:
                result[group][field] = median(values)
    return result

industry_medians = compute_median(industry_data)
sector_medians = compute_median(sector_data)

with open("json/average/industry/data.json", 'wb') as f:
    f.write(orjson.dumps(industry_medians))

with open("json/average/sector/data.json", 'wb') as f:
    f.write(orjson.dumps(sector_medians))
