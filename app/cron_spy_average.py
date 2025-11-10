import os
import orjson
from pathlib import Path

BASE_PATH = Path(__file__).parent / "json"
os.makedirs(BASE_PATH / "spy-average", exist_ok=True)

with open(BASE_PATH / "stock-screener/data.json", 'rb') as f:
    stock_data = orjson.loads(f.read())

# Growth metrics requiring current/previous calculation
GROWTH_METRICS = {
    'growthRevenue': 'revenue',
    'growthEPS': 'eps',
    'growthOperatingIncome': 'operatingIncome',
    'growthFreeCashFlow': 'freeCashFlow',
    'growthNetIncome': 'netIncome'
}

# Note: Now including negative values in all averages for realistic market data
# POSITIVE_RATIOS set removed - no longer filtering out negative values
POSITIVE_RATIOS = set()

def safe_float(val):
    try:
        return float(val) if val not in (None, '', 'None') else None
    except (ValueError, TypeError):
        return None

# Calculate total market cap and normalize weights in single pass
total_cap = sum(
    mc for item in stock_data
    if (mc := safe_float(item.get('marketCap'))) and mc > 0
)

if total_cap == 0:
    us_market_data = {}
else:
    # Filter and prepare holdings with weights
    holdings = [
        (item, mc / total_cap)
        for item in stock_data
        if (mc := safe_float(item.get('marketCap'))) and mc > 0
    ]

    # Extract numeric fields from multiple samples to get complete field set
    # (some stocks may have null values for certain fields)
    numeric_fields = set()
    for data, _ in holdings[:100]:  # Sample first 100 stocks
        numeric_fields.update({
            k for k, v in data.items()
            if isinstance(v, (int, float)) and k != 'marketCap'
        })

    # Initialize accumulators
    weighted_sums = {}
    valid_counts = {}
    growth_current = {base: 0.0 for base in GROWTH_METRICS.values()}
    growth_previous = {base: 0.0 for base in GROWTH_METRICS.values()}
    growth_valid = {base: 0.0 for base in GROWTH_METRICS.values()}

    # Single pass through all holdings
    for data, weight in holdings:
        for field in numeric_fields:
            val = safe_float(data.get(field))
            if field in GROWTH_METRICS:
                base_field = GROWTH_METRICS[field]
                current = safe_float(data.get(base_field))
                if current and val is not None:
                    growth_dec = val / 100
                    prev = current / (1 + growth_dec) if growth_dec != -1 else 0
                    growth_current[base_field] += current * weight
                    growth_previous[base_field] += prev * weight
                    growth_valid[base_field] += weight
            elif val is not None:
                # Include all values (positive and negative) for realistic averages
                weighted_sums[field] = weighted_sums.get(field, 0) + val * weight
                valid_counts[field] = valid_counts.get(field, 0) + weight

    # Calculate final averages
    us_market_data = {
        field: round(weighted_sums[field] / valid_counts[field], 1)
        for field in numeric_fields
        if field not in GROWTH_METRICS and valid_counts.get(field, 0) > 0
    }

    # Add growth metrics
    for gfield, base in GROWTH_METRICS.items():
        if growth_valid[base] > 0 and growth_previous[base] > 0:
            us_market_data[gfield] = round(
                ((growth_current[base] - growth_previous[base]) / growth_previous[base]) * 100, 1
            )

with open(BASE_PATH / "spy-average/data.json", 'wb') as f:
    f.write(orjson.dumps(us_market_data))
