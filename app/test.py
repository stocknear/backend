import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import orjson
from scipy.stats import skew, norm

# Load SPY data from JSON file
with open("json/historical-price/adj/SPY.json", "rb") as file:
    data = orjson.loads(file.read())
    data = sorted(data, key=lambda x: x['date'], reverse=False)[-252:]

# Convert to DataFrame
df_spy = pd.DataFrame(data)

# Ensure Date is datetime & sort
df_spy['date'] = pd.to_datetime(df_spy['date'])
df_spy = df_spy.sort_values('date').set_index('date')

# Compute daily log returns
df_spy['log_return'] = np.log(df_spy['adjClose'] / df_spy['adjClose'].shift(1))
df_spy = df_spy.dropna()

print(df_spy.head())

mean_return = df_spy['log_return'].mean()
median_return = df_spy['log_return'].median()

print(f"Mean daily log return: {mean_return:.6f}")
print(f"Median daily log return: {median_return:.6f}")

sample_skewness = df_spy['log_return'].skew()

print(f"Skewness of SPY daily log returns: {sample_skewness:.4f}")


# Plotting the distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_spy['log_return'], bins=100, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of Daily Log Returns - SPY (Last 20 Years)')
plt.xlabel('Daily Log Return')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
