import math
from scipy.stats import norm

def calculate_call_profit_probability(current_price, strike, premium, T, sigma, r):
    breakeven = strike + premium
    # Correct d2 for profit probability
    d1 = (math.log(current_price / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return norm.cdf(d2)

# Your example
current_price = 11.9
strike = 12
premium = 0
T = 1 / 365
sigma = 0.7
r = 0.05

pop = calculate_call_profit_probability(current_price, strike, premium, T, sigma, r)
print(f"Profit Probability ≈ {pop * 100:.2f}%")