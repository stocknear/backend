import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from tqdm import tqdm

#correlated_stocks = ['AQN', 'PACB', 'ZI', 'IPG', 'EW']
ticker = 'GME'
start_date = datetime(2024,5,1)
end_date = datetime.today()
df = yf.download(ticker, start=start_date, end=end_date, interval="1d").reset_index()
#df = df.rename(columns={'Adj Close': 'close', 'Date': 'date'})
df['daily_return'] = df['Adj Close'].pct_change()
df = df.dropna()



fig, ax = plt.subplots(figsize=(14,5))

ax.plot(df['Date'], df['daily_return']*100, linestyle='--', marker='o',color='blue',label='Daily Returns')

legend = ax.legend(loc="best", shadow=True, fontsize=15)
plt.xlabel("Date",fontsize = 14)
plt.ylabel("Percentage %", fontsize=15)
plt.grid(True)
plt.savefig('daily_return.png')






fig, ax = plt.subplots(figsize=(14,5))

days = 365

#delta t
dt = 1/365

mu = df['daily_return'].mean()

sigma = df['daily_return'].std()

#Function takes in stock price, number of days to run, mean and standard deviation values
def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        
        #Shock and drift formulas taken from the Monte Carlo formula
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        #New price = Old price + Old price*(shock+drift)
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
        
    return price

start_price = df['Adj Close'].iloc[-1] #Taken from above



for run in tqdm(range(200)):
    ax.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for GME')
plt.savefig('simulation.png')




fig, ax = plt.subplots(figsize=(14,5))
runs = 10000

simulations = np.zeros(runs)
for run in tqdm(range(runs)):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]

q = np.percentile(simulations,1)

plt.hist(simulations,bins=200)

plt.figtext(0.6,0.8,s="Start price: $%.2f" %start_price)

plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())

plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (start_price -q,))

plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)

plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Gamestop Stock after %s days" %days, weight='bold')
plt.savefig('histogram.png')

