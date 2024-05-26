import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, date, timedelta
import time
import sqlite3
import concurrent.futures
import json
from tqdm import tqdm
import argparse

#source https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662

def parse_args():
    parser = argparse.ArgumentParser(description='Process stock or ETF data.')
    parser.add_argument('--db', choices=['stocks', 'etf'], required=True, help='Database name (stocks or etf)')
    parser.add_argument('--table', choices=['stocks', 'etfs'], required=True, help='Table name (stocks or etfs)')
    return parser.parse_args()

class GeometricBrownianMotion:
	def __init__(self, data, pred_ndays):
		#self.start = start
		#self.end = end
		#self.ticker = ticker
		self.data = data
		self.days = pred_ndays
		self.num_sim = 1000
	
		self.percentile = 0.01
		

		np.random.seed(42)
		

	def run(self):

		self.data['date'] = pd.to_datetime(self.data['date'])
		dt = self.days/self.num_sim
		lr = np.log(1+self.data['close'].pct_change())
		u = lr.mean()
		sigma = lr.std()
		drift = u -sigma**2.0 / 2.0
		Z = norm.ppf(np.random.rand(self.days, self.num_sim)) #days, trials
		dr = np.exp(drift *dt + sigma * Z * np.sqrt(dt))
	
		#Calculating the stock price for every trial
		new_prediction = np.zeros_like(dr)
		new_prediction[0] = self.data['close'].iloc[-1]
		for t in range(1, self.days):
		    new_prediction[t] = new_prediction[t-1]*dr[t]
		

		#future_dates = pd.DataFrame([self.data['date'].iloc[-1] + timedelta(days=d) for d in range(0, self.days)])
		#future_dates = future_dates.reset_index()
		#future_dates['date'] = future_dates[0]
		

		#new_prediction=pd.concat([future_dates['Date'], pd.DataFrame(new_prediction)],axis=1)
		
		new_prediction = pd.DataFrame(new_prediction)

		percentile_price = pd.DataFrame()
		# Compute percentile of (99%,50%,1%) formula (100-1,100-50,100-1)
		#Likelihood that value x does not drop x-y is 99 % in the next d days

		for i in range(len(new_prediction)):
			next_price = new_prediction.iloc[i, :]
			next_price = sorted(next_price, key=int)
			pp = np.percentile(next_price, [1, 50, 100-self.percentile])

			# Concatenate the new data to the existing DataFrame
			df_temp = pd.DataFrame({'min': pp[0], 'mean': pp[1], 'max': pp[2]}, index=[0])
			percentile_price = pd.concat([percentile_price, df_temp], ignore_index=True)
		
		#percentile_price = pd.concat([future_dates['date'],percentile_price],axis=1)
		#dates_formatted =future_dates['date'].dt.strftime("%Y-%m-%d").tolist()
		dict_price = {
            #'date': dates_formatted,
            'min': percentile_price['min'].tolist()[-1],
            'mean': percentile_price['mean'].tolist()[-1],
            'max': percentile_price['max'].tolist()[-1]
        }

		'''	
		fig,ax = plt.subplots()
		ax.plot(self.data['date'],self.data['date'],color='purple')
		ax.plot(percentile_price['date'],percentile_price['brown_mean'],color='black')
		ax.plot(percentile_price['date'],percentile_price['brown_max'],color='green')
		ax.plot(percentile_price['date'],percentile_price['brown_min'],color='red')

		plt.fill_between(percentile_price['date'],percentile_price['brown_max'],percentile_price['brown_mean'],alpha=0.3,color='green')
		#plt.fill_between(percentile_price['date'],percentile_price['brown_mean'],percentile_price['brown_min'],alpha=0.3,color='red')
		plt.xlabel('%s days in the future' % self.days)
		plt.ylabel('Stock price prediction')
		plt.show()
		'''

		#return percentile_price[['date','mean']], percentile_price[['Date','max']], percentile_price[['Date','min']]
		
		return dict_price
		


def create_column(con):
    """
    Create the 'pricePrediction' column if it doesn't exist in the db table.
    """
    query_check = f"PRAGMA table_info({table_name})"
    cursor = con.execute(query_check)
    columns = [col[1] for col in cursor.fetchall()]

    if 'pricePrediction' not in columns:
    	print('yellow')
    	query = f"ALTER TABLE {table_name} ADD COLUMN pricePrediction TEXT"
    	con.execute(query)
    	con.commit()

def update_database(pred_dict, symbol, con):
    query = f"UPDATE {table_name} SET pricePrediction = ? WHERE symbol = ?"
    pred_json = json.dumps(pred_dict)  # Convert the pred dictionary to JSON string
    con.execute(query, (pred_json, symbol))
    con.commit()




def process_symbol(ticker):
    try:
        query_template = """
            SELECT
                date, close
            FROM
                "{ticker}"
            WHERE
                date BETWEEN ? AND ?
        """

        query = query_template.format(ticker=ticker)
        df = pd.read_sql_query(query, con, params=(start_date, end_date))
        time_list = [7,30,90,180]

        pred_dict = {}
        try:
        	for time_period in time_list:
        		if time_period == 7:
        			pred_dict['1W'] = GeometricBrownianMotion(df, time_period).run()
        		elif time_period == 30:
        			pred_dict['1M'] = GeometricBrownianMotion(df, time_period).run()
        		elif time_period == 90:
        			pred_dict['3M'] = GeometricBrownianMotion(df, time_period).run()
        		elif time_period == 180:
        			pred_dict['6M'] = GeometricBrownianMotion(df, time_period).run()

        except:
        	pred_dict = {'1W': {'min': 0, 'mean': 0, 'max': 0}, '1M': {'min': 0, 'mean': 0, 'max': 0}, '3M': {'min': 0, 'mean': 0, 'max': 0}, '6M': {'min': 0, 'mean': 0, 'max': 0}}

        create_column(con)
        update_database(pred_dict, ticker, con)

    except:
        print(f"Failed create price prediction for {ticker}")


args = parse_args()
db_name = args.db
table_name = args.table

con = sqlite3.connect(f'backup_db/{db_name}.db')

symbol_query = f"SELECT DISTINCT symbol FROM {table_name}"

symbol_cursor = con.execute(symbol_query)
symbols = [symbol[0] for symbol in symbol_cursor.fetchall()]

start_date = datetime(1970, 1, 1)
end_date = datetime.today()

# Number of concurrent workers
num_processes = 4 # You can adjust this based on your system's capabilities
futures = []

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for symbol in symbols:
        futures.append(executor.submit(process_symbol, symbol))

    # Use tqdm to wrap around the futures for progress tracking
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Processing"):
        pass
con.close()


'''
query_template = """
    SELECT
        date, close
    FROM
        {ticker}
    WHERE
        date BETWEEN ? AND ?
"""
ticker = 'AMD'
start_date = datetime(2020,1,1)
end_date = datetime.today()
con = sqlite3.connect('stocks.db')
query = query_template.format(ticker=ticker)
df = pd.read_sql_query(query, con, params=(start_date, end_date))
#Compute the logarithmic returns
GeometricBrownianMotion(df).run()
'''