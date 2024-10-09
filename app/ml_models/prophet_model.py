import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from datetime import datetime
import yfinance as yf
import asyncio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import matplotlib.pyplot as plt


async def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        df = df.reset_index()
        df = df[['Date', 'Adj Close']]
        df = df.rename(columns={"Date": "ds", "Adj Close": "y"})
        if len(df) > 252*2: #At least 2 years of history is necessary
            #df['y'] = df['y'].rolling(window=200).mean()
            #df = df.dropna()
            return df
    except Exception as e:
    	print(e)


class PricePredictor:
    def __init__(self, predict_ndays=365):
    	self.predict_ndays = predict_ndays
    	self.model = Prophet(
            interval_width = 0.8,
            daily_seasonality=True,
            yearly_seasonality = True,
      		)

    def run(self, df):
    	self.model.fit(df)
    	future = self.model.make_future_dataframe(periods=self.predict_ndays)
    	forecast = self.model.predict(future)

    	# Apply rolling average to smooth the upper bound
    	rolling_window = 200
    	forecast['smoothed_upper'] = round(forecast['yhat_upper'].rolling(window=rolling_window, min_periods=1).mean(),2)
    	forecast['smoothed_lower'] = round(forecast['yhat_lower'].rolling(window=rolling_window, min_periods=1).mean(),2)
    	forecast['smoothed_mean'] = round(forecast['yhat'].rolling(window=rolling_window, min_periods=1).mean(),2)

    	actual_values = df['y'].values
    	predicted_values = forecast['yhat'].values[:-self.predict_ndays]

    	rmse = round(np.sqrt(mean_squared_error(actual_values, predicted_values)),2)
    	mape = round(np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100)
    	r2 = round(r2_score(actual_values, predicted_values)*100)

    	print("RMSE:", rmse)
    	print("MAPE:", mape)
    	print("R2 Score:", r2)
    	pred_date_list = forecast['ds'][-1200-self.predict_ndays:].dt.strftime('%Y-%m-%d').tolist()
    	upper_list = forecast['smoothed_upper'][-1200-self.predict_ndays:].tolist()
    	lower_list = forecast['smoothed_lower'][-1200-self.predict_ndays:].tolist()
    	mean_list = forecast['smoothed_mean'][-1200-self.predict_ndays:].tolist()

    	historical_date_list = df['ds'][-1200:].dt.strftime('%Y-%m-%d').tolist()
    	historical_price_list = round(df['y'][-1200:],2).tolist()

    	return {'rmse': rmse,'mape': mape,'r2Score':r2, 'historicalPrice': historical_price_list, 'predictionDate': pred_date_list, 'upperBand': upper_list, 'lowerBand': lower_list, 'meanResult': mean_list}



#Test Mode
async def main():
    for ticker in ['NVDA']:
        start_date = datetime(2000, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        df = await download_data(ticker, start_date, end_date)
        data = PricePredictor().run(df)
        print(data)

# Run the main function
#asyncio.run(main())




# Plotting
'''
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(forecast['ds'][-1200-predict_ndays:], forecast['smoothed_mean'][-1200-predict_ndays:], color='blue', label='Predicted')
ax.fill_between(forecast['ds'][-1200-predict_ndays:], forecast['smoothed_lower'][-1200-predict_ndays:], forecast['smoothed_upper'][-1200-predict_ndays:], color='gray', alpha=0.5, label='Confidence Interval')
ax.plot(df['ds'][-1200:], df['y'][-1200:], color='black', label='Actual')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Forecasted Prices for {}'.format(ticker))
ax.legend()
ax.grid(True)
plt.show()
'''