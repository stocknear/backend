import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from datetime import datetime
import yfinance as yf
import asyncio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt


def get_monthly_historical_data(df):
    # Ensure the date column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Get the last available date in your data
    last_date = df['ds'].max()
    # Calculate the date one year ago
    one_year_ago = last_date - pd.DateOffset(years=2)
    
    # Filter data for the last year
    last_year_df = df[df['ds'] > one_year_ago]
    
    # Group by year-month and select the last record in each month
    monthly_df = last_year_df.groupby(last_year_df['ds'].dt.to_period('M')).apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    
    # Rename columns to the desired output format
    monthly_df = monthly_df[['ds', 'y']].rename(columns={'ds': 'date', 'y': 'close'})
    
    # Format the date as a string in YYYY-MM-DD format
    monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')
    
    # Convert to list of dictionaries
    return monthly_df.to_dict(orient='records')


async def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        df = df.reset_index()
        df = df[['Date', 'Adj Close']]
        df = df.rename(columns={"Date": "ds", "Adj Close": "y"})
        if len(df) > 252 * 2:  # At least 2 years of history is necessary
            # df['y'] = df['y'].rolling(window=200).mean()
            # df = df.dropna()
            return df
    except Exception as e:
        print(e)


class PricePredictor:
    def __init__(self, predict_ndays=365):
        self.predict_ndays = predict_ndays
        self.model = Prophet(
            interval_width=0.8,
            daily_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale= 0.1,
            seasonality_prior_scale=0.1,
        )
        #self.model.add_regressor('volume')


    def run(self, df):
        df = df.copy()
    
        self.model.fit(df)
        future = self.model.make_future_dataframe(periods=self.predict_ndays)
        forecast = self.model.predict(future)

        # Apply rolling average to smooth the forecast intervals
        rolling_window = 200
        forecast['smoothed_upper'] = forecast['yhat_upper'].rolling(window=rolling_window, min_periods=1).mean().round(2)
        forecast['smoothed_lower'] = forecast['yhat_lower'].rolling(window=rolling_window, min_periods=1).mean().round(2)
        forecast['smoothed_mean']  = forecast['yhat'].rolling(window=rolling_window, min_periods=1).mean().round(2)

        # Actual and predicted values for evaluation (optional)
        actual_values = df['y'].values
        predicted_values = forecast['yhat'].values[:-self.predict_ndays]

        # Extract forecast values for plotting or analysis (if needed)
        pred_date_list = forecast['ds'][-1200 - self.predict_ndays:].dt.strftime('%Y-%m-%d').tolist()
        upper_list = forecast['smoothed_upper'][-1200 - self.predict_ndays:].tolist()
        lower_list = forecast['smoothed_lower'][-1200 - self.predict_ndays:].tolist()
        mean_list = forecast['smoothed_mean'][-1200 - self.predict_ndays:].tolist()

        historical_date_list = df['ds'][-1200:].dt.strftime('%Y-%m-%d').tolist()
        historical_price_list = df['y'][-1200:].round(2).tolist()

        metrics_dict = {
            'mse': mean_squared_error(actual_values, predicted_values),
            'mae': mean_absolute_error(actual_values, predicted_values),
            'r2': r2_score(actual_values, predicted_values)}
        print("Metrics:", metrics_dict)

        # Get monthly historical data and round the close value
        monthly_historical_data = get_monthly_historical_data(df)
        monthly_historical_data = [{**item, 'close': round(item['close'], 2)} for item in monthly_historical_data]

       
        future_forecast = forecast[forecast['ds'] > df['ds'].max()]['smoothed_mean']
        median_price = round(np.mean([
            forecast['smoothed_lower'].iloc[-1],
            forecast['smoothed_mean'].iloc[-1],
            forecast['smoothed_upper'].iloc[-1]
        ]), 2)

        # Latest actual price from the dataset
        latest_price = round(df['y'].iloc[-1], 2)

        return {
            'pastPriceList': monthly_historical_data,
            'avgPriceTarget': mean_list[-1],
            'highPriceTarget': upper_list[-1],
            'lowPriceTarget': lower_list[-1],
            'medianPriceTarget': float(median_price),
        }



# Test Mode
async def main():
    for ticker in ['NVDA']:
        start_date = datetime(2015, 1, 1).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        df = await download_data(ticker, start_date, end_date)
        if df is not None:
            data = PricePredictor().run(df)

# Run the main function
# asyncio.run(main())

# Plotting (optional)
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
