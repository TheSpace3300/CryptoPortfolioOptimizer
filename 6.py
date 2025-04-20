from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd


series = fetch_ohlcv('ETH/USDT')
date = pd.to_datetime(series.index)
series = series.asfreq('D')
data = series.values
forecast_horizon = 7

best_model_name, best_mae, forecast = select_best_model(data, forecast_horizon)
last_date = series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

forecast_series = pd.Series(forecast, index=forecast_dates)
print("📈 Прогноз на 7 дней:")
print(forecast_series)


