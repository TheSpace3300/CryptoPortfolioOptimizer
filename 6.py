from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd


series = fetch_ohlcv('ETH/USDT')
series.index = pd.to_datetime(series.index)
series = series.asfreq('D')
data = series.values

best_model_name, best_rmse, forecast = select_best_model(data, forecast_horizon=7)
print("📈 Прогноз на 7 дней:", forecast)