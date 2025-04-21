from statsmodels.tsa.vector_ar.var_model import forecast

from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd
from data.download_data import data_forecast

pairs = ['TRUMP/USDT', 'BTC/USDT']

forecast_series = data_forecast(pairs, forecast_horizon=7)


