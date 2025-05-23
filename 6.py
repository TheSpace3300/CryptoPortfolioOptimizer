from statsmodels.tsa.vector_ar.var_model import forecast

from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd
from data.download_data import data_forecast
from models.LSTM import train_lstm
from models.arima_model import train_arima
from models.holt_winters import train_holt_winters

pairs = ['BTC/USDT', 'ETH/USDT', 'APEX/USDT', 'MNT/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'GRASS/USDT', 'PEPE/USDT', 'NEAR/USDT', 'ADA/USDT', 'BNB/USDT', 'TRUMP/USDT', 'SUN/USDT', 'TRX/USDT']


forecast_series = data_forecast(pairs, forecast_horizon=7)


