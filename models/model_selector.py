from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from utils.features import extract_features
import numpy as np
import pandas as pd

from models.LSTM import train_lstm
from models.arima_model import train_arima
from models.holt_winters import train_holt_winters


def select_best_model(data, forecast_horizon):
    results = []
    results.append(train_lstm(data.copy(), forecast_horizon=forecast_horizon))
    results.append(train_arima(data.copy(), forecast_horizon=forecast_horizon))
    results.append(train_holt_winters(data.copy(), forecast_horizon=forecast_horizon))

    best_model = min(results, key=lambda x: x[1])
    print(f"✅ Лучшая модель: {best_model[0]} (RMSE: {best_model[1]:.4f})")
    return best_model
