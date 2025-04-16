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
    for train_func in [train_lstm, train_arima, train_holt_winters]:
        model_name = train_func.__name__.replace("train_", "").upper()
        try:
            name, rmse, forecast = train_func(data.copy(), forecast_horizon=forecast_horizon)
            results.append((name, rmse, forecast))
            print(f"{model_name} успешно обучена. RMSE = {rmse:.4f}")
        except Exception as e:
            print(f"{model_name} не смогла обучиться: {e}")

    if not results:
        raise RuntimeError("Ни одна из моделей не смогла обучиться.")

    best_model = min(results, key=lambda x: x[1])
    print(f"✅ Лучшая модель: {best_model[0]} (RMSE: {best_model[1]:.4f})")
    return best_model
