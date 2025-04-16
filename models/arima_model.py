import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_arima(data, forecast_horizon=7):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    rmse = sqrt(mean_squared_error(test[:len(forecast)], forecast))
    return "ARIMA", rmse, forecast[:forecast_horizon]