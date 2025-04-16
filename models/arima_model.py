import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller

def train_arima(data, forecast_horizon=7):
    result = adfuller(data)
    train_size = forecast_horizon
    train, test = data[:-train_size], data[-train_size:]

    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    ar_forecast = model_fit.forecast(steps=forecast_horizon)

    mae = mean_absolute_error(test, ar_forecast)
    return "ARIMA", mae, ar_forecast
