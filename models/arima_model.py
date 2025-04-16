import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import adfuller

def train_arima(data, forecast_horizon=7):
    result = adfuller(data)

    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    ar_forecast = model_fit.forecast(steps=len(data))

    y_true = data
    y_pred = ar_forecast
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return "ARIMA", rmse, ar_forecast[:forecast_horizon]