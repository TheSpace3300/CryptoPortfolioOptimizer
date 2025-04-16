import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_holt_winters(data, forecast_horizon=7):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    model = ExponentialSmoothing(train, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(len(test))

    rmse = sqrt(mean_squared_error(test[:len(forecast)], forecast))
    return "Holt-Winters", rmse, forecast[:forecast_horizon]
