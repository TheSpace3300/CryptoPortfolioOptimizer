import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_holt_winters(data, forecast_horizon=7):
    train_size = forecast_horizon
    train, test = data[:-train_size], data[-train_size:]

    model = ExponentialSmoothing(train,
                                 trend='add',
                                 seasonal='add',
                                 seasonal_periods=12)
    model_fit = model.fit()
    hw_forecast = model_fit.forecast(steps=forecast_horizon)

    mae = mean_absolute_error(test, hw_forecast)
    return "Holt-Winters", mae, hw_forecast
