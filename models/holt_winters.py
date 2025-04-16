import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_holt_winters(data, forecast_horizon=7):
    model = ExponentialSmoothing(data,
                                 trend='add',  # Используем аддитивный тренд
                                 seasonal='add',  # Используем аддитивную сезонность
                                 seasonal_periods=12)  # Указываем период сезонности (например, для месячных данных: 12 месяцев)
    model_fit = model.fit()
    hw_forecast = model_fit.forecast(steps=12)
    y_true = data[-12:]
    y_pred = hw_forecast

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return "Holt-Winters", rmse, hw_forecast[:forecast_horizon]
