from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller

def train_arima(data, forecast_horizon=7):
    # Проверка на стационарность
    result = adfuller(data)
    p_value = result[1]

    d = 0
    diff_data = data.copy()

    # Если ряд нестационарен, дифференцируем
    while p_value > 0.05:
        diff_data = np.diff(diff_data)
        result = adfuller(diff_data)
        p_value = result[1]
        d += 1

    train_size = forecast_horizon
    train, test = diff_data[:-train_size], diff_data[-train_size:]

    model = ARIMA(train, order=(1, d, 1))
    model_fit = model.fit()
    ar_forecast = model_fit.forecast(steps=forecast_horizon)

    # Восстановление масштаба прогноза, если была дифференциация
    if d > 0:
        last_values = data[-forecast_horizon - d + 1: -d + 1] if d > 1 else data[-forecast_horizon:]
        restored_forecast = []
        prev = last_values[0]
        for i in range(forecast_horizon):
            pred = ar_forecast[i] + prev
            restored_forecast.append(pred)
            prev = pred
        ar_forecast = np.array(restored_forecast)

        ar_forecast = restored_forecast
    else:
        ar_forecast = ar_forecast

    mae = mean_absolute_error(data[-forecast_horizon:], ar_forecast)

    final_model = ARIMA(diff_data, order=(1, d, 1))
    final_model_fit = final_model.fit()
    ar_forecast_future = final_model_fit.forecast(steps=forecast_horizon)
    if d > 0:
        last_values = data[-d:]
        prev = last_values[0]
        restored_future = []
        for i in range(forecast_horizon):
            pred = ar_forecast_future[i] + prev
            restored_future.append(pred)
            prev = pred
        ar_forecast_future = np.array(restored_future)

    return "ARIMA", mae, ar_forecast_future
