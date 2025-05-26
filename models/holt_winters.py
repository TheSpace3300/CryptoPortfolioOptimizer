from sklearn.metrics import mean_absolute_error
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

    final_model = ExponentialSmoothing(data,
                                 trend='add',
                                 seasonal='add',
                                 seasonal_periods=12)
    final_model_fit = final_model.fit()
    hw_forecast_future = final_model_fit.forecast(steps=forecast_horizon)

    return "Holt-Winters", mae, hw_forecast_future
