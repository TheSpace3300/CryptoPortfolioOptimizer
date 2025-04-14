from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, steps=30):
    arima_model = ARIMA(series, order=(1, 1, 1)).fit()
    forecast = arima_model.forecast(steps=steps)
    return forecast