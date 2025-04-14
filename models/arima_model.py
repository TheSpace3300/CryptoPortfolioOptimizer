from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, steps=30):
    model = ARIMA(series, order=(5, 1, 0)).fit()
    return model.forecast(steps=steps)