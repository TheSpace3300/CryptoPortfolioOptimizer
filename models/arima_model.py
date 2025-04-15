import pandas as pd
import pmdarima as pm

def arima_forecast(series: pd.Series, steps: int = 30):
    model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=steps)
    return forecast