import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_forecast(series: pd.Series, steps: int = 30):
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated"
    ).fit()
    forecast = model.forecast(steps)
    return forecast
