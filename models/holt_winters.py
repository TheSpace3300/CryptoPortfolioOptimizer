from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_forecast(series, steps=30):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=7)
    hw_fit = model.fit()
    return hw_fit.forecast(steps=30)