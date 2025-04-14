from services.fetch_data import fetch_ohlcv
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from models.model_selector import get_model_predictions
from utils.metrics import rmse
from utils.sharpe import sharpe_ratio,neg_sharpe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

from scipy.optimize import minimize

PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'DOGE/USDT']
results = {}

for pair in PAIRS:
    series = fetch_ohlcv(pair)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('D')
    train, test = series[:-30], series[-30:]

    model_preds = get_model_predictions(train, steps=30)

    best_model = None
    best_rmse = float('inf')
    best_forecast = None

    for model_name, forecast in model_preds.items():
        forecast = np.array(forecast)
        if forecast.ndim == 0 or np.isnan(forecast).any():
            continue
        error = rmse(test, forecast)
        if error < best_rmse:
            best_rmse = error
            best_forecast = forecast
            best_model = model_name

    if best_forecast is None:
        continue

    returns = np.diff(np.log(best_forecast + 1e-8))
    sharpe = sharpe_ratio(returns)
    results[pair] = sharpe

    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(test.index, best_forecast, label=f'Forecast ({best_model})')
    plt.title(f'{pair} — Прогноз модели: {best_model}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{pair.replace("/", "_")}_forecast.png')
    plt.close()

positive_sharpes = {k: v for k, v in results.items() if v > 0}
total = sum(positive_sharpes.values())
allocation = {k: v / total for k, v in positive_sharpes.items()}

print("\n💼 Итоговый портфель:")
for pair, weight in allocation.items():
    print(f"{pair}: {weight:.2%} — Коэффициент Шарпа: {results.get(pair, 0):.2f}")
