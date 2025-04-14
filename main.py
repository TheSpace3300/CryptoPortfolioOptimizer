from services.fetch_data import fetch_ohlcv
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from models.model_selector import select_best_model
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

    arima_pred = arima_forecast(train, steps=30)
    hw_pred = holt_winters_forecast(train, steps=30)

    best_forecast = select_best_model(test, arima_pred, hw_pred, train)
    returns = np.diff(np.log(best_forecast + 1e-8))
    sharpe = sharpe_ratio(returns)
    results[pair] = sharpe

total = sum(v for v in results.values() if v > 0)
allocation = {k: v / total for k, v in results.items() if v > 0}

print("\n💼 Итоговый портфель:")
for pair, weight in allocation.items():
    print(f"{pair}: {weight:.2%} — Коэффициент Шарпа: {results.get(pair, 0):.2f}")

