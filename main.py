from services.fetch_data import fetch_ohlcv
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from models.model_selector import select_best_model_ml
from utils.metrics import rmse
from utils.sharpe import sharpe_ratio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

PAIRS = ['TRUMP/USDT']
results = {}

# Создаем папку для графиков, если не существует
os.makedirs('plots', exist_ok=True)

for pair in PAIRS:
    print(f"\n🔍 Обработка пары: {pair}")
    series = fetch_ohlcv(pair)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('D')
    train, test = series[:-30], series[-30:]

    arima_pred = arima_forecast(train, steps=30)
    hw_pred = holt_winters_forecast(train, steps=30)

    best_forecast = select_best_model_ml(train, test, arima_pred, hw_pred)
    best_model = best_forecast
    if not isinstance(best_forecast, (np.ndarray, pd.Series)):
        continue

    best_forecast = np.array(best_forecast)
    returns = np.diff(np.log(best_forecast + 1e-8))
    returns = returns[np.isfinite(returns)]  # фильтрация NaN и Inf
    if len(returns) == 0:
        print(f"⚠️ Прогноз {pair} дал пустой returns. Пропускаем.")
        continue

    sharpe = sharpe_ratio(returns)
    print(f"📊 Коэффициент Шарпа для {pair}: {sharpe:.2f}")

    if sharpe <= 0:
        print(f"⚠️ Шарп {sharpe:.2f} для {pair} ≤ 0. Пропускаем.")
        continue

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

if not results:
    print("\n❌ Нет результатов. Все Шарпы ≤ 0 или returns пустой.")
else:
    total = sum(results.values())
    allocation = {k: v / total for k, v in results.items()}

    print("\n💼 Итоговый портфель:")
    for pair, weight in allocation.items():
        print(f"{pair}: {weight:.2%} — Коэффициент Шарпа: {results.get(pair, 0):.2f}")

