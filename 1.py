from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from services.fetch_data import fetch_ohlcv
from utils.metrics import rmse
from utils.sharpe import sharpe_ratio
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

PAIRS = ['BTC/USDT']
results = {}

# Создаем папку для графиков, если не существует
os.makedirs('plots', exist_ok=True)

for pair in PAIRS:
    print(f"\n🔍 Обработка пары: {pair}")
    series = fetch_ohlcv(pair)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('D')

plt.plot(series)
plt.title("Временной ряд")
plt.show()

result = adfuller(series)
print(series)
print(f'Статистика теста: {result[0]}')
print(f'p-значение: {result[1]}')

model = ARIMA(series, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary)

ar_forecast = model_fit.forecast(steps=88)
print(f'Прогноз: {ar_forecast}')

data = series[-88:]
print(data)

y_true = data  # Реальные данные (пример)
y_pred = ar_forecast  # Прогнозируемые данные

mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

plt.plot(series, label='Исторические данные')
plt.plot(ar_forecast, label='Прогноз', color='red')
plt.legend()
plt.show()