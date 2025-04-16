import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

# Пример: создадим синтетический временной ряд с трендом и сезонностью
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

# Построим модель Holt-Winters с учётом уровня, тренда и сезонности
model = ExponentialSmoothing(series,
                             trend='add',     # Используем аддитивный тренд
                             seasonal='add',  # Используем аддитивную сезонность
                             seasonal_periods=12)  # Указываем период сезонности (например, для месячных данных: 12 месяцев)

# Обучаем модель
model_fit = model.fit()

# Смотрим на результаты обучения
print(model_fit.summary())

# Прогнозируем на 12 шагов вперед
forecast = model_fit.forecast(steps=12)

# Выведем прогноз
print(forecast)

from sklearn.metrics import mean_absolute_error

# Допустим, у нас есть реальные данные для сравнения (в реальных задачах используйте ваши реальные данные)
y_true = series[-12:]
y_pred = forecast

# Вычислим MAE
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')

# Визуализируем фактические данные и прогнозы
plt.plot(series, label='Исторические данные')
plt.plot(forecast, label='Прогноз', color='red')
plt.title('Holt-Winters прогнозирование')
plt.legend()
plt.show()


