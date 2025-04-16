from keras.src.layers import Bidirectional
from statsmodels.tsa.vector_ar.var_model import forecast
from tensorflow.python.ops.numpy_ops.np_array_ops import reshape

from services.fetch_data import fetch_ohlcv
from utils.sharpe import sharpe_ratio
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout, Bidirectional

PAIRS = ['BTC/USDT']
results = {}

# Создаем папку для графиков, если не существует
os.makedirs('plots', exist_ok=True)

for pair in PAIRS:
    print(f"\n🔍 Обработка пары: {pair}")
    series = fetch_ohlcv(pair)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('D')

# Преобразуем данные в pandas Series
prices = series

# Визуализируем данные
plt.plot(prices)
plt.title("Временной ряд")
plt.show()

# Нормализуем данные (преобразуем данные в диапазон [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(prices.values.reshape(-1, 1))

# Визуализируем нормализованные данные
plt.plot(data_scaled)
plt.title("Нормализованный временной ряд")
plt.show()

test_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:-test_size]
test_data = data_scaled[-test_size:]

time_step = 60  # Количество предыдущих шагов, которые модель будет использовать для предсказания
forecast_horizon = 10

# Функция для подготовки данных в форму [samples, time steps, features]
def create_dataset(data, time_step=1, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - time_step - forecast_horizon + 1):
        X.append(data[i:(i + time_step), 0])  # Предсказание на основе предыдущих time_step значений
        y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])      # Значение, которое мы хотим предсказать
    return np.array(X), np.array(y)


X_train, y_train = create_dataset(train_data, time_step, forecast_horizon)
X_test, y_test = create_dataset(test_data, time_step, forecast_horizon)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Создаем модель LSTM
model1 = Sequential()
# Добавляем слой LSTM
model1.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model1.add(Dropout(0.3))
model1.add(LSTM(units=64))
model1.add(Dropout(0.3))
# Добавляем полносвязный слой для прогноза
model1.add(Dense(units=forecast_horizon))
# Компилируем модель
model1.compile(optimizer='adam', loss='mean_squared_error')

# Обучаем модель
model1.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

predictions = model1.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_original = scaler.inverse_transform(y_test)

plt.figure(figsize=(14, 5))
plt.plot(y_test_original[0], color='blue', label='Истинные значения')
plt.plot(predictions[0],color='red', label='Прогнозируемые значения')
plt.title('Прогнозирование с использованием модели LSTM')
#plt.legend()
plt.show()


### 7. **Оценка модели**
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Оценка модели (MSE, RMSE, MAE)
mse = mean_squared_error(y_test_original, predictions)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test_original, predictions)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

last_data = prices.values[-time_step:]
last_data.mean()

last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
X_input = last_data_scaled.reshape(1, time_step, 1)
predicted_scaled = model1.predict(X_input)
predicted_values = scaler.inverse_transform(predicted_scaled)
print(f'Predicted Price: {predicted_values}')