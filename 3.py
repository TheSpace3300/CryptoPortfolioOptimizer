import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import Bidirectional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from services.fetch_data import fetch_ohlcv
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from models.model_selector import select_best_model_ml
from utils.metrics import rmse
from utils.sharpe import sharpe_ratio
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
data = pd.Series(series)

# Визуализируем данные
plt.plot(data)
plt.title("Временной ряд")
plt.show()

# Нормализуем данные (преобразуем данные в диапазон [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Визуализируем нормализованные данные
plt.plot(scaled_data)
plt.title("Нормализованный временной ряд")
plt.show()

# Функция для подготовки данных в форму [samples, time steps, features]
def create_dataset(data, time_step, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - time_step - forecast_horizon):
        X.append(data[i:(i + time_step), 0])  # Предсказание на основе предыдущих time_step значений
        y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])      # Значение, которое мы хотим предсказать
    return np.array(X), np.array(y)

# Преобразуем данные в подходящий формат для LSTM
time_step = 60  # Количество предыдущих шагов, которые модель будет использовать для предсказания
forecast_horizon = 10
X, y = create_dataset(scaled_data, time_step, forecast_horizon)

# Переформатируем X в форму, подходящую для LSTM: [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Разделим данные на обучающую и тестовую выборки
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Создаем модель LSTM
model = Sequential()
# Добавляем слой LSTM
model.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model.add(Dropout(0.3))
model.add(LSTM(units=64))
model.add(Dropout(0.3))
# Добавляем полносвязный слой для прогноза
model.add(Dense(units=forecast_horizon))
# Компилируем модель
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучаем модель
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Строим график обучения
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title("Потери на обучении и валидации")
plt.show()

y_pred = model.predict(X_test)

def rescale_batch(batch, scaler, feature_dim):
    rescaled = []
    for row in batch:
        row_reshaped = np.hstack([row.reshape(-1, 1), np.zeros((row.shape[0], feature_dim - 1))])
        row_rescaled = scaler.inverse_transform(row_reshaped)[:, 0]
        rescaled.append(row_rescaled)
    return np.array(rescaled)

feature_dim = scaled_data.shape[1]
y_pred_rescaled = rescale_batch(y_pred, scaler, feature_dim)
y_test_rescaled = rescale_batch(y_test, scaler, feature_dim)

# Обратная трансформация нормализованных данных
#y_pred = scaler.inverse_transform(y_pred)
#y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(y_test_rescaled[0], label='Истинные значения')
plt.plot(y_pred_rescaled[0], label='Прогнозируемые значения', color='red')
plt.title('Прогнозирование с использованием модели LSTM')
plt.legend()
plt.show()


### 7. **Оценка модели**
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Оценка модели (MSE, RMSE, MAE)
mse = mean_squared_error(y_test_rescaled, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

last_data = data.values[-time_step:]
last_data.mean()

last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
X_input = last_data_scaled.reshape(1, time_step, 1)
predicted_scaled = model.predict(X_input)
predicted_values = scaler.inverse_transform(predicted_scaled)
print(f'Predicted Price: {predicted_values}')
