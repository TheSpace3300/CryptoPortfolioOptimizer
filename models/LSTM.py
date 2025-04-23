import numpy as np
import pandas as pd
from ccxt.static_dependencies.ethereum.utils.units import units
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def create_lstm_dataset(data, time_step, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - time_step - forecast_horizon + 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])
    return np.array(X), np.array(y)

def train_lstm(data, forecast_horizon):
    min_required = forecast_horizon + 10
    if len(data) < min_required:
        raise ValueError(f"Слишком мало данных для прогноза LSTM (нужно хотя бы {min_required}, а есть {len(data)})")
    time_step = min(60, len(data) - forecast_horizon - 1)
    if time_step < 1:
        raise ValueError("Недостаточно данных даже для одного шага.")

    train_data = data[:-forecast_horizon]
    test_data = data[-forecast_horizon:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    full_scaled = scaler.fit_transform(data.reshape(-1, 1))


    X_train, y_train = create_lstm_dataset(train_scaled, time_step, forecast_horizon)

    if len(X_train) == 0:
        raise ValueError("Недостаточно данных после разбиения на train/test для LSTM.")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    last_input = train_scaled[-time_step:]
    X_input = last_input.reshape(1, time_step, 1)
    predicted_scaled = model.predict(X_input)
    predicted = scaler.inverse_transform(predicted_scaled)[0]

    mae = mean_absolute_error(test_data, predicted)

    X_full, y_full = create_lstm_dataset(full_scaled, time_step, forecast_horizon)
    X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

    model_full = Sequential()
    model_full.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model_full.add(Dropout(0.2))
    model_full.add(LSTM(100))
    model_full.add(Dropout(0.2))
    model_full.add(Dense(25))
    model_full.add(Dense(forecast_horizon))
    model_full.compile(optimizer='adam', loss='mean_squared_error')
    model_full.fit(X_full, y_full, epochs=10, batch_size=16, verbose=0)

    last_input_full = full_scaled[-time_step:]
    X_input_full = last_input_full.reshape(1, time_step, 1)
    forecast_future_scaled = model_full.predict(X_input_full)
    forecast_future = scaler.inverse_transform(forecast_future_scaled)[0]

    return "LSTM", mae, forecast_future