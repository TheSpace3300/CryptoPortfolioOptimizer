import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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
    time_step = 60
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    test_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:-test_size]
    test_data = data_scaled[-test_size:]

    X_train, y_train = create_lstm_dataset(train_data, time_step, forecast_horizon)
    X_test, y_test = create_lstm_dataset(test_data, time_step, forecast_horizon)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test_original = scaler.inverse_transform(y_test)

    rmse = sqrt(mean_squared_error(y_test_original, predictions))


    last_data = data[-time_step:]
    last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
    X_input = last_data_scaled.reshape(1, time_step, 1)
    predicted_scaled = model.predict(X_input)
    predicted_values = scaler.inverse_transform(predicted_scaled)[0]


    return "LSTM", rmse, predicted_values[:forecast_horizon]