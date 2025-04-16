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
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = create_lstm_dataset(scaled_data, time_step, forecast_horizon)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1))))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(
        np.hstack([y_test, np.zeros((y_test.shape[0], 1))]))[forecast_horizon]
    predictions_rescaled = scaler.inverse_transform(
        np.hstack([predictions, np.zeros((predictions.shape[0], 1))]))[forecast_horizon]

    rmse = sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    return "LSTM", rmse, predictions_rescaled