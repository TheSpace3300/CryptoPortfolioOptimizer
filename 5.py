import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

from services.fetch_data import fetch_ohlcv

# ---------------------------- Конфигурация ---------------------------- #
PAIRS = ['BTC/USDT']
TIME_STEP = 60
FORECAST_HORIZON = 10
EPOCHS = 80
BATCH_SIZE = 32
SAVE_FIG = True
PLOTS_DIR = 'plots'

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------- Функции ---------------------------- #
def create_dataset(data, time_step, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - time_step - forecast_horizon + 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[(i + time_step):(i + time_step + forecast_horizon), 0])
    return np.array(X), np.array(y)

def prepare_data(prices, time_step, forecast_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(prices.values.reshape(-1, 1))

    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    X_train, y_train = create_dataset(train_data, time_step, forecast_horizon)
    X_test, y_test = create_dataset(test_data, time_step, forecast_horizon)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler, prices

def create_model(input_shape, forecast_horizon):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'\n📊 Оценка модели:')
    print(f'MSE  = {mse:.4f}')
    print(f'RMSE = {rmse:.4f}')
    print(f'MAE  = {mae:.4f}')

def forecast_future(model, prices, scaler, time_step):
    last_data = prices[-time_step:]
    last_scaled = scaler.transform(last_data.values.reshape(-1, 1))
    input_seq = last_scaled.reshape(1, time_step, 1)
    predicted_scaled = model.predict(input_seq)
    predicted = scaler.inverse_transform(predicted_scaled)
    print(f'\n🔮 Прогноз на следующие {predicted.shape[1]} дней: {predicted.flatten()}')

def plot_predictions(y_true, y_pred, title, save_path=None):
    plt.figure(figsize=(14, 5))
    plt.plot(y_true[0], label='Истинные значения', color='blue')
    plt.plot(y_pred[0], label='Прогнозируемые значения', color='red')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ---------------------------- Основной цикл ---------------------------- #
for pair in PAIRS:
    print(f"\n🚀 Обработка пары: {pair}")
    series = fetch_ohlcv(pair)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('D')

    prices = pd.Series(series)
    plt.plot(prices)
    plt.title(f'{pair} — Временной ряд')
    if SAVE_FIG:
        plt.savefig(f"{PLOTS_DIR}/{pair.replace('/', '_')}_raw.png")
    plt.show()

    X_train, y_train, X_test, y_test, scaler, prices = prepare_data(prices, TIME_STEP, FORECAST_HORIZON)

    model = create_model((X_train.shape[1], 1), FORECAST_HORIZON)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test)

    plot_predictions(y_test_original, predictions,
                     f'{pair} — LSTM прогноз',
                     f"{PLOTS_DIR}/{pair.replace('/', '_')}_forecast.png" if SAVE_FIG else None)

    evaluate_model(y_test_original, predictions)
    forecast_future(model, prices, scaler, TIME_STEP)