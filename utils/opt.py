import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from data.download_data import data_raw

# Шаг 2: Предобработка данных
def preprocess_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


# Шаг 3: Создание и обучение модели LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Шаг 4: Прогнозирование будущих цен
def predict_future_prices(model, last_sequence, scaler, days=30):
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(days):
        # Предсказываем следующее значение
        prediction = model.predict(current_sequence)

        # Сохраняем предсказанное значение
        future_predictions.append(prediction[0, 0])

        # Обновляем текущую последовательность, добавляя новое предсказание
        # Убедимся, что prediction имеет правильную форму
        prediction = np.reshape(prediction, (1, 1, 1))  # Форма: (1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], prediction, axis=1)

    # Инвертируем масштабирование для получения реальных значений
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

# Шаг 3: Оптимизация портфеля
def optimize_portfolio(expected_returns, covariance_matrix, total_investment):
    num_assets = len(expected_returns)

    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        # Минимизируем риск (волатильность) при фиксированной доходности
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = [1 / num_assets] * num_assets

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    allocation = optimal_weights * total_investment
    return allocation, optimal_weights


# Основная функция
def create_investment_portfolio(pairs, investment_amount, forecast_days=5):
    # Скачиваем данные
    data = data_raw(pairs)

    predictions = {}
    returns = {}
    for pair in pairs:
        pair_data = data[pair].dropna()

        # Предобработка данных
        X, y, scaler = preprocess_data(pair_data)
        model = build_lstm_model((X.shape[1], 1))

        # Обучение модели
        model.fit(X, y, batch_size=32, epochs=20, verbose=1)

        # Прогнозирование будущих цен
        last_sequence = X[-1].reshape(1, -1, 1)
        future_prices = predict_future_prices(model, last_sequence, scaler, days=forecast_days)

        current_price = pair_data.values[-1]
        future_price = future_prices[-1]
        predicted_return = (future_price - current_price) / current_price

        predictions[pair] = future_price
        returns[pair] = predicted_return

    # Используем предсказанные доходности
    expected_returns = np.array(list(returns.values()))

    # Используем историческую ковариационную матрицу для волатильности
    historical_returns = pd.DataFrame({pair: data[pair].pct_change().dropna() for pair in pairs})
    covariance_matrix = historical_returns.cov().values

    # Оптимизация портфеля
    allocation, weights = optimize_portfolio(expected_returns, covariance_matrix, investment_amount)

    # Вывод результатов
    portfolio = {pairs[i]: {
        'allocation': allocation[i],
        'weight': weights[i],
        'predicted_return': expected_returns[i],
        'predicted_price': predictions[pairs[i]]
    } for i in range(len(pairs))}

    return portfolio
