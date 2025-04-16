import numpy as np
import pandas as pd
from scipy.optimize import minimize
from services.fetch_data import fetch_ohlcv
from models.LSTM import train_lstm

def load_crypto_data(pairs):
    df_dict = {}
    for pair in pairs:
        df = fetch_ohlcv(pair)
        df_dict[pair] = df
    return pd.DataFrame(df_dict)

# Шаг 3: Оптимизация портфеля
def optimize_portfolio(forecasts, total_investment):
    tickers = list(forecasts.keys())
    prices = np.array(list(forecasts.values()))

    # Целевая функция: минимизация отклонения от равновесного распределения
    def objective(weights):
        portfolio_value = np.dot(weights, prices)
        return np.sum((weights * portfolio_value - total_investment / len(tickers))**2)

    # Ограничения: сумма весов = 1, веса >= 0
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in tickers]

    # Начальные веса: равномерное распределение
    initial_weights = np.ones(len(tickers)) / len(tickers)

    # Оптимизация
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x

    # Рассчитываем количество акций
    shares = (optimized_weights * total_investment) / prices
    return dict(zip(tickers, shares.astype(int))), dict(zip(tickers, optimized_weights))

# Шаг 4: Создание портфеля
def create_portfolio(pairs, total_investment, forecast_horizon):
    data = load_crypto_data(pairs)

    # Прогнозирование
    forecasts = train_lstm(data, forecast_horizon)

    # Оптимизация портфеля
    shares, weights = optimize_portfolio(forecasts, total_investment)

    # Вывод результатов
    return shares, weights

def make_portfolio_minvol(portfolio):
    ans = ['📊 Твой крипто-портфель:']
    for symbol in portfolio.keys():
        ans.append(f"{symbol} — {portfolio[symbol]} единиц")
    return '\n'.join(ans)