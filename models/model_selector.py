from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from utils.features import extract_features
import numpy as np
import pandas as pd

def select_best_model_ml(train: pd.Series, test: pd.Series, arima_pred: np.ndarray, hw_pred: np.ndarray):
    """
    Обучает классификатор на основе простых фичей ряда, чтобы выбрать лучшую модель
    """
    X_train = extract_features(train)

    # Целевая переменная: какая модель дает меньшую ошибку
    error_arima = mean_squared_error(test, arima_pred)
    error_hw = mean_squared_error(test, hw_pred)

    y_train = [0 if error_arima < error_hw else 1]  # 0 = ARIMA лучше, 1 = Holt-Winters лучше

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Прогноз по фичам
    X_test = extract_features(pd.concat([train, test]))
    prediction = model.predict(X_test)[0]

    # Возвращаем соответствующий прогноз
    return hw_pred if prediction > 0.5 else arima_pred

