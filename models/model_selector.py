import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from utils.metrics import rmse
from utils.features import extract_features
import joblib
import os

MODEL_PATH = "models/model_choice_clf.pkl"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train_selector_model(X: pd.DataFrame, y: pd.Series):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf

def predict_model_choice(series: pd.Series) -> str:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель выбора не обучена. Запусти обучение.")

    clf = joblib.load(MODEL_PATH)
    feats = extract_features(series).to_frame().T
    pred = clf.predict(feats)[0]
    return "arima" if pred == 1 else "holt"

def select_best_model(test_series, arima_pred, hw_pred, train_series=None):
    if train_series is not None:
        try:
            method = predict_model_choice(train_series)
            return arima_pred if method == "arima" else hw_pred
        except Exception as e:
            print("ML выбор модели недоступен:", e)

    arima_error = rmse(test_series, arima_pred)
    hw_error = rmse(test_series, hw_pred)
    return arima_pred if arima_error < hw_error else hw_pred

def generate_training_data(pairs, fetch_func):
    X = []
    y = []
    for pair in pairs:
        series = fetch_func(pair)
        train, test = series[:-30], series[-30:]

        arima_pred = arima_forecast(train, steps=30)
        hw_pred = holt_winters_forecast(train, steps=30)

        arima_error = rmse(test, arima_pred)
        hw_error = rmse(test, hw_pred)

        target = 1 if arima_error < hw_error else 0
        X.append(extract_features(train))
        y.append(target)

    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    return X_df, y_series
