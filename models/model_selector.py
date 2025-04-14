import numpy as np
import pandas as pd
from models.arima_model import arima_forecast
from models.holt_winters import holt_winters_forecast
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

MODEL_PATH = 'models/model_choice_clf.pkl'

def get_model_predictions(series: pd.Series, steps: int = 30) -> dict:
    """Возвращает прогнозы от всех моделей для сравнения."""
    predictions = {}

    # ARIMA
    try:
        ar_forecast = arima_forecast(series, steps=steps)
        predictions['arima'] = arima_forecast
    except Exception as e:
        print(f"ARIMA error: {e}")
        predictions['arima'] = np.full(steps, np.nan)

    # Holt-Winters
    try:
        hw_forecast = holt_winters_forecast(series, steps=steps)
        predictions['holt_winters'] = hw_forecast
    except Exception as e:
        print(f"Holt-Winters error: {e}")
        predictions['holt_winters'] = np.full(steps, np.nan)

    return predictions

def generate_training_data(pairs, fetch_func, history_days=90, forecast_days=30):
    X, y = [], []

    for symbol in pairs:
        df = fetch_func(symbol, history_days)
        if df is None or df.empty:
            continue

        df['return'] = df['close'].pct_change().dropna()
        series = df['close'].dropna()

        if len(series) < forecast_days * 2:
            continue

        train = series[:-forecast_days]
        test = series[-forecast_days:]

        model_preds = get_model_predictions(train, steps=forecast_days)

        mse_scores = {
            model: mean_squared_error(test, preds)
            for model, preds in model_preds.items()
            if not np.isnan(preds).any()
        }

        if not mse_scores:
            continue

        best_model = min(mse_scores, key=mse_scores.get)
        X.append([series.mean(), series.std()])
        y.append(best_model)

    return np.array(X), np.array(y)

def train_selector_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, 'models/label_encoder.pkl')
    return clf

def load_selector_model():
    try:
        clf = joblib.load(MODEL_PATH)
        le = joblib.load('models/label_encoder.pkl')
        return clf, le
    except Exception:
        print("ML выбор модели недоступен: Модель выбора не обучена. Запусти обучение.")
        return None, None
