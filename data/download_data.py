from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd

def data_forecast(pairs, forecast_horizon=7):
    results = {}

    for pair in pairs:
        series = fetch_ohlcv(pair)
        series = series.asfreq('D')
        data = series.values

        best_model_name, best_mae, forecast = select_best_model(data, forecast_horizon)

        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_series = pd.Series(forecast, index=forecast_dates)

        results[pair] = {
            'model': best_model_name,
            'mae': best_mae,
            'forecast': forecast_series
        }

        print(f"📈 Прогноз на {forecast_horizon} дней для {pair} с моделью {best_model_name} (MAE: {best_mae:.4f}):")
        print(forecast_series)
        print()

    return results

def data_raw(pairs, timeframe='1d', limit=365):
    df_dict = {}
    for pair in pairs:
        series = fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        series = series.asfreq('D')  # приведение к ежедневной частоте
        df_dict[pair] = series
    # Объединение в единый DataFrame по индексу (timestamp)
    df_all = pd.DataFrame(df_dict)
    return df_all