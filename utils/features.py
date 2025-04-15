import numpy as np
import pandas as pd

def extract_features(series: pd.Series, steps: int = 30):
    """
    Извлечение простых статистических признаков из временного ряда
    """
    returns = np.diff(np.log(series + 1e-8))
    features = {
        "mean": np.mean(series),
        "std": np.std(series),
        "min": np.min(series),
        "max": np.max(series),
        "skew": pd.Series(series).skew(),
        "kurt": pd.Series(series).kurt(),
        "last_return": returns[-1] if len(returns) > 0 else 0
    }
    return np.array(list(features.values())).reshape(1, -1)
