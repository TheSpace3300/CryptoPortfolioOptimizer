import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import STL

def extract_features(series: pd.Series) -> pd.Series:
    values = series.values
    returns = np.diff(np.log(values + 1e-8))

    stl = STL(series, period=7, robust=True).fit()
    trend_strength = 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid)
    seasonal_strength = 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)

    features = {
        "mean": np.mean(values),
        "std": np.std(values),
        "volatility": np.std(returns),
        "skewness": skew(returns),
        "kurtosis": kurtosis(returns),
        "acf1": acf(returns, nlags=1)[1],
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "length": len(values)
    }
    return pd.Series(features)
