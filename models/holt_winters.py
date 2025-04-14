import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class HoltWinters:
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)

        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen)
            )

        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                    self.series[self.slen * j + i] - season_averages[j]
                )
            seasonals[i] = sum_of_vals_over_avg / n_seasons

        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                self.PredictedDeviation.append(0)

                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )
                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )
                continue

            if i >= len(self.series):
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)
            else:
                val = self.series[i]
                last_smooth = smooth
                smooth = self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i]) +
                    (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )
            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

def timeseriesCVscore(x, series):
    errors = []
    values = series.values.flatten()
    alpha, beta, gamma = x
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(values):
        model = HoltWinters(values[train_idx], slen=7, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test_idx))
        model.triple_exponential_smoothing()
        predictions = model.result[-len(test_idx):]
        actual = values[test_idx]
        error = mean_squared_error(predictions, actual)
        errors.append(error)

    return np.mean(errors)

def holt_winters_forecast(series: pd.Series, steps: int = 30):
    opt = minimize(lambda x: timeseriesCVscore(x, series),
                   x0=[0.5, 0.5, 0.5],
                   method="TNC",
                   bounds=((0, 1), (0, 1), (0, 1)))

    alpha_final, beta_final, gamma_final = opt.x

    hw = HoltWinters(series.values.flatten(),
                     slen=7,
                     alpha=alpha_final,
                     beta=beta_final,
                     gamma=gamma_final,
                     n_preds=steps)
    hw.triple_exponential_smoothing()
    hw_pred = hw.result[-steps:]
    return hw_pred