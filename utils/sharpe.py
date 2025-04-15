import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365):
    """
    Расчет годового коэффициента Шарпа.

    :param returns: массив доходностей (обычно лог-доходности)
    :param risk_free_rate: безрисковая ставка (в тех же временных единицах, что и доходности)
    :param periods_per_year: сколько периодов в году (например, 365 для дневных данных)
    :return: коэффициент Шарпа
    """
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        return 0

    sharpe = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)
    return sharpe

def neg_sharpe(weights, returns_df):
    portfolio_return = np.sum(returns_df.mean() * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights)))
    return -portfolio_return / portfolio_std