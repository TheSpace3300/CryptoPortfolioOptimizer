import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def neg_sharpe(weights, returns_df):
    portfolio_return = np.sum(returns_df.mean() * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights)))
    return -portfolio_return / portfolio_std