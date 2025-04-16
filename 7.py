from services.fetch_data import fetch_ohlcv
from models.model_selector import select_best_model
import pandas as pd
from utils.opt import create_portfolio, make_portfolio_minvol

if __name__ == "__main__":
    crypto_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    investment = 10000
    shares, weights = create_portfolio(crypto_pairs, investment, forecast_horizon=7)
    print(make_portfolio_minvol(shares))