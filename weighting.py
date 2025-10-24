# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:21:25 2025

@author: Colby Jaskowiak
"""
# Fourth
import pandas as pd
import numpy as np
import riskfolio as rp
import yfinance as yf

#%%
# Calculate Various Weights
def calculate_equal_weights(tickers):
    n = len(tickers)
    return pd.Series([1/n] * n, index=tickers)

def calculate_inverse_vol_weights(price_data, lookback_days=60):
    returns = price_data.pct_change().dropna()
    recent = returns[-lookback_days:]
    vol = recent.std()
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()

def calculate_sharpe_opt_weights(returns):
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='hist')
    weights = port.optimization(model='Classic', rm='MSV', obj='Sharpe', rf=0, l=0, hist=True)
    return weights.iloc[:, 0]

#%%
# Weight Loading
def load_portfolio_weights(method='equal', filepath=r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\data\portfolio.csv', start='2023-01-01'):
    portfolio = pd.read_csv(filepath)
    tickers = portfolio['Ticker'].tolist()

    if method == 'equal':
        return calculate_equal_weights(tickers)

    # Download price data
    price_data = yf.download(tickers, start=start)['Close']

    if method == 'inverse_vol':
        return calculate_inverse_vol_weights(price_data)

    elif method == 'sharpe_opt':
        returns = price_data.pct_change().dropna()
        return calculate_sharpe_opt_weights(returns)

    else:
        raise ValueError("Invalid weighting method. Choose from 'equal', 'inverse_vol', or 'sharpe_opt'.")

#%%