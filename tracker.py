# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 00:55:48 2025

@author: Colby Jaskowiak
"""
# Sixth Script
import pandas as pd
import yfinance as yf
import numpy as np
import riskfolio as rp

#%%

from weighting import (
    calculate_equal_weights,
    calculate_inverse_vol_weights,
    calculate_sharpe_opt_weights
)

from metrics import (
    calculate_daily_returns,
    calculate_cumulative_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_sortino_ratio
)

from visualizations import plot_portfolio_value, plot_asset_allocation
from tracker import fetch_price_data, calculate_portfolio_value
from weighting import load_portfolio_weights
from reporting import generate_pdf_report
from config import weighting_mode

#%%
# Functions

def load_portfolio(filepath=None):
    if filepath is None:
        filepath = r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\data\portfolio.csv'
    return pd.read_csv(filepath)

def fetch_price_data(tickers, start='2023-01-01'):
    data = yf.download(tickers, start=start)['Close']
    return data

def calculate_portfolio_value(price_data, weights):
    normalized = price_data / price_data.iloc[0]
    portfolio_value = (normalized * weights).sum(axis=1)
    return portfolio_value

#%%
# Load tickers from portfolio CSV
portfolio = load_portfolio()
tickers = portfolio['Ticker'].tolist()

# Download price data
prices = fetch_price_data(tickers)
spy_price_series = prices['SPY']
normalized_spy = spy_price_series / spy_price_series.iloc[0]
returns = prices.pct_change().dropna()

# Calculate weights dynamically
if weighting_mode == 'equal':
    weights_series = calculate_equal_weights(tickers)
elif weighting_mode == 'inverse_vol':
    weights_series = calculate_inverse_vol_weights(prices)
elif weighting_mode == 'sharpe_opt':
    returns = prices.pct_change().dropna()
    weights_series = calculate_sharpe_opt_weights(returns)
else:
    raise ValueError(f"Invalid weighting mode: {weighting_mode}")

# Extract values for use in calculations
weights = weights_series.values
tickers = weights_series.index.tolist()
weights_series = load_portfolio_weights(method=weighting_mode)

portfolio_value = calculate_portfolio_value(prices, weights)
daily_returns = calculate_daily_returns(portfolio_value)

#%%
print("Portfolio Weights:")
print(weights_series.round(4))

#%%
# Run Test
if __name__ == '__main__':
    
    print(portfolio_value.tail())
    print("Cumulative Return:", round(calculate_cumulative_return(portfolio_value), 4))
    print("Annual Volatility:", round(calculate_annualized_volatility(daily_returns), 4))
    print("Sharpe Ratio:", round(calculate_sharpe_ratio(daily_returns), 4))
    print("Sortino Ratio:", round(calculate_sortino_ratio(daily_returns), 4))
    print("Max Drawdown:", round(calculate_max_drawdown(portfolio_value), 4))
    print("CAGR:", round(calculate_cagr(portfolio_value), 4))
    
    # Visualizations
    plot_portfolio_value(portfolio_value)
    plot_asset_allocation(weights, tickers)
    
    # Normalize SPY for comparison if it's in the tickers
    if 'SPY' in prices.columns:
        benchmark_series = prices['SPY'] / prices['SPY'].iloc[0]
        benchmark_series = benchmark_series.loc[portfolio_value.index]  # align dates
    else:
        benchmark_series = None

#%%
# Create metrics dict for reporting
metrics = {
    'Cumulative Return': calculate_cumulative_return(portfolio_value),
    'Annual Volatility': calculate_annualized_volatility(daily_returns),
    'Sharpe Ratio': calculate_sharpe_ratio(daily_returns),
    'Sortino Ratio': calculate_sortino_ratio(daily_returns),
    'Max Drawdown': calculate_max_drawdown(portfolio_value),
    'CAGR': calculate_cagr(portfolio_value)
}

# Export report
output_path = r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\reports\portfolio_report.pdf'
generate_pdf_report(output_path, portfolio_value, weights, tickers, metrics, weighting_mode=weighting_mode, daily_returns=daily_returns, returns_df=returns, benchmark_series=benchmark_series)
print(f"Report saved to: {output_path}")

#%%