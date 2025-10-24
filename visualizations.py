# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 01:01:26 2025

@author: Colby Jaskowiak
"""

# Third Script
import numpy as np
import matplotlib.pyplot as plt

#%%
# 1. Portfolio Value Time-Series Chart

def plot_portfolio_value(portfolio_value):
    plt.figure(figsize=(10,5))
    plt.plot(portfolio_value, label='Portfolio Value', linewidth=2)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value (Normalized)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
# 2. Asset Allocation Pie Chart

def plot_asset_allocation(weights, tickers):
    plt.figure(figsize=(6,6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title('Asset Allocation')
    plt.tight_layout()
    plt.show()

#%%
# 3. Rolling Sharpe Ratio Chart

def plot_rolling_sharpe_ratio(daily_returns, window=60):

    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    plt.figure(figsize=(10, 4))
    rolling_sharpe.plot(color='purple', linewidth=1.5)
    plt.title(f'Rolling Sharpe Ratio (Window={window})')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%