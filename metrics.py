# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 01:01:12 2025

@author: Colby Jaskowiak
"""
# Second script to run
import pandas as pd
import numpy as np

#%%
# Functions

def calculate_daily_returns(portfolio_value):
    return portfolio_value.pct_change().dropna()

def calculate_cumulative_return(portfolio_value):
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1

def calculate_annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.03):
    excess_returns = daily_returns - (risk_free_rate / 252)
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_max_drawdown(portfolio_value):
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return drawdown.min()

def calculate_cagr(portfolio_value):
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
    num_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    return total_return ** (1 / num_years) - 1

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.03):
    excess_returns = daily_returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('nan')  # Avoid divide by zero
    return float(excess_returns.mean() / downside_std * np.sqrt(252))

#%%