# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 01:02:18 2025

@author: Colby Jaskowiak
"""
# Seventh (not needed usually)
import pandas as pd
import os
from datetime import datetime, timedelta

#%%
from tracker import load_portfolio, fetch_price_data, calculate_portfolio_value
from metrics import (
    calculate_daily_returns,
    calculate_cumulative_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_sortino_ratio
)
from reporting import generate_pdf_report
from weighting import load_portfolio_weights
from config import weighting_mode

#%%
def update_portfolio():
    # --- File paths ---
    csv_path = r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\data\portfolio.csv'
    history_path = r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\data\portfolio_history.csv'
    report_path = r'C:\Users\Colby Jaskowiak\OneDrive\Documents\aa New Projects Folder\1. Automated Portfolio Tracker\reports\portfolio_report.pdf'

    # --- Load portfolio ---
    portfolio = load_portfolio(csv_path)
    tickers = portfolio['Ticker'].tolist()
    weights_series = load_portfolio_weights(method=weighting_mode)
    weights = weights_series.values

    # --- Load existing history ---
    if os.path.exists(history_path):
        history = pd.read_csv(history_path, index_col=0, parse_dates=True)
        history.index = pd.to_datetime(history.index)

        if 'Portfolio Value' in history.columns:
            history = history['Portfolio Value']
        else:
            history = history.iloc[:, 0]

        last_date = history.index[-1].date()
        start_date = last_date + timedelta(days=1)
    else:
        history = pd.Series(dtype='float64')
        start_date = datetime(2023, 1, 1).date()

    # --- Prevent updating too early ---
    today = datetime.now().date()
    max_safe_date = today - timedelta(days=1)
    if start_date > max_safe_date:
        print("No new data available yet. Market hasn't updated.")
        return

    # --- Fetch new data ---
    price_data = fetch_price_data(tickers, start=start_date)
    price_data.index = pd.to_datetime(price_data.index)
    returns = price_data.pct_change().dropna()

    if price_data.empty or price_data.shape[0] < 2:
        print("Not enough new data to update portfolio.")
        return

    # --- Calculate new values ---
    new_values = calculate_portfolio_value(price_data, weights)
    new_values.name = 'Portfolio Value'
    new_values.index = pd.to_datetime(new_values.index)

    new_entries = new_values[~new_values.index.isin(history.index)]
    if not new_entries.empty:
        updated_history = pd.concat([history, new_entries])
    else:
        updated_history = history

    if isinstance(updated_history, pd.DataFrame) and updated_history.shape[1] == 1:
        updated_history = updated_history.iloc[:, 0]

    # --- Save history and report ---
    updated_history.to_frame(name='Portfolio Value').to_csv(history_path)

    daily_returns = calculate_daily_returns(updated_history)
    metrics = {
        'Cumulative Return': calculate_cumulative_return(updated_history),
        'Annual Volatility': calculate_annualized_volatility(daily_returns),
        'Sharpe Ratio': calculate_sharpe_ratio(daily_returns),
        'Sortino Ratio': calculate_sortino_ratio(daily_returns),
        'Max Drawdown': calculate_max_drawdown(updated_history),
        'CAGR': calculate_cagr(updated_history)
    }

    # --- Normalize SPY if available ---
    if 'SPY' in price_data.columns:
        benchmark_series = price_data['SPY'] / price_data['SPY'].iloc[0]
        benchmark_series = benchmark_series.loc[new_values.index]
    else:
        benchmark_series = None

    # --- Generate Report ---
    generate_pdf_report(
        output_path=report_path,
        portfolio_value=updated_history,
        weights=weights,
        tickers=tickers,
        metrics_dict=metrics,
        weighting_mode=weighting_mode,
        daily_returns=daily_returns,
        returns_df=returns,
        benchmark_series=benchmark_series
    )

    print("Update complete. Report saved.")

# --- Only run if executed directly ---
if __name__ == '__main__':
    update_portfolio()

#%%