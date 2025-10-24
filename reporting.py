# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 01:01:47 2025

@author: Colby Jaskowiak
"""
# Fifth Script
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import textwrap
import numpy as np

#%%
# Weighting method explanations
WEIGHTING_EXPLANATIONS = {
    'equal': (
        "Equal Weighting:\n"
        "Each asset in the portfolio is allocated the same weight, regardless of its volatility or expected return. "
        "This simple approach ensures diversification without relying on historical data."
    ),
    'inverse_vol': (
        "Inverse Volatility Weighting:\n"
        "Assets are weighted inversely proportional to their historical volatility. "
        "Less volatile assets receive higher weights, which can reduce overall portfolio risk."
    ),
    'sharpe_opt': (
        "Sharpe-Optimized Weighting:\n"
        "Weights are optimized to maximize the portfolio's Sharpe ratio. "
        "This method uses historical returns and covariances to find the risk-adjusted return-maximizing allocation."
    )
}

#%%
# Set consistent styling
rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def generate_pdf_report(output_path, portfolio_value, weights, tickers, metrics_dict,
                        weighting_mode='N/A', daily_returns=None, returns_df=None, benchmark_series=None):
    with PdfPages(output_path) as pdf:
        # --- Title Page ---
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        now = datetime.now().strftime("%B %d, %Y")
        lines = [
            "Portfolio Performance Report",
            f"Date: {now}",
            f"Weighting Method: {weighting_mode.capitalize()}",
            "",
            "This report summarizes portfolio performance, allocation, and key metrics."
        ]
        for i, line in enumerate(lines):
            plt.text(0.5, 0.9 - 0.05 * i, line, ha='center', va='top', fontsize=14 if i == 0 else 12)
        pdf.savefig()
        plt.close()

        # --- Portfolio Value Over Time ---
        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_value.index, portfolio_value.values, label='Portfolio Value', linewidth=2, color='teal')
        
        if benchmark_series is not None:
            plt.plot(benchmark_series.index, benchmark_series.values, label='SPY Benchmark', linestyle='--', color='gray')
    
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value (Normalized)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Weighting Method Summary Page (Wrapped) ---
        explanation = WEIGHTING_EXPLANATIONS.get(weighting_mode.lower(), "No explanation available for the selected weighting method.")
        wrapped_lines = textwrap.wrap(explanation, width=80)

        plt.figure(figsize=(8.5, 6))
        plt.axis('off')
        plt.title('Weighting Method Summary', fontsize=16, pad=20)
        for i, line in enumerate(wrapped_lines):
            plt.text(0.05, 0.9 - 0.05 * i, line, fontsize=12, ha='left')
        pdf.savefig()
        plt.close()

        # --- Asset Allocation Pie Chart ---
        plt.figure(figsize=(6, 6))
        colors = plt.cm.Set3(range(len(tickers)))
        plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140,
                colors=colors, textprops={'fontsize': 10})
        plt.title('Asset Allocation')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Performance Metrics Summary ---
        plt.figure(figsize=(6, 4))
        plt.axis('off')
        metric_lines = [f'{k}: {float(v):.4f}' for k, v in metrics_dict.items()]
        for i, line in enumerate(metric_lines):
            plt.text(0, 1 - 0.15 * i, line, fontsize=11)
        plt.title('Performance Metrics', fontsize=14, loc='left')
        pdf.savefig()
        plt.close()
        
        # --- Drawdown Chart ---
        cumulative_max = portfolio_value.cummax()
        drawdowns = (portfolio_value - cumulative_max) / cumulative_max
        
        plt.figure(figsize=(10, 4))
        plt.plot(drawdowns.index, drawdowns.values, color='red', linewidth=2)
        plt.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)
        plt.title('Portfolio Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # --- Correlation Heatmap ---
        if returns_df is not None and not returns_df.empty:
            corr = returns_df.corr()
        
            plt.figure(figsize=(8, 6))
            plt.imshow(corr, cmap='coolwarm', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Correlation')
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title('Asset Correlation Heatmap')
            for i in range(len(corr.columns)):
                for j in range(len(corr.index)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # --- Rolling Sharpe Ratio ---
        if daily_returns is not None and not daily_returns.empty:
            rolling_mean = daily_returns.rolling(window=60).mean()
            rolling_std = daily_returns.rolling(window=60).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
            
            plt.figure(figsize=(10, 4))
            rolling_sharpe.plot(color='purple', linewidth=1.5)
            plt.title('Rolling Sharpe Ratio (60-day window)')
            plt.xlabel('Date')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

#%%