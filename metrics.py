# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 02:40:28 2025

@author: User
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def calculate_metrics(returns, risk_free_rate: float = 0.0, trades: pd.Series = None):
    metrics = {}
    returns = returns.dropna()

    # Basic returns
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    metrics["Total Return"] = total_return
    metrics["Annualized Return"] = ann_return

    # Risk metrics
    vol = returns.std()
    ann_vol = vol * np.sqrt(252)
    metrics["Volatility"] = vol
    metrics["Annualized Volatility"] = ann_vol

    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    metrics["Max Drawdown"] = drawdown.min()
    metrics["Calmar Ratio"] = ann_return / abs(drawdown.min()) if drawdown.min() != 0 else np.nan

    # Risk-adjusted
    sharpe = (returns.mean() - risk_free_rate) / vol * np.sqrt(252) if vol > 0 else 0
    downside_vol = returns[returns < 0].std()
    sortino = (returns.mean() - risk_free_rate) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
    metrics["Sharpe Ratio"] = sharpe
    metrics["Sortino Ratio"] = sortino

    # Trade stats (if trades available)
    if trades is not None and len(trades) > 0:
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        metrics["Number of Trades"] = len(trades)
        metrics["Win Rate (%)"] = len(wins) / len(trades) * 100
        metrics["Average Win"] = wins.mean() if len(wins) > 0 else 0
        metrics["Average Loss"] = losses.mean() if len(losses) > 0 else 0
        metrics["Profit Factor"] = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
        metrics["Payoff Ratio"] = abs(metrics["Average Win"] / metrics["Average Loss"]) if metrics["Average Loss"] != 0 else np.inf
        metrics["Expectancy"] = trades.mean()
    
    return metrics
