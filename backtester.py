# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 02:51:54 2025

@author: User
"""

import pandas as pd
import numpy as np
from metrics import calculate_metrics

class Backtester:
    def __init__(self, data: pd.DataFrame, signal_col: str = "signal", price_col: str = "close"):
        self.data = data.copy()
        self.signal_col = signal_col
        self.price_col = price_col
        self.results = None
        self.trades = None

    def run_backtest(self, initial_capital: float = 100_000, position_size: float = 1.0):
        df = self.data.copy()
        df["returns"] = df[self.price_col].pct_change().fillna(0)
        
        # Strategy returns = signal * asset returns
        df["strategy_returns"] = df[self.signal_col].shift(1) * df["returns"]
        
        # Equity curve
        df["equity_curve"] = (1 + df["strategy_returns"] * position_size).cumprod() * initial_capital
        
        self.results = df

        # Build trade log
        self.trades = self._generate_trade_log(df, initial_capital, position_size)
        
        return df

    def _generate_trade_log(self, df: pd.DataFrame, initial_capital: float, position_size: float):
        """
        Parse signal changes into trade entries/exits.
        """
        trades = []
        position = 0
        entry_price = 0
        entry_index = None

        for i, row in df.iterrows():
            signal = row[self.signal_col]
            price = row[self.price_col]

            # Entry
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_index = i

            # Exit (when signal goes flat or flips)
            elif position != 0 and signal != position:
                exit_price = price
                pnl = (exit_price - entry_price) / entry_price * position * initial_capital * position_size
                trades.append({
                    "Entry Date": entry_index,
                    "Exit Date": i,
                    "Direction": "Long" if position == 1 else "Short",
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL": pnl,
                    "Return (%)": pnl / (initial_capital * position_size) * 100,
                    "Holding Period": (i - entry_index).days if hasattr(i, "days") else 1
                })
                position = 0
                entry_price = 0
                entry_index = None

        return pd.DataFrame(trades)

    def evaluate(self):
        if self.results is None:
            raise ValueError("Run backtest first with run_backtest()")

        return calculate_metrics(self.results["strategy_returns"], trades=self.trades["PnL"] if self.trades is not None else None)
