import json
import numpy as np
import optuna
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import concurrent.futures

"""
12 May 2026
Base class for ticker parameter optimization. Each strategy (portfolio) has its own subclass of this.

ONLY IMPLEMENTED FOR portfolio_3 currently

Given 1 jan 2015 to 31 dec 2025, parameters are trained from 01/01/2015 to 31/12/2022,
01/01/2023 to 31/12/2023 used as validation set, 2024-2025 is holdout test set.
"""

class TickerParamOptim(ABC):

    def __init__(
        self,
        cache_dir: str = 'src/backtest/data/backfill_cache', # local ticker data cache as of May 11 2026
        params_path: str = 'src/portfolios/portfolio_3/ticker_params.json', 
        train_end: str = '2022-12-31',
        val_start: str = '2023-01-01',
        val_end: str = '2023-12-31'
    ):
        self.cache_dir = Path(cache_dir)
        self.params_path = Path(params_path)
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end

        self.vix_df = None # initialized once for each run() call
    
    def load_data(self, ticker:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns cached ticker data and cached vix data as pandas dataframes

        Use ticker name and self.cache_dir to get ticker data. Make sure invalid
        file name characters are removed before looking.
        """
        safe = ticker.replace('^','_').replace('/','_')

        ticker_df = pd.read_parquet(self.cache_dir / f'{safe}.parquet')
        ticker_df = ticker_df.set_index('timestamp').sort_index()

        vix_df = pd.read_parquet(self.cache_dir / '_VIX.parquet')
        vix_df = vix_df.set_index('timestamp').sort_index()

        return ticker_df, vix_df
    
    def compute_sharpe(self, returns: pd.Series) -> float:
        """
        Computes risk-adjusted performance of a given ticker.

        Input raw returns from optimizer iteration for a ticker, compute
        sqrt(total bars per year) * average return / return std.
        """
        returns = returns.dropna()
        if len(returns) < 30 or returns.std() == 0:
            return -999.0
        bars_per_year = 252 * 13 # 13 30-min bars per ticker per day in cache
        return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

    def run(self, ticker: str, n_trials: int=150, save: bool=True) -> dict:


        ticker_df, self.vix_df = self.load_data(ticker)

        # TPEsampler by default
        # MedianPruner is default pruner
        study = optuna.create_study(
            direction='maximize', # maximizing return
        )

        study.optimize(
            lambda trial: self._objective(trial, ticker_df),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best = study.best_params
        if save:
            self.save_results(ticker, best, study.best_value)
        print(f"[{ticker}] Best Sharpe: {study.best_value:.4f} | Params: {best}")
        return ticker, best, study.best_value

    def _objective(self, trial, ticker_df: pd.DataFrame) -> float:
        """
        This method defines the optimizers objective function and is whats used
        by the Optuna study.optimize() call in run().

        Uses whole ticker_df up to val_end. This allows indicators to be warmed up
        before evaluating params on validation set. Evaluation uses only the
        validation section of ticker_df.
        """
        params = self.suggest_params(trial) # anon func, defined in subclasses. Contains the params being tuned

        # test + val data
        test_val_df = ticker_df[:self.val_end]

        # add indicators to test_val_df. Is an anon function, defined in subclasses
        # in order to use only the indicators needed for a given strategy.
        df_ind = self.compute_indicators(test_val_df, self.vix_df)

        # simulate signal, anon func, based on the strategy being used
        returns = self.simulate_signals(df_ind, params)

        # Extract only validation performance, return risk-adjusted performance
        val_returns = returns[self.val_start:self.val_end]
        return self.compute_sharpe(val_returns)
    
    def save_results(self, ticker:str, params:dict, sharpe:float) -> None:
        """
        Saves best parameters for a ticker for a strategy
        """
        existing = {}

        # check if path already exists
        if self.params_path.exists():
            with open(self.params_path) as file:
                existing = json.load(file)
        
        existing[ticker] = {
            **params,                               # store best params from optimizer 
            'sharpe_validation': round(sharpe, 4),  # Store risk-adjusted performance on validation set
            'optimized_at': str(date.today()),      # store date of this optimization
            'train_period': '2010-01-01/2022-12-31 (5-fold CV)',
            'validation_period': '2015,2017,2019,2021,2023'
        }
        with open(self.params_path, 'w') as file:
            json.dump(existing, file, indent=2)
    

    """
    Abstract methods to be defined by subclasses (i.e., strategy-specific methods)
    """
    @abstractmethod
    def suggest_params(self, trial) -> dict:
        # Defines what parameters to optimize and what values to test
        ...
    
    @abstractmethod
    def compute_indicators(self, df:pd.DataFrame, vix_df:pd.DataFrame) -> pd.DataFrame:
        # Compute indicators needed for signal logic for a strategy
        ...
    
    @abstractmethod
    def simulate_signals(self, df:pd.DataFrame, params: dict) -> pd.Series:
        # Apply signal logic to ticker data after indicators have been added to it
        ...