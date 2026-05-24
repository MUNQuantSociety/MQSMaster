import argparse
import numpy as np
import pandas as pd
from src.backtest.optimizer import TickerParamOptim

"""
Subclass of src/backtest/optimizer.py for portfolio_3

NOTE THIS OPTIMIZER DOES NOT INPUT VALUES FROM strategy.py, therefore:
    - indicator periods are hardcoded
    - assumes 1 bar per day of VIX
    - assumes 13 bars per day for all other tickers
"""

PARAMS_PATH = 'src/portfolios/portfolio_3/ticker_params.json'

class Portfolio3Optimizer(TickerParamOptim):
    def __init__(self, **kwargs):
        super().__init__(params_path=PARAMS_PATH, **kwargs)
    
    def suggest_params(self, trial):
        """
        Just returns a dict of params to be optimized in the form of optuna suggestions
        """
        params = {
            'ATR_BAND_MULT': trial.suggest_float('ATR_BAND_MULT', 0.5, 3.0, step=0.1),
            'MOMENTUM_THRESHOLD': trial.suggest_float('MOMENTUM_THRESHOLD', 0.3, 3.0, step=0.1),
            'BASE_CONF': trial.suggest_float('BASE_CONF', 0.3, 0.9, step=0.05),
            'STOP_LOSS_ATR_MULT': trial.suggest_float('STOP_LOSS_ATR_MULT', 1.0, 6.0, step=0.5),
            'REVERSAL_THRESHOLD': trial.suggest_int('REVERSAL_THRESHOLD', 1, 5)
        }
        return params
    
    def compute_indicators(self, df, vix_df):
        """
        Computes vwap atr roc and sma50 for portfolio_3.

        Uses same rolling day counts as are hardcoded in portfolio_3's strategy.py. 
            vwap 20 day
            atr 14 day
            roc 10 day
            vix ema 10 day

        NOTE ^^ these are 'periods' not days. converting periods to days should be done
        """
        df = df.copy()

        # computing VWAP -> volume * average high/low/close / volume, uses rolling 20
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # computing ATR -> previous ATR * (n-1) + current TR (max of high-low, abs(high-prev close), abs(low-prev close))
        prev_close = df['close_price'].shift(1)
        tr = pd.concat([
            df['high_price'] - df['low_price'],     # high - low
            (df['high_price'] - prev_close).abs(),  # abs(high - prev close)
            (df['low_price'] - prev_close).abs(),   # abs(low - prev close)
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # computing ROC -> ROC = ((Current Price - Price 'period' bars ago) / Price 'period' bars ago)
        df['roc'] = (df['close_price'] - df['close_price'].shift(10)) / df['close_price'].shift(10) * 100

        # compiting sma50
        df['sma50'] = df['close_price'].rolling(50).mean()

        # computing vix ema -> currently using 30 min bars for tickers but only 1 per day for VIX.
        # Each stock bar needs to know most recent vix, so have to fill it in with previous days closing.
        # vix_raw is used in strategy too, so must be added to this df for optimizing individual tickers.
        vix_ema = vix_df['close_price'].ewm(span=10, min_periods=10).mean().rename('vix_ema')
        vix_raw = vix_df['close_price'].rename('vix_raw')

        vix_combined = pd.concat([vix_ema, vix_raw], axis=1)
        combined = df.join(vix_combined, how='outer').sort_index()
        combined[['vix_ema','vix_raw']] = combined[['vix_ema','vix_raw']].ffill()
        df[['vix_ema','vix_raw']] = combined[['vix_ema','vix_raw']].reindex(df.index)


        return df

    def simulate_signals(self, df, params):
        """
        Emulates OnData() in strategy.py to get ticker performance, returns a Series of
        returns. 
        """
        # get params for this instance
        ATR_BAND_MULT = params.get('ATR_BAND_MULT')
        MOMENTUM_THRESHOLD = params.get('MOMENTUM_THRESHOLD')
        BASE_CONF = params.get('BASE_CONF')
        STOP_LOSS_ATR_MULT = params.get('STOP_LOSS_ATR_MULT')
        REVERSAL_THRESHOLD = int(params.get('REVERSAL_THRESHOLD'))

        # convert df columns to numpy arrays for efficiency
        close_price = df['close_price'].to_numpy(dtype=float)
        vwap = df['vwap'].to_numpy(dtype=float)
        atr = df['atr'].to_numpy(dtype=float)
        roc = df['roc'].to_numpy(dtype=float)
        sma50 = df['sma50'].to_numpy(dtype=float)
        vix_raw = df['vix_raw'].to_numpy(dtype=float)
        vix_ema = df['vix_ema'].to_numpy(dtype=float)
        all_indicators = [close_price, vwap, atr, roc, sma50, vix_ema, vix_raw]

        # init state variables
        n = len(df)
        position = 0 # 0=flat, 1=long
        entry_price = 0.0
        entry_regime = None # high vs low
        last_signal = None
        streak_dir = None
        streak_count = 0
        returns = np.zeros(n)

        # Simulation begins, loops for each bar and immitates strategy.py's OnData()
        for i in range(1,n):
            # Skip invalid rows
            if np.isnan([arr[i] for arr in all_indicators]).any():
                continue

            # Record return for this bar based on position entering the bar
            returns[i] = (close_price[i] - close_price[i-1]) / close_price[i-1] * position

            # Regime and bands
            is_high_vol = vix_raw[i] > vix_ema[i] * 1.05
            upper = vwap[i] + ATR_BAND_MULT * atr[i]
            lower = vwap[i] - ATR_BAND_MULT * atr[i]

            signal    = "HOLD"
            is_forced = False

            # Stop loss overrides regime signal
            if position > 0 and close_price[i] <= entry_price - STOP_LOSS_ATR_MULT * atr[i]:
                signal    = "SELL"
                is_forced = True

            if not is_forced:
                if is_high_vol:
                    if close_price[i] > upper:
                        signal = "SELL"
                    elif close_price[i] < lower:
                        signal = "BUY"
                    elif position > 0 and close_price[i] >= vwap[i] and entry_regime == "high_vol":
                        signal    = "SELL"
                        is_forced = True  # mean reversion exit
                else:
                    if roc[i] > MOMENTUM_THRESHOLD and close_price[i] > sma50[i]:
                        signal = "BUY"
                    elif roc[i] < -MOMENTUM_THRESHOLD:
                        signal = "SELL"

            # Long-only: no new shorts
            if signal == "SELL" and position <= 0:
                continue

            # Update streak
            if signal == streak_dir and signal != "HOLD":
                streak_count += 1
            else:
                streak_dir   = signal
                streak_count = 1

            # Suppress direction reversals until confirmed by REVERSAL_THRESHOLD consecutive bars
            if (
                not is_forced
                and signal != "HOLD"
                and last_signal is not None
                and last_signal != "HOLD"
                and signal != last_signal
                and streak_count < REVERSAL_THRESHOLD
            ):
                continue

            # Execute
            if signal == "BUY" and position == 0:
                position     = 1
                entry_price  = close_price[i]
                entry_regime = "high_vol" if is_high_vol else "low_vol"
                last_signal  = "BUY"
            elif signal == "SELL" and position > 0:
                position     = 0
                entry_price  = 0.0
                entry_regime = None
                last_signal  = "SELL"

        return pd.Series(returns, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Optimize per-ticker parameters for Portfolio 3")
    parser.add_argument("--tickers",  nargs="+", required=True, help="One or more ticker symbols to optimize")
    parser.add_argument("--n-trials", type=int,  default=150,   help="Number of Optuna trials per ticker")
    args = parser.parse_args()

    optimizer = Portfolio3Optimizer()
    for ticker in args.tickers:
        print(f"\n{'='*50}\nOptimizing {ticker} ({args.n_trials} trials)\n{'='*50}")
        optimizer.run(ticker, n_trials=args.n_trials)


if __name__ == "__main__":
    main()