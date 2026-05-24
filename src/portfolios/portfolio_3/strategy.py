import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

try:
    from portfolios.portfolio_BASE.strategy import BasePortfolio
    from portfolios.strategy_api import StrategyContext
except ImportError as rel_err:
    logging.warning(
        "Base Portfolio and strategy_api relative import failed; using absolute import. Details: %s",
        rel_err,
    )
    try:
        from src.portfolios.portfolio_BASE.strategy import BasePortfolio
        from src.portfolios.strategy_api import StrategyContext
    except ImportError as abs_err:
        logging.error(
            "Failed to import BasePortfolio and StrategyContext from both relative and absolute paths. Details: %s",
            abs_err,
        )
        raise

class RegimeAdaptiveStrategy(BasePortfolio):
    """
    Adaptive strategy that switches between momentum and mean-reversion (VWAP/ATR fades)
    based on the VIX, using the OnData framework.

    All logic is contained within __init__, generate_signals_and_trade, and OnData.
    """

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        # --- Base Class Initialization ---
        if config_dict is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        
        # loading parameters from optimizer
        params_path = os.path.join(os.path.dirname(__file__), "ticker_params.json")
        with open(params_path) as file:
            self.ticker_params = json.load(file)
        

        super().__init__(
            db_connector, executor, debug, config_dict, backtest_start_date
        )
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.portfolio_id}"
        )

        # --- Strategy Properties ---
        self.interval_seconds = self.poll_interval
        self.last_decision_time = {}  # Cooldown timer per ticker

        # --- Order Tracking ---
        # last_signal: last EXECUTED signal per ticker ("BUY", "SELL", or "HOLD")
        self.last_signal = {}

        # raw_signal_streak: (direction, consecutive_count) of the raw signal per ticker.
        # Accumulates even on suppressed cycles so hysteresis can confirm reversals.
        self.raw_signal_streak = {}

        # order_log: rolling log (last 10) of executed orders per ticker.
        # Each entry: {"timestamp": ts, "signal": str, "price": float, "confidence": float, ...}
        self.order_log = {}

        # --- Trade History (used for history-based confidence scaling) ---
        # entry_price: most recent buy price per ticker, used for stop-loss and PnL tracking.
        self.entry_price = {}

        # Track entry regime (high vs low) alongside entry_price, prevents MR exits while in momentum regime.
        self.entry_regime = {}

        # trade_results: rolling list of (exit_price - entry_price) per ticker, capped at 5.
        # Positive = win, negative = loss. Drives history_factor in confidence formula.
        self.trade_results = {}

        # Number of consecutive same-direction bars required before flipping direction.
        self.REVERSAL_THRESHOLD = 2

        # --- Signal Parameters (tune these between runs) ---
        # ATR multiplier for VWAP fade bands. Higher = fewer but higher-conviction signals.
        # 1.0 triggers on routine intraday moves; 2.0 requires a meaningful dislocation.
        self.ATR_BAND_MULT = 1.1

        # 10-day ROC, threshold increased to 1.0, filter out noise.
        self.MOMENTUM_THRESHOLD = 1.3

        # Base trade confidence. Scales position size: 0.6 -> ~60% of one bar's allocation. 
        self.BASE_CONF = 0.65

        # Stop-loss multiplier: exit if price drops this many ATRs below the recorded entry.
        # 1.5 gives a buffer of roughly 1.5x the recent daily range before cutting the loss.
        self.STOP_LOSS_ATR_MULT = 3

        # *---------------------------------------------------
        # * 1. DEFINE YOUR INDICATORS HERE
        # *---------------------------------------------------
        indicator_definitions = {
            "vwap": (
                "VWAP",
                {"period": 20, "price_col": "close_price", "vol_col": "volume"},
            ),
            "atr": (
                "AverageTrueRange",
                {
                    "period": 14,
                    "high_col": "high_price",
                    "low_col": "low_price",
                    "close_col": "close_price",
                },
            ),
            "momentum_pct": (
                "RateOfChange",
                {"period": 10, "price_col": "close_price", "mode": "percentage"},
            ),
            "sma50": (
                "SimpleMovingAverage",
                {"period": 50, "price_col": "close_price"},
            ),
        }

        self.RegisterIndicatorSet(indicator_definitions)

        # VIX EMA for adaptive regime detection (sole determinant of is_high_vol).
        # The old is_market_open override has been removed: on daily bars (all timestamped
        # at 9:30am) it permanently matched the 9:30-10:00 window, forcing the strategy
        # into mean-reversion mode 100% of the time and making the momentum branch
        # unreachable dead code.
        # Fix 
        self.vix_ema = self.AddIndicator(
            "ExponentialMovingAverage",
            "^VIX",
            period=10
        )

        self.logger.info("RegimeAdaptiveStrategy initialized for OnData framework.")
        self.logger.info(f"Registered indicators: {list(indicator_definitions.keys())} + VIX EMA(10)"  )

    
    def generate_signals_and_trade(
        self, data: Dict[str, pd.DataFrame], current_time: Optional[datetime] = None
    ):
        """
        Overrides BasePortfolio.generate_signals_and_trade, which isn't designed
        to handle our ATR and VWAP implementations.
        """

        # Code from original
        market_data_df = data.get("MARKET_DATA")
        if market_data_df is not None and not market_data_df.empty:
            if self._last_processed_timestamp is not None:
                new_data = market_data_df[
                    market_data_df["timestamp"] > self._last_processed_timestamp
                ]
            else:
                new_data = (
                    market_data_df.sort_values("timestamp")
                    .groupby("ticker")
                    .last()
                    .reset_index()
                )
            if not new_data.empty:
                for timestamp, group in new_data.sort_values("timestamp").groupby("timestamp"):
                    for row in group.itertuples():
                        for indicator in self._indicators:
                            if indicator.ticker != row.ticker:
                                continue
                            price_col = getattr(indicator, "price_col", "close_price")
                            vol_col   = getattr(indicator, "vol_col",   "volume")
                            high_col  = getattr(indicator, "high_col",  "high_price")
                            low_col   = getattr(indicator, "low_col",   "low_price")

                            # New
                            price_val = getattr(row, price_col, None)
                            if price_val is None or not pd.notna(price_val):
                                continue

                            # Build kwargs so all OHLCV data arrives in one call (instead of per-ticker)
                            update_kwargs = {}
                            if hasattr(indicator, "vol_col"):
                                v = getattr(row, vol_col, None)
                                if v is not None and pd.notna(v):
                                    update_kwargs["volume"] = float(v)
                            if hasattr(indicator, "high_col"):
                                v = getattr(row, high_col, None)
                                if v is not None and pd.notna(v):
                                    update_kwargs[high_col] = float(v)
                            if hasattr(indicator, "low_col"):
                                v = getattr(row, low_col, None)
                                if v is not None and pd.notna(v):
                                    update_kwargs[low_col] = float(v)

                            indicator.Update(row.timestamp, float(price_val), **update_kwargs)

        # Update the last-processed timestamp.
        if current_time is not None:
            self._last_processed_timestamp = current_time
        else:
            if market_data_df is not None and not market_data_df.empty:
                try:
                    self._last_processed_timestamp = market_data_df["timestamp"].max()
                except Exception:
                    self.logger.warning(
                        "Could not update last processed timestamp from market data."
                    )

        context = StrategyContext(
            market_data_df=market_data_df,
            cash_df=data.get("CASH_EQUITY"),
            positions_df=data.get("POSITIONS"),
            port_notional_df=data.get("PORT_NOTIONAL"),
            current_time=current_time,
            executor=self.executor,
            portfolio_config=self.portfolio_config_dict,
        )

        self.OnData(context)


    """
    Replacement for context.buy() / context.sell() that fixes the executor's
    equal-weight fallback. Weight fetched would be 0.0, fallback to 1/#oftickers
    and include VIX ticker (should not). 

    Just makes sure weight calculations are correct.

    Is modular design, changes to buy/sell should be made within this method
    """
    def _execute_order(
        self,
        context: StrategyContext,
        ticker: str,
        signal_type: str,
        confidence: float,
        latest_price: float,
        trade_ts: datetime,
        n_tradeable: int,
    ):
        asset_data = context.Market[ticker]
        if not asset_data.Exists or asset_data.Close is None or asset_data.Close <= 0:
            self.logger.warning(
                "No valid market data for %s at %s, skipping...", ticker, trade_ts
            )
            return

        current_weight = context.Portfolio.get_asset_weight(ticker, latest_price)
        # Correct weight if 0.0 returned above
        ticker_weight = current_weight if current_weight != 0.0 else 1.0 / max(n_tradeable, 1)

        context._executor.execute_trade(
            portfolio_id=context._portfolio_config["id"],
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            arrival_price=latest_price,
            cash=context.Portfolio.cash,
            positions=context._positions_df,
            port_notional=context.Portfolio.total_value,
            ticker_weight=ticker_weight,
            timestamp=trade_ts,
        )

    def OnData(self, context: StrategyContext):
        """
        This method is called for each new data point (e.g., each minute bar).
        All strategy logic is contained here.
        """
        try:
            trade_ts = context.time
            positions_dict = context.Portfolio.positions
        except Exception as e:
            self.logger.error(f"Error accessing context properties: {e}")
            return

        # Get VIX data for OnData() call
        try:
            vix_asset = context.Market["^VIX"]

            # Skip if no VIX data for date/time
            if not vix_asset.Exists or vix_asset.Close is None:
                self.logger.warning(f"No VIX data found at {trade_ts}. Skipping cycle.")
                return

            vix_value = vix_asset.Close
        
        except Exception as e:
            self.logger.error(f"Error getting VIX data: {e}")
            raise
        
        # Number of tradeable tickers
        #NOTE this is defined in other variables -> need to remove this and make one variable for whole class
        n_stocks = len([t for t in self.tickers if t != "^VIX"])

        # Loop through all tickers except vix
        for ticker in self.tickers:
            if ticker == "^VIX":
                continue

            try:
                # set params, if not found in ticker_params.json then revert to hardcoded values
                p = self.ticker_params.get(ticker, {})
                ATR_BAND_MULT = p.get('ATR_BAND_MULT', self.ATR_BAND_MULT)
                MOMENTUM_THRESHOLD = p.get('MOMENTUM_THRESHOLD', self.MOMENTUM_THRESHOLD)
                BASE_CONF = p.get('BASE_CONF', self.BASE_CONF)
                STOP_LOSS_ATR_MULT = p.get('STOP_LOSS_ATR_MULT', self.STOP_LOSS_ATR_MULT)
                REVERSAL_THRESHOLD = int(p.get('REVERSAL_THRESHOLD', self.REVERSAL_THRESHOLD))



                # Check cooldown time -> skip iteration if last decision for ticker was made to recently
                last_decision = self.last_decision_time.get(ticker)
                if last_decision and (trade_ts - last_decision) < timedelta(
                    seconds=self.interval_seconds
                ):
                    continue

                # Fetch indicators for ticker
                vwap_ind = self.vwap[ticker]
                atr_ind = self.atr[ticker]
                momentum_ind = self.momentum_pct[ticker]
                sma50_ind = self.sma50[ticker]

                asset_data = context.Market[ticker]

                # Check if indicator data is ready
                indicators_to_check = [vwap_ind, atr_ind, momentum_ind, sma50_ind]
                if not all(ind.IsReady for ind in indicators_to_check):
                    continue
                if not asset_data.Exists:
                    self.logger.debug(f"No market data for {ticker} at {trade_ts}")
                    continue


                # Fetch values from indicators
                vwap_v = vwap_ind.Current
                atr_v = atr_ind.Current
                momentum_v = momentum_ind.Current
                sma50_v = sma50_ind.Current
                latest_price = asset_data.Close
                quantity = positions_dict.get(ticker, 0.0)

                all_values = [vwap_v, atr_v, momentum_v, sma50_v, latest_price, quantity]

                # Skip iteration if missing ticker value(s)
                if any(v is None for v in all_values):
                    self.logger.debug(f"Skipping {ticker} due to None values.")
                    continue

                # Determine volatility regime using vix, removed is_market_open check.
                # Vix must be >5% above its EMA before switching to high_vol. Increase
                # value for more stability since vix ema reduced to 10-day.
                if self.vix_ema.IsReady:
                    is_high_vol = vix_value > self.vix_ema.Current * 1.05
                else:
                    is_high_vol = vix_value > 20

                # Upper and lower bands WITH multiplier param
                upper_band = vwap_v + atr_v * ATR_BAND_MULT
                lower_band = vwap_v - atr_v * ATR_BAND_MULT

                # Initialize regime bools
                signal = "HOLD"
                is_mean_reversion_exit = False
                is_stop_loss_exit = False

                # Stop loss -> force sell regardless of regime signal.
                # Checked first so it overrides any BUY signal that would average into a loss.
                if quantity > 0 and ticker in self.entry_price:
                    entry = self.entry_price[ticker]
                    if latest_price <= entry - STOP_LOSS_ATR_MULT * atr_v:
                        signal = "SELL"
                        is_stop_loss_exit = True
                        self.logger.debug(
                            f"[{ticker}] Stop-loss triggered: price {latest_price:.2f} "
                            f"<= entry {entry:.2f} - {STOP_LOSS_ATR_MULT}*ATR({atr_v:.2f})"
                        )

                if not is_stop_loss_exit:
                    if is_high_vol:
                        # Regime: High Volatility -> Fade (Mean Reversion)
                        if latest_price > upper_band:
                            # Price significantly above VWAP -> sell
                            signal = "SELL"
                        elif latest_price < lower_band:
                            # Price significantly below VWAP -> buy
                            signal = "BUY"
                        elif (
                            quantity > 0
                            and latest_price >= vwap_v
                            and self.entry_regime.get(ticker) == "high_vol"
                        ):
                            # Price has reverted to VWAP; only exit if the position was during current
                            # high vol regime to avoid prematurely closing momentum trades.
                            signal = "SELL"
                            is_mean_reversion_exit = True

                    else:
                        # Regime: Low Volatility -> Momentum
                        # Require price above SMA(50) to confirm the trend before buying.
                        if momentum_v > MOMENTUM_THRESHOLD and latest_price > sma50_v:
                            signal = "BUY"
                        elif momentum_v < -MOMENTUM_THRESHOLD:
                            signal = "SELL"

                # Fix issue of sell creating shorts -> sell signals only
                # close existing longs.
                if signal == "SELL" and quantity <= 0:
                    continue

                ### Buy/Sell history/weights begin here ###

                # Update raw signal streak
                prev_dir, prev_count = self.raw_signal_streak.get(ticker, (None, 0))
                streak_count = (prev_count + 1) if signal == prev_dir else 1
                self.raw_signal_streak[ticker] = (signal, streak_count)

                # Suppresses signal if checks aren't met (if a forced exit or outside of threshold
                # or signal hasn't changed or signal is a hold)
                is_forced_exit = is_mean_reversion_exit or is_stop_loss_exit
                last_exec = self.last_signal.get(ticker)
                if (
                    not is_forced_exit
                    and signal != "HOLD"
                    and last_exec is not None
                    and last_exec != "HOLD"
                    and signal != last_exec
                    and streak_count < REVERSAL_THRESHOLD
                ):
                    self.logger.debug(
                        f"[{ticker}] Reversal {last_exec}->{signal} suppressed "
                        f"(streak={streak_count}/{REVERSAL_THRESHOLD})"
                    )
                    continue

           
                # Prevents re-buying/re-selling when at or near target weight.
                # Target weight comes from config.json 
                target_weight = (
                    (self.portfolio_weights or {}).get(ticker)
                    or 1.0 / max(n_stocks, 1)
                )
                current_weight = context.Portfolio.get_asset_weight(ticker, latest_price)
                if signal == "BUY" and current_weight >= target_weight * 0.9:
                    self.logger.debug(
                        f"[{ticker}] BUY suppressed: weight {current_weight:.3f} "
                        f">= {target_weight * 0.9:.3f} (already at/near target long)"
                    )
                    continue

                if signal == "SELL" and current_weight <= -(target_weight * 0.9):
                    self.logger.debug(
                        f"[{ticker}] SELL suppressed: weight {current_weight:.3f} "
                        f"<= {-(target_weight * 0.9):.3f} (already at/near target short)"
                    )
                    continue

                
                # Compute is_reversal once so it is available in both confidence and logging.
                # Confidence calculation:
                #   - closes long if forced exit by MR or stoploss
                #   - Adds streak bonus to base confidence
                #   - adds history_factor based on success of recent 2-3 trades (can be adjusted below -> 'lookback')
                #   - add strength_factor based on either price vs vwap or momentum depending on regime
                #   - adds reversal_factor based on reversal checks below
                is_reversal = (
                    not is_forced_exit
                    and last_exec is not None
                    and last_exec != "HOLD"
                    and signal != "HOLD"
                    and signal != last_exec
                )

                if is_forced_exit:
                    confidence = 0.5
                else:
                    streak_bonus = min(0.05 * (streak_count - 1), 0.2)
                    reversal_factor = 0.7 if is_reversal else 1.0

                    if is_high_vol:
                        ref_band = upper_band if signal == "SELL" else lower_band
                        deviation = abs(latest_price - ref_band) / (atr_v + 1e-9)
                        strength_factor = min(1.0 + deviation * 0.5, 1.5)
                    else:
                        strength_factor = min(
                            1.0 + abs(momentum_v) / MOMENTUM_THRESHOLD * 0.5, 1.5
                        )

                    # History factor
                    recent_results = self.trade_results.get(ticker, [])
                    if len(recent_results) >= 2:
                        lookback = recent_results[-3:] #ADJUST THIS if want further lookback
                        win_rate = sum(1 for p in lookback if p > 0) / len(lookback)
                        history_factor = 0.7 + 0.3 * win_rate
                    else:
                        history_factor = 1.0

                    # Combine weights, take min of 1.0 and calculated conf to prevent >100% conf
                    confidence = round(
                        min(
                            1.0,
                            max(
                                0.05,
                                (BASE_CONF + streak_bonus)
                                * reversal_factor
                                * strength_factor
                                * history_factor,
                            ),
                        ),
                        4,
                    )

                # --- 3e. Execute Trade (if signal) ---
                if signal != "HOLD":
                    self.logger.debug(
                        f"[{ticker}] Executing {signal} | "
                        f"mean_rev_exit={is_mean_reversion_exit} | "
                        f"stop_loss_exit={is_stop_loss_exit} | "
                        f"streak={streak_count} | "
                        f"reversal={is_reversal} | "
                        f"confidence={confidence:.4f}"
                    )

                    if signal == "BUY":
                        self._execute_order(
                            context, ticker, "BUY", confidence, latest_price, trade_ts, n_stocks
                        )
                        # # Record entry price and entry regime for stop-loss and win/loss tracking.
                        self.entry_price[ticker] = latest_price
                        self.entry_regime[ticker] = "high_vol" if is_high_vol else "low_vol"

                    elif signal == "SELL":
                        self._execute_order(
                            context, ticker, "SELL", confidence, latest_price, trade_ts, n_stocks
                        )
                        # Record completed trade result for history_factor scaling.
                        if ticker in self.entry_price:
                            pnl = latest_price - self.entry_price[ticker]
                            if ticker not in self.trade_results:
                                self.trade_results[ticker] = []
                            self.trade_results[ticker].append(pnl)
                            if len(self.trade_results[ticker]) > 5:
                                self.trade_results[ticker] = self.trade_results[ticker][-5:]
                            del self.entry_price[ticker]
                        # remove stored entry regime when sold
                        self.entry_regime.pop(ticker, None)

                    self.last_decision_time[ticker] = trade_ts
                    self.last_signal[ticker] = signal

                    if ticker not in self.order_log:
                        self.order_log[ticker] = []
                    self.order_log[ticker].append({
                        "timestamp": trade_ts,
                        "signal": signal,
                        "price": latest_price,
                        "confidence": confidence,
                        "mean_reversion_exit": is_mean_reversion_exit,
                        "stop_loss_exit": is_stop_loss_exit,
                    })
                    if len(self.order_log[ticker]) > 10:
                        self.order_log[ticker] = self.order_log[ticker][-10:]

            except Exception as e:
                self.logger.error(f"[{ticker}] Error during OnData decision loop: {e}")