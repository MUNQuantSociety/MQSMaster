# MQS Trading System - Main Backtest Module
"""
How to run a backtest:
1. Load your portfolio through imports such as:
    `from portfolios.portfolio_n.strategy import StrategyClass`.
2. Setup main args with start date, end date, initial capital,slippage, and backtest mode (event/fast).
3. Add class to classes list comment out unused strategies for faster testing.
4. Run python -m src.main_backtest
"""

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import RLock, cpu_count
from typing import List, Optional, Type

import pandas as pd
from tqdm import tqdm

from src.backtest.backtest_engine import BacktestEngine
from src.common.database.MQSDBConnector import MQSDBConnector
from src.portfolios.portfolio_1.strategy import VolMomentum
from src.portfolios.portfolio_2.strategy import MomentumStrategy
from src.portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
from src.portfolios.portfolio_4.strategy import TrendRotateStrategy
from src.portfolios.portfolio_5.strategy import RBPStrategy
from src.portfolios.portfolio_BASE.strategy import BasePortfolio
from src.portfolios.portfolio_dummy.strategy import CrossoverRmiStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
"""
Backtest configuration parameters:
- START_DATE: The start date for the backtest in "YYYY-MM-DD" format.
- END_DATE: The end date for the backtest in "YYYY-MM-DD" format.
- INITIAL_CAPITAL: The initial capital for the backtest as a float.
- SLIPPAGE: The slippage to be applied to trade executions, expressed as a decimal (e.g., 0.000001 for 0.1 basis point).
- BACKTEST_MODE: The mode of backtesting to be used, either "event" for event-driven backtesting or "fast" for a faster approximation mode.
- DEFAULT_PORTFOLIO_CLASSES: A list of default portfolio strategy classes to be used in the backtest if no specific classes are provided.
- AVAILABLE_PORTFOLIO_CLASSES: A list of all available portfolio strategy classes that can be used in the backtest.
- BACKTEST_NUM_BATCHES: An optional integer specifying the number of batches to use for parallel backtest execution. If set to None, the batch count will be automatically determined based on the number of CPU cores and the number of portfolios.
"""
START_DATE = "2025-01-01"
END_DATE = "2025-09-05"
INITIAL_CAPITAL = 1000000.0
SLIPPAGE = 0.000001  # 0.1 basis point
BACKTEST_MODE = ""  # or "fast"
BACKTEST_NUM_BATCHES = None  # Set to an integer to override auto batch(for best results use the number of cores on your machine).
DEFAULT_PORTFOLIO_CLASSES = [
    VolMomentum,
    MomentumStrategy,
    RegimeAdaptiveStrategy,
    TrendRotateStrategy,
    CrossoverRmiStrategy,
    RBPStrategy,
]

AVAILABLE_PORTFOLIO_CLASSES = [
    VolMomentum,
    MomentumStrategy,
    RegimeAdaptiveStrategy,
    TrendRotateStrategy,
    CrossoverRmiStrategy,
    RBPStrategy,
]

# Adapters for external vectorized backtest approximations of the above strategies.
"""
quick: True/False flag to control whether the adapter should use faster approximations (e.g. fewer lookback periods, simpler logic) to speed up execution during fast-mode backtests. Adapters can choose how to interpret this flag based on the complexity of the underlying strategy and the desired tradeoff between accuracy and speed.

mc_enabled: True/False flag to control whether the adapter should prepare its output in a way that is compatible with Monte Carlo simulation. This may involve ensuring that the target_weights and signal_strength outputs are structured appropriately for resampling and path generation during the Monte Carlo process. Adapters can choose how to interpret this flag based on the specific requirements of their strategy and the intended use of the Monte Carlo results.

mc_n_sims: Integer specifying the number of Monte Carlo simulation paths that will be generated during fast-mode backtests when mc_enabled is True. Adapters can use this information to optimize their output preparation, such as by precomputing certain values or structuring their outputs in a way that facilitates efficient resampling and path generation for the specified number of simulations.

mc_method: String specifying the method of Monte Carlo simulation (e.g. "bootstrap", "parametric") that will be used during fast-mode backtests when mc_enabled is True. Adapters can use this information to tailor their output preparation to the specific requirements of the chosen Monte Carlo method, such as by ensuring that their outputs are compatible with the resampling techniques or distributional assumptions associated with the specified method.

mc_block_size: Integer specifying the block size to be used for block bootstrapping during Monte Carlo simulation when mc_enabled is True and mc_method is "bootstrap". Adapters can use this information to structure their outputs in a way that facilitates efficient block resampling during the Monte Carlo process, such as by organizing their outputs into contiguous blocks of the specified size.

mc_seed: Integer seed value for random number generation during Monte Carlo simulation when mc_enabled is True. Adapters can use this information to ensure that any stochastic elements of their output preparation are reproducible and consistent across different runs of the backtest, which can be important for debugging and result validation purposes.

mc_plot_percentiles: List of integers specifying the percentiles to be plotted during Monte Carlo simulation when mc_enabled is True. Adapters can use this information to prepare their outputs in a way that facilitates the generation of percentile plots during the Monte Carlo process, such as by structuring their outputs to allow for easy calculation and visualization of the specified percentiles.
"""

FAST_MODE_CONFIG = {
    "years_back": 3,
    "benchmark_label": "SPY",
    "quick": True,
    "mc_enabled": True,
    "mc_n_sims": 10000,
    "mc_method": "bootstrap",
    "mc_block_size": 5,  # Preserve short-term autocorrelation for block bootstrap.
    "mc_seed": None,
    "mc_plot_percentiles": [10, 50, 90],
}

def _resolve_fast_mode_config(
    fast_config=None,
    *,
    fast_years_back=None,
    fast_benchmark_label=None,
):
    """Resolve fast-mode config; when both are provided, fast_config values take precedence."""
    resolved = deepcopy(FAST_MODE_CONFIG)

    if fast_years_back is not None:
        resolved["years_back"] = int(fast_years_back)
    if fast_benchmark_label is not None:
        resolved["benchmark_label"] = str(fast_benchmark_label)

    if fast_config:
        if (
            "years_back" in fast_config
            and fast_years_back is not None
            and int(fast_config["years_back"]) != int(fast_years_back)
        ):
            warnings.warn(
                "fast_config['years_back'] overrides fast_years_back.",
                UserWarning,
                stacklevel=2,
            )
        if (
            "benchmark_label" in fast_config
            and fast_benchmark_label is not None
            and str(fast_config["benchmark_label"]) != str(fast_benchmark_label)
        ):
            warnings.warn(
                "fast_config['benchmark_label'] overrides fast_benchmark_label.",
                UserWarning,
                stacklevel=2,
            )
        resolved.update(dict(fast_config))

    return resolved


def _init_backtest(
    portfolio_classes=DEFAULT_PORTFOLIO_CLASSES,
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=INITIAL_CAPITAL,
    slippage=SLIPPAGE,
    backtest_mode=BACKTEST_MODE,
    fast_config=None,
    fast_years_back=None,
    fast_benchmark_label=None,
):

    if portfolio_classes is None or len(portfolio_classes) == 0:
        portfolio_classes: List[Type[BasePortfolio]] = list(
            AVAILABLE_PORTFOLIO_CLASSES[:2]
        )  # Default to first 2 portfolios.

    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    if backtest_mode == "fast":
        # Fast-mode tuning (single source of truth):
        # - Legacy fast_years_back / fast_benchmark_label values are still accepted.
        # - Any provided fast_config keys override defaults and legacy values.
        resolved_fast_config = _resolve_fast_mode_config(
            fast_config=fast_config,
            fast_years_back=fast_years_back,
            fast_benchmark_label=fast_benchmark_label,
        )
        return (
            portfolio_classes,
            start_date,
            end_date,
            initial_capital,
            slippage,
            backtest_mode,
            resolved_fast_config,
        )
    elif backtest_mode != "event":
        warnings.warn(
            "currently using 'event'. Set `backtest_mode` explicitly to silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
    backtest_mode = "event"
    return (
        portfolio_classes,
        start_date,
        end_date,
        initial_capital,
        slippage,
        backtest_mode,
        None,
    )

def run_backtest(
    portfolio_classes,
    start_date,
    end_date,
    initial_capital,
    slippage,
    backtest_mode,
    resolved_fast_config,
    progress_position=0,
):
    try:
        # Pin each worker's tqdm output to a dedicated terminal row.
        os.environ["TQDM_POSITION"] = str(int(progress_position))
        os.environ["TQDM_DESC"] = f"Running Backtest [{int(progress_position) + 1}]"

        dbconn = MQSDBConnector()

        backtest_engine = BacktestEngine(db_connector=dbconn, backtest_executor=None)

        backtest_engine.setup(
            portfolio_classes=portfolio_classes,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            slippage=slippage,  # 0.1 basis point
            backtest_mode=backtest_mode,
            fast_config=resolved_fast_config,
        )

        trade_log = backtest_engine.run()
        return trade_log

    except Exception as e:
        logging.error(f"Error during backtest execution: {e}", exc_info=True)
        raise


def _resolve_num_batches(
    portfolio_count: int,
    requested_num_batches: Optional[int] = None,
) -> int:
    if portfolio_count <= 0:
        return 0

    if requested_num_batches is not None:
        try:
            requested = int(requested_num_batches)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid requested_num_batches=%r; using auto batch count.",
                requested_num_batches,
            )
        else:
            if requested < 1:
                logging.warning(
                    "requested_num_batches must be >= 1 (got %s); using 1.",
                    requested,
                )
                requested = 1
            return min(portfolio_count, requested)

    num_batches = BACKTEST_NUM_BATCHES
    if num_batches is not None:
        try:
            parsed_env = int(num_batches)
            if parsed_env < 1:
                raise ValueError("must be >= 1")
            return min(portfolio_count, parsed_env)
        except ValueError:
            logging.warning(
                "Invalid BACKTEST_NUM_BATCHES=%r; using auto batch count.",
                num_batches,
            )

    cpus = cpu_count()
    if cpus:
        return min(portfolio_count, cpus)
    return min(portfolio_count, 4)

# Default to 4 batches if CPU count is unavailable.
def _partition_portfolios_evenly(
    portfolio_classes: List[Type[BasePortfolio]],
    num_batches: int,
) -> List[List[Type[BasePortfolio]]]:
    if num_batches <= 0:
        return []

    n = len(portfolio_classes)
    base_size = n // num_batches
    remainder = n % num_batches

    batches: List[List[Type[BasePortfolio]]] = []
    start = 0
    for i in range(num_batches):
        batch_size = base_size + (1 if i < remainder else 0)
        end = start + batch_size
        batches.append(portfolio_classes[start:end])
        start = end

    return batches



def main(num_batches: Optional[int] = None):
    """
    Main entry point for the MQS Trading System backtests.
    using ProcessPoolExecutor to run multiple backtests in parallel.
    """
    try:
        (
            portfolio_classes,
            start_date,
            end_date,
            initial_capital,
            slippage,
            backtest_mode,
            resolved_fast_config,
        ) = _init_backtest()

        resolved_num_batches = _resolve_num_batches(
            len(portfolio_classes),
            requested_num_batches=num_batches,
        )
        selected_portfolios = _partition_portfolios_evenly(
            portfolio_classes,
            resolved_num_batches,
        )
        futures = []
        non_empty_batches = [batch for batch in selected_portfolios if len(batch) > 0]

        if len(non_empty_batches) == 0:
            logging.warning("No non-empty portfolio batches to run.")
            return futures

        tqdm_lock = RLock()

        with ProcessPoolExecutor(
            max_workers=len(non_empty_batches),
            initializer=tqdm.set_lock,
            initargs=(tqdm_lock,),
        ) as executor:
            for i, batch_portfolios in enumerate(non_empty_batches):
                logging.info(f"Submitting backtest for portfolios: {batch_portfolios}")

                future = executor.submit(
                    run_backtest,
                    batch_portfolios,
                    start_date,
                    end_date,
                    initial_capital,
                    slippage,
                    backtest_mode,
                    resolved_fast_config,
                    i,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    trade_log = future.result()
                    logging.info("Backtest completed successfully.")
                    if len(trade_log) == 0:
                        logging.info("No trades executed in this batch.")
                        continue
                    elif trade_log is None:
                        raise ValueError("Backtest returned None instead of a trade log.")
                    for portfolio in range(len(trade_log)):
                        if trade_log[portfolio] is None or len(trade_log[portfolio]) == 0:
                            logging.info("No trades executed for portfolio")
                            continue
                        for trade in trade_log[portfolio]:
                            print(f"{trade}")
                except Exception as e:
                    logging.error(f"Backtest failed with error: {e}", exc_info=True)

    finally:
        logging.info("===== DONE =====")


if __name__ == "__main__":
    main()
