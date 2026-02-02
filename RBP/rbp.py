"""
Relevance-Based Prediction (RBP) and Relevance-Based Importance (RBI) Implementation

This module implements the RBP prediction methodology and RBI variable importance framework
as described in the research paper. The core idea is to weight past observations by their
relevance to the current prediction task, where relevance combines similarity and informativeness.

Key Components:
- MarketDataFetcher: Retrieves historical market data from Financial Modeling Prep API
- FeatureEngineer: Transforms raw market data into predictive features
- MahalanobisDistanceCalculator: Computes statistical distances between observations
- RelevanceCalculator: Determines how relevant past observations are to current predictions
- PredictionWeightCalculator: Converts relevance scores into observation weights
- RBPPredictor: Main prediction engine that generates forecasts
- RBICalculator: Computes variable importance scores for interpretability
"""

import itertools
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import dotenv
import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed

dotenv.load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")


class MarketDataFetcher:
    """
    Fetches historical market data from Financial Modeling Prep (FMP) API.

    The fetcher handles API rate limits by breaking requests into yearly chunks
    and includes automatic retry logic for failed requests.
    """

    def __init__(self, api_key: Optional[str] = FMP_API_KEY):
        """
        Initialize the fetcher with the FMP API key.

        Args:
            api_key: Financial Modeling Prep API key
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3/historical-price-full"

    def fetch_data(
        self,
        ticker_symbols: List[str],
        lookback_days: int,
        delay_between_requests: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch daily end-of-day historical market data for specified tickers.

        This method loops through years to bypass API limitations on historical data range.
        Each year is fetched separately with a delay between requests to respect rate limits.

        Args:
            ticker_symbols: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            lookback_days: Number of days of historical data to retrieve
            delay_between_requests: Seconds to wait between API calls (default 0.5)

        Returns:
            DataFrame with columns: timestamp, ticker, open_price, high_price,
                                   low_price, close_price, volume, etc.
        """
        # --- 1. Prepare Parameters ---
        tickers_list = ticker_symbols

        if not tickers_list:
            logging.warning("No tickers provided. Returning empty DataFrame.")
            return pd.DataFrame()

        tickers_str = ",".join(tickers_list)

        # Calculate start and end years for the loop
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        start_year = start_date.year
        end_year = end_date.year  # This will be the current year

        all_dfs = []  # List to hold DataFrame for each year

        logging.info(
            "Starting yearly data fetch from %s to %s for %s...",
            start_year,
            end_year,
            tickers_str,
        )

        # --- 2. Loop Through Each Year ---
        for year in range(start_year, end_year + 1):
            # Define the date range for this specific year
            from_date_loop = f"{year}-01-01"
            to_date_loop = f"{year}-12-31"

            # Override 'to_date' if we are in the current year
            if year == end_year:
                to_date_loop = end_date.date().isoformat()

            logging.info("--- Fetching data for year: %s ---", year)

            raw_year = self._fetch_single_year(
                tickers_str, from_date_loop, to_date_loop
            )
            if not raw_year.empty:
                all_dfs.append(raw_year)
            else:
                logging.warning("No data returned from API for %s.", year)

            # Wait between calls to avoid API rate limits
            time.sleep(delay_between_requests)

        # --- 3. Combine All DataFrames ---
        if not all_dfs:
            logging.warning("No data fetched for any year. Returning empty DataFrame.")
            return pd.DataFrame()

        raw_df = pd.concat(all_dfs, ignore_index=True)

        # --- 4. Convert to DataFrame and Clean (Run once on the final DF) ---
        df = self._process_raw_data(raw_df)

        logging.info(
            "Successfully processed %s TOTAL data points from all years.", len(df)
        )
        return df

    def _fetch_single_year(
        self, tickers_string: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """
        Fetch data for a single year from the FMP API.

        Args:
            tickers_string: Comma-separated ticker symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with raw API response data for the specified period
        """
        url = f"{self.base_url}/{ tickers_string}"
        params = {"from": from_date, "to": to_date, "apikey": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises exception for 4xx/5xx status codes
            data = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {from_date} to {to_date}: {e}")
            return pd.DataFrame()

        historical_records = []

        # Handle single ticker response format
        if isinstance(data, dict) and "historical" in data:
            logging.info("Processing single ticker response")
            for record in data["historical"]:
                record["symbol"] = tickers_string
                historical_records.append(record)

        # Handle multi-ticker response format
        elif isinstance(data, dict) and "historicalStockList" in data:
            logging.info("Processing multi-ticker response")
            for stock_data in data["historicalStockList"]:
                ticker_symbol = stock_data["symbol"]
                for record in stock_data["historical"]:
                    record["symbol"] = ticker_symbol
                    historical_records.append(record)
        else:
            logging.warning("No valid data found in API response.")

        if historical_records:
            logging.info(f"Parsed {len(historical_records)} data points.")
            return pd.DataFrame(historical_records)

        return pd.DataFrame()

    def _process_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean raw API data into standardized format.

        Standardizes column names, converts data types, handles missing values,
        and sorts by time and ticker.

        Args:
            raw_data: Raw DataFrame from API responses

        Returns:
            Cleaned and standardized DataFrame
        """
        # Standardize column names to match internal conventions
        raw_data.rename(
            columns={
                "date": "timestamp",
                "close": "close_price",
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "symbol": "ticker",
            },
            inplace=True,
        )

        # Convert timestamp to datetime
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])

        # Convert numeric columns to proper types, coercing errors to NaN
        numeric_columns = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "adjClose",
            "volume",
            "unadjustedVolume",
            "change",
            "changePercent",
            "vwap",
        ]
        for column in numeric_columns:
            if column in raw_data.columns:
                raw_data[column] = pd.to_numeric(raw_data[column], errors="coerce")

        # Remove rows with missing critical data
        raw_data.dropna(subset=["timestamp", "ticker", "close_price"], inplace=True)

        # Sort chronologically by ticker
        raw_data.sort_values(["timestamp", "ticker"], inplace=True)

        return raw_data


class FeatureEngineer:
    """
    Transforms raw market data into predictive features and target variables.

    Creates momentum and volatility features from historical prices, along with
    forward-looking target returns for supervised learning.
    """

    def engineer_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer predictive features (X) and target variable (Y) from market data.

        Predictive Features (X):
        - past_return_21d: 1-month (21 trading days) past return
        - past_vol_21d: 1-month past volatility (standard deviation of daily returns)
        - past_return_63d: 3-month (63 trading days) past return
        - past_vol_63d: 3-month past volatility
        - past_return_252d: 1-year (252 trading days) past return

        Target Variable (Y):
        - target_return_21d: 1-month forward-looking return

        Args:
            market_data: DataFrame with timestamp, ticker, and price columns

        Returns:
            DataFrame with original data plus engineered features and target
        """
        logging.info("Starting feature engineering...")

        # Work on a copy to avoid modifying original data
        data = market_data.copy()

        # Ensure proper chronological ordering for time-series operations
        data.sort_values(["ticker", "timestamp"], inplace=True)

        # Calculate daily returns as foundation for volatility measures
        daily_returns = data.groupby("ticker")["close_price"].pct_change()

        # Group by ticker to apply operations within each stock independently
        grouped_by_ticker = data.groupby("ticker")

        # --- Momentum Features (Past Returns) ---
        # pct_change(N) calculates the percentage change from N periods ago
        data["past_return_21d"] = grouped_by_ticker["close_price"].pct_change(21)
        data["past_return_63d"] = grouped_by_ticker["close_price"].pct_change(63)
        data["past_return_252d"] = grouped_by_ticker["close_price"].pct_change(252)

        # --- Volatility Features (Past Standard Deviation of Returns) ---
        # Using transform ensures the rolling calculation aligns with original index
        data["past_vol_21d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(21).std()
        )

        data["past_vol_63d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(63).std()
        )

        # --- Target Variable (Forward-Looking Return) ---
        # shift(-21) pulls future values back to current row
        # This is what we're trying to predict
        data["target_return_21d"] = (
            grouped_by_ticker["close_price"].shift(-21) / data["close_price"] - 1
        )

        # Remove rows with NaN values created by rolling windows or forward shifts
        data.dropna(inplace=True)

        logging.info(
            f"Feature engineering complete. {len(data)} rows remain after NaN removal."
        )

        return data


class MahalanobisDistanceCalculator:
    """
    Computes Mahalanobis distances between observations.

    The Mahalanobis distance is a measure of the distance between a point and a distribution,
    accounting for correlations between variables. It's computed as:
    d²(x, y) = (x - y)ᵀ Σ⁻¹ (x - y)
    where Σ⁻¹ is the inverse covariance matrix.
    """

    def __init__(self, training_data: pd.DataFrame):
        """
        Initialize calculator with training data statistics.

        Args:
            training_data: DataFrame of training observations (features only)
        """
        self.feature_columns = training_data.columns.tolist()
        self.mean_vector = training_data.mean().values  # Shape: (K,)
        self.inverse_covariance_matrix = self._compute_inverse_covariance(training_data)

        logging.info(
            f"Initialized Mahalanobis calculator with {len(self.feature_columns)} features"
        )

    def _compute_inverse_covariance(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute inverse covariance matrix with regularization for stability.

        The inverse covariance (precision matrix) is used in Mahalanobis distance.
        If the covariance matrix is singular (non-invertible), we add a small
        amount of ridge regularization to stabilize it.

        Args:
            data: DataFrame of observations

        Returns:
            Inverse covariance matrix of shape (K, K)
        """
        covariance_matrix = data.cov().values

        try:
            inverse_cov = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            logging.warning(
                "Covariance matrix is singular. Adding ridge regularization."
            )
            # Add small ridge term (1e-6 * Identity) to diagonal for stability
            regularized_cov = (
                covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-6
            )
            inverse_cov = np.linalg.inv(regularized_cov)

        return inverse_cov

    def calculate_distance(
        self, observation_1: np.ndarray, observation_2: np.ndarray
    ) -> float:
        """
        Calculate squared Mahalanobis distance between two observations.

        Formula: d²(x₁, x₂) = (x₁ - x₂)ᵀ Σ⁻¹ (x₁ - x₂)

        Args:
            observation_1: First observation vector of shape (K,)
            observation_2: Second observation vector of shape (K,)

        Returns:
            Squared Mahalanobis distance (scalar)
        """
        difference = observation_1 - observation_2  # Shape: (K,)

        # Matrix multiplication: (1,K) @ (K,K) @ (K,1) -> scalar
        # This is the quadratic form: xᵀ A x
        distance_squared = difference @ self.inverse_covariance_matrix @ difference

        return float(distance_squared)


class RelevanceCalculator:
    """
    Calculates relevance scores for past observations relative to prediction tasks.

    Relevance combines two concepts:
    1. Similarity: How similar is the past observation to the current task?
    2. Informativeness: How informative/unusual is the observation compared to average?

    Formula: r(i,t) = -0.5 * sim(xᵢ, xₜ) + 0.5 * [info(xᵢ) + info(xₜ)]
    """

    def __init__(self, distance_calculator: MahalanobisDistanceCalculator):
        """
        Initialize relevance calculator.

        Args:
            distance_calculator: Initialized Mahalanobis distance calculator
        """
        self.distance_calculator = distance_calculator
        self.mean_vector = distance_calculator.mean_vector
        self.inverse_cov_matrix = distance_calculator.inverse_covariance_matrix

    def calculate_relevance(
        self, past_observation: np.ndarray, current_task: np.ndarray
    ) -> float:
        """
        Calculate relevance of a past observation to a current prediction task.

        Relevance balances similarity (closer is better) with informativeness
        (more extreme/unusual observations are more valuable).

        Args:
            past_observation: Historical observation vector (K,)
            current_task: Current prediction task vector (K,)

        Returns:
            Relevance score (higher = more relevant)
        """
        # Similarity: Distance between past observation and current task
        # Lower distance = higher similarity = higher relevance
        similarity_component = self.distance_calculator.calculate_distance(
            past_observation, current_task
        )

        # Informativeness of past observation: Distance from mean
        # More extreme observations are more informative
        informativeness_past = self.distance_calculator.calculate_distance(
            past_observation, self.mean_vector
        )

        # Informativeness of current task: Distance from mean
        # More unusual prediction tasks benefit from more data
        informativeness_current = self.distance_calculator.calculate_distance(
            current_task, self.mean_vector
        )

        # Combined relevance formula from research paper
        relevance = -0.5 * similarity_component + 0.5 * (
            informativeness_past + informativeness_current
        )

        return relevance

    def calculate_relevance_for_all_past_observations(
        self, current_task: np.ndarray, past_observations: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate relevance scores for all past observations against one task.

        Args:
            current_task: Current prediction task vector (K,)
            past_observations: DataFrame of all past observations (N, K)

        Returns:
            Series of relevance scores indexed like past_observations
        """
        # Use apply with axis=1 to iterate over rows
        # .values converts each row to numpy array for efficiency
        relevance_scores = past_observations.apply(
            lambda past_obs: self.calculate_relevance(past_obs.values, current_task),
            axis=1,
        )

        return relevance_scores


class PredictionWeightCalculator:
    """
    Converts relevance scores into observation weights for predictions.

    Supports two weighting schemes:
    1. Linear: All observations weighted proportional to relevance
    2. Censored: Only most relevant observations used, with adjusted weights
    """

    def calculate_weights(
        self, relevance_scores: pd.Series, censoring_quantile: float = 0.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate observation weights from relevance scores.

        If censoring_quantile = 0.0, uses linear weighting (Equation 6):
            w(i,t) = 1/N + (1/(N-1)) * r(i,t)

        If censoring_quantile > 0.0, uses censored weighting (Equations 7-9):
            Only observations above the quantile threshold are retained,
            with weights adjusted by scaling factor λ².

        Args:
            relevance_scores: Series of relevance scores for all observations
            censoring_quantile: Quantile threshold for censoring (0.0 = no censoring,
                               0.2 = keep top 80%, 0.5 = keep top 50%, etc.)

        Returns:
            Tuple of (weights, retained_mask)
            - weights: Series of observation weights
            - retained_mask: Boolean series indicating which observations are retained
        """
        num_observations = len(relevance_scores)

        # Handle edge case of insufficient data
        if num_observations < 2:
            logging.warning("Not enough observations to calculate weights.")
            return (
                pd.Series(np.nan, index=relevance_scores.index),
                pd.Series(False, index=relevance_scores.index),
            )

        # --- Linear Regression Case (No Censoring) ---
        if censoring_quantile == 0.0:
            weights = (1 / num_observations) + (
                1 / (num_observations - 1)
            ) * relevance_scores
            retained_mask = pd.Series(True, index=relevance_scores.index)
            return weights, retained_mask

        # --- Censored Case ---

        # Determine relevance threshold value
        relevance_threshold = relevance_scores.quantile(censoring_quantile)

        # Identify retained observations (above threshold)
        retained_mask = relevance_scores >= relevance_threshold
        num_retained = retained_mask.sum()

        # Handle edge case where too few observations retained
        if num_retained < 2:
            logging.warning(
                f"Censoring quantile {censoring_quantile} resulted in < 2 retained obs. "
                f"Reverting to linear weights."
            )
            weights = (1 / num_observations) + (
                1 / (num_observations - 1)
            ) * relevance_scores
            retained_mask = pd.Series(True, index=relevance_scores.index)
            return weights, retained_mask

        # Calculate censored weight components
        retention_rate = num_retained / num_observations  # φ in paper
        retained_scores = relevance_scores[retained_mask]
        mean_retained_relevance = retained_scores.mean()

        # Calculate scaling factor λ² (Equation 9)
        # This preserves the variance structure after censoring
        variance_full = (relevance_scores**2).sum() / (num_observations - 1)
        variance_retained = (retained_scores**2).sum() / (num_retained - 1)

        if variance_retained == 0:
            lambda_squared = 1.0
        else:
            lambda_squared = variance_full / variance_retained

        # Calculate censored weights (Equation 7)
        # δ(i,t) * r(i,t) is just the score where retained, 0 otherwise
        delta_times_relevance = relevance_scores.where(retained_mask, 0.0)

        weights = 1 / num_observations + (lambda_squared / (num_retained - 1)) * (
            delta_times_relevance - retention_rate * mean_retained_relevance
        )

        return weights, retained_mask


class FitCalculator:
    """
    Calculates reliability metrics for predictions.

    Fit measures how well relevance weights correlate with outcomes,
    indicating prediction reliability.
    """

    def calculate_fit(self, weights: pd.Series, outcomes: pd.Series) -> float:
        """
        Calculate fit as squared correlation between weights and outcomes.

        Fit (ρ²) measures how well the relevance-based weights correlate with
        actual outcomes, serving as a reliability metric.

        Formula: ρ² = [corr(weights, outcomes)]²

        Args:
            weights: Series of observation weights
            outcomes: Series of past outcomes (target values)

        Returns:
            Fit score (0 to 1, higher = more reliable)
        """
        # Align indices to ensure matching observations
        weights_aligned, outcomes_aligned = weights.align(outcomes)

        # Handle cases where correlation cannot be computed
        if weights_aligned.std() == 0 or outcomes_aligned.std() == 0:
            return 0.0

        # Calculate Pearson correlation coefficient
        correlation_matrix = np.corrcoef(weights_aligned, outcomes_aligned)
        correlation = correlation_matrix[0, 1]

        if np.isnan(correlation):
            return 0.0

        # Return squared correlation (R²)
        return correlation**2

    def calculate_asymmetry(
        self, weights: pd.Series, outcomes: pd.Series, retained_mask: pd.Series
    ) -> float:
        """
        Calculate asymmetry between retained and censored subsamples.

        Asymmetry measures the difference in weight-outcome correlation between
        retained (high relevance) and censored (low relevance) observations.
        Higher asymmetry indicates better discrimination.

        Formula: asymmetry = 0.5 * (ρ₊ - ρ₋)²
        where ρ₊ = correlation for retained, ρ₋ = correlation for censored

        Args:
            weights: Series of observation weights
            outcomes: Series of past outcomes
            retained_mask: Boolean series indicating retained observations

        Returns:
            Asymmetry score (≥ 0)
        """
        # Align all series to same index
        weights_aligned, outcomes_aligned = weights.align(outcomes)
        weights_aligned, mask_aligned = weights_aligned.align(retained_mask)
        outcomes_aligned, mask_aligned = outcomes_aligned.align(mask_aligned)

        # Split into retained and censored subsamples
        weights_retained = weights_aligned[mask_aligned]
        outcomes_retained = outcomes_aligned[mask_aligned]

        weights_censored = weights_aligned[~mask_aligned]
        outcomes_censored = outcomes_aligned[~mask_aligned]

        # Calculate correlation for retained subsample
        correlation_retained = 0.0
        if (
            len(weights_retained) >= 2
            and weights_retained.std() > 0
            and outcomes_retained.std() > 0
        ):
            corr_matrix = np.corrcoef(weights_retained, outcomes_retained)
            correlation_retained = corr_matrix[0, 1]
            if np.isnan(correlation_retained):
                correlation_retained = 0.0

        # Calculate correlation for censored subsample
        correlation_censored = 0.0
        if (
            len(weights_censored) >= 2
            and weights_censored.std() > 0
            and outcomes_censored.std() > 0
        ):
            corr_matrix = np.corrcoef(weights_censored, outcomes_censored)
            correlation_censored = corr_matrix[0, 1]
            if np.isnan(correlation_censored):
                correlation_censored = 0.0

        # Asymmetry formula from paper (Equation 13)
        asymmetry = 0.5 * (correlation_retained - correlation_censored) ** 2

        return asymmetry

    def calculate_adjusted_fit(
        self, fit: float, asymmetry: float, num_features: int
    ) -> float:
        """
        Calculate adjusted fit score incorporating asymmetry.

        Adjusted fit = K * (fit + asymmetry)
        where K is the number of features used in the prediction.

        Args:
            fit: Base fit score (ρ²)
            asymmetry: Asymmetry score
            num_features: Number of features used (K)

        Returns:
            Adjusted fit score
        """
        return num_features * (fit + asymmetry)


class GridCell:
    """
    Represents a single cell in the RBP prediction grid.

    Each cell corresponds to a specific combination of:
    - Variable subset (which features to use)
    - Relevance threshold (how much censoring to apply)
    """

    def __init__(self, feature_names: Tuple[str, ...], censoring_quantile: float):
        """
        Initialize a grid cell.

        Args:
            feature_names: Tuple of feature names used in this cell
            censoring_quantile: Relevance threshold for this cell
        """
        self.feature_names = feature_names
        self.censoring_quantile = censoring_quantile
        self.num_features = len(feature_names)
        self.prediction = None
        self.adjusted_fit = None

    def __repr__(self) -> str:
        return (
            f"GridCell(features={self.feature_names}, "
            f"threshold={self.censoring_quantile}, "
            f"prediction={self.prediction:.4f if self.prediction else 'None'}, "
            f"adj_fit={self.adjusted_fit:.4f if self.adjusted_fit else 'None'})"
        )


class RBPPredictor:
    """
    Main Relevance-Based Prediction engine.

    Generates predictions by creating a grid of models with different feature
    combinations and relevance thresholds, then combining them based on reliability.
    """

    def __init__(
        self,
        feature_columns: List[str],
        censoring_quantiles: List[float] = [0.0, 0.2, 0.5, 0.8],
    ):
        """
        Initialize RBP predictor.

        Args:
            feature_columns: List of feature names available for prediction
            censoring_quantiles: List of relevance thresholds to try
        """
        self.feature_columns = feature_columns
        self.censoring_quantiles = censoring_quantiles
        self.num_features = len(feature_columns)

        # Generate all non-empty feature combinations (2^K - 1)
        self.feature_combinations = self._generate_feature_combinations()

        logging.info(
            f"Initialized RBP predictor with {len(self.feature_combinations)} "
            f"feature combinations and {len(censoring_quantiles)} thresholds"
        )

    def _generate_feature_combinations(self) -> List[Tuple[str, ...]]:
        """
        Generate all possible non-empty subsets of features.

        Returns:
            List of tuples, each containing a combination of feature names
        """
        all_combinations = []
        for subset_size in range(1, self.num_features + 1):
            for combination in itertools.combinations(
                self.feature_columns, subset_size
            ):
                all_combinations.append(combination)
        return all_combinations

    def predict_single_task(
        self,
        current_task: pd.Series,
        training_features: pd.DataFrame,
        training_outcomes: pd.Series,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Generate prediction for a single task using full RBP grid.

        Process:
        1. For each (feature_subset, threshold) combination:
           - Calculate relevance using only those features
           - Compute weights and make prediction
           - Calculate reliability (adjusted fit)
        2. Combine all grid predictions weighted by their reliability

        Args:
            current_task: Series containing current prediction task features
            training_features: DataFrame of past observations (N, K)
            training_outcomes: Series of past outcomes (N,)

        Returns:
            Tuple of (final_prediction, grid_results_dataframe)
        """
        grid_results = []

        # Iterate through all grid cells (theta combinations)
        for feature_combination in self.feature_combinations:
            feature_list = list(feature_combination)
            num_features_in_cell = len(feature_list)

            # Extract feature subset for this cell
            training_subset = training_features[feature_list]
            task_subset = current_task[feature_list].values

            # Initialize components for this feature subset
            distance_calc = MahalanobisDistanceCalculator(training_subset)
            relevance_calc = RelevanceCalculator(distance_calc)
            weight_calc = PredictionWeightCalculator()
            fit_calc = FitCalculator()

            # Calculate relevance using this feature subset
            relevance_scores = (
                relevance_calc.calculate_relevance_for_all_past_observations(
                    task_subset, training_subset
                )
            )

            # Try each censoring threshold for this feature subset
            for quantile in self.censoring_quantiles:
                # Create grid cell
                cell = GridCell(feature_combination, quantile)

                # Calculate weights for this cell
                weights, retained_mask = weight_calc.calculate_weights(
                    relevance_scores, quantile
                )

                # Generate cell prediction (weighted average of outcomes)
                # Align weights and outcomes before computing weighted sum
                aligned_outcomes, aligned_weights = training_outcomes.align(
                    weights, join="inner"
                )
                cell.prediction = (aligned_weights * aligned_outcomes).sum()

                # Calculate cell reliability
                fit = fit_calc.calculate_fit(weights, training_outcomes)
                asymmetry = fit_calc.calculate_asymmetry(
                    weights, training_outcomes, retained_mask
                )
                cell.adjusted_fit = fit_calc.calculate_adjusted_fit(
                    fit, asymmetry, num_features_in_cell
                )

                # Store cell results
                grid_results.append(
                    {
                        "features": cell.feature_names,
                        "censoring_quantile": cell.censoring_quantile,
                        "num_features": cell.num_features,
                        "prediction": cell.prediction,
                        "adjusted_fit": cell.adjusted_fit,
                    }
                )

        # Convert to DataFrame
        grid_df = pd.DataFrame(grid_results)

        # Combine predictions weighted by reliability
        final_prediction = self._combine_grid_predictions(grid_df)

        return final_prediction, grid_df

    def _combine_grid_predictions(self, grid_results: pd.DataFrame) -> float:
        """
        Combine grid predictions weighted by adjusted fit (reliability).

        Formula: ŷ = Σ(ψ_θ * ŷ_θ)
        where ψ_θ = adjusted_fit_θ / Σ(adjusted_fit)

        Args:
            grid_results: DataFrame with columns: prediction, adjusted_fit

        Returns:
            Final composite prediction
        """
        # Clip adjusted fits at 0 to ensure non-negative weights
        adjusted_fits = grid_results["adjusted_fit"].clip(lower=0)

        sum_adjusted_fits = adjusted_fits.sum()

        # Handle case where all fits are zero
        if sum_adjusted_fits == 0:
            logging.warning("All adjusted fits are zero. Prediction unreliable.")
            return 0.0

        # Calculate reliability weights (ψ_θ)
        reliability_weights = adjusted_fits / sum_adjusted_fits

        # Weighted average prediction
        final_prediction = (reliability_weights * grid_results["prediction"]).sum()

        return final_prediction


class RBICalculator:
    """
    Calculates Relevance-Based Importance (RBI) scores for variables.

    RBI measures how much each variable contributes to prediction reliability
    by comparing adjusted fit with vs. without that variable.
    """

    def calculate_rbi_for_task(
        self, grid_results: pd.DataFrame, all_feature_names: List[str]
    ) -> pd.Series:
        """
        Calculate RBI score for each variable for a single prediction task.

        RBI for variable k = average adjusted_fit when k is included -
                            average adjusted_fit when k is excluded

        Higher RBI indicates the variable contributes more to reliability.

        Args:
            grid_results: DataFrame from predict_single_task with columns:
                         features, adjusted_fit
            all_feature_names: List of all possible feature names

        Returns:
            Series of RBI scores indexed by feature name
        """
        rbi_scores = {}

        adjusted_fit_series = grid_results["adjusted_fit"]

        for feature_name in all_feature_names:
            # Identify grid cells that include this feature
            includes_feature_mask = grid_results["features"].apply(
                lambda feature_tuple: feature_name in feature_tuple
            )

            # Average adjusted fit when feature is included
            fits_with_feature = adjusted_fit_series[includes_feature_mask]
            avg_fit_with = fits_with_feature.mean()
            if pd.isna(avg_fit_with):
                avg_fit_with = 0.0

            # Average adjusted fit when feature is excluded
            fits_without_feature = adjusted_fit_series[~includes_feature_mask]
            avg_fit_without = fits_without_feature.mean()
            if pd.isna(avg_fit_without):
                avg_fit_without = 0.0

            # RBI = marginal contribution to reliability
            rbi_scores[feature_name] = avg_fit_with - avg_fit_without

        return pd.Series(rbi_scores)


class RBPPipeline:
    """
    End-to-end pipeline for RBP prediction and RBI analysis.

    Orchestrates data fetching, feature engineering, training, prediction,
    and importance calculation.
    """

    def __init__(
        self,
        fmp_api_key: str,
        feature_columns: List[str],
        target_column: str = "target_return_21d",
    ):
        """
        Initialize RBP pipeline.

        Args:
            fmp_api_key: API key for Financial Modeling Prep
            feature_columns: List of feature column names to use
            target_column: Name of target variable column
        """
        self.data_fetcher = MarketDataFetcher(fmp_api_key)
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.predictor = RBPPredictor(feature_columns)
        self.rbi_calculator = RBICalculator()

    def run_batch_predictions(
        self,
        test_features: pd.DataFrame,
        test_outcomes: pd.Series,
        training_features: pd.DataFrame,
        training_outcomes: pd.Series,
        n_jobs: int = -1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run RBP predictions for all test tasks in parallel.

        Args:
            test_features: DataFrame of test prediction tasks (M, K)
            test_outcomes: Series of actual test outcomes (M,)
            training_features: DataFrame of training observations (N, K)
            training_outcomes: Series of training outcomes (N,)
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            Tuple of (predictions_df, rbi_scores_df)
            - predictions_df: DataFrame with actual vs predicted values
            - rbi_scores_df: DataFrame with RBI scores for each task
        """
        logging.info(f"Starting batch predictions for {len(test_features)} tasks...")

        # Materialize iterator for parallel processing
        task_items = list(test_features.iterrows())

        # Run predictions in parallel
        parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._process_single_task)(
                task_index, task_features, training_features, training_outcomes
            )
            for task_index, task_features in task_items
        )

        # Filter out failed tasks
        valid_results = [result for result in parallel_results if result is not None]

        if len(valid_results) == 0:
            raise RuntimeError("All tasks failed. Check logs for errors.")

        # Separate predictions and RBI scores
        prediction_rows, rbi_series_list = zip(*valid_results)

        # Build result DataFrames
        predictions_df = pd.DataFrame(prediction_rows).set_index("task_index")

        # Add actual outcomes to predictions
        predictions_df["y_actual"] = test_outcomes

        rbi_scores_df = pd.DataFrame(rbi_series_list)

        logging.info("Batch predictions complete.")

        return predictions_df, rbi_scores_df

    def _process_single_task(
        self,
        task_index,
        task_features: pd.Series,
        training_features: pd.DataFrame,
        training_outcomes: pd.Series,
    ) -> Optional[Tuple[Dict, pd.Series]]:
        """
        Process a single prediction task (helper for parallel execution).

        Args:
            task_index: Index of the task
            task_features: Feature values for this task
            training_features: All training features
            training_outcomes: All training outcomes

        Returns:
            Tuple of (prediction_dict, rbi_scores) or None if failed
        """
        try:
            # Generate prediction and grid
            prediction, grid_results = self.predictor.predict_single_task(
                task_features, training_features, training_outcomes
            )

            # Calculate RBI scores
            rbi_scores = self.rbi_calculator.calculate_rbi_for_task(
                grid_results, self.feature_columns
            )
            rbi_scores.name = task_index

            # Package results
            result_dict = {"task_index": task_index, "y_pred_rbp": prediction}

            return result_dict, rbi_scores

        except Exception as e:
            logging.error(f"Failed to process task {task_index}: {e}")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================


def train_test_split_by_date(
    data: pd.DataFrame, feature_columns: List[str], target_column: str, split_date: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train/test sets based on date.

    Args:
        data: DataFrame with timestamp column and features
        feature_columns: List of feature column names
        target_column: Name of target column
        split_date: Date string (YYYY-MM-DD) for splitting

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """

    split_timestamp = pd.to_datetime(split_date)

    train_mask = data["timestamp"] < split_timestamp
    test_mask = ~train_mask

    train_data = data[train_mask]
    test_data = data[test_mask]

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    logging.info(
        f"Train: {len(X_train)} samples, Test: {len(X_test)} samples "
        f"(split at {split_date})"
    )

    return X_train, y_train, X_test, y_test


def main():
    """
    Main function to demonstrate RBP prediction pipeline.
    """
    # Example usage of RBPPipeline
    fmp_api_key = "YOUR_FMP_API_KEY"
    feature_columns = [
        "past_return_21d",
        "past_return_63d",
        "past_return_252d",
        "past_vol_21d",
        "past_vol_63d",
    ]
    target_column = "target_return_21d"
    pipeline = RBPPipeline(fmp_api_key, feature_columns, target_column)
    # 1. Load tickers from tickers.json in parent directory
    script_dir = os.path.dirname(__file__)
    # Look for tickers.json in the parent directory
    ticker_file_path = os.path.abspath(
        os.path.join(script_dir, "..", "src/orchestrator/backfill/tickers.json")
    )
    try:
        with open(ticker_file_path, "r") as f:
            tickers = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load tickers from {ticker_file_path}: {e}")
        return
    # 2. Fetch and prepare data
    # Calculate lookback days from 2015-01-01 to 2023-12-31 (approximately 9 years)
    lookback_days = 365 * 9

    # Fetch data for all tickers at once
    all_data = pipeline.data_fetcher.fetch_data(
        ticker_symbols=tickers, lookback_days=lookback_days
    )

    if all_data.empty:
        logging.error("No data fetched for any tickers.")
        return
    # 3. Feature engineering
    engineered_data = pipeline.feature_engineer.engineer_features(all_data)
    # 4. Train/test split
    X_train, y_train, X_test, y_test = train_test_split_by_date(
        engineered_data, feature_columns, target_column, "2022-01-01"
    )
    # 5. Run RBP predictions
    predictions_df, rbi_scores_df = pipeline.run_batch_predictions(
        X_test, y_test, X_train, y_train, n_jobs=-1
    )
    # 6. Display results
    print("Predictions:")
    print(predictions_df.head())
    print("\nRBI Scores:")
    print(rbi_scores_df.head())


if __name__ == "__main__":
    main()
