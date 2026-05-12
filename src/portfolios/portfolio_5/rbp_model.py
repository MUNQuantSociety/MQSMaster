"""
Relevance-Based Prediction (RBP) Model.

Implements the RBP framework from the notebook (setup.ipynb) as a reusable class.
Core concepts:
  - Mahalanobis distance for measuring observation similarity
  - Relevance scoring (similarity + informativeness)
  - Grid search over variable subsets × censoring thresholds
  - Composite prediction via reliability-weighted averaging
  - RBI (Relevance-Based Importance) for variable importance
"""

import itertools
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RBPModel:
    """
    Encapsulates the full RBP pipeline: feature engineering, training
    statistics computation, relevance calculation, grid prediction,
    and composite prediction with RBI scoring.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "target_return_21d",
        relevance_thresholds: Optional[List[float]] = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.relevance_thresholds = relevance_thresholds or [0.0, 0.2, 0.5, 0.8]

        # Populated after fit()
        self._X_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._train_stats: Dict[str, np.ndarray] = {}
        self._is_fitted = False

    # ── Feature Engineering ──────────────────────────────────────────

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers predictive features (X) and target variable (Y).

        X variables:
          - past_return_21d:  1-month past return
          - past_vol_21d:     1-month past volatility
          - past_return_63d:  3-month past return
          - past_vol_63d:     3-month past volatility
          - past_return_252d: 1-year past return

        Y variable:
          - target_return_21d: 1-month future return
        """
        data = df.copy()
        if "timestamp" not in data.columns:
            if data.index.name == "timestamp":
                data = data.reset_index()
            else:
                data = data.reset_index().rename(columns={"index": "timestamp"})
        elif data.index.name == "timestamp":
        # Column already exists; drop the index label to prevent ambiguity
            data = data.reset_index(drop=True)
        data.sort_values(["ticker", "timestamp"], inplace=True)

        daily_returns = data.groupby("ticker")["close_price"].pct_change()
        grouped = data.groupby("ticker")

        # Past returns (momentum)
        data["past_return_21d"] = grouped["close_price"].pct_change(21)
        data["past_return_63d"] = grouped["close_price"].pct_change(63)
        data["past_return_252d"] = grouped["close_price"].pct_change(252)

        # Past volatility
        data["past_vol_21d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(21).std()
        )
        data["past_vol_63d"] = daily_returns.groupby(data["ticker"]).transform(
            lambda x: x.rolling(63).std()
        )

        # Target: 21-day future return
        data["target_return_21d"] = (
            grouped["close_price"].shift(-21) / data["close_price"] - 1
        )

        data.dropna(inplace=True)
        logger.info("Feature engineering complete. %d rows remain.", len(data))
        return data

    # ── Training Statistics ──────────────────────────────────────────

    @staticmethod
    def _compute_training_statistics(
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes mean vector and inverse covariance matrix for Mahalanobis distance."""
        # inside def _compute_training_statistics(X: pd.DataFrame):
        if X.shape[0] < 2:
        # not enough rows to compute covariance reliably
            x_mean = X.mean().fillna(0.0).values
            inv_cov = np.eye(X.shape[1]) * 1.0  
            # conservative identity covariance inverse
            return x_mean, inv_cov

        cov_matrix = X.cov().values

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix — adding jitter to stabilize.")
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov_matrix)

        x_mean = X.mean().values
        return x_mean, inv_cov

    # ── Relevance Calculations ───────────────────────────────────────

    @staticmethod
    def _mahalanobis_distance_sq(
        vec1: np.ndarray, vec2: np.ndarray, inv_cov: np.ndarray
    ) -> float:
        """Squared Mahalanobis distance between two vectors."""
        diff = vec1 - vec2
        return float(diff @ inv_cov @ diff)

    @staticmethod
    def _relevance(
        x_i: np.ndarray,
        x_t: np.ndarray,
        x_mean: np.ndarray,
        inv_cov: np.ndarray,
    ) -> float:
        """
        Relevance of past observation x_i to current task x_t (Eq. 1).
        r_it = -0.5 * sim(x_i, x_t) + 0.5 * (info(x_i) + info(x_t))
        """
        md = RBPModel._mahalanobis_distance_sq
        sim = md(x_i, x_t, inv_cov)
        info_i = md(x_i, x_mean, inv_cov)
        info_t = md(x_t, x_mean, inv_cov)
        return -0.5 * sim + 0.5 * (info_i + info_t)

    @staticmethod
    def _relevance_scores_for_task(
        x_t: np.ndarray,
        X_train: pd.DataFrame,
        x_mean: np.ndarray,
        inv_cov: np.ndarray,
    ) -> pd.Series:
        """Relevance of every training row against a single task vector."""
        return X_train.apply(
            lambda row: RBPModel._relevance(row.values, x_t, x_mean, inv_cov),
            axis=1,
        )

    # ── Prediction Weights ───────────────────────────────────────────

    @staticmethod
    def _prediction_weights(
        relevance_scores: pd.Series, threshold_quantile: float = 0.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes observation weights from relevance scores.
        threshold_quantile=0.0 → linear regression weights (Eq. 6).
        threshold_quantile>0   → censored RBP weights (Eqs. 7-9).
        """
        N = len(relevance_scores)
        if N < 2:
            return (
                pd.Series(np.nan, index=relevance_scores.index),
                pd.Series(False, index=relevance_scores.index),
            )

        # Linear case
        if threshold_quantile == 0.0:
            weights = (1 / N) + (1 / (N - 1)) * relevance_scores
            mask = pd.Series(True, index=relevance_scores.index)
            return weights, mask

        # Censored case
        r_star = relevance_scores.quantile(threshold_quantile)
        mask = relevance_scores >= r_star
        n = mask.sum()

        if n < 2:
            weights = (1 / N) + (1 / (N - 1)) * relevance_scores
            mask = pd.Series(True, index=relevance_scores.index)
            return weights, mask

        phi = n / N
        retained = relevance_scores[mask]
        r_sub_avg = retained.mean()

        var_full = (relevance_scores**2).sum() / (N - 1) if N > 1 else 1.0
        var_retained = (retained**2).sum() / (n - 1) if n > 1 else 1.0
        lambda_sq = var_full / var_retained if var_retained != 0 else 1.0

        delta_r = relevance_scores.where(mask, 0.0)
        weights = (1 / N) + (lambda_sq / (n - 1)) * (delta_r - phi * r_sub_avg)
        return weights, mask

    # ── Fit / Asymmetry / Adjusted Fit ───────────────────────────────

    @staticmethod
    def _fit(weights: pd.Series, outcomes: pd.Series) -> float:
        """Fit = squared correlation of weights and outcomes (Eq. 11)."""
        w, o = weights.align(outcomes)
        if w.std() == 0 or o.std() == 0:
            return 0.0
        rho = np.corrcoef(w, o)[0, 1]
        return 0.0 if np.isnan(rho) else rho**2

    @staticmethod
    def _asymmetry(
        weights: pd.Series, outcomes: pd.Series, mask: pd.Series
    ) -> float:
        """Asymmetry per Eq. 13."""
        w, o = weights.align(outcomes)
        w, m = w.align(mask)
        o, m = o.align(m)

        def _safe_corr(a, b):
            if len(a) < 2 or a.std() == 0 or b.std() == 0:
                return 0.0
            r = np.corrcoef(a, b)[0, 1]
            return 0.0 if np.isnan(r) else r

        rho_plus = _safe_corr(w[m], o[m])
        rho_minus = _safe_corr(w[~m], o[~m])
        return 0.5 * (rho_plus - rho_minus) ** 2

    @staticmethod
    def _adjusted_fit(fit: float, asymmetry: float, K: int) -> float:
        """Adjusted fit per Eq. 14."""
        return K * (fit + asymmetry)

    # ── Grid Processing ──────────────────────────────────────────────

    @staticmethod
    def _variable_combinations(columns: List[str]) -> List[Tuple[str, ...]]:
        """All 2^K - 1 non-empty subsets of variable names."""
        combos = []
        for k in range(1, len(columns) + 1):
            combos.extend(itertools.combinations(columns, k))
        return combos

    def _process_grid_for_task(
        self, x_t_series: pd.Series
    ) -> pd.DataFrame:
        """
        Builds the full prediction grid for a single task.
        Returns DataFrame with columns: [params, prediction, adj_fit].
        """
        var_combos = self._variable_combinations(list(self._X_train.columns))
        results = []

        for combo_tuple in var_combos:
            combo = list(combo_tuple)
            K = len(combo)
            X_sub = self._X_train[combo]
            x_t_sub = x_t_series[combo].values

            x_mean_sub, inv_cov_sub = self._compute_training_statistics(X_sub)
            rel_scores = self._relevance_scores_for_task(
                x_t_sub, X_sub, x_mean_sub, inv_cov_sub
            )

            for thresh in self.relevance_thresholds:
                weights, mask = self._prediction_weights(rel_scores, thresh)
                y_hat = (weights * self._y_train.align(weights)[0]).sum()
                fit = self._fit(weights, self._y_train)
                asym = self._asymmetry(weights, self._y_train, mask)
                adj_fit = self._adjusted_fit(fit, asym, K)

                results.append(
                    {
                        "params": {"vars": combo_tuple, "thresh": thresh, "K": K},
                        "prediction": y_hat,
                        "adj_fit": adj_fit,
                    }
                )

        return pd.DataFrame(results)

    # ── Composite Prediction & RBI ───────────────────────────────────

    @staticmethod
    def _composite_prediction(
        grid_df: pd.DataFrame,
    ) -> Tuple[float, pd.Series]:
        """
        Composite grid prediction (Eqs. 15-16).
        Returns (y_hat_grid, psi_weights).
        """
        adj_fits = grid_df["adj_fit"].clip(lower=0)
        total = adj_fits.sum()

        if total == 0:
            return 0.0, pd.Series(0.0, index=grid_df.index)

        psi = adj_fits / total
        y_hat = (psi * grid_df["prediction"]).sum()
        return y_hat, psi

    @staticmethod
    def _rbi_for_task(
        grid_df: pd.DataFrame, all_variables: List[str]
    ) -> pd.Series:
        """
        RBI scores for every variable for a single task (Eq. 18).
        Marginal contribution approach.
        """
        adj_fits = grid_df["adj_fit"]
        scores = {}

        for var_k in all_variables:
            includes_k = grid_df["params"].apply(lambda p: var_k in p["vars"])
            avg_with = adj_fits[includes_k].mean() or 0.0
            avg_without = adj_fits[~includes_k].mean() or 0.0
            scores[var_k] = avg_with - avg_without

        return pd.Series(scores)

    # ── Public API ───────────────────────────────────────────────────

    def fit(self, processed_data: pd.DataFrame, split_date: str = "2020-01-01"):
        """
        Fits the model on training data (before split_date).
        processed_data should already have features engineered.
        """
        split_ts = pd.to_datetime(split_date, utc=True)
        data_ts = pd.to_datetime(processed_data["timestamp"], utc=True)
        train = processed_data[data_ts < split_ts]

        self._X_train = train[self.feature_cols]
        self._y_train = train[self.target_col]

        x_mean, inv_cov = self._compute_training_statistics(self._X_train)
        self._train_stats = {"x_mean": x_mean, "inv_cov": inv_cov}
        self._is_fitted = True

        logger.info(
            "RBP model fitted on %d training rows with %d features.",
            len(self._X_train),
            len(self.feature_cols),
        )

    def predict(self, x_t_series: pd.Series) -> Tuple[float, pd.Series]:
        """
        Generates a composite prediction and RBI scores for a single task.

        Args:
            x_t_series: A Series with feature_cols as index.

        Returns:
            (prediction, rbi_scores)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        grid_df = self._process_grid_for_task(x_t_series)
        y_hat, _ = self._composite_prediction(grid_df)
        rbi = self._rbi_for_task(grid_df, self.feature_cols)
        return y_hat, rbi
