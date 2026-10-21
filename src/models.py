"""Forecasting models. All learn the *change* in memory over the horizon
(memory(t+h) - memory(t)); this removes the level/trend so tree models do not
have to extrapolate beyond the memory range seen in training."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Persistence(BaseEstimator, RegressorMixin):
    """Predict that memory stays where it is now."""
    def fit(self, X, y=None): return self
    def predict(self, X): return np.zeros(len(X))


class MovingAverage(BaseEstimator, RegressorMixin):
    """Predict that memory reverts to its trailing 1-hour mean."""
    def fit(self, X, y=None): return self
    def predict(self, X): return -X["mem_dev_mean12"].to_numpy()


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Predict that the next change equals the change over the same window yesterday."""
    def fit(self, X, y=None): return self
    def predict(self, X): return X["seas_delta"].to_numpy()


def get_models(seed: int = 42) -> dict:
    return {
        "Persistence": Persistence(),
        "Moving average (1h)": MovingAverage(),
        "Seasonal naive (24h)": SeasonalNaive(),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        ),
        "MLP (64-32)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=seed),
        ),
    }


LEARNED = ("Ridge regression", "Random Forest", "Gradient Boosting", "MLP (64-32)")


def quantile_model(q: float = 0.95, seed: int = 42) -> HistGradientBoostingRegressor:
    """Upper-bound forecaster: predicts the q-quantile of the future change."""
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=seed,
    )
