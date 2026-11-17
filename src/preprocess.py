"""Shared preprocessing pipelines and target scaling for Task 1 and Task 2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def get_feature_lists(task: str = "task2") -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for 'task1' or 'task2'."""
    if task == "task2":
        return list(C.NUM_FEATURES_T2), list(C.CAT_FEATURES)
    if task == "task1":
        return list(C.NUM_FEATURES_T1), list(C.CAT_FEATURES)
    raise ValueError(f"Unknown task '{task}'; expected 'task1' or 'task2'.")


def make_preprocessor(task: str = "task2", sparse_output: bool = False) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for the requested task.

    Numeric columns use median imputation with a missingness indicator followed by
    standard scaling. Categorical columns use OneHotEncoder(handle_unknown='ignore')
    so unseen Ghanaian lenders or loan types never crash inference (Decision D-5).
    """
    num_cols, cat_cols = get_feature_lists(task)
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def load_features(path: Path | None = None, split: str | None = None) -> pd.DataFrame:
    """Load the shared feature table built by `python -m src.features`."""
    target_path = path or C.FEATURES_PARQUET
    df = pd.read_parquet(target_path)
    for col in C.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


class TargetLogScaler:
    """Standardize log1p-transformed loan amounts using training-fold statistics only.

    Forward: z = (log(1 + y) - mu) / sigma
    Inverse: y_hat = max(0, exp(z_hat * sigma + mu) - 1)
    """

    def __init__(self) -> None:
        self.mu_: float = 0.0
        self.sigma_: float = 1.0
        self._is_fitted: bool = False

    def fit(self, y: np.ndarray | pd.Series) -> TargetLogScaler:
        arr = np.asarray(y, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError("TargetLogScaler expects non-negative loan amounts.")
        log_y = np.log1p(arr)
        self.mu_ = float(np.mean(log_y))
        std = float(np.std(log_y))
        self.sigma_ = std if std > 0.0 else 1.0
        self._is_fitted = True
        return self

    def transform(self, y: np.ndarray | pd.Series) -> np.ndarray:
        self._check_fitted()
        arr = np.asarray(y, dtype=np.float64)
        return (np.log1p(arr) - self.mu_) / self.sigma_

    def fit_transform(self, y: np.ndarray | pd.Series) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, z: np.ndarray | pd.Series) -> np.ndarray:
        self._check_fitted()
        arr = np.asarray(z, dtype=np.float64)
        log_y = np.clip(arr * self.sigma_ + self.mu_, a_min=-20.0, a_max=30.0)
        return np.maximum(0.0, np.expm1(log_y))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TargetLogScaler must be fitted before transform/inverse_transform.")
