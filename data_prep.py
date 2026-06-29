"""
Data loading and preprocessing.

Responsibilities:
  1. Load the raw CSV.
  2. Build the modelling target (default = 1 for bad credit risk).
  3. Split into train / test in a stratified, reproducible way.
  4. Build a preprocessing pipeline (scale numerics, one-hot encode categoricals)
     that is FIT ON TRAIN ONLY -- this prevents leakage of test information.

The preprocessing lives in a scikit-learn ColumnTransformer so it can be bolted
onto any estimator inside a single Pipeline. That matters: it means the exact
same transforms are applied at train and at inference, with no chance of the
columns drifting out of sync.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def load_raw() -> pd.DataFrame:
    """Read the CSV and build the binary default target."""
    df = pd.read_csv(C.DATA_PATH)
    # 1=good, 0=bad  ->  default=1 when bad, 0 when good.
    df[C.TARGET] = (df[C.RAW_TARGET] == 0).astype(int)
    df = df.drop(columns=[C.RAW_TARGET])
    return df


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Resolve numeric vs categorical feature lists from the actual schema."""
    feature_cols = [c for c in df.columns if c != C.TARGET]
    numeric = [c for c in C.NUMERIC_FEATURES if c in feature_cols]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale numerics, one-hot encode categoricals. Fit happens later, on train."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ],
        remainder="drop",
    )


def get_data():
    """
    Return everything downstream code needs:
      X_train, X_test, y_train, y_test, preprocessor, (numeric, categorical)
    """
    df = load_raw()
    numeric, categorical = split_columns(df)

    X = df.drop(columns=[C.TARGET])
    y = df[C.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=C.TEST_SIZE,
        random_state=C.RANDOM_STATE,
        stratify=y,                      # keep the 30% default rate in both splits
    )

    preprocessor = build_preprocessor(numeric, categorical)
    return X_train, X_test, y_train, y_test, preprocessor, (numeric, categorical)


def feature_names_after_transform(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable feature names after one-hot encoding (for interpretability)."""
    return list(fitted_preprocessor.get_feature_names_out())
