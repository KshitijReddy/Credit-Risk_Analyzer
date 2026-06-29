"""
Model training.

We train three models that climb a deliberate ladder:

  1. Logistic Regression  -- the credit-risk WORKHORSE. Banks favour it because
     every coefficient is an interpretable, monotonic effect, which regulators
     can audit and which maps cleanly onto a points-based scorecard. This is our
     baseline AND, often, the model that actually ships.

  2. Random Forest        -- a non-linear ensemble; captures interactions a
     linear model misses, at the cost of interpretability.

  3. XGBoost              -- gradient-boosted trees; usually the strongest raw
     ranker on tabular data. The model to beat on AUC.

Class imbalance (30% default) is handled WITHOUT resampling the data:
  - LogReg / RF use class_weight="balanced"
  - XGBoost uses scale_pos_weight = n_neg / n_pos
This keeps the pipeline simple and avoids the leakage traps of oversampling
before cross-validation.

Every model is wrapped in a Pipeline with the preprocessor so the transforms are
re-fit inside each CV fold -- no test/fold leakage.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as C


def build_models(y_train) -> dict:
    """Instantiate the three estimators with imbalance handling baked in."""
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos  # weight the rare (default) class up in XGBoost

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=C.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def train_all(X_train, y_train, preprocessor):
    """
    Fit each model on the full training set and also report cross-validated AUC
    so we know the headline number isn't a lucky split.
    Returns: dict[name] -> {"pipeline": fitted_pipeline, "cv_auc_mean", "cv_auc_std"}
    """
    results = {}
    for name, est in build_models(y_train).items():
        pipe = make_pipeline(preprocessor, est)

        cv = cross_val_score(
            pipe, X_train, y_train,
            cv=C.CV_FOLDS, scoring="roc_auc", n_jobs=-1,
        )
        pipe.fit(X_train, y_train)

        results[name] = {
            "pipeline": pipe,
            "cv_auc_mean": cv.mean(),
            "cv_auc_std": cv.std(),
        }
        joblib.dump(pipe, C.MODEL_DIR / f"{name}.joblib")
        print(f"  {name:22s}  CV AUC = {cv.mean():.3f} (+/- {cv.std():.3f})")

    return results
