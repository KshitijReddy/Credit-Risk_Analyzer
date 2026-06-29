"""
Interpretability and the scorecard -- the part that makes this a CREDIT project
and not just another classifier.

Two reasons interpretability is non-negotiable in lending:
  1. Regulation: a declined applicant is often legally entitled to the reasons
     ("adverse action"). A black box that can't explain itself can't be deployed.
  2. Trust: risk teams sign off on models they can reason about.

We provide three views:
  * Logistic coefficients  -- the direct, signed effect of each feature.
  * SHAP values            -- per-feature contributions for the best model,
                              which work even for the tree ensembles.
  * A points-based SCORECARD -- converting PD into a familiar credit score via
                              the standard log-odds -> points transform.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data_prep import feature_names_after_transform


# ----------------------------------------------------------------------------
# Logistic coefficients
# ----------------------------------------------------------------------------
def logistic_coefficients(pipeline) -> pd.DataFrame:
    """Signed coefficients -> which features push toward / away from default."""
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = feature_names_after_transform(prep)
    coefs = model.coef_[0]
    out = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    return out


# ----------------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------------
def shap_summary(pipeline, X_train, X_test, path, max_display=15):
    """SHAP global feature importance for a tree model. Saves a bar plot."""
    import shap

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = feature_names_after_transform(prep)

    X_test_t = prep.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_t)
    # Some SHAP versions return a list per class; take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(max_display)
    )

    plt.figure(figsize=(7, 5))
    plt.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
    plt.xlabel("mean(|SHAP value|)  -- average impact on predicted PD")
    plt.title("Global feature importance (SHAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return imp.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Scorecard: PD -> points
# ----------------------------------------------------------------------------
def pd_to_score(pd_prob: np.ndarray) -> np.ndarray:
    """
    Convert probability of default into a credit score using the standard
    log-odds -> points transform.

        odds  = P(good) / P(bad)
        score = offset + factor * ln(odds)

    where factor and offset are calibrated so that:
        - score = BASE_SCORE at BASE_ODDS:1
        - every PDO points doubles the odds
    """
    pd_prob = np.clip(pd_prob, 1e-6, 1 - 1e-6)
    odds_good = (1 - pd_prob) / pd_prob

    factor = C.PDO / np.log(2)
    offset = C.BASE_SCORE - factor * np.log(C.BASE_ODDS)
    return offset + factor * np.log(odds_good)


def score_distribution_plot(pd_prob, y_true, path):
    """Show that scores separate goods from bads -- the visual KS story."""
    scores = pd_to_score(pd_prob)
    y = np.asarray(y_true)
    plt.figure(figsize=(7, 4.5))
    plt.hist(scores[y == 0], bins=30, alpha=0.6, label="good (repaid)", density=True)
    plt.hist(scores[y == 1], bins=30, alpha=0.6, label="default", density=True)
    plt.xlabel("Credit score (higher = lower risk)")
    plt.ylabel("Density")
    plt.title("Score distribution: goods vs defaults")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return scores
