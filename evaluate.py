"""
Evaluation -- with the metrics credit risk actually cares about.

Beyond accuracy (which is misleading on imbalanced data) we compute:

  * ROC-AUC        -- ranking quality: P(model scores a random defaulter above a
                      random non-defaulter).
  * Gini = 2*AUC-1 -- the SAME information rescaled to [0,1]; the number a credit
                      risk team will actually ask you for.
  * KS statistic   -- the maximum separation between the cumulative good and bad
                      distributions. The classic scorecard "how well does it
                      separate?" metric.
  * Precision/Recall on the DEFAULT class -- because catching defaulters (recall)
                      is the expensive thing to get wrong.

We also pick the decision threshold by a BUSINESS COST, not the default 0.5:
approving a defaulter is much costlier than declining a good customer, so we
search the threshold that minimises expected cost.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from . import config as C

# Business cost assumption (illustrative, documented so it can be challenged):
# a False Negative (approve someone who defaults) costs ~5x a False Positive
# (decline someone who would have repaid).
COST_FN = 5.0
COST_FP = 1.0


def ks_statistic(y_true, y_prob) -> float:
    """Max gap between cumulative bad-rate and good-rate across score order."""
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def best_cost_threshold(y_true, y_prob) -> tuple[float, float]:
    """Threshold on PD that minimises expected misclassification cost."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_cost = 0.5, np.inf
    y = np.asarray(y_true)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost


def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    """Compute the full metric suite for one fitted pipeline on the test set."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]   # P(default)
    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    ks = ks_statistic(y_test, y_prob)
    thr, cost = best_cost_threshold(y_test, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    return {
        "name": name,
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "threshold": thr,
        "expected_cost": cost,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "report": classification_report(
            y_test, y_pred, target_names=["good", "default"], digits=3
        ),
        "confusion": confusion_matrix(y_test, y_pred),
    }


def plot_roc(results: dict, y_test, path):
    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC -- Probability of Default models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(r, path):
    cm = r["confusion"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["good", "default"]); ax.set_yticklabels(["good", "default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix -- {r['name']}\n(threshold={r['threshold']:.2f})")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
