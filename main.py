"""
End-to-end credit-risk pipeline.

Run:  python -m src.main      (from the project root)

Stages:
  1. Load + split + preprocess
  2. Train logistic / random forest / xgboost (with CV)
  3. Evaluate on held-out test set (AUC, Gini, KS, cost-optimal threshold)
  4. Interpret the best model (SHAP + logistic coefficients)
  5. Build a points-based scorecard from predicted PD
All figures land in outputs/figures/, all models in outputs/models/.
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import config as C
from . import data_prep, evaluate, interpret, train

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)


def main():
    print("=" * 70)
    print("CREDIT RISK ANALYSIS  --  Probability of Default modelling")
    print("=" * 70)

    # 1. Data ----------------------------------------------------------------
    print("\n[1] Loading and splitting data ...")
    X_train, X_test, y_train, y_test, preprocessor, (num, cat) = data_prep.get_data()
    print(f"    train={len(X_train)}  test={len(X_test)}  "
          f"| numeric={len(num)}  categorical={len(cat)}")
    print(f"    train default rate = {y_train.mean():.1%}  "
          f"test default rate = {y_test.mean():.1%}")

    # 2. Train ---------------------------------------------------------------
    print("\n[2] Training models (5-fold CV AUC) ...")
    trained = train.train_all(X_train, y_train, preprocessor)

    # 3. Evaluate ------------------------------------------------------------
    print("\n[3] Test-set evaluation ...")
    results = {}
    for name, info in trained.items():
        results[name] = evaluate.evaluate_model(name, info["pipeline"], X_test, y_test)

    summary = pd.DataFrame([
        {
            "model": r["name"],
            "AUC": round(r["auc"], 3),
            "Gini": round(r["gini"], 3),
            "KS": round(r["ks"], 3),
            "thr*": round(r["threshold"], 2),
            "cost@thr*": int(r["expected_cost"]),
        }
        for r in results.values()
    ]).sort_values("AUC", ascending=False).reset_index(drop=True)
    print("\n" + summary.to_string(index=False))

    best_name = summary.iloc[0]["model"]
    best = results[best_name]
    print(f"\n    Best ranker by AUC: {best_name}")
    print("\n    Classification report (cost-optimal threshold):")
    print("    " + best["report"].replace("\n", "\n    "))

    # Plots
    evaluate.plot_roc(results, y_test, C.FIG_DIR / "roc_curves.png")
    evaluate.plot_confusion(best, C.FIG_DIR / "confusion_best.png")

    # 4. Interpret -----------------------------------------------------------
    print("\n[4] Interpretability ...")
    lr = trained["logistic_regression"]["pipeline"]
    coefs = interpret.logistic_coefficients(lr)
    print("\n    Top logistic drivers of DEFAULT (positive = raises risk):")
    print("    " + coefs.head(8).to_string(index=False).replace("\n", "\n    "))

    # SHAP on the best tree model (fall back to xgboost if best is logistic)
    tree_name = best_name if best_name in ("random_forest", "xgboost") else "xgboost"
    try:
        imp = interpret.shap_summary(
            trained[tree_name]["pipeline"], X_train, X_test,
            C.FIG_DIR / "shap_importance.png",
        )
        print(f"\n    Top SHAP features ({tree_name}):")
        print("    " + imp.head(8).to_string(index=False).replace("\n", "\n    "))
    except Exception as e:
        print(f"    [SHAP skipped: {e}]")

    # 5. Scorecard -----------------------------------------------------------
    print("\n[5] Building scorecard from predicted PD ...")
    scores = interpret.score_distribution_plot(
        best["y_prob"], y_test, C.FIG_DIR / "score_distribution.png"
    )
    sc = pd.Series(scores)
    print(f"    Score range: {sc.min():.0f} - {sc.max():.0f}  "
          f"(median {sc.median():.0f})")
    print(f"    Mean score | repaid  = {sc[y_test.values == 0].mean():.0f}")
    print(f"    Mean score | default = {sc[y_test.values == 1].mean():.0f}")

    print("\n" + "=" * 70)
    print("Done. Figures -> outputs/figures/   Models -> outputs/models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
