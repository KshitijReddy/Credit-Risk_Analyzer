# Credit Risk Analyzer — Probability of Default & Scorecard

**A machine-learning system that predicts how likely a loan applicant is to
default, explains *why* each decision is made, and converts the prediction into a
usable credit score — built the way a regulated lender actually needs it.**

> **Result:** 0.805 ROC-AUC (Gini 0.61, KS 0.54) on held-out data, recalling **83%
> of defaulters** at a business-cost-optimized threshold, with full per-applicant
> explainability via SHAP and a points-based credit scorecard.

---

## What problem is this solving?

A lender's real question is not "good or bad — yes or no?" but **"how likely is
this person to default, and can I justify the decision?"** Those two needs —
accurate *ranking* of risk and *explainability* of each decision — drive every
design choice here, and they're what separate a credit model from a generic
classifier.

The system does three things:

1. **Predicts** the probability of default (PD) for each applicant.
2. **Explains** which factors drove each prediction (required by law when you
   decline someone).
3. **Scores** — converts PD into a familiar points-based credit score.

The data is the real **Statlog German Credit** dataset: 1,000 applicants, 20
features (checking-account status, credit history, loan amount/duration/purpose,
savings, employment, etc.), with a ~30% default rate.

---

## The core concepts (own these for an interview)

### 1. Why we model "probability of default," and why default = the positive class

The target is reframed so that **default = 1** (the event we care about catching).
This matters because it makes the key metrics mean what you'd expect: **recall**
becomes "of the people who actually defaulted, how many did we catch?" — the
expensive thing to get right. If you left default = 0, recall would describe the
wrong group.

### 2. Class imbalance — why accuracy is a trap

Only ~30% of applicants default, so a lazy model that predicts "everyone repays"
scores **70% accuracy while catching zero defaulters** — useless. This is why the
project never reports accuracy as a headline and instead handles the imbalance
explicitly (class weighting for Logistic Regression / Random Forest, and
`scale_pos_weight` for XGBoost, which up-weights the rare default class).

### 3. Why ROC-AUC, Gini, and KS — the credit-risk metric trio

- **ROC-AUC (0.805)** — the probability the model scores a random *real* defaulter
  as riskier than a random non-defaulter. It measures **ranking quality**, which is
  exactly what you need to rank applicants by risk. 0.5 = random, 1.0 = perfect.
- **Gini (0.61)** — literally `2 × AUC − 1`; the same information rescaled, and the
  number a credit team asks for by name.
- **KS statistic (0.54)** — the maximum separation between the cumulative
  distributions of goods and bads across the score range; the classic scorecard
  "how well does this separate the two groups?" measure.

### 4. The cost-optimal threshold — why not 0.5

A model outputs a probability; you still need a cutoff to approve or decline. The
default 0.5 is **wrong here because the two mistakes aren't equally costly**:
approving someone who then defaults (a false negative) is far more expensive than
declining someone who'd have repaid (a false positive). Assuming a 5:1 cost ratio,
the project searches for the threshold that **minimises total expected cost** —
which pushes default recall up to **83%** (catching five of every six bad
applicants). This is the difference between a model that's statistically good and
one that's tuned to the actual business decision.

### 5. Interpretability and "adverse action" — why a black box can't ship

In lending, a declined applicant is often **legally entitled to the reasons** (the
"adverse-action" requirement). So a model that can't explain itself can't be
deployed. The project surfaces *why* via two complementary lenses:

- **Logistic coefficients** — the direct, signed effect of each feature (positive =
  raises default risk).
- **SHAP values** — per-prediction feature contributions that work even for the
  tree models.

Both independently flag the same top drivers: **checking-account status, loan
duration, loan amount, credit history, and savings** — which also passes a sanity
check against domain intuition.

### 6. The scorecard — turning PD into points

Lenders don't show staff a raw probability like 0.23; they use a familiar score.
The project converts PD to points with the standard **log-odds transform**:

```
odds  = P(good) / P(bad)
score = offset + factor · ln(odds)
```

calibrated so a chosen base score corresponds to a base odds ratio, and every fixed
number of points (PDO — "points to double the odds") doubles the odds of being
good. Higher score = lower risk. This is a real industry artifact, not a toy — it
shows you understand how a model gets *operationalised*.

---

## How it works — the pipeline end to end

The code is modular (one module per responsibility), and `main.py` runs the stages
in order:

**1. Data prep (`data_prep.py`)** — load the data, reframe the target to
`default = 1`, and split into train/test **stratified** so both halves keep the
~30% default rate. Build a preprocessing pipeline that scales numeric features and
one-hot encodes categoricals. Crucially, this preprocessing lives *inside* a
scikit-learn `Pipeline`, so it is **re-fit within each cross-validation fold** —
preventing test information from leaking into training (a subtle but classic
mistake).

**2. Train (`train.py`)** — fit three models, each wrapped with the preprocessor
and imbalance handling:
- **Logistic Regression** — the interpretable lending workhorse and baseline.
- **Random Forest** — a non-linear ensemble.
- **XGBoost** — gradient-boosted trees, usually the strongest tabular ranker.
Each is scored with **5-fold cross-validation** so the headline number isn't a
lucky split.

**3. Evaluate (`evaluate.py`)** — on the held-out test set, compute AUC, Gini, KS,
the cost-optimal threshold, the confusion matrix, and the classification report.

**4. Interpret (`interpret.py`)** — run SHAP on the tree model and pull the logistic
coefficients to produce the adverse-action drivers.

**5. Scorecard (`interpret.py`)** — convert predicted PDs to scores and plot the
score distributions of repaid vs. defaulted applicants to show they separate.

---

## Results

| Model | ROC-AUC | Gini | KS |
|---|---|---|---|
| **Logistic Regression** | **0.805** | **0.61** | **0.54** |
| XGBoost | 0.804 | 0.61 | 0.50 |
| Random Forest | 0.786 | 0.57 | 0.45 |

**The interpretable model won.** On a small (1,000-row), largely linear-signal
dataset, Logistic Regression matched XGBoost — which in a regulated context is the
*ideal* outcome: you get top performance *and* full transparency. At the
cost-optimal threshold, the model recalls **83% of defaulters**.

---

## Tech stack

- **Language:** Python
- **Core ML:** scikit-learn (pipelines, Logistic Regression, Random Forest,
  cross-validation, preprocessing, metrics), XGBoost
- **Interpretability:** SHAP
- **Data / viz:** pandas, NumPy, Matplotlib
- **Concepts:** supervised binary classification, class-imbalance handling,
  ROC-AUC / Gini / KS evaluation, cost-sensitive thresholding, model
  explainability, credit-scorecard construction, leakage-safe pipelines

---

## How to run

```bash
pip install -r requirements.txt
python -m src.main          # from the project root
```

Models are saved to `outputs/models/`, figures to `outputs/figures/`.

---

## Honest limitations & responsible-ML note

- **Fairness — the most important point.** The dataset contains features like sex /
  marital status and foreign-worker status. Using these in a real lending decision
  is **illegal** under fair-lending / anti-discrimination law. For this exercise
  they're left in, but in production they must be **removed** — and noticing this
  unprompted is itself a signal of responsible-ML thinking.
- **No probability calibration yet.** The PDs rank well (good AUC), but "ranks well"
  and "is numerically honest" aren't the same — a calibration curve + Brier score
  would confirm the probabilities mean what they say.
- **Small dataset.** 1,000 rows limits how complex a model can usefully get, which
  is part of why the simple model competes so well.
- **Natural extensions:** calibration, a FastAPI endpoint that scores one applicant
  and returns its top adverse-action reasons (making it a *deployed* piece), and
  population-stability monitoring for drift.

---

## What this project demonstrates

End-to-end, it shows the ability to: frame a business problem as the right ML target,
handle **class imbalance** correctly, evaluate with **domain-appropriate metrics**
(not just accuracy), tune the decision threshold to a **business cost**, deliver
**explainability** that meets a real regulatory need, and turn a model output into
an **operational artifact** (the scorecard) — plus the engineering discipline of a
**leakage-safe, modular pipeline**. It's a complete credit-risk workflow, not a
single trained classifier.

*Data: Statlog (German Credit Data), UCI Machine Learning Repository.*
