"""
Central configuration for the credit-risk project.

Keeping paths, column groups and hyper-parameters in one place means the rest
of the code never hard-codes a magic string or number -- you change behaviour
here, not by hunting through five files.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "german_credit.csv"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------
# Raw column `credit_risk`: 1 = good payer, 0 = bad payer.
# We MODEL DEFAULT, so the positive class (1) must be the "bad" outcome.
# default = 1 when the applicant is a bad credit risk.
RAW_TARGET = "credit_risk"
TARGET = "default"

# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------
# Continuous / count features -> scaled.
NUMERIC_FEATURES = [
    "duration", "amount", "age", "installment_rate",
    "present_residence", "number_credits", "people_liable",
]
# Everything else (string-coded) -> one-hot encoded.
# We resolve the exact list at runtime in data_prep so the code stays robust
# if the schema changes, but this documents intent.

# ----------------------------------------------------------------------------
# Split / CV
# ----------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# ----------------------------------------------------------------------------
# Scorecard scaling (industry-standard PD -> points transform)
# ----------------------------------------------------------------------------
# A scorecard maps log-odds to a points scale. The convention below means:
#   - a score of `BASE_SCORE` corresponds to BASE_ODDS:1 good:bad odds
#   - every `PDO` points the odds DOUBLE ("Points to Double the Odds")
BASE_SCORE = 600
BASE_ODDS = 50      # 50:1 good:bad at the base score
PDO = 20            # +20 points => odds of being good double
