"""Shared settings. Every script imports paths and the seed from here; never hard-code them."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
PREDS = ROOT / "preds"

# Raw files (unzip each zip into data/raw/<zip name>/)
CREDIT_DIR = RAW / "african-credit-scoring-challenge"
TRAIN_CSV = CREDIT_DIR / "Train.csv"
TEST_CSV = CREDIT_DIR / "Test.csv"
MACRO_CSV = CREDIT_DIR / "economic_indicators.csv"
FINDEX_CSV = RAW / "GHA_2021_FINDEX_v02_M_csv" / "micro_gha.csv"

FEATURES_PARQUET = PROCESSED / "features.parquet"

# Columns dropped by the leakage audit (docs/feature_audit.md, decision D-2). Draft until confirmed.
LEAKY_COLUMNS = ["Total_Amount_to_Repay", "Lender_portion_to_be_repaid"]
# Exact functions of other columns: dropped as redundant (docs/feature_audit.md).
REDUNDANT_COLUMNS = ["due_date", "Amount_Funded_By_Lender"]

# Temporal split: train on loans disbursed before this date, validate on loans on/after it (D-7).
TEMPORAL_CUTOFF = "2023-01-01"
N_FOLDS = 5

# Macro indicators complete for both Kenya and Ghana (the others are empty for Ghana).
MACRO_INDICATORS = {
    "Inflation, consumer prices (annual %)": "macro_inflation",
    "Official exchange rate (LCU per US$, period average)": "macro_fx",
    "Deposit interest rate (%)": "macro_deposit_rate",
    "Unemployment rate": "macro_unemployment",
}

# Shared feature, target, slice, and output schemas (single source of truth across src/ and notebooks/)
KEY_COLUMNS = ["ID", "split", "target", "customer_id", "tbl_loan_id", "disbursement_date"]
CAT_FEATURES = ["lender_id", "loan_type"]
FAIRNESS_SLICES = ["country_id", "lender_id", "loan_type", "is_new_borrower"]

NUM_FEATURES_T2 = [
    "log_amount_usd",
    "log_amount_local",
    "duration",
    "Lender_portion_Funded",
    "is_new_borrower",
    "n_lenders_on_loan",
    "n_past_loans",
    "days_since_last_loan",
    "n_past_labelled",
    "n_past_defaults",
    "past_default_rate",
    "days_since_last_default",
    "disb_month",
    "disb_dow",
    "disb_dom",
    *MACRO_INDICATORS.values(),
]

# Task 1 predicts loan amount, so amount-derived features are excluded to prevent target leakage.
NUM_FEATURES_T1 = [
    c for c in NUM_FEATURES_T2 if c not in ("log_amount_usd", "log_amount_local")
]

PRED_COLUMNS = [
    "ID",
    "customer_id",
    "country_id",
    "lender_id",
    "loan_type",
    "is_new_borrower",
    "split",
    "scheme",
    "fold",
    "model",
    "y_true",
    "y_pred",
]

# Default Ghanaian cost-matrix parameters (override via CLI or function args as data-driven values are finalized)
DEFAULT_PRINCIPAL_GHS = 300.0
DEFAULT_FEE_RATE = 0.15
DEFAULT_RECOVERY_RATE = 0.20
DEFAULT_COLLECTION_COST_GHS = 25.0
