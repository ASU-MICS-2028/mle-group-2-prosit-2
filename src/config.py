"""Shared settings. Every script imports paths, the seed and the feature lists from here; never hard-code them."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
PREDS = ROOT / "preds"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# Raw files (unzip each zip into data/raw/<zip name>/)
CREDIT_DIR = RAW / "african-credit-scoring-challenge"
TRAIN_CSV = CREDIT_DIR / "Train.csv"
TEST_CSV = CREDIT_DIR / "Test.csv"
MACRO_CSV = CREDIT_DIR / "economic_indicators.csv"
FINDEX_CSV = RAW / "GHA_2021_FINDEX_v02_M_csv" / "micro_gha.csv"

FEATURES_PARQUET = PROCESSED / "features.parquet"
MODELS_YAML = CONFIGS / "models.yaml"
COST_MATRIX_YAML = CONFIGS / "cost_matrix.yaml"

# Columns dropped by the leakage audit (docs/feature_audit.md, decision D-2). Draft until confirmed.
LEAKY_COLUMNS = ["Total_Amount_to_Repay", "Lender_portion_to_be_repaid"]
# Exact functions of other columns: dropped as redundant (docs/feature_audit.md).
REDUNDANT_COLUMNS = ["due_date", "Amount_Funded_By_Lender"]

# Temporal split: train on loans disbursed before this date, validate on loans on/after it (D-7).
TEMPORAL_CUTOFF = "2023-01-01"
N_FOLDS = 5

# Macro indicators complete for both Kenya and Ghana (the others are empty for Ghana).
# They stay in features.parquet for the Kenya -> Ghana shift study, but are NOT model inputs (D-18).
MACRO_INDICATORS = {
    "Inflation, consumer prices (annual %)": "macro_inflation",
    "Official exchange rate (LCU per US$, period average)": "macro_fx",
    "Deposit interest rate (%)": "macro_deposit_rate",
    "Unemployment rate": "macro_unemployment",
}
MACRO_FEATURES = list(MACRO_INDICATORS.values())

# Shared key, feature, slice and output schemas (single source of truth across src/, scripts/ and notebooks/)
KEY_COLUMNS = ["ID", "split", "target", "customer_id", "tbl_loan_id", "disbursement_date"]
CAT_FEATURES = ["lender_id", "loan_type"]
FAIRNESS_SLICES = ["country_id", "lender_id", "loan_type", "is_new_borrower"]

# Model inputs for Task 2 (default). D-18: one amount column only (USD), no macro columns.
#  - log_amount_local together with log_amount_usd encodes the exchange rate, i.e. the country.
#  - Train is Kenya only, so each macro column takes 4 yearly values (6-8% inflation, 110-140 KES/USD);
#    Ghana (31% inflation, 8 GHS/USD) is far outside that range and linear models extrapolate wildly.
NUM_FEATURES_T2 = [
    "log_amount_usd",
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
]

# Task 1 predicts the loan amount, so amount-derived features are excluded to prevent target leakage.
NUM_FEATURES_T1 = [c for c in NUM_FEATURES_T2 if c != "log_amount_usd"]
# Task 1 target: log1p(amount in USD), so Kenya and Ghana limits share one unit (D-18).
TARGET_T1 = "log_amount_usd"

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


def rel(path: Path) -> Path:
    """`path` relative to the project root for printing, or unchanged if it lives elsewhere."""
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return Path(path)
