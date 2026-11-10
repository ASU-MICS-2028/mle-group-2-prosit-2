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
