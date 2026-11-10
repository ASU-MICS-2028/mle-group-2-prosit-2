"""Build the one shared feature file and the fold splits.  Run:  python -m src.features

Everything here is time-safe: a loan's features use only information that existed on its
disbursement date. In particular a borrower's *past default* is counted only for loans whose
DUE date is before this loan's disbursement date, because an outcome is unknown until then.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from . import config as C


def load_raw() -> pd.DataFrame:
    """Train and Test stacked, with a `split` column; target is NaN for Test."""
    kw = dict(parse_dates=["disbursement_date", "due_date"])
    tr = pd.read_csv(C.TRAIN_CSV, **kw).assign(split="train")
    te = pd.read_csv(C.TEST_CSV, **kw).assign(split="test", target=np.nan)
    return pd.concat([tr, te], ignore_index=True)


def borrower_history(df: pd.DataFrame) -> pd.DataFrame:
    """Past-loan features per row, using backward merge_asof with no exact matches.

    Two-lender loans share a tbl_loan_id and count as ONE loan. Only Train loans have labels,
    so `n_past_labelled` says how many past loans the default rate is based on.
    """
    loans = df.groupby("tbl_loan_id", as_index=False).agg(
        customer_id=("customer_id", "first"), disb=("disbursement_date", "first"),
        due=("due_date", "first"), target=("target", "max"))

    starts = loans.sort_values("disb")[["customer_id", "disb"]].copy()
    starts["n_past_loans"] = starts.groupby("customer_id").cumcount() + 1
    starts["last_disb"] = starts["disb"]

    lab = loans[loans.target.notna()].sort_values("due").copy()
    lab["n_past_labelled"] = lab.groupby("customer_id").cumcount() + 1
    lab["n_past_defaults"] = lab.groupby("customer_id").target.cumsum()
    lab = lab[["customer_id", "due", "n_past_labelled", "n_past_defaults"]]

    dflt = loans[loans.target == 1].sort_values("due")[["customer_id", "due"]].rename(columns={"due": "last_default_due"})
    dflt["_k"] = dflt["last_default_due"]

    left = df[["ID", "customer_id", "disbursement_date"]].sort_values("disbursement_date")
    asof = dict(by="customer_id", direction="backward", allow_exact_matches=False)
    h = pd.merge_asof(left, starts, left_on="disbursement_date", right_on="disb", **asof)
    h = h.sort_values("disbursement_date")
    h = pd.merge_asof(h, lab, left_on="disbursement_date", right_on="due", **asof)
    h = h.sort_values("disbursement_date")
    h = pd.merge_asof(h, dflt, left_on="disbursement_date", right_on="_k", **asof)

    out = pd.DataFrame({"ID": h["ID"]})
    out["n_past_loans"] = h["n_past_loans"].fillna(0).astype(int)
    out["days_since_last_loan"] = (h["disbursement_date"] - h["last_disb"]).dt.days
    out["n_past_labelled"] = h["n_past_labelled"].fillna(0).astype(int)
    out["n_past_defaults"] = h["n_past_defaults"].fillna(0).astype(int)
    out["past_default_rate"] = out["n_past_defaults"] / out["n_past_labelled"].replace(0, np.nan)
    out["days_since_last_default"] = (h["disbursement_date"] - h["last_default_due"]).dt.days
    return out


def macro_by_country_year() -> pd.DataFrame:
    """One row per (country, year) with the complete indicators; year is the year the value describes."""
    m = pd.read_csv(C.MACRO_CSV)
    m = m[m["Indicator"].isin(C.MACRO_INDICATORS)]
    long = m.melt(id_vars=["Country", "Indicator"], var_name="year", value_name="v")
    long["year"] = long["year"].str.replace("YR", "").astype(int)
    long["Indicator"] = long["Indicator"].map(C.MACRO_INDICATORS)
    wide = long.pivot_table(index=["Country", "year"], columns="Indicator", values="v").reset_index()
    return wide.rename(columns={"Country": "country_id"})


def build() -> pd.DataFrame:
    df = load_raw()
    hist = borrower_history(df)
    df = df.merge(hist, on="ID", how="left", validate="one_to_one")

    # Calendar
    d = df["disbursement_date"]
    df["disb_month"], df["disb_dow"], df["disb_dom"] = d.dt.month, d.dt.dayofweek, d.dt.day

    # Macro: previous year's value (annual figures are only published after year-end)
    df["macro_year"] = d.dt.year - 1
    df = df.merge(macro_by_country_year(), left_on=["country_id", "macro_year"],
                  right_on=["country_id", "year"], how="left", validate="many_to_one").drop(columns=["year", "macro_year"])

    # Money: USD so Kenya and Ghana are comparable (D-6); log because amounts are very skewed
    df["log_amount_usd"] = np.log1p(df["Total_Amount"] / df["macro_fx"])
    df["log_amount_local"] = np.log1p(df["Total_Amount"])

    # Loan facts known at disbursement
    df["is_new_borrower"] = (df["New_versus_Repeat"] == "New Loan").astype(int)
    df["n_lenders_on_loan"] = df.groupby("tbl_loan_id")["ID"].transform("size")

    keep_keys = ["ID", "split", "target", "customer_id", "tbl_loan_id", "disbursement_date"]
    cats = ["country_id", "lender_id", "loan_type"]
    nums = ["log_amount_usd", "log_amount_local", "duration", "Lender_portion_Funded", "is_new_borrower",
            "n_lenders_on_loan", "n_past_loans", "days_since_last_loan", "n_past_labelled", "n_past_defaults",
            "past_default_rate", "days_since_last_default", "disb_month", "disb_dow", "disb_dom",
            *C.MACRO_INDICATORS.values()]
    out = df[keep_keys + cats + nums].copy()
    out["lender_id"] = out["lender_id"].astype(str)
    return add_folds(out)


def add_folds(df: pd.DataFrame) -> pd.DataFrame:
    """fold_group: GroupKFold by customer. fold_strat: naive stratified (for the inflation comparison only).
    temporal: 'train'/'valid' by disbursement date. All are -1 / 'test' for Test rows."""
    df = df.copy()
    df["fold_group"], df["fold_strat"] = -1, -1
    tr = df["split"] == "train"
    y, g = df.loc[tr, "target"].astype(int), df.loc[tr, "customer_id"]
    for k, (_, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(df.loc[tr], y, g)):
        df.loc[df.index[tr][va], "fold_group"] = k
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for k, (_, va) in enumerate(skf.split(df.loc[tr], y)):
        df.loc[df.index[tr][va], "fold_strat"] = k
    df["temporal"] = np.where(~tr, "test", np.where(df["disbursement_date"] < C.TEMPORAL_CUTOFF, "train", "valid"))
    return df


def validate(df: pd.DataFrame, n_check: int = 400) -> None:
    """Fail loudly if the file leaks. Called by main(); also safe to run in a notebook."""
    tr = df[df.split == "train"]
    # 1. No customer (hence no loan) on both sides of any GroupKFold split
    for k in range(C.N_FOLDS):
        va = set(tr.loc[tr.fold_group == k, "customer_id"])
        assert not va & set(tr.loc[tr.fold_group != k, "customer_id"]), f"customer in train and valid, fold {k}"
    # 2. No loan straddles folds
    assert (tr.groupby("tbl_loan_id").fold_group.nunique() == 1).all(), "a loan is split across folds"
    # 3. Every Train row is in exactly one validation fold, in both schemes
    for c in ("fold_group", "fold_strat"):
        assert tr[c].between(0, C.N_FOLDS - 1).all(), c
    # 4. Temporal split is ordered in time
    assert tr[tr.temporal == "train"].disbursement_date.max() < tr[tr.temporal == "valid"].disbursement_date.min()
    # 5. Brute-force check of history features on a random sample (reads raw data independently)
    raw = load_raw()
    loans = raw.groupby("tbl_loan_id").agg(cust=("customer_id", "first"), disb=("disbursement_date", "first"),
                                           due=("due_date", "first"), tgt=("target", "max"))
    sample = df.sample(min(n_check, len(df)), random_state=C.SEED)
    for _, r in sample.iterrows():
        mine = loans[(loans.cust == r.customer_id) & (loans.disb < r.disbursement_date)]
        known = loans[(loans.cust == r.customer_id) & (loans.due < r.disbursement_date) & loans.tgt.notna()]
        assert r.n_past_loans == len(mine), (r.ID, "n_past_loans")
        assert r.n_past_labelled == len(known), (r.ID, "n_past_labelled")
        assert r.n_past_defaults == known.tgt.sum(), (r.ID, "n_past_defaults")
    assert df["ID"].is_unique


def main() -> None:
    df = build()
    validate(df)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.FEATURES_PARQUET, index=False)
    print(f"Saved {C.FEATURES_PARQUET}  shape={df.shape}  (all validation checks passed)")


if __name__ == "__main__":
    main()
