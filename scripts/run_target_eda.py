"""Print the target and leakage EDA tables (Henry's explore_target notebook, as a script).

    python -m scripts.run_target_eda        (or: make eda-target)
"""
from src import config as C
from src.analysis import compute_target_eda, load_target_eda_source

TABLES = ["by_tenure", "by_year", "by_month", "by_lender", "by_loan_type", "shift_2023_summary", "shift_2023_by_lender"]


def main() -> None:
    df, source = load_target_eda_source()
    res = compute_target_eda(df)
    print(f"Source: {source}  rows={res['n_rows']}  defaults={res['n_defaults']} ({res['default_rate'] * 100:.2f}%)")
    if res["extra_pct_by_target"] is not None:
        print("\n--- Repayment markup (extra_pct) by outcome: the leakage check ---")
        print(res["extra_pct_by_target"].to_string(index=False))
    C.TABLES.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        print(f"\n--- {name} ---")
        print(res[name].head(15).to_string(index=False))
        res[name].to_csv(C.TABLES / f"eda_target_{name}.csv", index=False)
    print(f"\nTables saved to {C.TABLES.relative_to(C.ROOT)}/eda_target_*.csv")


if __name__ == "__main__":
    main()
