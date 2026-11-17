"""Print the target and leakage EDA tables (Henry's explore_target notebook, as a script).

    python -m scripts.run_target_eda        (or: make eda-target)
"""
from src import config as C
from src.analysis import compute_target_eda, load_target_eda_source
from src.console import console

TABLES = ["by_tenure", "by_year", "by_month", "by_lender", "by_loan_type", "shift_2023_summary", "shift_2023_by_lender"]


def main() -> None:
    console.title("Target EDA")
    with console.step("load data and compute default-rate tables"):
        df, source = load_target_eda_source()
        res = compute_target_eda(df)
    console.info(f"{source}: {res['n_rows']:,} rows, {res['n_defaults']:,} defaults ({res['default_rate']:.2%})")
    if res["extra_pct_by_target"] is not None:
        console.table(res["extra_pct_by_target"], "repayment markup (%) by outcome: the leakage check")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        res[name].to_csv(C.TABLES / f"eda_target_{name}.csv", index=False)
    console.table(res["by_year"], "default rate by year")
    console.table(res["shift_2023_summary"], "before vs after 2023")
    console.done(f"{len(TABLES)} tables in {C.rel(C.TABLES)}/eda_target_*.csv")


if __name__ == "__main__":
    main()
