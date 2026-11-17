"""Build data/processed/features.parquet and the fold splits; fails loudly on any leakage.

    python -m scripts.build_features        (or: make features; `python -m src.features` also works)
"""
from src import config as C
from src import features as F
from src.console import console


def main() -> None:
    console.title("Build features")
    with console.step("load Train + Test, build history, calendar, macro, folds"):
        df = F.build()
    with console.step("leakage checks: folds, time order, history re-derived"):
        F.validate(df)
    with console.step(f"write {C.rel(C.FEATURES_PARQUET)}"):
        C.PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.FEATURES_PARQUET, index=False)
    n_train, n_test = int((df["split"] == "train").sum()), int((df["split"] == "test").sum())
    console.done(f"{len(df):,} rows x {df.shape[1]} columns ({n_train:,} train, {n_test:,} test); all checks passed")


if __name__ == "__main__":
    main()
