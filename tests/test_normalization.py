"""Normalization and denormalization, checked number by number.

Features: median imputation + missing indicator + standard scaling, learned from training rows only.
Task 1 target: z = (log1p(y) - mu) / sigma on the training fold; back: y = expm1(z * sigma + mu), clipped.
Scorer: the model predicts USD; the limit is USD x the same exchange rate used to build the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.processing import TargetLogScaler, get_feature_lists, make_preprocessor
from src.training import fit_predict, iter_splits


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num, cat = get_feature_lists("task2")
    df = pd.DataFrame({c: rng.normal(10 * (i + 1), i + 1, n) for i, c in enumerate(num)})
    df.loc[rng.random(n) < 0.3, "days_since_last_loan"] = np.nan
    df["lender_id"] = rng.choice(["A", "B", "C"], n)
    df["loan_type"] = rng.choice(["Type_1", "Type_7"], n)
    return df[num + cat]


def test_numeric_columns_are_standardized_on_the_training_rows() -> None:
    df = _frame()
    num, _ = get_feature_lists("task2")
    pre = make_preprocessor("task2")
    X = pre.fit_transform(df)
    scaled = X[:, : len(num)]
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-9)


def test_scaler_and_imputer_never_see_validation_rows() -> None:
    train, valid = _frame(seed=1), _frame(seed=2)
    num, cat = get_feature_lists("task2")
    valid[num] = valid[num] + 1000  # a shifted validation fold
    pre = make_preprocessor("task2").fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    assert np.allclose(imputer.statistics_, train[num].median().to_numpy())
    assert np.allclose(scaler.mean_[: len(num)], pre.named_transformers_["num"].named_steps["impute"].transform(train[num])[:, : len(num)].mean(axis=0))
    assert pre.transform(valid)[:, : len(num)].mean() > 10, "validation must be scaled with training statistics, not its own"


def test_missing_values_get_training_median_and_an_indicator_column() -> None:
    df = _frame()
    pre = make_preprocessor("task2").fit(df)
    names = list(pre.get_feature_names_out())
    assert "num__missingindicator_days_since_last_loan" in names
    row = df.head(1).copy()
    row["days_since_last_loan"] = np.nan
    out = pd.Series(pre.transform(row)[0], index=names)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    j = list(pre.named_transformers_["num"].named_steps["impute"].get_feature_names_out()).index("days_since_last_loan")
    median = df["days_since_last_loan"].median()
    assert out["num__days_since_last_loan"] == pytest.approx((median - scaler.mean_[j]) / scaler.scale_[j])
    assert out["num__missingindicator_days_since_last_loan"] != 0


def test_categories_one_hot_and_unseen_become_all_zero() -> None:
    df = _frame()
    pre = make_preprocessor("task2").fit(df)
    names = list(pre.get_feature_names_out())
    cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
    seen = pre.transform(df.head(5))[:, cat_idx]
    assert np.allclose(seen.sum(axis=1), 2)  # one lender + one loan type
    unseen = df.head(1).assign(lender_id="NEW", loan_type="Type_3")
    assert np.allclose(pre.transform(unseen)[:, cat_idx], 0)


def test_target_scaler_formula_and_round_trip() -> None:
    y = np.array([5.0, 40.0, 120.0, 900.0, 15000.0])
    s = TargetLogScaler().fit(y)
    log_y = np.log1p(y)
    assert s.mu_ == pytest.approx(log_y.mean()) and s.sigma_ == pytest.approx(log_y.std())
    z = s.transform(y)
    assert np.allclose(z, (log_y - log_y.mean()) / log_y.std())
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)
    assert np.allclose(s.inverse_transform(z), y)


def test_target_scaler_rejects_negative_amounts_and_unfitted_use() -> None:
    with pytest.raises(ValueError):
        TargetLogScaler().fit([-1.0, 2.0])
    with pytest.raises(RuntimeError):
        TargetLogScaler().transform([1.0])


def test_task1_cv_denormalizes_into_usd_and_is_fold_safe(synthetic_features) -> None:
    """Predictions come back on the dollar scale, within the training range of each fold."""
    df = synthetic_features[synthetic_features["split"] == "train"].reset_index(drop=True)
    num, cat = get_feature_lists("task1")
    y = np.expm1(df[C.TARGET_T1].to_numpy())
    for _, tr, va in iter_splits(df, "fold_group"):
        pred = fit_predict("ridge", "task1", df.loc[tr, num + cat], y[tr], df.loc[va, num + cat], df.loc[tr, "customer_id"].to_numpy())
        assert pred.min() >= y[tr].min() - 1e-6 and pred.max() <= y[tr].max() + 1e-6
        assert 0.1 < np.median(pred) / np.median(y[tr]) < 10, "prediction should be on the same (USD) scale as the target"


def test_usd_to_local_conversion_matches_features(synthetic_features, monkeypatch) -> None:
    """The scorer's exchange rate is the previous year's, exactly as in features.py."""
    from src.serving import schema

    monkeypatch.setattr(schema, "_fx_table", lambda: {("Kenya", 2022): 117.866, ("Ghana", 2022): 8.2724})
    row, fx = schema.prepare_single_application(
        {"customer_id": "c", "country_id": "Ghana", "lender_id": "x", "loan_type": "Type_1", "Total_Amount": 827.24,
         "duration": 7, "disbursement_date": "2023-05-01", "New_versus_Repeat": "Repeat Loan"}
    )
    assert fx == pytest.approx(8.2724)
    assert np.expm1(row["log_amount_usd"].iloc[0]) == pytest.approx(100.0)  # 827.24 GHS = 100 USD
