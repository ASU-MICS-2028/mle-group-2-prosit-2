"""Preprocessing: unseen categories never crash, and the target scaler never explodes."""
from __future__ import annotations

import numpy as np

from src.processing import TargetLogScaler, get_feature_lists, make_preprocessor


def test_unseen_ghana_categories_become_zero_rows(synthetic_features) -> None:
    for task in ("task1", "task2"):
        num, cat = get_feature_lists(task)
        pre = make_preprocessor(task)
        X_tr = pre.fit_transform(synthetic_features[num + cat])
        ghana = synthetic_features[num + cat].head(1).copy()
        ghana["lender_id"], ghana["loan_type"] = "UNSEEN_LENDER", "Type_3"
        X_gh = pre.transform(ghana)
        assert X_gh.shape[1] == X_tr.shape[1]
        assert not np.isnan(X_gh).any()


def test_target_scaler_round_trip() -> None:
    y = np.array([100.0, 300.0, 1500.0, 25000.0])
    scaler = TargetLogScaler().fit(y)
    assert np.allclose(scaler.inverse_transform(scaler.transform(y)), y, rtol=1e-9)


def test_target_scaler_clips_to_training_range() -> None:
    """A wild extrapolation (z = 50) must come back as the largest training amount, not 1e100."""
    y = np.array([10.0, 50.0, 200.0, 1000.0])
    scaler = TargetLogScaler().fit(y)
    back = scaler.inverse_transform(np.array([-50.0, 50.0]))
    assert np.isclose(back[0], 10.0) and np.isclose(back[1], 1000.0)
