"""Bootstrap 95% confidence intervals on PR-AUC and F1, resampling customers rather than rows.

A customer's loans are correlated (median 6 loans each), so resampling single rows pretends we
have more independent evidence than we do and gives intervals that are too narrow.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from .. import config as C


def bootstrap_ci(
    y_true,
    y_prob,
    threshold: float,
    groups=None,
    n_boot: int = 500,
    seed: int = C.SEED,
    noise_floor: float = 0.0263,
) -> dict[str, float | bool]:
    """95% intervals for PR-AUC and F1. Pass `groups=customer_id` for a cluster bootstrap.

    `beats_baseline` is True when the PR-AUC lower bound clears the majority-class baseline
    (the default rate) by more than the brief's noise floor of 0.0263.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if groups is None:
        units = [np.array([i]) for i in range(len(yt))]
    else:
        g = np.asarray(groups)
        order = np.argsort(g, kind="stable")
        _, starts = np.unique(g[order], return_index=True)
        units = np.split(order, starts[1:])

    pr, f1 = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = np.concatenate([units[j] for j in rng.integers(0, len(units), size=len(units))])
        if len(np.unique(yt[idx])) < 2:
            continue
        pr[i] = average_precision_score(yt[idx], yp[idx])
        f1[i] = f1_score(yt[idx], (yp[idx] > threshold).astype(int), zero_division=0)

    pr_low, pr_high = np.nanpercentile(pr, [2.5, 97.5])
    f1_low, f1_high = np.nanpercentile(f1, [2.5, 97.5])
    baseline = float(yt.mean())
    return {
        "pr_auc": float(average_precision_score(yt, yp)),
        "pr_auc_ci_low": float(pr_low),
        "pr_auc_ci_high": float(pr_high),
        "f1": float(f1_score(yt, (yp > threshold).astype(int), zero_division=0)),
        "f1_ci_low": float(f1_low),
        "f1_ci_high": float(f1_high),
        "baseline_pr_auc": baseline,
        "beats_baseline": bool(pr_low > baseline + noise_floor),
        "resampled_unit": "customer" if groups is not None else "row",
    }
