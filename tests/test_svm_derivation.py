"""Checks docs/derivations/svm.md on small 2-D toy data against scikit-learn's SVC (#32).

Each test verifies one claim of the derivation numerically:
  margin = 2/||w||, w = sum(alpha_i y_i x_i), sum(alpha_i y_i) = 0, strong duality,
  box constraint 0 <= alpha_i <= C_i, and the RBF decision function from the dual coefficients.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.svm import SVC

RNG = np.random.default_rng(0)


def _separable() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated blobs, labels in {-1, +1} (the derivation's convention)."""
    pos = RNG.normal([2.5, 2.5], 0.6, size=(25, 2))
    neg = RNG.normal([-2.5, -2.5], 0.6, size=(25, 2))
    return np.vstack([pos, neg]), np.r_[np.ones(25), -np.ones(25)]


def _overlapping() -> tuple[np.ndarray, np.ndarray]:
    """Blobs that overlap, so some points must violate the margin (soft margin)."""
    pos = RNG.normal([1.0, 1.0], 1.0, size=(60, 2))
    neg = RNG.normal([-1.0, -1.0], 1.0, size=(60, 2))
    return np.vstack([pos, neg]), np.r_[np.ones(60), -np.ones(60)]


def _linear_fit(X, y, C, class_weight=None) -> SVC:
    return SVC(kernel="linear", C=C, tol=1e-10, class_weight=class_weight).fit(X, y)


def _alpha_y(svc: SVC) -> np.ndarray:
    """sklearn stores alpha_i * y_i for the support vectors in dual_coef_."""
    return svc.dual_coef_.ravel()


def test_hard_margin_support_vectors_sit_on_the_margin() -> None:
    X, y = _separable()
    svc = _linear_fit(X, y, C=1e6)               # huge C ~ no slack allowed (hard margin)
    f = svc.decision_function(X)
    assert np.isclose((y * f).min(), 1.0, atol=1e-6)            # closest point has y*f = 1
    sv_f = y[svc.support_] * f[svc.support_]
    assert np.allclose(sv_f, 1.0, atol=1e-6)                    # every support vector is exactly on it


def test_margin_width_is_two_over_norm_w_and_is_the_maximum() -> None:
    X, y = _separable()
    svc = _linear_fit(X, y, C=1e6)
    w = svc.coef_.ravel()
    svm_margin = 2.0 / np.linalg.norm(w)
    # Independent brute force: for each unit direction u, the best parallel-lines gap is
    #   min(u.x over +) - max(u.x over -). The largest gap over all directions is the max margin.
    angles = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    U = np.c_[np.cos(angles), np.sin(angles)]
    proj = X @ U.T
    gap = proj[y == 1].min(axis=0) - proj[y == -1].max(axis=0)
    assert np.isclose(gap.max(), svm_margin, rtol=1e-3)


def test_w_is_the_weighted_sum_of_support_vectors_and_alphas_balance() -> None:
    X, y = _overlapping()
    svc = _linear_fit(X, y, C=1.0)
    w_from_dual = _alpha_y(svc) @ svc.support_vectors_
    assert np.allclose(w_from_dual, svc.coef_.ravel(), atol=1e-8)   # w = sum alpha_i y_i x_i
    assert np.isclose(_alpha_y(svc).sum(), 0.0, atol=1e-8)          # sum alpha_i y_i = 0


def test_strong_duality_primal_equals_dual() -> None:
    X, y = _overlapping()
    C = 1.0
    svc = _linear_fit(X, y, C=C)
    w = svc.coef_.ravel()
    f = svc.decision_function(X)
    primal = 0.5 * w @ w + C * np.maximum(0.0, 1.0 - y * f).sum()    # margin term + C * hinge losses
    alpha = np.abs(_alpha_y(svc))
    dual = alpha.sum() - 0.5 * w @ w                                  # sum alpha - 1/2 ||w||^2
    assert np.isclose(primal, dual, rtol=1e-6)


def test_kkt_alpha_zero_inside_alpha_C_violating_between_on_margin() -> None:
    X, y = _overlapping()
    C = 1.0
    svc = _linear_fit(X, y, C=C)
    alpha = np.zeros(len(y))
    alpha[svc.support_] = np.abs(_alpha_y(svc))
    yf = y * svc.decision_function(X)
    eps = 1e-6
    assert np.all(alpha <= C + eps) and np.all(alpha >= 0)           # box constraint 0 <= alpha <= C
    assert np.all(yf[alpha < eps] >= 1 - 1e-5)                       # alpha = 0  -> correct side, outside margin
    assert np.all(yf[alpha > C - eps] <= 1 + 1e-5)                   # alpha = C  -> on or inside margin / misclassified
    mid = (alpha > eps) & (alpha < C - eps)
    assert mid.any() and np.allclose(yf[mid], 1.0, atol=1e-5)        # 0 < alpha < C -> exactly on the margin


def test_class_weight_scales_the_box_constraint() -> None:
    X, y = _overlapping()
    C, a_pos, a_neg = 1.0, 20.0, 1.0                                 # 'balanced'-style: rare class costs more
    svc = _linear_fit(X, y, C=C, class_weight={1: a_pos, -1: a_neg})
    alpha = np.zeros(len(y))
    alpha[svc.support_] = np.abs(_alpha_y(svc))
    assert alpha[y == 1].max() <= C * a_pos + 1e-6                   # alpha_i <= C * weight(y_i)
    assert alpha[y == -1].max() <= C * a_neg + 1e-6
    assert alpha[y == 1].max() > C * a_neg + 1e-3                    # and the rare class really uses the bigger box


def test_rbf_decision_function_is_a_kernel_sum_over_support_vectors() -> None:
    X, y = _overlapping()
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-10).fit(X, y)
    gamma = 1.0 / (X.shape[1] * X.var())                              # sklearn's gamma='scale'
    assert np.isclose(svc._gamma, gamma)
    sq = ((X[:, None, :] - svc.support_vectors_[None, :, :]) ** 2).sum(-1)
    K = np.exp(-gamma * sq)                                           # k(x, x_i) = exp(-gamma ||x - x_i||^2)
    by_hand = K @ _alpha_y(svc) + svc.intercept_[0]                   # f(x) = sum alpha_i y_i k(x_i, x) + b
    assert np.allclose(by_hand, svc.decision_function(X), atol=1e-8)


def test_rbf_kernel_equals_feature_map_inner_product_in_the_series_limit() -> None:
    """exp(-g||x-z||^2) = exp(-g||x||^2) exp(-g||z||^2) exp(2g x.z); truncating the last series
    converges to the kernel, showing it is an inner product in a (very) high-dimensional space."""
    x, z, g = np.array([0.4, -0.3]), np.array([0.1, 0.5]), 0.7
    exact = np.exp(-g * np.sum((x - z) ** 2))
    t = 2 * g * x @ z
    series = sum(t**k / math.factorial(k) for k in range(25))
    assert np.isclose(np.exp(-g * x @ x) * np.exp(-g * z @ z) * series, exact, atol=1e-12)
