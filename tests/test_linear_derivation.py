"""Checks docs/derivations/linear.md: normal equations, Ridge MAP closed form, gradients and convexity (#32)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

RNG = np.random.default_rng(42)
N, D = 120, 5
X_RAW = RNG.normal(size=(N, D))
TRUE_W = np.array([1.5, -2.0, 0.8, 0.0, 3.1])
TRUE_B = 4.2
Y_RAW = X_RAW @ TRUE_W + TRUE_B + RNG.normal(scale=0.5, size=N)


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def test_normal_equations_match_sklearn_ols() -> None:
    X_aug = _with_intercept(X_RAW)
    theta_closed = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ Y_RAW)

    ols = LinearRegression(fit_intercept=True).fit(X_RAW, Y_RAW)
    assert np.allclose(theta_closed[0], ols.intercept_, atol=1e-10)
    assert np.allclose(theta_closed[1:], ols.coef_, atol=1e-10)


def test_ridge_map_closed_form_matches_sklearn_ridge() -> None:
    alpha = 10.0  # same setting as configs/models.yaml task1.ridge
    X_aug = _with_intercept(X_RAW)
    penalty = np.diag([0.0] + [alpha] * D)  # intercept is left unpenalized
    theta_ridge = np.linalg.solve(X_aug.T @ X_aug + penalty, X_aug.T @ Y_RAW)

    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(X_RAW, Y_RAW)
    assert np.allclose(theta_ridge[0], ridge.intercept_, atol=1e-10)
    assert np.allclose(theta_ridge[1:], ridge.coef_, atol=1e-10)


def test_analytic_gradient_matches_pytorch_autograd() -> None:
    alpha = 10.0
    X_t = torch.tensor(X_RAW, dtype=torch.float64)
    y_t = torch.tensor(Y_RAW, dtype=torch.float64)
    w_t = torch.tensor(RNG.normal(size=D), dtype=torch.float64, requires_grad=True)

    residual = X_t @ w_t - y_t
    loss = 0.5 * torch.sum(residual**2) + 0.5 * alpha * torch.sum(w_t**2)
    loss.backward()

    analytic = X_t.T @ (X_t @ w_t.detach() - y_t) + alpha * w_t.detach()
    assert torch.allclose(w_t.grad, analytic, atol=1e-10)


def test_hessian_is_psd_and_ridge_stabilizes_collinearity() -> None:
    # Build a nearly collinear matrix (like n_past_loans and n_past_labelled)
    collinear = np.column_stack([X_RAW[:, 0], X_RAW[:, 0] + 1e-6 * RNG.normal(size=N), X_RAW[:, 1]])
    gram = collinear.T @ collinear
    eigs_ols = np.linalg.eigvalsh(gram)
    assert eigs_ols.min() >= -1e-12

    lam = 10.0
    gram_ridge = gram + lam * np.eye(collinear.shape[1])
    eigs_ridge = np.linalg.eigvalsh(gram_ridge)
    assert np.allclose(eigs_ridge, eigs_ols + lam, atol=1e-10)
    assert np.linalg.cond(gram_ridge) < np.linalg.cond(gram)


def test_log_retransformation_underpredicts_mean_by_jensens_inequality() -> None:
    mu, sigma = 4.0, 1.2
    log_y = RNG.normal(loc=mu, scale=sigma, size=200_000)
    y_orig = np.exp(log_y)
    assert np.exp(log_y.mean()) < y_orig.mean()
    assert np.isclose(y_orig.mean() / np.exp(log_y.mean()), np.exp(0.5 * sigma**2), rtol=0.03)
