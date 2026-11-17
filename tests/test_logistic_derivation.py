"""Checks docs/derivations/logistic.md: the analytic BCE gradient equals PyTorch autograd (#32)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
N, D = 200, 6
X = torch.randn(N, D, dtype=torch.float64)
y = (torch.rand(N, dtype=torch.float64) < 0.2).to(torch.float64)
w0 = torch.randn(D, dtype=torch.float64) * 0.5


def _autograd_gradient(pos_weight: float | None) -> torch.Tensor:
    w = w0.clone().requires_grad_(True)
    pw = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float64)
    loss = nn.BCEWithLogitsLoss(pos_weight=pw)(X @ w, y)   # mean over the n loans
    loss.backward()
    return w.grad


def test_gradient_is_xt_sigma_minus_y() -> None:
    analytic = X.T @ (torch.sigmoid(X @ w0) - y) / N
    assert torch.allclose(_autograd_gradient(None), analytic, atol=1e-12)


def test_weighted_gradient_matches_pos_weight() -> None:
    c = 54.0
    a = torch.where(y == 1, torch.tensor(c, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    analytic = X.T @ (a * (torch.sigmoid(X @ w0) - y)) / N
    assert torch.allclose(_autograd_gradient(c), analytic, atol=1e-12)


def test_stable_form_equals_textbook_bce() -> None:
    z = X @ w0
    p = torch.sigmoid(z)
    textbook = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    stable = torch.log1p(torch.exp(z)) - y * z
    assert torch.allclose(textbook, stable, atol=1e-12)


def test_hessian_is_positive_semidefinite() -> None:
    p = torch.sigmoid(X @ w0)
    hessian = X.T @ torch.diag(p * (1 - p)) @ X / N
    assert np.linalg.eigvalsh(hessian.numpy()).min() >= -1e-12
