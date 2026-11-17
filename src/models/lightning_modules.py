"""PyTorch Lightning models: logistic regression, MLP (Task 2) and linear regression (Task 1).

Same architecture and settings as Henry's notebook (02_classification): one linear layer for
logistic regression, two hidden layers (64, 32) with ReLU and dropout for the MLP, Adam with an
L2 weight decay, and BCEWithLogitsLoss with pos_weight so each default counts ~54 times.
"""
from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import torch
from torch import nn


class _BaseLit(L.LightningModule):
    """Shared training/validation steps and Adam optimiser for every model in this file."""

    loss_fn: nn.Module

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


class LitLogReg(_BaseLit):
    """Logistic regression = one linear layer; BCE loss is the negative log-likelihood (MLE)."""

    def __init__(self, n_features: int, pos_weight: float, lr: float = 1e-2, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(n_features, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)  # logits; sigmoid gives the probability


class LitMLP(_BaseLit):
    """Logistic regression with hidden layers in between, so it can learn feature interactions."""

    def __init__(
        self,
        n_features: int,
        pos_weight: float,
        hidden: Sequence[int] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        size_in = n_features
        for size_out in hidden:
            layers.extend([nn.Linear(size_in, size_out), nn.ReLU(), nn.Dropout(dropout)])
            size_in = size_out
        layers.append(nn.Linear(size_in, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class LitLinearReg(_BaseLit):
    """Linear regression = one linear layer with MSE loss (MLE under Gaussian noise)."""

    def __init__(self, n_features: int, lr: float = 1e-2, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(n_features, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)
