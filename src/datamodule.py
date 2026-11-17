"""Shared PyTorch Lightning DataModule and neural architectures for Task 1 and Task 2."""
from __future__ import annotations

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from . import config as C


def make_loader(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 1024,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap float32 numpy arrays into a vectorized-batch PyTorch DataLoader."""
    x_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
    ds = (
        TensorDataset(x_tensor)
        if y is None
        else TensorDataset(x_tensor, torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32))
    )
    base_sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    batch_sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=False)
    return DataLoader(ds, batch_size=None, sampler=batch_sampler)


class LoanDataModule(L.LightningDataModule):
    """Lightning DataModule handing batched fold arrays with customer-grouped early stopping."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_train: np.ndarray | None = None,
        val_fraction: float = 0.1,
        batch_size: int = 1024,
        seed: int = C.SEED,
    ) -> None:
        super().__init__()
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.groups_train = groups_train
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.X_fit: np.ndarray | None = None
        self.y_fit: np.ndarray | None = None
        self.X_stop: np.ndarray | None = None
        self.y_stop: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.groups_train is not None and len(np.unique(self.groups_train)) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed)
            fit_idx, stop_idx = next(splitter.split(self.X_train, self.y_train, self.groups_train))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.seed)
            fit_idx, stop_idx = next(splitter.split(self.X_train, self.y_train))
        self.X_fit, self.y_fit = self.X_train[fit_idx], self.y_train[fit_idx]
        self.X_stop, self.y_stop = self.X_train[stop_idx], self.y_train[stop_idx]

    def train_dataloader(self) -> DataLoader:
        assert self.X_fit is not None and self.y_fit is not None
        return make_loader(self.X_fit, self.y_fit, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.X_stop is not None and self.y_stop is not None
        return make_loader(self.X_stop, self.y_stop, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return make_loader(self.X_val, None, batch_size=self.batch_size, shuffle=False)


class LitLogReg(L.LightningModule):
    """Single-layer logistic regression trained with weighted BCEWithLogitsLoss (MLE)."""

    def __init__(
        self,
        n_features: int,
        pos_weight: float,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(n_features, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class LitMLP(L.LightningModule):
    """Two-hidden-layer MLP with ReLU and Dropout for Task 2 default classification."""

    def __init__(
        self,
        n_features: int,
        pos_weight: float,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        size_in = n_features
        for size_out in hidden:
            layers.extend([nn.Linear(size_in, size_out), nn.ReLU(), nn.Dropout(dropout)])
            size_in = size_out
        layers.append(nn.Linear(size_in, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class LitLinearReg(L.LightningModule):
    """Single-layer linear regression trained with MSELoss (Gaussian MLE) for Task 1."""

    def __init__(
        self,
        n_features: int,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(n_features, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", self.loss_fn(self(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def fit_predict_lightning_clf(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    groups_train: np.ndarray | None = None,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 1024,
    seed: int = C.SEED,
) -> np.ndarray:
    """Fit LitLogReg or LitMLP deterministically on training rows and return probabilities."""
    L.seed_everything(seed, workers=False)
    dm = LoanDataModule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=np.zeros(len(X_val), dtype=np.float32),
        groups_train=groups_train,
        batch_size=batch_size,
        seed=seed,
    )
    dm.setup()
    assert dm.y_fit is not None
    n_pos = float(np.sum(dm.y_fit == 1))
    n_neg = float(np.sum(dm.y_fit == 0))
    pos_weight = n_neg / max(n_pos, 1.0)

    if model_type == "lit_logreg":
        model: L.LightningModule = LitLogReg(X_train.shape[1], pos_weight=pos_weight)
    elif model_type == "lit_mlp":
        model = LitMLP(X_train.shape[1], pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported Lightning classifier '{model_type}'.")

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.astype(np.float64)


def fit_predict_lightning_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    groups_train: np.ndarray | None = None,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 1024,
    seed: int = C.SEED,
) -> np.ndarray:
    """Fit LitLinearReg deterministically on standardized log-target and return raw predictions."""
    L.seed_everything(seed, workers=False)
    dm = LoanDataModule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=np.zeros(len(X_val), dtype=np.float32),
        groups_train=groups_train,
        batch_size=batch_size,
        seed=seed,
    )
    dm.setup()
    model = LitLinearReg(X_train.shape[1])
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32)).cpu().numpy()
    return preds.astype(np.float64)
