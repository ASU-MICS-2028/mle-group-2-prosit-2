"""Fit a Lightning model on one fold and return its predictions, with the same loop for every model."""
from __future__ import annotations

import logging
import warnings

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping

from .. import config as C
from ..models import model_params
from ..models.data_module import LoanDataModule
from ..models.lightning_modules import LitLinearReg, LitLogReg, LitMLP


def _quiet_lightning() -> None:
    """Silence Lightning's per-fit banners and tips so notebooks show results, not log spam."""
    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")


def _fit(model: L.LightningModule, dm: LoanDataModule, max_epochs: int, patience: int) -> L.LightningModule:
    _quiet_lightning()
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
    return model


def _datamodule(X_train, y_train, X_predict, groups_train, seed) -> tuple[LoanDataModule, dict]:
    loop = model_params("lightning")
    dm = LoanDataModule(
        X_train,
        y_train,
        X_predict,
        groups_train=groups_train,
        early_stop_fraction=loop["early_stop_fraction"],
        batch_size=loop["batch_size"],
        seed=seed,
    )
    dm.setup()
    return dm, loop


def fit_predict_lightning_clf(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    groups_train: np.ndarray | None = None,
    seed: int = C.SEED,
) -> np.ndarray:
    """Train LitLogReg or LitMLP and return default probabilities for X_predict."""
    L.seed_everything(seed, workers=False, verbose=False)
    dm, loop = _datamodule(X_train, y_train, X_predict, groups_train, seed)
    pos_weight = float(np.sum(dm.y_fit == 0)) / max(float(np.sum(dm.y_fit == 1)), 1.0)
    params = model_params(model_name, "task2")
    if model_name == "lit_logreg":
        model: L.LightningModule = LitLogReg(X_train.shape[1], pos_weight=pos_weight, **params)
    elif model_name == "lit_mlp":
        model = LitMLP(X_train.shape[1], pos_weight=pos_weight, **params)
    else:
        raise ValueError(f"Unsupported Lightning classifier '{model_name}'.")
    model = _fit(model, dm, loop["max_epochs"], loop["patience"])
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X_predict, dtype=np.float32)))
    return torch.sigmoid(logits).numpy().astype(np.float64)


def fit_predict_lightning_reg(
    X_train: np.ndarray,
    z_train: np.ndarray,
    X_predict: np.ndarray,
    groups_train: np.ndarray | None = None,
    seed: int = C.SEED,
) -> np.ndarray:
    """Train LitLinearReg on the standardized log target and return standardized predictions."""
    L.seed_everything(seed, workers=False, verbose=False)
    dm, loop = _datamodule(X_train, z_train, X_predict, groups_train, seed)
    model = _fit(LitLinearReg(X_train.shape[1], **model_params("lit_linear", "task1")), dm,
                 loop["max_epochs"], loop["patience"])
    with torch.no_grad():
        preds = model(torch.tensor(np.asarray(X_predict, dtype=np.float32)))
    return preds.numpy().astype(np.float64)
