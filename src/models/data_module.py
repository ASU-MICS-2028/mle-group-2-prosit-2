"""Lightning DataModule: hands one fold's preprocessed arrays to any Lightning model in batches."""
from __future__ import annotations

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset

from .. import config as C


def make_loader(
    X: np.ndarray, y: np.ndarray | None = None, batch_size: int = 1024, shuffle: bool = False
) -> DataLoader:
    """Wrap float32 arrays in a DataLoader that slices whole batches at once (fast on CPU)."""
    tensors = [torch.tensor(np.asarray(X, dtype=np.float32))]
    if y is not None:
        tensors.append(torch.tensor(np.asarray(y, dtype=np.float32)))
    ds = TensorDataset(*tensors)
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, batch_size=None, sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False))


class LoanDataModule(L.LightningDataModule):
    """Training rows are split again by customer: 90% to fit, 10% to decide when to stop early.

    The validation fold is never seen during training; it is only scored at the end (predict).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_predict: np.ndarray,
        groups_train: np.ndarray | None = None,
        early_stop_fraction: float = 0.1,
        batch_size: int = 1024,
        seed: int = C.SEED,
    ) -> None:
        super().__init__()
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.X_predict = np.asarray(X_predict, dtype=np.float32)
        self.groups_train = groups_train
        self.early_stop_fraction = early_stop_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.X_fit = self.y_fit = self.X_stop = self.y_stop = None

    def setup(self, stage: str | None = None) -> None:
        if self.X_fit is not None:
            return
        if self.groups_train is not None and len(np.unique(self.groups_train)) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.early_stop_fraction, random_state=self.seed)
            fit_idx, stop_idx = next(splitter.split(self.X_train, self.y_train, self.groups_train))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=self.early_stop_fraction, random_state=self.seed)
            fit_idx, stop_idx = next(splitter.split(self.X_train, self.y_train))
        self.X_fit, self.y_fit = self.X_train[fit_idx], self.y_train[fit_idx]
        self.X_stop, self.y_stop = self.X_train[stop_idx], self.y_train[stop_idx]

    def train_dataloader(self) -> DataLoader:
        return make_loader(self.X_fit, self.y_fit, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return make_loader(self.X_stop, self.y_stop, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return make_loader(self.X_predict, None, batch_size=self.batch_size, shuffle=False)
