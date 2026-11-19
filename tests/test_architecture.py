"""Every model is the architecture we say it is: layers, losses, class weighting, settings from config."""
from __future__ import annotations

import lightning as L
import numpy as np
import pytest
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import nn

from src.models import build_estimator, model_params
from src.models.data_module import LoanDataModule, make_loader
from src.models.lightning_modules import LitLinearReg, LitLogReg, LitMLP
from src.models.svm import SubsampledSVC, stratified_subsample
from src.training.lightning_runner import fit_predict_lightning_clf, fit_predict_lightning_reg

# ----------------------------------------------------------------- registry -> estimator types
EXPECTED = {
    ("task2", "majority"): DummyClassifier,
    ("task2", "logreg"): LogisticRegression,
    ("task2", "knn"): KNeighborsClassifier,
    ("task2", "svm_rbf"): SubsampledSVC,
    ("task2", "decision_tree"): DecisionTreeClassifier,
    ("task2", "random_forest"): RandomForestClassifier,
    ("task2", "lightgbm"): LGBMClassifier,
    ("task1", "mean"): DummyRegressor,
    ("task1", "ols"): LinearRegression,
    ("task1", "ridge"): Ridge,
    ("task1", "lasso"): Lasso,
    ("task1", "decision_tree_reg"): DecisionTreeRegressor,
    ("task1", "lightgbm_reg"): LGBMRegressor,
    ("task1", "lightgbm_reg_q85"): LGBMRegressor,
}


@pytest.mark.parametrize(("task", "name"), list(EXPECTED))
def test_registry_builds_the_right_estimator_with_config_settings(task: str, name: str) -> None:
    est = build_estimator(name, task, pos_weight=40.0)
    assert isinstance(est, EXPECTED[(task, name)])
    params = est.get_params()
    for key, value in model_params(name, task).items():
        assert params[key] == value, f"{name}.{key} does not match configs/models.yaml"


def test_imbalance_is_handled_in_every_classifier_loss() -> None:
    for name in ("logreg", "decision_tree", "random_forest"):
        assert build_estimator(name, "task2").get_params()["class_weight"] == "balanced"
    assert build_estimator("lightgbm", "task2", pos_weight=53.6).get_params()["scale_pos_weight"] == 53.6
    assert build_estimator("svm_rbf", "task2").fit(*_toy(400)).svc_.estimator.class_weight == "balanced"


def test_trees_are_pruned() -> None:
    for name, task in (("decision_tree", "task2"), ("decision_tree_reg", "task1")):
        p = build_estimator(name, task).get_params()
        assert p["max_depth"] is not None and p["ccp_alpha"] > 0


def _toy(n: int = 300, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 1.2).astype(np.float32)
    return X, y


def test_svm_subsample_keeps_the_default_rate_and_size() -> None:
    X, y = _toy(5000)
    Xs, ys = stratified_subsample(X, y, max_rows=1000)
    assert len(Xs) == 1000
    assert abs(ys.mean() - y.mean()) < 0.01
    probs = SubsampledSVC(max_train_rows=500).fit(X, y).predict_proba(X[:10])
    assert probs.shape == (10, 2) and np.allclose(probs.sum(axis=1), 1)


# ----------------------------------------------------------------- Lightning architectures
def test_logistic_regression_is_one_linear_layer() -> None:
    m = LitLogReg(n_features=6, pos_weight=54.0)
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 1 and linears[0].in_features == 6 and linears[0].out_features == 1
    assert sum(p.numel() for p in m.parameters()) == 6 + 1  # one weight per input + bias
    assert isinstance(m.loss_fn, nn.BCEWithLogitsLoss)
    assert float(m.loss_fn.pos_weight) == pytest.approx(54.0)
    assert m(torch.zeros(5, 6)).shape == (5,)  # logits, one per row


def test_logistic_loss_is_weighted_negative_log_likelihood() -> None:
    m = LitLogReg(n_features=1, pos_weight=3.0)
    logits, y = torch.tensor([0.4, -1.2]), torch.tensor([1.0, 0.0])
    p = torch.sigmoid(logits)
    manual = -(3.0 * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    assert float(m.loss_fn(logits, y)) == pytest.approx(float(manual), rel=1e-6)


def test_mlp_layers_match_config_and_dropout_is_off_when_predicting() -> None:
    cfg = model_params("lit_mlp", "task2")
    m = LitMLP(n_features=10, pos_weight=5.0, **cfg)
    linears = [mod for mod in m.net if isinstance(mod, nn.Linear)]
    assert [lin.out_features for lin in linears] == [*cfg["hidden"], 1]
    assert linears[0].in_features == 10
    assert sum(isinstance(mod, nn.ReLU) for mod in m.net) == len(cfg["hidden"])
    assert all(mod.p == cfg["dropout"] for mod in m.net if isinstance(mod, nn.Dropout))
    x = torch.randn(64, 10)
    m.train()
    assert not torch.allclose(m(x), m(x)), "dropout should randomise outputs during training"
    m.eval()
    assert torch.allclose(m(x), m(x)), "dropout must be off when predicting"


def test_linear_regression_uses_mse() -> None:
    m = LitLinearReg(n_features=4)
    assert isinstance(m.loss_fn, nn.MSELoss)
    assert sum(p.numel() for p in m.parameters()) == 5


def test_optimizer_is_adam_with_weight_decay_from_config() -> None:
    cfg = model_params("lit_logreg", "task2")
    opt = LitLogReg(3, 1.0, **cfg).configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == cfg["lr"] and opt.defaults["weight_decay"] == cfg["weight_decay"]


def test_datamodule_early_stopping_split_never_shares_a_customer() -> None:
    X, y = _toy(1000)
    groups = np.repeat(np.arange(200), 5)
    dm = LoanDataModule(X, y, X[:10], groups_train=groups, early_stop_fraction=0.1)
    dm.setup()
    fit_rows = {tuple(r) for r in dm.X_fit}
    assert len(dm.X_fit) + len(dm.X_stop) == len(X)
    assert not any(tuple(r) in fit_rows for r in dm.X_stop)
    batch = next(iter(make_loader(X, y, batch_size=128, shuffle=True)))
    assert batch[0].shape == (128, 6) and batch[0].dtype == torch.float32


def test_lightning_models_learn_a_simple_signal() -> None:
    X, y = _toy(2000)
    groups = np.arange(2000)
    from sklearn.metrics import roc_auc_score

    for name in ("lit_logreg", "lit_mlp"):
        p = fit_predict_lightning_clf(name, X[:1500], y[:1500], X[1500:], groups_train=groups[:1500])
        assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()
        assert roc_auc_score(y[1500:], p) > 0.85, name
    z = X[:, 0] * 2 + 1
    pred = fit_predict_lightning_reg(X[:1500], z[:1500], X[1500:], groups_train=groups[:1500])
    assert np.corrcoef(pred, z[1500:])[0, 1] > 0.95


def test_lightning_training_is_reproducible() -> None:
    X, y = _toy(800)
    a = fit_predict_lightning_clf("lit_logreg", X[:600], y[:600], X[600:])
    b = fit_predict_lightning_clf("lit_logreg", X[:600], y[:600], X[600:])
    assert np.allclose(a, b)
    L.seed_everything(0, verbose=False)
