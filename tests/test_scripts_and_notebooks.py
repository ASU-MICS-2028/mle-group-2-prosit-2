"""Entry points exist and validate their inputs; notebooks run top to bottom (opt-in, needs real data)."""
from __future__ import annotations

import importlib
import os
import sys

import pytest

from src import config as C

SCRIPTS = ["build_features", "run_target_eda", "train", "evaluate", "make_figures", "make_eda_report", "score"]
NOTEBOOKS = ["01_eda", "02_classification", "03_regression", "04_transfer_and_fairness"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_has_a_main(name: str) -> None:
    module = importlib.import_module(f"scripts.{name}")
    assert callable(module.main)
    assert module.__doc__ and "python -m scripts." in module.__doc__, "docstring must say how to run it"


def test_train_rejects_unknown_models(monkeypatch) -> None:
    from scripts import train

    monkeypatch.setattr(sys, "argv", ["train", "--task", "task2", "--models", "not_a_model"])
    monkeypatch.setattr(train, "load_features", lambda: None)
    with pytest.raises(SystemExit, match="Unknown task2 models"):
        train.main()


def test_makefile_targets_point_at_existing_scripts() -> None:
    text = (C.ROOT / "Makefile").read_text(encoding="utf-8")
    for name in SCRIPTS:
        if f"scripts.{name}" in text:
            assert (C.ROOT / "scripts" / f"{name}.py").exists()
    for target in ("features", "train", "eval", "eda", "figures", "score", "test", "lint", "all"):
        assert f"\n{target}:" in text or text.startswith(f"{target}:"), f"make {target} missing"


@pytest.mark.skipif(os.environ.get("RUN_NOTEBOOKS") != "1" or not C.FEATURES_PARQUET.exists(),
                    reason="set RUN_NOTEBOOKS=1 with real data (make test-notebooks)")
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_top_to_bottom(name: str) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(C.ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(C.ROOT / "notebooks")}}).execute()
