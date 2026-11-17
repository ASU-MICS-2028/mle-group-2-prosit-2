"""Minimal, consistent terminal output for every command in scripts/.

    from src.console import console
    console.title("Train models")
    with console.step("task2 logreg fold_group"):
        ...
    console.done("27 prediction files in preds/")

Each step prints one line when it finishes:  `  ✓ task2 logreg fold_group   4.1s`.
In an interactive terminal the line shows `…` while the step runs, then is replaced.
Library warnings (sklearn, LightGBM, Lightning) are silenced so only our lines appear.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager

WIDTH = 58


class _Console:
    def __init__(self) -> None:
        self._configured = False
        self.tty = sys.stdout.isatty()

    def setup(self) -> None:
        if self._configured:
            return
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        warnings.filterwarnings("ignore")
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        for name in ("lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric", "matplotlib", "shap"):
            logging.getLogger(name).setLevel(logging.ERROR)
        self._configured = True

    def title(self, text: str) -> None:
        self.setup()
        print(f"\n{text}\n{'─' * len(text)}", flush=True)

    @contextmanager
    def step(self, label: str):
        self.setup()
        t0 = time.perf_counter()
        if self.tty:
            print(f"  … {label}", end="\r", flush=True)
        try:
            yield
        except BaseException:
            print(f"  ✗ {label:<{WIDTH}} failed", flush=True)
            raise
        print(f"  ✓ {label:<{WIDTH}} {time.perf_counter() - t0:5.1f}s", flush=True)

    def info(self, text: str) -> None:
        print(f"    {text}", flush=True)

    def table(self, df, title: str | None = None, max_rows: int = 20, floatfmt: str = "{:,.3f}") -> None:
        """A compact, aligned table (floats rounded)."""
        if title:
            print(f"\n  {title}")
        shown = df.head(max_rows).copy()
        for col in shown.columns:
            if shown[col].dtype.kind == "f":
                shown[col] = shown[col].map(lambda v: "" if v != v else floatfmt.format(v))
        for line in shown.to_string(index=False).splitlines():
            print(f"    {line}")
        if len(df) > max_rows:
            print(f"    … {len(df) - max_rows} more rows")

    def done(self, text: str) -> None:
        print(f"\n✔ {text}\n", flush=True)


console = _Console()
