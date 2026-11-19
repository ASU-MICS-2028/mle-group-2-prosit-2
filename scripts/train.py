"""Train models on the shared folds and write predictions to preds/ (and, optionally, the scorer artifacts).

    python -m scripts.train --task task2 --models majority logreg lightgbm --schemes fold_group temporal
    python -m scripts.train --task task1 --models mean ridge lightgbm_reg --schemes fold_group
    python -m scripts.train --save-artifacts            # calibrated scorer bundle in artifacts/

Model names and settings: src/models/__init__.py and configs/models.yaml.
"""
from __future__ import annotations

import argparse

from src import config as C
from src.console import console
from src.models import TASK1_MODELS, TASK2_MODELS
from src.processing import load_features
from src.training import fit_and_save_deployment_artifacts, predict_test_split, run_cv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train shared Task 1 and Task 2 models.")
    p.add_argument("--task", choices=["task1", "task2"], default="task2")
    p.add_argument("--models", nargs="+", default=None, help=f"task2: {TASK2_MODELS}; task1: {TASK1_MODELS}")
    p.add_argument("--schemes", nargs="+", default=["fold_group"], choices=["fold_group", "fold_strat", "temporal"])
    p.add_argument("--predict-test", action="store_true", help="also fit on all of Train and predict Test")
    p.add_argument("--save-artifacts", action="store_true", help="save the calibrated scorer bundle")
    p.add_argument("--clf-artifact-model", default="logreg")
    p.add_argument("--reg-artifact-model", default="lightgbm_reg_q85")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    explicit_models = args.models is not None
    models = args.models or (["majority", "logreg"] if args.task == "task2" else ["mean", "ridge"])
    valid = TASK2_MODELS if args.task == "task2" else TASK1_MODELS
    unknown = sorted(set(models) - set(valid))
    if unknown:
        raise SystemExit(f"Unknown {args.task} models {unknown}. Options: {valid}")

    df = load_features()
    jobs = [] if (args.save_artifacts and not explicit_models) else [
        (m, s) for m in models for s in args.schemes + (["test"] if args.predict_test else [])
    ]
    if jobs:
        console.title(f"Train {args.task}: {len(models)} model(s) x {len(args.schemes)} scheme(s)")
        for i, (model, scheme) in enumerate(jobs, 1):
            with console.step(f"[{i:>2}/{len(jobs)}] {model:<18} {scheme}"):
                if scheme == "test":
                    predict_test_split(df, model_name=model, task=args.task)
                else:
                    run_cv(df, model_name=model, task=args.task, scheme=scheme)
    if args.save_artifacts:
        console.title("Scorer artifacts")
        with console.step(f"calibrated {args.clf_artifact_model} + {args.reg_artifact_model} limits"):
            paths = fit_and_save_deployment_artifacts(df, clf_model=args.clf_artifact_model, reg_model=args.reg_artifact_model)
    summary = [f"{len(jobs)} prediction file(s) in {C.rel(C.PREDS)}"] if jobs else []
    if args.save_artifacts:
        summary.append(f"{paths['clf_path'].name} and {paths['reg_path'].name} in {C.rel(C.ARTIFACTS)}")
    console.done("; ".join(summary))


if __name__ == "__main__":
    main()
