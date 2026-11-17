# AI-use log

The report needs an AI-use declaration, and each of us must be able to explain and reproduce our work without AI. Log substantial AI use here as it happens.

| Date | Who | Tool | What for | What we checked or changed | Understood? |
| --- | --- | --- | --- | --- | --- |
| 2026-11-07 | Perfect | Claude Code | Drafted the docs system and README tracking section | Checked against the brief PDF; open items are proposals | Yes |
| 2026-11-10 | Henry | Claude Code | Explained P2's role, logistic regression, ROC-AUC/PR-AUC and shared features; gave step-by-step notebook code (scikit-learn, shared features, Lightning); diagnosed the Mac MPS/float32 error and the wrong notebook kernel; drafted the explainer in `docs/learning/`, PR #41 text and these log entries | Typed and ran every cell myself with Run All on `.venv`; results match Claude's test runs (0.845 / 0.170); the Lightning and scikit-learn versions agree | Yes |
| 2026-11-16 | Bernard | Antigravity AI | Rescaffolded project with shared `src/` pipeline (`preprocess`, `datamodule`, `train`, `evaluate`, `fairness`, `score`), automated `tests/` guardrails, GitHub Actions (`ci.yml`, `pr-issue-check.yml`), `.pre-commit-config.yaml`, `Makefile`, and streamlined `README.md` / `docs/DATA_AND_PIPELINE_GUIDE.md` on `p3/13-30-senior-scaffolding` | Reviewed every module, verified zero edits to P1/P2 files, ran `ruff`, `pytest`, `src.train`, `src.evaluate`, and `src.score` end-to-end on `features.parquet` | Yes |
| 2026-11-17 | Bernard | Antigravity AI | Converted `explore_target_henry.ipynb` into `src/eda_target.py`, refactored `02_classification.ipynb`, and created thin script-backed `03_regression.ipynb` and `04_transfer_and_fairness.ipynb` so only the 4 Page-6 deliverable notebooks remain | Executed `src.eda_target`, ran all 3 deliverable notebooks top-to-bottom via `jupyter nbconvert --execute`, and verified `ruff` + `pytest` | Yes |
