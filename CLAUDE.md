# Instructions for Claude Code in this repo

This is the ICS553 Prosit 2 group project (see README.md). Four people share it, so the docs must always match the code.

## Docs ship with code (required)

Before you tell the user a task is done, or draft a commit or PR, update every doc the change touches:

| If the change… | Update |
| --- | --- |
| adds or changes code, a notebook or results | `docs/WORKLOG.md` (new entry at the top) and `docs/PROGRESS.md` (status rows, done table, session log, "Last updated") |
| changes a notebook's state | the status row in README.md "Status" table |
| makes or changes a data or modelling choice | `docs/DECISIONS.md` (new row; mark old ones `superseded`) |
| finishes part of a learning outcome or deliverable | tick it in `docs/LEARNING_OUTCOMES.md` and `docs/DELIVERABLES.md` |
| finishes a step in someone's plan | tick it in `docs/P*_PLAN.md` |
| uses AI for substantial work | `docs/AI_USE_LOG.md` |

Never leave a title like "uncommitted" or "not yet pushed" in the docs after the work is pushed. Describe state as of the last merge to `main`, and name the PR.

## Workflow
One issue, one branch (`<role>/<issue>-<task>`), one pull request; nothing goes straight to `main`. The GitHub Action `Docs freshness` fails a PR that changes `src/`, `notebooks/` or `requirements.txt` without touching `docs/WORKLOG.md` and `docs/PROGRESS.md`.
