# Individual reflection · Henry Kofi Brandoh (P2, classification and evaluation)

## My contribution

I owned Task 2, predicting whether a borrower will default, and its evaluation, in `notebooks/02_classification.ipynb`: a logistic regression baseline, rebuilt on the team's shared features and by hand in PyTorch Lightning (both ROC-AUC ≈ 0.845; PRs #40, #41); an MLP (PR #45); a validation study comparing random, by-customer and by-time splits (#14); the problem framing, a cost matrix in cedis measured from Ghana's own loans (decline above 3.6% risk), and a study of whether calibration holds on 2023+ loans (PRs #60, #61); and the logistic regression derivation with an autograd check (PR #62). Bernard added the calibration, threshold and confidence-interval sections (§7–10); I reviewed his restructuring (PR #47) and reran and verified them.

## What I learned

I started unsure of the basics. My first questions were what "classification and evaluation" meant, what a default is, why we train on `Train.csv`, and what ROC-AUC and PR-AUC measure.

The lesson that changed my thinking most was that **accuracy can lie**. Only 1.83% of loans default, so a model that approves everyone is 98% accurate and completely useless. I now judge a model by whether it finds the rare cases, using PR-AUC compared with the default rate (0.018), not with 0.5. I also saw that two repayment columns predict default almost perfectly (AUC 0.98) only because they are recorded after the outcome. Before this prosit I would have trusted a high score; now a score that looks too good is the first thing I question.

Other lessons followed from that. Models scoring about 0.85 on cross-validation fell to about 0.63–0.65 on loans from 2023 onwards, because the lenders, loan terms and borrowers changed. Weighting defaults made raw probabilities far too high, so calibration was needed before any decision. Tied to Ghana's real costs, the honest rule is to decline anyone above 3.6% risk, and on 2024 loans that meant declining everyone: a pricing question for the business, not a modelling one.

## What was hard

My notebook first ran on the wrong Python environment, and later the code crashed on my Mac because two libraries clashed. Git was the bigger challenge. My first PR was merged before my follow-up work was pushed, and later a teammate's PR rewrote my notebook before it was restored. At first I did not fully understand what had happened to my work, and that pushed me to learn how branches, pull requests and merges actually work, so I can protect my work instead of being confused by it. Under deadline pressure I also merged some of my own later PRs without a recorded review, which I would not repeat.

## How I used AI

I used Claude Code throughout, recorded in `docs/AI_USE_LOG.md`: for plain-language explanations, step-by-step code (early on I typed and ran every cell myself), debugging, and in the final days for larger parts of the notebook, the derivation and the documentation, which I checked against the outputs. It also made mistakes that checking caught, such as an early claim that the MLP was clearly better, which customer-level confidence intervals did not support. Honestly, I can now explain the concepts myself (PR-AUC, leakage, the time split, calibration and the cost threshold), but I could not rewrite all of the later code alone, and I need to revise the derivation before the debrief.

## What I would do differently

First, **ask about the basics on day one**: clarify my role and the core ideas at the start instead of picking them up mid-project. Second, **test on future data from the start**: the by-time split changed every conclusion, and running it early would have shaped which models and claims I trusted.
