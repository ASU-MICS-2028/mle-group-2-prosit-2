# Individual Reflection: Henry Kofi Brandoh

Role: P2, Classification and Evaluation · Project: NkwaPay credit scoring (ICS553 Prosit 2) · Notebook: `02_classification.ipynb`

## 1. My contribution

I owned Task 2, predicting whether a borrower will default, and the evaluation that decides whether those predictions can be trusted. I built a logistic regression baseline (ROC-AUC 0.79, PR-AUC 0.19), rebuilt it on the team's shared features, and implemented it a second time by hand in PyTorch Lightning, where matching scores (≈ 0.845) confirmed the training loop was correct (PRs #40, #41). I added an MLP (PR #45) and a validation study comparing random, by-customer and by-time splits (#14). I then wrote the problem framing, a cost matrix in cedis measured from Ghana's own loans, and a study of whether calibration holds on 2023+ loans (PRs #60, #61), and the logistic regression derivation, checked against PyTorch autograd (PR #62). Bernard added the calibration, threshold and confidence-interval sections; I reviewed his restructuring (PR #47) and reran and verified them.

## 2. What I learned

I began unsure of the basics. My first questions were what "classification and evaluation" meant, what a default is, why we train on `Train.csv`, and what ROC-AUC and PR-AUC measure.

The lesson that changed my thinking most is that accuracy can lie. Only 1.83% of loans default, so a model that approves everyone is 98% accurate and catches no defaulter at all. I learned to ask whether a model finds the rare cases, which is what PR-AUC measures, and to compare it with the default rate (0.018) rather than with 0.5: my baseline's 0.19 was about ten times better than chance. I also found that two repayment columns predict default almost perfectly (AUC 0.98) only because they are recorded *after* the outcome. This changed how I read results: before this prosit I would have trusted a high score; now a score that looks too good is the first thing I question, and I check what the model could not have known at decision time.

That habit explained the later results. Models scoring about 0.85 on cross-validation fell to about 0.63–0.65 on loans from 2023 onwards, because the main lender's share fell from 97% to 49%, loans became 2.5 times longer and defaults rose from 1.3% to 9.5%. Weighting defaults made raw probabilities average about 0.30 against a true 0.018, which my derivation traced to a shift of log 54 ≈ 4 in every log-odds, so calibration was essential. Tied to Ghana's real costs (losing 90 GHS by refusing a good borrower versus 2,416 GHS by approving a defaulter), the honest rule is to decline above 3.6% risk. On 2024 loans that meant declining everyone, which showed me that some problems are pricing decisions for the business, not modelling problems.

## 3. What was hard

My notebook first ran on the wrong Python environment, and later the code crashed on my Mac because two libraries clashed. Git was the bigger challenge. My first PR was merged before my follow-up work was pushed, so `main` showed an outdated version, and later a teammate's PR rewrote my notebook before it was restored. At first I did not fully understand what had happened to my work. That confusion pushed me to learn how branches, pull requests and merges actually work, so that I can protect my work rather than be surprised by changes to it. I also recognise that under deadline pressure I merged some of my own PRs without a recorded review, the same shortcut that had affected me.

## 4. How I used AI

I used Claude Code throughout, and every use is recorded in `docs/AI_USE_LOG.md`. It explained concepts in plain language, gave step-by-step code (early on I typed and ran every cell myself and compared my results with its test runs), and diagnosed environment errors. In the final days it wrote larger parts of the notebook, the derivation and the documentation, which I checked against the outputs. It also made mistakes that checking caught, such as an early claim that the MLP was clearly better than logistic regression, which customer-level confidence intervals did not support. Honestly, I can now explain the concepts myself (PR-AUC, leakage, the time split, calibration and the cost threshold), but I could not yet rewrite all of the later code alone, and I need to revise the derivation before the debrief.

## 5. What I would do differently

Ask about the basics on day one. I spent the first part of the project working out what my role and the core metrics meant; clarifying them at the start would have let me contribute to decisions sooner. Test on future data from the start. The by-time split changed almost every conclusion, from which model looked best to what we could promise NkwaPay. Running it early would have shaped which models I trusted and which claims I made.
