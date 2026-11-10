# Logistic regression baseline, explained simply

**Author:** Henry (P2, classification and evaluation) · **Issue:** #16 · **Branch:** `p2/logreg-baseline` · **Notebook:** `notebooks/02_classification.ipynb`

This note explains what the logistic regression baseline does, why each step is there, and how to read its scores. No maths background needed.

## Status

| Part | State |
| --- | --- |
| Data loading (Step 1) | ✅ In the notebook |
| scikit-learn baseline (Steps 2–6) | ✅ In the notebook; results below |
| Switch to Perfect's shared features and folds (PR #39) | ⬜ Next |
| PyTorch Lightning version | ⬜ After that |

---

## 1. What problem are we solving?

A lender receives a loan application and must decide: **will this person pay us back?**

A **default** means the borrower **didn't pay the loan back**. In the data, `target = 1` means defaulted and `target = 0` means repaid. About 1 in 55 loans defaulted.

We want a model that looks at a loan's details and gives a **probability of default**: how likely it is that the money won't come back. For example, "this loan has a 12% chance of default". This is called **classification**: putting each loan into one of two groups, *defaults* (1) or *repays* (0).

Checking how trustworthy those predictions are is called **evaluation**. Both are P2's job.

## 2. How logistic regression thinks

Think of it as a **points system**.

| Fact about the loan | Points (made-up example) |
| --- | --- |
| Borrower is new (first loan) | +3 (risky) |
| Borrower is repeat | −1 (safer) |
| Loan is large | +0.5 |
| Lender A | +1 |

1. Add up the points for a loan to get a **risk score**.
2. Pass the score through an S-shaped curve (the **sigmoid**), which turns any number into a probability between 0 and 1. Very negative scores give close to 0%, a score of 0 gives 50%, and very positive scores give close to 100%.

**We don't choose the points ourselves.** The model learns them from past loans whose outcome we already know. That learning is called **training**:

> guess each loan's probability → compare with what really happened → adjust the points to be less wrong → repeat.

The learned points are called **weights**. A big positive weight pushes towards "default"; a big negative weight pushes towards "repays".

## 3. Which data, and why

**We train on `Train.csv`** (68,654 loans, 1,258 defaults, i.e. 1.83%).

- **Why `Train.csv`?** It's the only file with the answer column (`target`). A model can only learn from examples where it can check its guesses.
- **Why not `Test.csv`?** It has no `target` column. It's only used at the very end to make the competition submission.

## 4. The steps, one by one

### Step 1 · Load the data
Read `data/raw/Train.csv` and check its size (68,654 rows) and the default rate (1.83%).
**Why:** 1.83% is the baseline to beat. It also warns us that defaults are rare, which affects every later choice.

### Step 2 · Split clues (X) from the answer (y)

| Used as clues (features) | Why |
| --- | --- |
| `Total_Amount` | Loan size |
| `duration` | Loan length in days |
| `Lender_portion_Funded` | The lender's share of the loan |
| `New_versus_Repeat` | Big signal: about 21% of new borrowers default vs 1.7% of repeat borrowers |
| `loan_type`, `lender_id`, `country_id` | Kind of loan, who lent it, which country |

| Left out on purpose | Why |
| --- | --- |
| `Total_Amount_to_Repay`, `Lender_portion_to_be_repaid` | **Leaks.** Each one alone predicts default almost perfectly (AUC 0.98), most likely because late penalties are added *after* someone defaults. Using them is like reading the last page of the novel: the model would look brilliant in testing and fail in real life. |
| `ID`, `customer_id`, `tbl_loan_id` | Just labels for rows, customers and loans, with no information about risk. (`customer_id` is kept aside for splitting; see Step 5.) |
| `due_date`, `Amount_Funded_By_Lender` | Can be calculated from other columns, so they add nothing new. |

The answer (y) is `target`: 1 = defaulted, 0 = repaid.

`lender_id` is turned into text so the model treats it as a *name*. Otherwise it would think lender 267278 is "bigger" than lender 100.

### Step 3 · Clean up the clues (preprocessing)

| What we do | Why |
| --- | --- |
| **Log of money amounts** (`log1p`) | A few loans are millions while most are a few thousand. The log shrinks the giants so they don't dominate. |
| **Scaling** (`StandardScaler`) | Puts every number on the same scale, so no feature matters more just because its numbers are bigger. |
| **One-hot encoding** of categories | The model only understands numbers. "Lender A" becomes its own column of 1s and 0s. |
| `handle_unknown="ignore"` | Some Ghana lenders and loan types never appear in training. This stops the model crashing when it meets one. |

### Step 4 · Build the model
A scikit-learn `Pipeline` that does the preprocessing, then `LogisticRegression(class_weight="balanced")`.

- **Why a pipeline?** The cleaning is re-learned from the training rows only each time. If the scaler peeked at the test rows, information would leak and the score would look better than it is.
- **Why `class_weight="balanced"`?** Only about 1 in 55 loans defaults. Without help, the model can score 98% accuracy by saying "nobody defaults", which is useless. Balanced weighting makes each default count about 54× more. (The Lightning version will do the same with `pos_weight ≈ 53.6`.)

### Step 5 · Test it fairly (cross-validation, grouped by customer)

1. Split the loans into **5 groups** (folds).
2. Train on 4 groups and test on the 1 the model hasn't seen.
3. Repeat 5 times so each group is the test once, then average the scores.

We use **`GroupKFold` by `customer_id`**: all of one customer's loans stay in the same group.
**Why:** otherwise the model could be tested on a borrower it already studied. That's like seeing the exam questions in advance, and the score would be too optimistic.

### Step 6 · Score it
We report two scores (explained in Section 5).

## 5. Results (rough features, from `02_classification.ipynb`)

| Metric | Average over 5 folds | Random guessing would score |
| --- | --- | --- |
| **ROC-AUC** | **0.79** | 0.50 |
| **PR-AUC** | **0.19** | 0.018 (the default rate) |

The score was about the same in every fold (ROC-AUC 0.77–0.81), so it's stable.

## 6. What the scores mean

Both scores grade how well the model **ranks** loans from riskiest to safest. A good model puts the real defaulters near the top.

**ROC-AUC: "Does it rank defaulters above repayers?"**
Pick one random defaulter and one random repayer. ROC-AUC is the chance the model gives the defaulter the higher risk. **Ours is 0.79: right 79% of the time.** A coin flip gets 0.50.

**PR-AUC: "When it flags someone, is it usually right, and does it catch most defaulters?"**
It balances two questions:
- **Precision:** of the loans we flagged as risky, how many really defaulted?
- **Recall:** of all the real defaulters, how many did we flag?

Compare PR-AUC with the default rate, not with 0.5. **Ours is 0.19 vs 0.018 for random guessing, so about 10× better than chance.**

**Why we treat PR-AUC as the main score:** defaults are rare, so ROC-AUC can look flattering. For example, wrongly flagging 5% of ~67,000 good borrowers sounds small but is about 3,350 people. PR-AUC focuses on the rare defaulters, so it's the more honest score here.

## 7. Why this baseline matters to the team

- **It's the number to beat.** Every other model (SVM, trees, LightGBM, MLP) should beat PR-AUC 0.19 under the same splits, or it isn't worth the extra complexity.
- **It proves the pipeline works** end to end: load, clean, train, test fairly, score.
- **It shows the leak is real.** Adding `Total_Amount_to_Repay` back makes the score jump to nearly perfect, which is a clear example for the report.

## 8. Next steps

1. Switch to Perfect's shared feature file and fold splits (PR #39) so results are comparable with everyone else's.
2. Save predictions (ID, fold, true answer, predicted probability) to `preds/` so `src/evaluate.py` can score every model the same way.
3. Build the same model in PyTorch Lightning and check its weights match scikit-learn's.
4. Later: calibration, the cost-based approve/decline cutoff, and confidence intervals.

## Glossary

| Word | Meaning |
| --- | --- |
| Default | The borrower didn't pay the loan back (`target = 1`). About 1 in 55 loans in our data. |
| Feature | A clue the model uses (loan amount, lender…) |
| Label / target | The answer we predict (defaulted or not) |
| Weight | The "points" the model learns for each feature |
| Sigmoid | The S-curve that turns a score into a probability |
| Training | The model learning its weights from past loans |
| Leak | A column that secretly contains the answer |
| Cross-validation | Testing on parts of the data the model didn't train on, in rotation |
| GroupKFold | Cross-validation that keeps each customer's loans together |
| Class imbalance | One outcome is rare (defaults: 1.83%) |
