# Derivation: logistic regression, from log-odds to the gradient

**Owner:** Henry (P2) · **Model in the code:** `LitLogReg` in `notebooks/02_classification.ipynb` §3 (`nn.Linear` + `BCEWithLogitsLoss(pos_weight≈54)`) · **Checked by:** `tests/test_logistic_derivation.py` (analytic gradient = PyTorch autograd)

The chain in one line: **log-odds are linear → the sigmoid gives a probability → the Bernoulli likelihood of the labels → its negative log is binary cross-entropy (BCE) → the gradient is Xᵀ(σ(Xw) − y)**.

---

## 1. Setup

For loan *i* we have a feature vector **x**ᵢ ∈ ℝᵈ (after preprocessing; a constant 1 is appended so the bias is one of the weights) and a label yᵢ ∈ {0, 1} (1 = defaulted). Stack the n loans into a matrix **X** (n × d) and a vector **y**.

## 2. Step 1: model the log-odds as linear

A probability lives in (0, 1), but a weighted sum **w**ᵀ**x** can be any real number. The **odds** p / (1 − p) live in (0, ∞), and the **log-odds** live in (−∞, ∞), the same range as a linear function. So we assume:

  log( pᵢ / (1 − pᵢ) ) = zᵢ = **w**ᵀ**x**ᵢ

Each weight wⱼ is the change in log-odds when feature j rises by one unit, so e^wⱼ multiplies the odds of default. This is what "points per feature" means in our learning report.

## 3. Step 2: invert to get the sigmoid

Solve log(p / (1 − p)) = z for p:

  p / (1 − p) = eᶻ  ⇒  p = eᶻ (1 − p)  ⇒  p (1 + eᶻ) = eᶻ  ⇒  **p = σ(z) = 1 / (1 + e⁻ᶻ)**

Its derivative, needed below, has a neat form:

  σ′(z) = e⁻ᶻ / (1 + e⁻ᶻ)² = σ(z) · (1 − σ(z))

## 4. Step 3: the likelihood of the labels

Each label is a Bernoulli draw with probability pᵢ = σ(zᵢ). One formula covers both outcomes:

  P(yᵢ | **x**ᵢ, **w**) = pᵢ^yᵢ · (1 − pᵢ)^(1 − yᵢ)  (gives pᵢ if yᵢ = 1, and 1 − pᵢ if yᵢ = 0)

Assuming the loans are independent given the features, the likelihood of all labels is the product:

  L(**w**) = ∏ᵢ pᵢ^yᵢ (1 − pᵢ)^(1 − yᵢ)

**Maximum likelihood** picks the **w** that makes the observed defaults and repayments most probable.

## 5. Step 4: the negative log-likelihood is BCE

A product of 68,654 small numbers underflows, and sums are easier to differentiate, so take the log (it is increasing, so the maximiser does not change), flip the sign to get something to *minimise*, and divide by n to get an average:

  J(**w**) = −(1/n) Σᵢ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]

This is exactly **binary cross-entropy**. So *minimising BCE = maximising the likelihood*: the loss we train with is not an arbitrary choice; it is what maximum likelihood gives for Bernoulli labels.

**Stable form used by PyTorch.** Substituting pᵢ = σ(zᵢ) and simplifying:

  −[ y log σ(z) + (1 − y) log(1 − σ(z)) ] = log(1 + eᶻ) − y·z

This needs only z (the "logit"), never p, which is why the code feeds raw scores to `BCEWithLogitsLoss` and never applies the sigmoid first: it avoids log(0) when σ(z) rounds to 0 or 1.

## 6. Step 5: the gradient

Differentiate one loan's loss ℓᵢ = log(1 + e^zᵢ) − yᵢ zᵢ with respect to zᵢ:

  ∂ℓᵢ/∂zᵢ = e^zᵢ / (1 + e^zᵢ) − yᵢ = σ(zᵢ) − yᵢ

The prediction error itself. By the chain rule, since zᵢ = **w**ᵀ**x**ᵢ gives ∂zᵢ/∂**w** = **x**ᵢ:

  ∇J(**w**) = (1/n) Σᵢ (σ(zᵢ) − yᵢ) **x**ᵢ = **(1/n) Xᵀ (σ(Xw) − y)**

Read it as: each loan pushes the weights in the direction of its own features, scaled by how wrong the prediction was. A defaulter (y = 1) predicted at 0.02 pushes hard; a repayer predicted at 0.01 barely moves anything.

**Training** repeats **w** ← **w** − η ∇J(**w**) (we use Adam, which adapts η per weight, on batches of 1,024 loans).

## 7. Step 6: why there is one best answer (convexity)

Differentiate again: ∂²ℓᵢ/∂zᵢ² = σ(zᵢ)(1 − σ(zᵢ)) > 0, so the Hessian is

  H = (1/n) Xᵀ S X,  S = diag(σ(zᵢ)(1 − σ(zᵢ)))

For any vector **v**, **v**ᵀH**v** = (1/n) Σᵢ sᵢ (**v**ᵀ**x**ᵢ)² ≥ 0. H is positive semi-definite, so J is **convex**: there are no bad local minima. That is why scikit-learn's solver and our Lightning loop with Adam land on essentially the same model (ROC-AUC 0.844 vs 0.846, `02` §3). There is **no closed-form solution**, unlike linear regression, because σ is non-linear, so we iterate.

## 8. Step 7: the version we actually train (`pos_weight`)

Only 1.83% of loans default, so plain BCE lets the model look good by predicting "repays" everywhere. `BCEWithLogitsLoss(pos_weight=c)` multiplies the defaulters' term by c ≈ 67,396 / 1,258 ≈ 54:

  ℓᵢ = −[ c · yᵢ log σ(zᵢ) + (1 − yᵢ) log(1 − σ(zᵢ)) ]

Differentiating as before:

- yᵢ = 1: ∂ℓᵢ/∂zᵢ = −c (1 − σ(zᵢ)) = c (σ(zᵢ) − 1)
- yᵢ = 0: ∂ℓᵢ/∂zᵢ = σ(zᵢ)

so with weights aᵢ = c if yᵢ = 1 and 1 otherwise:

  **∇J(w) = (1/n) Xᵀ [ a ⊙ (σ(Xw) − y) ]**

Same shape, but each defaulter's error counts 54 times. Adam's `weight_decay` λ adds λ**w** to this gradient (an L2 penalty, like scikit-learn's default).

## 9. What the derivation explains in our results

**Why the raw probabilities were far too high (`02` §8).** Weighting every defaulter by c is the same as pretending the data had c times as many defaults. The model then learns the log-odds of that imaginary population, roughly the true log-odds **plus log c ≈ log 54 ≈ 4.0**. A loan whose real default risk is 1.8% (log-odds −4.0) gets a raw score near log-odds 0, i.e. about 50%. That is why the raw class-weighted scores averaged 0.28–0.32 against a true 0.018, and why we **calibrate afterwards** (Platt scaling fits a new slope and intercept on the log-odds and removes the shift: ECE 0.30 → 0.002). Ranking is unaffected because adding a constant to every log-odds keeps the order, so ROC-AUC and PR-AUC barely change.

**Why coefficients can differ while predictions agree (`02` §3).** When two features carry the same information (e.g. `n_past_loans` and `n_past_labelled`), X has nearly dependent columns, H is close to singular, and many weight vectors give almost the same Xw. The loss surface is flat along those directions, so two optimisers can stop at different weights with the same predictions.

## 10. Checked against autograd

`tests/test_logistic_derivation.py` builds random X, y and w, computes the loss with PyTorch's `BCEWithLogitsLoss` (with and without `pos_weight`), lets **autograd** compute the gradient, and checks it equals the formulas above:

- plain: (1/n) Xᵀ(σ(Xw) − y)
- weighted: (1/n) Xᵀ[a ⊙ (σ(Xw) − y)]

It also checks that the stable form log(1 + eᶻ) − yz equals the textbook BCE, and that the Hessian Xᵀ S X / n has no negative eigenvalues (convexity).
