# Gradient boosting for default prediction, from first principles

Owner: Eugene (P4). Used by: LightGBM in `src/models/ensembles.py`, settings in `configs/models.yaml` (Task 2 `lightgbm`, Task 1 `lightgbm_reg`, which is the deployed limit model).

## 1. The idea in one line

Build the model as a **sum of small trees**, adding one tree at a time, where each new tree is fitted to **the direction that most reduces the loss** of the model so far. That is gradient descent, but taken in the space of functions instead of the space of weights.

## 2. Setup

We have loans $(x_i, y_i)$, $i = 1..n$, with $y_i \in \{0, 1\}$ (1 = defaulted). The model outputs a score $F(x)$ on the **log-odds** scale, and the default probability is

$$p_i = \sigma(F(x_i)) = \frac{1}{1 + e^{-F(x_i)}}.$$

The loss is the same binary cross-entropy as logistic regression (the negative log-likelihood of a Bernoulli label, see `logistic.md`):

$$L = \sum_{i=1}^{n} \ell(y_i, F_i), \qquad \ell(y, F) = -\big[\,y \log \sigma(F) + (1-y)\log(1-\sigma(F))\,\big].$$

Logistic regression restricts $F$ to a linear function $w^\top x$. Boosting instead builds

$$F_M(x) = F_0 + \nu \sum_{m=1}^{M} f_m(x),$$

where each $f_m$ is a small regression tree and $\nu \in (0, 1]$ is the learning rate (shrinkage).

## 3. Step 0: the best constant

Before any tree, pick the constant $F_0$ that minimises $L$. Setting $\partial L / \partial F_0 = \sum_i (\sigma(F_0) - y_i) = 0$ gives $\sigma(F_0) = \bar{y}$, so

$$F_0 = \log \frac{\bar{y}}{1 - \bar{y}}.$$

The starting model predicts the overall default rate for everyone. In our Train set $\bar y = 0.0183$, so $F_0 = \log(0.0183/0.9817) \approx -3.98$.

## 4. The gradient of the loss with respect to the model's output

Treat each prediction $F_i$ as a free parameter. Using $\sigma'(F) = \sigma(F)(1 - \sigma(F))$:

$$\frac{\partial \ell}{\partial F_i} = -\Big[y_i \frac{\sigma'(F_i)}{\sigma(F_i)} - (1-y_i)\frac{\sigma'(F_i)}{1-\sigma(F_i)}\Big] = -\big[y_i(1-p_i) - (1-y_i)p_i\big] = p_i - y_i.$$

The **negative gradient**, called the *pseudo-residual*, is therefore

$$r_i = -\frac{\partial \ell}{\partial F_i} = y_i - p_i.$$

It is simply "actual minus predicted probability". A defaulter the model rated at 2% has $r = 0.98$ (push its score up hard). A good payer rated at 2% has $r = -0.02$ (push down a little).

The second derivative (needed in step 6) is

$$h_i = \frac{\partial^2 \ell}{\partial F_i^2} = p_i (1 - p_i).$$

## 5. Step m: fit a tree to the pseudo-residuals

Gradient descent would move each $F_i$ by $-\nu \,\partial \ell / \partial F_i = \nu r_i$. But we cannot move each training point separately: that would memorise the training set and say nothing about new applicants. So we **fit a regression tree $f_m$ to the $r_i$** by least squares. The tree groups loans with similar features into leaves $R_{jm}$, $j = 1..J$, and the update generalises to any new applicant that lands in the same leaf.

## 6. The value in each leaf (Newton step)

Inside leaf $R_j$ every loan gets the same increment $\gamma_j$. Choose it to minimise the loss of that leaf:

$$\gamma_j = \arg\min_\gamma \sum_{i \in R_j} \ell(y_i, F_{m-1,i} + \gamma).$$

There is no closed form, so take one Newton step from $\gamma = 0$ using the second-order Taylor expansion $\ell(F + \gamma) \approx \ell(F) - r\gamma + \tfrac12 h \gamma^2$. Setting the derivative to zero:

$$\gamma_j = \frac{\sum_{i \in R_j} r_i}{\sum_{i \in R_j} h_i} = \frac{\sum_{i \in R_j} (y_i - p_i)}{\sum_{i \in R_j} p_i(1-p_i)}.$$

LightGBM adds an L2 penalty $\lambda$ on leaf values, which puts $\lambda$ in the denominator: $\gamma_j = \sum r_i / (\sum h_i + \lambda)$. The same quantities also score candidate splits: a split is worth making when $\frac{(\sum_L r)^2}{\sum_L h + \lambda} + \frac{(\sum_R r)^2}{\sum_R h + \lambda} - \frac{(\sum r)^2}{\sum h + \lambda}$ is large.

## 7. Update with shrinkage

$$F_m(x) = F_{m-1}(x) + \nu \sum_{j} \gamma_j \,\mathbf{1}[x \in R_j].$$

A small $\nu$ (our LightGBM uses $\nu = 0.05$, 300 trees and 31 leaves per tree, `configs/models.yaml`) means each tree only corrects part of the error. More trees are needed, but the model overfits much less. Repeat steps 4 to 7 for $m = 1..M$, stopping when validation loss stops improving.

## 8. Worked example (checked numerically)

Four loans, one defaulter: $y = (1, 0, 0, 0)$, learning rate $\nu = 0.1$, and a one-split tree that separates loan 1 from the rest.

| Step | Value |
|---|---|
| Start, $\bar y = 0.25$ | $F_0 = \log(0.25/0.75) = -1.0986$, so $p = 0.25$ for all |
| Pseudo-residuals $r = y - p$ | $(0.75, -0.25, -0.25, -0.25)$ |
| Hessians $h = p(1-p)$ | $0.1875$ each |
| Leaf A {loan 1}: $\gamma_A = 0.75 / 0.1875$ | $4.000$ |
| Leaf B {loans 2-4}: $\gamma_B = -0.75 / 0.5625$ | $-1.333$ |
| New scores $F_1 = F_0 + 0.1\gamma$ | $(-0.699, -1.232, -1.232, -1.232)$ |
| New probabilities | $(0.332, 0.226, 0.226, 0.226)$ |
| Log-loss | $2.249 \rightarrow 1.870$ |

The defaulter's probability rises and the payers' fall, and the loss drops. (Numbers verified with a short Python script.)

## 9. How this relates to our project

- **Same loss as logistic regression.** Both minimise binary cross-entropy, so both produce probabilities that can be calibrated and fed into the cost threshold $C_{FP}/(C_{FP}+C_{FN})$. Boosting can capture interactions (for example "first-time borrower **and** a long term") that a linear model cannot.
- **Class imbalance.** With 1.83% defaults, `scale_pos_weight` multiplies the defaulters' $r_i$ and $h_i$. This raises recall but **inflates predicted probabilities**, which is why the predictions must be recalibrated (Platt scaling in deployment, `src/evaluation/calibration.py`) before any threshold or fairness rate is read off them.
- **Explanations.** Because $F$ is a sum of trees, per-feature contributions (SHAP values, via `shap.TreeExplainer`) add up exactly to the log-odds. `src/serving/reasons.py` uses them for the top-3 reasons when the deployed model is a tree ensemble (today it is the logistic regression, so it uses weight × value instead).
- **What boosting cannot fix.** Trees split only on values seen in training. Ghana's unseen lenders and loan types fall into an "unknown" bucket, and Ghana's loan sizes (about 10x Kenya's in USD) sit beyond every split point, so the model extrapolates flat. This is a covariate-shift problem, not something a stronger learner solves (see `04_transfer_and_fairness.ipynb`).
