# Derivation: linear regression, from Gaussian likelihood to Ridge as MAP

**Owner:** Bernard (P3) · **Models in the code:** `ols`, `ridge` (`alpha=10.0`), `lasso` (`alpha=0.01`) in [`src/models/linear.py`](../../src/models/linear.py), and `LitLinear` in [`src/models/lightning_modules.py`](../../src/models/lightning_modules.py) · **Settings:** `task1` in [`configs/models.yaml`](../../configs/models.yaml) · **Checked by:** [`tests/test_linear_derivation.py`](../../tests/test_linear_derivation.py) (normal equations, Ridge closed form, gradients, convexity and collinearity verified against scikit-learn and PyTorch)

The chain in one line: **Gaussian noise around a linear mean → maximise likelihood → minimise squared error (OLS) → set gradient to zero → normal equations → add a Gaussian prior on weights → MAP gives Ridge regression**.

---

## 1. Setup

For loan *i* ∈ {1, …, *n*}, we have a preprocessed feature vector **x**ᵢ ∈ ℝᵈ (numeric features standardized and categoricals one-hot encoded, excluding amount columns so Task 1 has no target leakage; see `NUM_FEATURES_T1` in `src/config.py`) and a continuous target yᵢ ∈ ℝ. In Task 1, yᵢ is the standardized log loan amount in USD produced by `TargetLogScaler`:

  yᵢ = (log(1 + amount_usdᵢ) − μ) / s

Stack the *n* loans into the design matrix **X** ∈ ℝⁿˣᵈ (with a leading column of 1s when absorbing the intercept *b* into **w**, or keeping *b* separate so regularisation does not shrink it) and target vector **y** ∈ ℝⁿ.

---

## 2. Step 1: Gaussian noise model and likelihood

Linear regression assumes the observed target is a linear function of the features plus independent, zero-mean Gaussian noise with constant variance σ² > 0:

  yᵢ = **w**ᵀ**x**ᵢ + εᵢ,    εᵢ ~ N(0, σ²)  i.i.d.

Equivalently, the conditional density of yᵢ given **x**ᵢ and **w** is:

  p(yᵢ | **x**ᵢ, **w**) = (1 / √(2πσ²)) · exp( −(yᵢ − **w**ᵀ**x**ᵢ)² / (2σ²) )

Assuming conditional independence across loans, the joint likelihood of **y** given **X** and **w** is the product:

  L(**w**) = ∏ᵢ p(yᵢ | **x**ᵢ, **w**) = (2πσ²)⁻ⁿ/² · exp( −(1 / (2σ²)) Σᵢ (yᵢ − **w**ᵀ**x**ᵢ)² )

---

## 3. Step 2: negative log-likelihood gives least squares (OLS)

Maximising L(**w**) is equivalent to minimising the negative log-likelihood −log L(**w**), since the logarithm is strictly increasing:

  −log L(**w**) = (n / 2) log(2πσ²) + (1 / (2σ²)) Σᵢ (yᵢ − **w**ᵀ**x**ᵢ)²

The first term does not depend on **w**, and 1 / (2σ²) is a positive constant. Dropping both leaves the **ordinary least squares (OLS)** objective (proportional to the mean squared error `MSELoss` used in `LitLinear`):

  J(**w**) = ½ ‖**y** − **X****w**‖₂² = ½ (**y** − **X****w**)ᵀ(**y** − **X****w**)

So **minimising squared error is maximum likelihood estimation (MLE) under Gaussian noise**. If the errors are heavy-tailed rather than Gaussian, squared error over-penalises extreme points — which is why we log-transform `Total_Amount` before fitting (`TargetLogScaler`).

---

## 4. Step 3: gradient, convexity and the normal equations

Expand J(**w**):

  J(**w**) = ½ (**y**ᵀ**y** − 2**w**ᵀ**X**ᵀ**y** + **w**ᵀ**X**ᵀ**X****w**)

Differentiate with respect to **w** using ∂(**a**ᵀ**w**)/∂**w** = **a** and ∂(**w**ᵀ**A****w**)/∂**w** = 2**A****w** for symmetric **A** = **X**ᵀ**X**:

  ∇J(**w**) = **X**ᵀ**X****w** − **X**ᵀ**y** = **X**ᵀ(**X****w** − **y**)

Differentiate a second time to get the Hessian:

  **H** = ∇²J(**w**) = **X**ᵀ**X**

For any vector **v** ∈ ℝᵈ, **v**ᵀ**H****v** = **v**ᵀ**X**ᵀ**X****v** = ‖**X****v**‖₂² ≥ 0. Therefore **H** is positive semi-definite and J(**w**) is **convex**: any stationary point is a global minimum.

Setting the gradient to zero gives the **normal equations**:

  **X**ᵀ**X****w** = **X**ᵀ**y**

When **X** has full column rank, **X**ᵀ**X** is strictly positive definite and invertible, yielding the unique closed-form OLS solution:

  **ŵ**_OLS = (**X**ᵀ**X**)⁻¹ **X**ᵀ**y**

**Why the normal equations fail under collinearity.** If two columns of **X** are linearly dependent — or nearly so, such as `n_past_loans` and `n_past_labelled`, or redundant one-hot indicators — then **X**ᵀ**X** has zero (or tiny) eigenvalues. Its inverse either does not exist or has huge entries, causing the estimated weights to explode in opposite directions and hurting generalisation (scikit-learn's `LinearRegression` avoids division by zero via SVD pseudoinverse, but variance still explodes on nearly collinear columns).

---

## 5. Step 4: Ridge regression as MAP with a Gaussian prior

Place an independent, zero-mean Gaussian prior on the feature weights (excluding the intercept *b*, which shifts the overall mean):

  p(**w**) = ∏ⱼ (1 / √(2πτ²)) · exp( −wⱼ² / (2τ²) )

By Bayes' theorem, the posterior is p(**w** | **X**, **y**) ∝ p(**y** | **X**, **w**) · p(**w**). Maximising the posterior (**MAP estimation**) is equivalent to minimising the negative log-posterior:

  −log p(**w** | **X**, **y**) = const + (1 / (2σ²)) ‖**y** − **X****w**‖₂² + (1 / (2τ²)) ‖**w**‖₂²

Multiply by σ² and define the regularisation strength **λ = σ² / τ² > 0** (called `alpha` in scikit-learn's `Ridge`):

  J_Ridge(**w**) = ½ ‖**y** − **X****w**‖₂² + (λ / 2) ‖**w**‖₂²

Taking the gradient and setting it to zero:

  ∇J_Ridge(**w**) = **X**ᵀ**X****w** − **X**ᵀ**y** + λ**w** = (**X**ᵀ**X** + λ**I**)**w** − **X**ᵀ**y** = **0**

  **ŵ**_Ridge = (**X**ᵀ**X** + λ**I**)⁻¹ **X**ᵀ**y**

*(With an unpenalized intercept, the diagonal matrix **D** = diag(0, 1, …, 1) replaces **I** so the intercept is untouched.)*

**Why Ridge fixes collinearity.** If the eigenvalues of **X**ᵀ**X** are μ₁, …, μ_d ≥ 0, then the eigenvalues of **X**ᵀ**X** + λ**I** are μ₁ + λ, …, μ_d + λ ≥ λ > 0. The matrix is **always strictly positive definite and invertible**, its condition number drops, and correlated weights are shrunk smoothly toward zero. A tighter prior (smaller τ²) corresponds to a larger λ and stronger shrinkage.

**Connection to Lasso and `LitLinear`.**
- Replacing the Gaussian prior with a zero-mean **Laplace prior** p(wⱼ) ∝ exp(−|wⱼ| / *b*) turns the penalty into λ‖**w**‖₁ (**Lasso**, `alpha=0.01`). Because |wⱼ| has a sharp corner at zero, Lasso drives weak weights to *exact* zero (feature selection), though it has no closed-form solution and is solved iteratively via coordinate descent.
- Our PyTorch Lightning model `LitLinear` trains `nn.Linear` with `MSELoss` and Adam `weight_decay=0.0001` (an L2 penalty on the weights), minimising the same convex objective as Ridge iteratively on mini-batches.

---

## 6. What the derivation explains in our Task 1 results

From [`reports/tables/task1_model_summary.csv`](../../reports/tables/task1_model_summary.csv) (target: loan amount in USD, evaluated via 5-fold `fold_group` and `temporal` 2023+ split):

| Model | `fold_group` RMSE (log) | `fold_group` R² (log) | `fold_group` RMSE (USD) | `temporal` RMSE (log) | `temporal` RMSE (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mean` | 1.2097 | −0.0001 | 1,275.2 | 1.5402 | 2,242.2 |
| `ols` | 1.1094 | 0.1588 | 2,088.9 | 1.0156 | 3,360.0 |
| `ridge` (`alpha=10`) | 1.1115 | 0.1556 | 1,745.1 | 1.1693 | 4,189.9 |
| `lasso` (`alpha=0.01`) | 1.1240 | 0.1366 | 1,896.3 | 1.3184 | 3,817.7 |
| `lit_linear` | 1.1146 | 0.1509 | 1,765.4 | 1.1181 | 6,632.9 |
| `decision_tree_reg` | 1.0894 | 0.1889 | 949.9 | 0.8695 | 2,069.7 |
| `lightgbm_reg` | 1.0763 | 0.2083 | 1,113.7 | 0.8501 | 1,842.0 |

1. **Convexity explains why OLS and `LitLinear` agree on the log scale.** On `fold_group`, closed-form `ols` (log RMSE 1.1094), closed-form `ridge` (1.1115), and gradient-based `lit_linear` (1.1146) converge to nearly the same log-scale accuracy because the squared-error objective is convex with no bad local minima.
2. **Regularisation tames dollar-scale variance on `fold_group`.** With correlated borrower-history columns and one-hot lenders/loan types, unregularised `ols` gets dollar RMSE \$2,089; Ridge (`alpha=10`) shrinks noisy weights and drops dollar RMSE to \$1,745 (−16%) at almost no cost in log RMSE (1.1094 → 1.1115).
3. **Why linear models lose to the mean in USD.** All linear models beat `mean` on the log scale they optimise (R² 0.14–0.16 vs 0.00), yet lose to `mean` on the dollar scale (RMSE \$1,745–\$2,089 vs \$1,275). Two effects drive this:
   - **Exponential amplification of linear extrapolation:** a small overshoot +Δ in log space multiplies the dollar prediction by e^Δ. For rare high-duration or unseen-combination loans, linear combinations extrapolate far past the typical amount before clipping (`TargetLogScaler`).
   - **Jensen's inequality on log-normal targets:** E[exp(Y)] = exp(μ + ½σ²) > exp(E[Y]), so predicting the conditional mean in log space gives the conditional *median* in dollars rather than the conditional mean, under-predicting the heavy right tail while linear extrapolation over-predicts outliers.
   Tree models (`decision_tree_reg`, `lightgbm_reg`) cannot extrapolate beyond leaf training ranges, so they beat `mean` on both scales (USD RMSE \$950 and \$1,114).

---

## 7. Checked against scikit-learn and PyTorch

[`tests/test_linear_derivation.py`](../../tests/test_linear_derivation.py) verifies each claim numerically on synthetic toy data:

1. **Normal equations = `LinearRegression`:** **ŵ**_OLS = (**X**ᵀ**X**)⁻¹**X**ᵀ**y** (with an intercept column) matches `sklearn.linear_model.LinearRegression().fit(X, y)` to 10⁻¹⁰.
2. **Ridge MAP closed form = `Ridge`:** (**X**ᵀ**X** + λ**D**)⁻¹**X**ᵀ**y** (unpenalized intercept) matches `sklearn.linear_model.Ridge(alpha=10.0)` to 10⁻¹⁰.
3. **Analytic gradient = PyTorch autograd:** (1/n)**X**ᵀ(**X****w** − **y**) + (λ/n)**w** matches autograd on half-MSE + L2 penalty to 10⁻¹².
4. **Convexity:** all eigenvalues of **H** = **X**ᵀ**X** are ≥ 0 (positive semi-definite), and adding λ**I** shifts every eigenvalue up by λ > 0, strictly reducing the condition number on collinear features.
5. **Log-scale retransformation bias:** on log-normal targets, exp(E[log Y]) systematically under-predicts E[Y] by factor exp(σ²/2).
