# Derivation: the support vector machine, from the widest street to the kernel trick

**Owner:** Perfect (P1) · **Model in the code:** `SubsampledSVC` in [`src/models/svm.py`](../../src/models/svm.py) (`SVC(kernel="rbf", class_weight="balanced")` on a 12,000-row stratified subsample, then Platt scaling) · **Settings:** `svm_rbf` in [`configs/models.yaml`](../../configs/models.yaml) · **Checked by:** [`tests/test_svm_derivation.py`](../../tests/test_svm_derivation.py) (each claim below is verified on 2-D toy data against scikit-learn's `SVC`)

The chain in one line: **widest street between the classes → minimise ‖w‖² → add slack for borrowers who cross the street (hinge loss) → Lagrangian → dual problem that only needs dot products between loans → swap the dot product for a kernel**.

---

## 1. Setup

Loan *i* has a feature vector **x**ᵢ ∈ ℝᵈ (after preprocessing) and a label **yᵢ ∈ {−1, +1}** (+1 = defaulted, −1 = repaid). This differs from the 0/1 labels used elsewhere in the repo; ±1 makes the algebra shorter, and the code converts for us.

A linear classifier scores a loan with

  f(**x**) = **w**ᵀ**x** + b,  and predicts +1 when f(**x**) > 0.

The set f(**x**) = 0 is a straight line (a "hyperplane" in more dimensions). Unlike logistic regression, f is **not** a probability. It is just a signed score, so the SVM needs an extra step (section 8) before the committee can read it as a risk.

## 2. Step 1: the widest street (hard margin)

The distance from a point **x** to the line f = 0 is |f(**x**)| / ‖**w**‖. (Reason: **w** is perpendicular to the line, and moving a distance t along the unit vector **w**/‖**w**‖ changes f by t‖**w**‖.)

Many lines separate two classes. The SVM picks the one with the **widest empty street** around it. Because (**w**, b) can be rescaled without moving the line, fix the scale so the closest points of each class satisfy

  yᵢ (**w**ᵀ**x**ᵢ + b) = 1,  and every other point has yᵢ (**w**ᵀ**x**ᵢ + b) ≥ 1.

One closest point is on each side, each at distance 1/‖**w**‖, so the **street is 2/‖w‖ wide**. Maximising 2/‖**w**‖ is the same as minimising ‖**w**‖², and we add a ½ so the derivative comes out clean:

  min over **w**, b:  ½‖**w**‖²  subject to  yᵢ (**w**ᵀ**x**ᵢ + b) ≥ 1 for every loan i.  (hard margin)

*Checked:* the test finds the widest street by brute force (trying 20,000 directions) and confirms it equals 2/‖**w**‖ from `SVC`. It also confirms the closest points sit exactly at yᵢ f(**x**ᵢ) = 1.

## 3. Step 2: let some borrowers cross the street (soft margin and hinge loss)

Real loans overlap: some defaulters look like good borrowers, so no line separates them and the hard-margin problem has no solution. Give each loan a **slack** ξᵢ ≥ 0, the amount by which it breaks the rule, and charge for it:

  min over **w**, b, **ξ**:  ½‖**w**‖² + Σᵢ Cᵢ ξᵢ  subject to  yᵢ (**w**ᵀ**x**ᵢ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0.

Cᵢ is the price of slack for loan *i*. For now Cᵢ = C for everyone.

**This is a regularised loss in disguise.** For fixed (**w**, b), the cheapest slack that satisfies both constraints is

  ξᵢ = max(0, 1 − yᵢ f(**x**ᵢ))  (the **hinge loss**: zero if the loan is on the right side and outside the street, growing linearly the further it is on the wrong side).

Substituting gives an unconstrained problem:

  min  ½‖**w**‖² + C Σᵢ max(0, 1 − yᵢ f(**x**ᵢ)).

So an SVM is "hinge loss + a penalty on ‖**w**‖²". Logistic regression is "log loss + a penalty on ‖**w**‖²". The only difference is the loss: the hinge is exactly zero for confident correct points, so **those points stop mattering**; log loss never reaches zero.

**What C does.** Large C: slack is expensive, the street narrows to fit the training points (risk of overfitting). Small C: slack is cheap, the street widens (risk of underfitting). Our setting is C = 1.

**Class weights (our code uses `class_weight="balanced"`).** Defaulters are 1.83% of loans, so a model that ignores them looks 98% accurate. We give each class its own price: Cᵢ = C · w_c where w_c = n / (2·n_c). With 1.83% defaults, **w₊ ≈ 27.3 and w₋ ≈ 0.51**, so one missed defaulter costs about 54× one wrongly flagged repayer. The same 54 appears as `pos_weight` in Henry's logistic regression.

## 4. Step 3: the Lagrangian

To solve a constrained problem, attach a multiplier to each constraint: αᵢ ≥ 0 for yᵢ(**w**ᵀ**x**ᵢ + b) ≥ 1 − ξᵢ, and μᵢ ≥ 0 for ξᵢ ≥ 0:

  L = ½‖**w**‖² + Σ Cᵢξᵢ − Σ αᵢ [ yᵢ(**w**ᵀ**x**ᵢ + b) − 1 + ξᵢ ] − Σ μᵢ ξᵢ.

At the optimum the derivatives of L with respect to the primal variables are zero:

- ∂L/∂**w** = **w** − Σ αᵢ yᵢ **x**ᵢ = 0  ⇒  **w = Σᵢ αᵢ yᵢ xᵢ**
- ∂L/∂b = −Σ αᵢ yᵢ = 0  ⇒  **Σᵢ αᵢ yᵢ = 0**
- ∂L/∂ξᵢ = Cᵢ − αᵢ − μᵢ = 0, and μᵢ ≥ 0  ⇒  **0 ≤ αᵢ ≤ Cᵢ**

The first line is the key insight: **the weight vector is a weighted sum of training loans.** Only loans with αᵢ > 0 appear in it. Those are the **support vectors**.

## 5. Step 4: the dual problem

Substitute **w** = Σ αᵢ yᵢ **x**ᵢ back into L. The ξ and b terms vanish by the three conditions above, and the ½‖**w**‖² and −**w**ᵀ**w** terms combine to −½‖**w**‖²:

  **max over α:  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)**  subject to  0 ≤ αᵢ ≤ Cᵢ,  Σᵢ αᵢ yᵢ = 0.

Two facts to notice:

1. The data enter **only through dot products xᵢᵀxⱼ** between pairs of loans.
2. The problem is concave with linear constraints, so the dual optimum equals the primal optimum (**strong duality**). At the solution, ½‖**w**‖² + CΣξᵢ = Σαᵢ − ½‖**w**‖².

*Checked:* the tests confirm w = Σαᵢyᵢ**x**ᵢ, Σαᵢyᵢ = 0, and that the primal and dual objectives agree to six digits (`SVC` stores αᵢyᵢ as `dual_coef_`).

## 6. Step 5: which loans matter (complementary slackness)

The optimum must also satisfy αᵢ·[yᵢ f(**x**ᵢ) − 1 + ξᵢ] = 0 and μᵢξᵢ = 0. Reading these off gives three kinds of loan:

| αᵢ | Where the loan is | Role |
| --- | --- | --- |
| = 0 | Correct side, outside the street (yᵢ f ≥ 1) | Irrelevant. Delete it and nothing changes |
| 0 < αᵢ < Cᵢ | **Exactly on the street edge** (yᵢ f = 1, ξᵢ = 0) | Support vector. Used to compute b |
| = Cᵢ | Inside the street or on the wrong side (yᵢ f ≤ 1, ξᵢ ≥ 0) | Support vector that pays slack |

So the model is defined by a (usually small) subset of loans, and b follows from any edge loan: b = yⱼ − Σᵢ αᵢ yᵢ **x**ᵢᵀ**x**ⱼ.

*Checked:* the test builds all three groups on overlapping toy data and confirms each one's position. With class weights it confirms αᵢ ≤ C·w_c separately for each class.

## 7. Step 6: the kernel trick

A straight line is too rigid for the loan data. The standard escape is to map each loan into a richer space, **x** ↦ φ(**x**), and draw the line there. The dual only ever uses dot products, so we never need φ itself, only the **kernel** k(**x**, **z**) = φ(**x**)ᵀφ(**z**). Replace xᵢᵀxⱼ by k(**x**ᵢ, **x**ⱼ):

  f(**x**) = Σᵢ αᵢ yᵢ k(**x**ᵢ, **x**) + b.

Our kernel is the **RBF (Gaussian)**:

  k(**x**, **z**) = exp(−γ ‖**x** − **z**‖²).

It is large when two loans are close and near zero when they are far apart, so f is a sum of bumps centred on the support vectors. γ sets the bump width. Large γ gives narrow bumps and a wiggly, overfit boundary. Small γ gives wide bumps and a nearly straight boundary.

**Why it is an inner product in a huge space.** Expand: exp(−γ‖**x**−**z**‖²) = exp(−γ‖**x**‖²) · exp(−γ‖**z**‖²) · exp(2γ **x**ᵀ**z**), and exp(t) = Σₖ tᵏ/k! with t = 2γ **x**ᵀ**z**. Each term (**x**ᵀ**z**)ᵏ is an inner product of degree-k polynomial features, so the RBF kernel is an inner product over polynomial features of **every** degree: infinitely many dimensions, computed at the cost of one exponential.

**Our γ.** `gamma="scale"` means γ = 1 / (d · Var(X)), where d is the number of features and Var(X) is the variance of all entries of the training matrix. It adapts to the data's scale.

*Checked:* the test rebuilds `SVC.decision_function` by hand from `dual_coef_`, `support_vectors_`, `intercept_` and this γ, and matches it to 1e-8. A second test confirms the series truncated at 25 terms equals the kernel to 1e-12.

## 8. From a score to a probability (Platt scaling)

f(**x**) is a distance-like score, not a risk. To feed the cost threshold (about 0.10 in cedis under D-28, see `configs/cost_matrix.yaml`; 0.036 under D-22) we need a probability. [`SubsampledSVC`](../../src/models/svm.py) fits **Platt scaling**: p = 1 / (1 + exp(A·f + B)), where A and B are learned by a one-feature logistic regression on held-out scores (`CalibratedClassifierCV(method="sigmoid", cv=3)`). It is monotone, so it never changes the ranking, and ROC-AUC and PR-AUC are unaffected. Only the probability values change.

## 9. Why the code trains on 12,000 rows

The dual has one variable per loan and needs the n × n kernel matrix, so time and memory grow roughly with n². Each training fold has about 55,000 loans (3 billion kernel entries). [`stratified_subsample`](../../src/models/svm.py) keeps at most 12,000 rows and preserves the default rate, which leaves **only about 220 defaulters** (12,000 × 1.83%) to learn from.

## 10. What we measured, and how to read it

From `reports/tables/task2_model_summary.csv`:

| Scheme | Model | ROC-AUC | PR-AUC |
| --- | --- | :---: | :---: |
| By customer (`fold_group`) | Majority | 0.500 | 0.018 |
| | kNN (k = 25) | 0.769 | 0.155 |
| | **SVM (RBF)** | **0.805** | **0.073** |
| By time (`temporal`) | Majority | 0.500 | 0.095 |
| | kNN | 0.600 | 0.140 |
| | **SVM (RBF)** | **0.570** | **0.116** |

Reading, with what is proven and what is only a guess (all numbers from the same table; other models' scores come from the same file):

- **Proven:** among our models the SVM is middling at ranking by customer. Its ROC-AUC (0.805) beats kNN (0.769) but trails logistic regression (0.843), LightGBM (0.849), the decision tree (0.849), the MLP (0.868) and Random Forest (0.871). On PR-AUC it is the **weakest model that learns anything** (0.073 against 0.155 to 0.207 for the others), less than half of kNN's.
- **Why the two metrics disagree (kNN vs SVM):** ROC-AUC rewards getting the whole ordering right, PR-AUC rewards getting the **top of the list** right, and at a 1.83% default rate the top of the list is what matters. The SVM orders the bulk of loans better than kNN but puts too many repayers at the top.
- **Proven:** by time the SVM falls to 0.570 ROC-AUC, close to chance and one of the lowest (decision tree 0.559, LightGBM 0.582, kNN 0.600, Random Forest 0.619, logistic regression 0.633, MLP 0.648). Every model loses ground; a model that ranks well within a period does not rank well on newer loans.
- **A likely cause, not tested:** the subsample leaves about 220 defaulters, so the SVM sees little of the rare class. A larger `max_train_rows` would test this, at a quadratic cost in time.
- **Read the by-time PR-AUC against its own floor:** the temporal validation set has a 9.5% default rate, so even "approve everyone" scores PR-AUC 0.095 there, not 0.018. On that floor the SVM's 0.116 is only slightly better than doing nothing.

## 11. You understand it when you can answer

- Why is the street 2/‖**w**‖ wide, and why does minimising ‖**w**‖² widen it?
- What is a support vector, and why can you delete every other loan without changing the model?
- What does hinge loss do that log loss does not, and what does that cost you when you need a probability?
- Why does the dual need only dot products, and what does that allow?
- With 1.83% defaults, what does `class_weight="balanced"` do to C, and where else in the project does the same number appear?
- Why can the SVM beat kNN on ROC-AUC yet score less than half of kNN's PR-AUC?
