# Machine Learning Hierarchy and Implementation Strategy for Nuclear Level Matching

## Level 1: The Foundation
### Decision Tree

- **Concept:** A non-parametric supervised method that predicts targets by learning simple if-then-else rules from features.

- **Core Components:**
  - **Root Node:** The starting point containing all data.
  - **Decision Node (Internal Node):** Where data is split based on a feature's value.
  - **Branches:** Connect nodes and represent decision outcomes.
  - **Leaf Node (Terminal Node):** Final outcomes or predictions.
  - **Splitting:** The process of dividing a node into two sub-nodes.

- **CART (Classification and Regression Trees):** Algorithm that builds trees using binary splits to minimize impurity (Gini impurity for classification, mean squared error for regression).

- **Characteristics:** Low bias and high variance. Deep trees are sensitive to small changes and can overfit.

- **Role:** The base estimator in ensemble methods.


## Level 2: The Strategy

Ensemble Learning: Reduce variance (Bagging) or bias (Boosting) by combining multiple weak learners.

### Strategy A: Bagging (Bootstrap Aggregating)

- **Logic:** Parallel execution to reduce variance.
- **Mechanism:** Generates M datasets via random sampling with replacement (bootstrap). Trains M independent trees. Final prediction is the average (regression) or majority vote (classification).
- **Key Algorithm:** Random Forest (Breiman, 2001).
- **Physics Limitation:** Averaging fails for "Hard Vetoes." If 1 tree detects a fatal Spin Mismatch (Probability approximately 0) but 99 trees see an Energy Match (Probability approximately 1), the average remains high (approximately 0.99). Physics requires the single veto to drive probability to 0.

### Strategy B: Boosting

- **Logic:** Sequential execution to reduce bias.
- **Mechanism:** Iterative improvement. Tree $m$ is trained to minimize errors (loss) of ensemble $F_{m-1}$.
- **Strength for Physics:** Mimics a "Veto" system. If Tree 1 predicts a match, Tree 2 can detect a specific violation (e.g., Parity Mismatch) and output a large negative correction, effectively suppressing match probability.

## Level 3: The Algorithm

Mathematical frameworks for implementing Boosting.

### Algorithm A: AdaBoost (Adaptive Boosting)

- **Mechanism:** Sample Reweighting. At step $m$, it increases the weights of misclassified observations.
- **Reference:** Freund and Schapire (1997).
- **Verdict:** Reject. Sensitive to noisy data both theoretically and empirically. In nuclear data, experimental outliers (large errors) receive exponential weight, causing the model to fixate on anomalies rather than general trends.

### Algorithm B: GBM (Gradient Boosting Machine)

- **Mechanism:** Functional Gradient Descent. At step $m$, a new tree is trained to predict the negative gradient (pseudo-residuals) of the loss function. It fits the error, not the data.
- **Reference:** Friedman (2001).
- **Verdict:** Superior. Optimizing differentiable loss functions (e.g., Log-Loss) makes it more robust to outliers than the exponential loss used in AdaBoost.

## Level 4: The Package (Software Libraries)

Major libraries implementing Gradient Boosting.

| Package | NaN Handling | Growth Strategy | Best Data Scale | Weakness for Physics | Verdict |
|---------|--------------|-----------------|-----------------|----------------------|---------|
| Scikit-learn GradientBoosting (Legacy) | Fails (Crashes) | Level-wise | Small | Requires imputation (bias risk); slow; lacks regularization. | Reject |
| LightGBM (Microsoft, 2017) | Native (Safe) | Leaf-wise | Huge (>100k) | "Greedy" growth overfits small data; creates unbalanced trees. | Reject |
| Scikit-learn HistGradientBoosting (2019) | Native (Safe) | Leaf-wise | Medium/Large | Less tunable regularization than XGBoost; defaults to greedy growth. | Acceptable |
| XGBoost (Chen and Guestrin, 2016) | Native (Safe) | Level-wise | Any | Minimal. Level-wise growth and L1/L2 regularization ideal for stability. | Best |
| CatBoost (Yandex, 2017) | Native (Safe) | Symmetric | Medium/Large | Slower training for pure numerical tasks; heavier dependency. | Alternative |

#### Growth Strategies

- **Leaf-wise Growth (LightGBM / HistGBDT):** Splits the single leaf with the highest error, regardless of tree balance. Can grow deep, lopsided, overfitted trees that "memorize" outliers in small datasets.
- **Level-wise Growth (XGBoost / Legacy GBDT):** Grows the tree layer-by-layer. Produces balanced trees, acting as a natural regularizer against experimental noise.


*   **The Challenge:** Nuclear datasets are inherently sparse; Level scheme is always incomplete and measured quantities sometimes have large uncertainties. Spin ($J$) and Parity ($\pi$) assignments are frequently unknown or tentative or many possibilities.

### 1. Native Handling of Missing Values (Sparsity Awareness)
*   **The XGBoost Solution:** XGBoost utilizes **Sparsity-Aware Split Finding**. Unlike legacy algorithms that fail on `NaN` or require bias-inducing imputation (guessing), XGBoost explicitly learns optimal default directions at tree split for `NaN ` based on other training labels. It treats "Unknown" as a distinct state, preserving the ambiguity inherent in experimental data.

### 2. Prevention of Overfitting on Small Datasets (Regularization)
*   **The Challenge:** Nuclear level schemes represent "small data" (typically $N < 500$ levels). Algorithms like LightGBM, designed for massive datasets, tend to grow deep, greedy trees that "memorize" experimental noise.
*   **The XGBoost Solution:** XGBoost defaults to **Level-wise growth**, constructing balanced and conservative trees. Combined with native **L1 (Lasso) and L2 (Ridge) regularization**, this architecture prevents overfitting and ensures the model captures general physics trends rather than statistical fluctuations.

### 3. Integration of Physical Logic with Statistical Matching

*   **The Challenge:** Nuclear level matching is not merely a numerical optimization problem; it is constrained by strict physical laws. Standard models often treat logical constraints (e.g., selection rules) as "soft" statistical features, potentially allowing a precise energy match to override a fatal physics violation (e.g., matching $J^\pi = 3^+$ with $4^-$).
*   **The XGBoost Solution:**
    *   **Statistical Compliance (Monotonicity):** A larger energy deviation penalizes the match probability, reflecting the statistical nature of experimental uncertainties.
    *   **Logical Compliance (Hard Vetoes):** As a sequential learner, XGBoost creates a hierarchy of decisions. It can learn that specific physical violations (like Parity Mismatch) act as absolute vetoes that nullify the probability, regardless of how perfect the energy agreement is.


## Level 5: The Implementation

### Feature Engineering

Feature engineering is the process of transforming raw data into meaningful input features that enable the machine learning model to detect patterns, relationships, and interactions. Feature engineering is one of the most critical steps in building high-performance tree-based models.

Well-engineered features can significantly boost model performance, leading to improved accuracy and predictive power. XGBoost is capable of handling complex data relationships.

Feature_Engineer.py

### Model Configuration

Level_Matcher.py