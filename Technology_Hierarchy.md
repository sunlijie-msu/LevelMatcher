# Machine Learning Hierarchy & Application Strategy

## Level 1: The Fundamental Unit
### Decision Tree
*   **Concept:** A single flowchart structure representing sequential decisions.
*   **Characteristics:** Weak and unstable on its own; prone to overfitting.
*   **Role:** The basic building block used to construct robust ensemble models.

## Level 2: The Strategy (Ensemble Learning)
*Goal: Combine multiple trees to create a powerful tool for regression or classification.*

### Strategy A: Bagging (Bootstrap Aggregating)
*   **Logic:** **Parallel** execution.
*   **Mechanism:** Trains $N$ trees independently on random data subsets. The final result is an average or vote.
*   **Key Algorithm:**
    *   **Random Forest:** Combines Bagging with Random Feature Selection.

### Strategy B: Boosting
*   **Logic:** **Sequential** execution.
*   **Mechanism:** Iterative correction. Tree $N$ targets the errors of Tree $N-1$.
*   **Key Algorithms:**
    *   **AdaBoost:** Adjusts **sample weights** (focuses on hard-to-classify data points).
    *   **Gradient Boosting:** Uses **gradient descent** to minimize error residuals (focuses on reducing loss).

## Level 3: Software Implementations (The Packages)
*Libraries implementing the Gradient Boosting algorithm.*

| Package | Key Characteristics |
| :--- | :--- |
| **XGBoost** | Optimized for performance; handles missing values (`NaN`) natively; widely used in science. |
| **LightGBM** | Histogram-based; leaf-wise growth; optimized for speed and large datasets. |
| **HistGradientBoosting** | Scikit-learn's native implementation of the histogram-based algorithm (similar to LightGBM). |
| **CatBoost** | Specialized for handling categorical data (text labels) directly without preprocessing. |

## Level 4: Application to Nuclear Level Matching
*Final Selection for Nuclear Data Reconciliation*

*   **Concept:** Decision Trees.
*   **Strategy:** **Boosting**
    *   *Reasoning:* Boosting handles **physics constraints** (e.g., Spin/Parity vetoes) effectively by applying strong negative corrections to invalid matches. Bagging (Random Forest) tends to "average out" these critical vetoes.
*   **Algorithm:** Gradient Boosting.
*   **Recommended Package:**
    1.  **HistGradientBoosting (Scikit-learn):** Best for standard Python environments; native `NaN` handling for missing experimental uncertainties.
    2.  **XGBoost:** Excellent alternative if external dependencies are permitted.