# Machine Learning Hierarchy & Application Strategy for Nuclear Level Matching

## Level 1: The Fundamental Unit of Tree-Based Models
### Decision Tree
*   **Concept:** A single flowchart structure representing sequential decisions.
*   **Characteristics:** Weak and unstable on its own; prone to overfitting.
*   **Role:** The basic building block used to construct robust ensemble models.

## Level 2: The Strategy (Ensemble Learning)
*Goal: Combine multiple trees to create a powerful tool for regression or classification.*

### Strategy A: Bagging (Bootstrap Aggregating)
*   **Logic:** **Parallel** execution.
*   **Mechanism:** Trains N trees independently on random data subsets. The final result is an average.
*   **Key algorithm:** Random Forest.
*   **Weakness for physics:** Averaging dilutes hard constraints. For example, an energy match combined with a spin veto may still yield a high average probability.

### Strategy B: Boosting
*   **Logic:** **Sequential** execution. Each new tree corrects errors made by previous trees (boosting).
*   **Mechanism:** Iterative correction; later trees focus on previous errors.

## Level 3: The Algorithms
*Specific approaches to implementing the boosting strategy.*

### Algorithm A: Adaptive Boosting (AdaBoost)
*   **Mechanism:** **Weight Adjustment**. Identifies misclassified examples and increases their weight for the next iteration.
*   **Structure:** Uses **Decision Stumps** (single-split trees).
*   **Verdict:** **Risk of Overfitting**. Highly sensitive to noise. Nuclear data outliers (experimental errors) may distort the model as it forces corrections on noisy points.

### Algorithm B: Gradient Boosting
*   **Mechanism:** **Residual Fitting**. Calculates the current model's error (residual) and trains the next tree specifically to predict and subtract that error.
*   **Structure:** Uses **Deep Trees** (typically depth 4–8).
*   **Verdict:** **Superior**. Robustly reduces error without over-fixating on outliers, making it stable for experimental datasets.

## Level 4: The Packages (Software Implementations)
*Major libraries implementing the Gradient Boosting algorithm.*

| Package | NaN Handling | Growth Strategy | Best Data Scale | Verdict for Nuclear Data |
| :--- | :--- | :--- | :--- | :--- |
| **Scikit-learn `GradientBoosting`** | **Fails** (Crashes) | Level-wise | Small/medium | **Reject** (Requires imputation, creating bias). |
| **LightGBM** | Native / safe | Leaf-wise | **Huge** (>100k) | **Reject** (Risk of overfitting small data). |
| **Scikit-learn `HistGradientBoosting`**| Native / safe | Leaf-wise | Medium/large | **Acceptable** (Good, but less tunable than XGBoost). |
| **XGBoost** | **Native / safe** | **Level-wise** | **Any** (Performs well on small datasets) | **Best** (Stable, robust regularization). |
| **CatBoost** | Native / safe | Oblivious trees | Categorical heavy | **Specialized** (Overkill for numerical energy data). |

*   **Leaf-wise (LightGBM):** Light Gradient Boosting Machine. Grows deep branches quickly. Great for massive data, but overfits noise in small nuclear datasets.
*   **Level-wise (XGBoost):** Grows balanced trees. More stable and conservative for small datasets with experimental uncertainties.

## Level 5: Application to Nuclear Level Matching

### Final recommendation

*   **Concept:** Decision Trees.
*   **Strategy:** **Boosting**
    *   *Reasoning:* Boosting handles **physics constraints** (e.g., Spin/Parity vetoes) effectively by applying strong negative corrections to invalid matches. Bagging (Random Forest) tends to "average out" these critical vetoes.
*   **Algorithm:** Gradient Boosting.
*   **Recommended package:** **XGBoost**: optimized implementation of gradient boosting. It combines multiple weak models into a stronger model.
*   **Why it is the best:**
    1.  **Sparsity awareness:** Nuclear data contains many missing values (unknown `J^pi`). XGBoost handles NaN natively; imputation is not required.
    2.  **Stability on small data:** Level-wise growth and regularization (L1/L2) reduce overfitting on small datasets (N < 500).
    3.  **Physics compliance:** Boosting enforces hard vetoes, such as selection rules, more effectively than bagging.


### XGBoost implementation

**Feature engineering:** Feature engineering is the process of transforming raw data into meaningful input features that make the machine learning model to detect patterns, relationships, and interactions.

**Model capabilities:** Well-engineered features can significantly boost model performance, leading to improved accuracy and predictive power. Feature engineering is one of the most critical steps in building high-performance tree-based models. XGBoost are capable of handling complex data relationships.

**Handling missing values:** Handling Missing Values: can be replaced with mean/median/mode using model-based imputation. XGBoost can inherently handle NaNs as missing.
