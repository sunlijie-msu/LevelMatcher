# LevelMatcher

Physics-aware nuclear level matching tool using XGBoost Regression and Anchor-Based Clustering. Matches energy levels across experimental datasets to generate "Adopted Levels" with probabilistic confidence scores.

## Core Features

*   **Soft Probability Matching:** Uses `XGBRegressor` to output continuous match probabilities (e.g., 79%) rather than binary classifications.
*   **Physics Veto:** Enforces strict selection rules. Mismatched Spin/Parity sets probability to 0.0 regardless of energy agreement.
*   **Anchor-Based Clustering:** Resolves multiplets using a strict hierarchy (Dataset A > B > C).
    *   **Constraint:** Clusters contain at most one level per dataset.
    *   **Tie-Breaking:** Prioritizes higher probability, then lower Z-score.
*   **Synthetic Training:** "Physics-informed" training data embedded directly in the script (`training_data_points`), eliminating external dependencies.

## Architecture

| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Model** | `XGBRegressor` | Predicts match probability based on Z-Score and Veto. |
| **Objective** | `binary:logistic` | Outputs soft probabilities (0.0 - 1.0). |
| **Constraints** | `monotone_constraints='(-1, -1)'` | Ensures probability decreases as Energy Diff or Veto increases. |
| **Clustering** | Custom Hierarchical | Groups levels around "Anchors" (A > B > C). |

## Usage

1.  **Edit Data:** Modify the `levels` list in `Level_Matcher_Gemini.py` to input your datasets (A, B, C).
2.  **Run:**
    ```bash
    python Level_Matcher_Gemini.py
    ```
3.  **Output:**
    *   **Adopted_E:** Weighted average energy of the cluster.
    *   **Sources:** List of matching levels with confidence scores (e.g., `A_3000(100%) + C_3005(81%)`).
    *   **Spin_Parity:** Adopted $J^\pi$ from the anchor level.

## Logic & Conventions

*   **Z-Score:** $\Delta E / \sqrt{\sigma_1^2 + \sigma_2^2}$.
    *   $Z < 2.0$: High probability match.
    *   $Z > 3.0$: Low probability match.
*   **Soft Source List:**
    *   Probabilities are calculated against the **Cluster Anchor**.
    *   Levels from datasets already present in the cluster are excluded from the candidate list.
