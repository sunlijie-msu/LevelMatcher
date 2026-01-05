# LevelMatcher

Physics-aware nuclear level matching tool using XGBoost Regression and Graph Clustering. Matches energy levels across experimental datasets to generate "Adopted Levels" and XREF lists.

## Core Features

*   **Physics-Informed XGBoost:** Uses `XGBRegressor` to output continuous match probabilities based on energy agreement (Z-Score) and physical properties (Spin/Parity Veto).
*   **Physics Veto:** Enforces strict selection rules. Mismatched Spin/Parity sets probability to 0.0 regardless of energy agreement.
*   **Graph Clustering (Greedy Merge):** Resolves multiplets by merging high-probability pairs into clusters.
    *   **Constraint:** Clusters contain at most one level per dataset (Dataset Conflict Resolution).
    *   **Anchor Selection:** The member with the lowest energy uncertainty ($ \Delta E $) defines the cluster's physical properties.
*   **Synthetic Training:** "Physics-informed" training data embedded directly in the script, eliminating external dependencies.

## Architecture

| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Model** | `XGBRegressor` | Predicts match probability based on Z-Score and Veto. |
| **Objective** | `binary:logistic` | Outputs soft probabilities (0.0 - 1.0). |
| **Constraints** | `monotone_constraints='(-1, -1)'` | Ensures probability decreases as Energy Diff or Veto increases. |
| **Clustering** | Greedy Graph Merge | Groups levels into unique physical states based on ML predictions. |

## Workflow

1.  **Input Data Ingestion:** Loads experimental datasets (A, B, C) with Energy, Uncertainty, Spin, and Parity.
2.  **Model Training:** Trains XGBoost on synthetic physics rules (Low Z-Score + No Veto = High Probability).
3.  **Pairwise Inference:** Calculates match probabilities for all inter-dataset pairs.
4.  **Graph Clustering:**
    *   Initializes every level as a unique cluster.
    *   Iteratively merges clusters connected by high-probability matches (>50%).
    *   Prevents merging if it would result in two levels from the same dataset in one cluster.
5.  **Adopted Level Generation:**
    *   **Energy:** Weighted average of all cluster members ($ w = 1/\sigma^2 $).
    *   **Spin/Parity:** Adopted from the "Anchor" (most precise member).
    *   **XREF:** List of datasets contributing to the level (derived strictly from cluster membership).

## Usage

1.  **Edit Data:** Modify the `levels` list in `Level_Matcher_Gemini.py` to input your datasets.
2.  **Run:**
    ```bash
    python Level_Matcher_Gemini.py
    ``` 
3.  **Output:**
    *   **Adopted_E:** Weighted average energy of the cluster.
    *   **XREF:** List of matching levels (e.g., `A_3000 + B_3000 + C_3005`).
    *   **Spin_Parity:** Adopted $ J^\pi $ from the anchor level.

## Logic & Conventions

*   **Z-Score:** $ \Delta E / \sqrt{\sigma_1^2 + \sigma_2^2} $.
    *   $ Z < 2.0 $: High probability match.
    *   $ Z > 3.0 $: Low probability match.
*   **XREF Generation:**
    *   Determined **solely** by the machine learning clustering results.
    *   No manual post-processing or "soft match" re-calculations.
