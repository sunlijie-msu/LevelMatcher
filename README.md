# Level Matcher

Physics-informed nuclear level matching tool using XGBoost Regression and Graph Clustering.
Matches energy levels across experimental datasets to generate "Adopted Levels" and XREF list with probabilistic confidence scores.

## Key Features

*   **Physics-Informed XGBoost:** Uses `XGBRegressor` to output continuous match probabilities based on energy agreement (Z-Score) and physical properties (Spin/Parity Veto).
*   **Physics Veto:** Enforces strict selection rules. Mismatched Spin/Parity sets probability to 0.0 regardless of energy agreement.
*   **Graph Clustering (Greedy Merge):** Resolves multiplets by merging high-probability pairs into clusters.
    *   **Consistency:** Validates merges against all existing members to ensure consistency (Clique-like).
    *   **Doublet Support:** Allows levels to belong to multiple clusters if they match both consistently.
    *   **Constraint:** Clusters contain at most one level per dataset (Dataset Conflict Resolution).
    *   **Anchor Selection:** The member with the lowest energy uncertainty ($ \Delta E $) defines the cluster's physical properties.
*   **Synthetic Training:** "Physics-informed" training data embedded directly in the script, eliminating external dependencies.

## Workflow
1.  **Data Ingestion:** Ingests datasets (A, B, C) with Energy, Uncertainty, Spin, and Parity.
2.  **Model Training:** Trains `XGBRegressor` to penalize high Z-scores and Spin/Parity mismatches.
3.  **Pairwise Inference:** Calculates pairwise probabilities for all cross-dataset level pairs.
4.  **Graph Clustering:**
    *   Greedy merge of high-probability pairs.
    *   Enforces **Dataset Uniqueness** (one level per dataset per cluster).
    *   Enforces **ML Consistency** (all members must match each other).
    *   Handles **Doublets** by allowing overlap when merges are blocked by dataset conflicts.
5.  **Adopted Level Generation:** Generates "Adopted Levels" with weighted average energy and XREF lists.

## Usage
1.  Edit `levels` in `Level_Matcher_Gemini.py`.
2.  Run: `python Level_Matcher_Gemini.py`
3.  View output: Adopted Energy, XREF (with probabilities), and Anchor Spin/Parity.

## Logic
*   **Z-Score:** $ |E_1 - E_2| / \sqrt{\sigma_1^2 + \sigma_2^2} $
*   **Veto:** Probability = 0.0 if Spin/Parity mismatch.
*   **Constraints:** Monotonic constraints ensure probability decreases as Z-score increases.
