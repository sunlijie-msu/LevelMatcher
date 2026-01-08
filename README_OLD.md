# Level Matcher

A physics-informed nuclear level matching tool employing XGBoost (eXtreme Gradient Boosting) regression with graph clustering techniques has been developed by the FRIB Nuclear Data Group.

This tool facilitates matching energy levels across experimental datasets, generating "Adopted Levels" and a cross-reference list with corresponding probabilistic confidence scores.


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
1.  Populate `test_dataset_A.json`, `test_dataset_B.json`, and `test_dataset_C.json` with experimental data.
2.  Run: `python Level_Matcher_Engine.py`
3.  View output: Adopted Energy, Cross-Reference (with probabilities), and Anchor Spin/Parity.

## Architecture
*   **`Level_Matcher_Engine.py`**: Main application logic. Handles model training, inference, graph clustering, and reporting.
*   **`data_parser.py`**: Core physics engine. Contains physics constants (`Scoring_Config`), feature extraction logic, and data ingestion (ENSDF JSON).
*   **Physics Logic**:
    *   **Energy Similarity**: Gaussian kernel of Z-Score.
    *   **Spin/Parity Similarity**: Weighted scoring system (Firm Match, Tentative Match, Conflict, Veto).
    *   **Quality Metrics**: Certainty (Tentativeness) and Specificity (Multiplicity) inputs for the ML model.

## Logic
*   **Z-Score:** $ |E_1 - E_2| / \sqrt{\sigma_1^2 + \sigma_2^2} $
*   **Consistency Check:** Spin/Parity strings are parsed and scored. Supports ranges (e.g., "1/2:7/2"), lists (e.g., "3/2, 5/2"), and tentative assignments (e.g., "(1/2, 3/2)+").
*   **Constraints:** Monotonic increasing constraints (1, 1, ...) ensure probability increases as similarity scores increase.
