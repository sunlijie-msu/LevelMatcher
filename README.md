# LevelMatcher

Nuclear structure analysis tool using **XGBoost Regression** and **Graph Clustering** to match energy levels across experimental datasets.

## Key Features
*   **Physics-Informed ML:** XGBoost model trained on synthetic physics rules (Z-Score + Spin/Parity Veto).
*   **Soft Probabilities:** Outputs continuous match probabilities (0.0-1.0) rather than binary classifications.
*   **Advanced Clustering:**
    *   **Consistency Check:** Merges only if all members form a valid clique (all pairs >50% prob).
    *   **Doublet Support:** Allows a single level to belong to multiple clusters if it bridges conflicting states (e.g., large uncertainty level matching two precise levels).
    *   **Anchor-Based:** Physical properties defined by the most precise level in the cluster.

## Workflow
1.  **Data:** Ingests datasets (A, B, C) with Energy, Uncertainty, Spin, and Parity.
2.  **Training:** Trains `XGBRegressor` to penalize high Z-scores and Spin/Parity mismatches.
3.  **Inference:** Calculates pairwise probabilities for all cross-dataset levels.
4.  **Clustering:**
    *   Greedy merge of high-probability pairs.
    *   Enforces **Dataset Uniqueness** (one level per dataset per cluster).
    *   Enforces **ML Consistency** (all members must match each other).
    *   Handles **Doublets** by allowing overlap when merges are blocked by dataset conflicts.
5.  **Output:** Generates "Adopted Levels" with weighted average energy and XREF lists.

## Usage
1.  Edit `levels` in `Level_Matcher_Gemini.py`.
2.  Run: `python Level_Matcher_Gemini.py`
3.  View output: Adopted Energy, XREF (with probabilities), and Anchor Spin/Parity.

## Logic
*   **Z-Score:** $ |E_1 - E_2| / \sqrt{\sigma_1^2 + \sigma_2^2} $
*   **Veto:** Probability = 0.0 if Spin/Parity mismatch.
*   **Constraints:** Monotonic constraints ensure probability decreases as Z-score increases.
