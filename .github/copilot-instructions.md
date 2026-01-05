# LevelMatcher Project Instructions

## Project Overview
Nuclear structure analysis tool that matches energy levels across separate experimental datasets (A, B, C). It uses **XGBoost Regression** for soft probability matching and **Anchor-Based Hierarchical Clustering** for resolving assignments.

## Core Architecture
- **Single-File Script:** The primary logic resides in `Level_Matcher_Gemini.py`.
- **Simplicity First:** Avoid creating new functions for simple logic. Keep the logic inline and use clear, concise comments to explain complex lines.
- **Pipeline:**
  1.  **Data Ingestion:** Hardcoded `pandas` DataFrame setup (Datasets A, B, C).
  2.  **Physics-Informed Training:** `XGBRegressor` trained on `training_data_points` (synthetic list of tuples `(Z_Score, Veto, Probability)`).
  3.  **Inference:** Predicts match probability (0.0-1.0) based on Z-Score and Physics Veto.
  4.  **Clustering:** Anchor-based approach (A > B > C) to form "Adopted Levels".

## Key Patterns & Conventions
- **Soft Labels:** Use `XGBRegressor(objective='binary:logistic')` to output continuous probabilities (e.g., 0.79) rather than binary classes.
- **Prediction Syntax:** `model.predict([[z, veto]])[0]`
  - `[[...]]`: The model expects a 2D array (batch of inputs).
  - `[0]`: The model returns a list of results; we take the first one for our single pair.
- **Physics Veto:** 
  - If Spin/Parity mismatch -> `Veto=1`.
  - `Veto=1` forces Probability to `0.0` in training data.
- **Monotonic Constraints:** `monotone_constraints='(-1, -1)'` ensures probability decreases as Z-Score increases or Veto flag is set.
- **ID Format:** `f"{row['DS']}_{int(row['E_level'])}"` (e.g., `A_3005`).

## Algorithms
- **Z-Score:** `abs(E1 - E2) / sqrt(err1^2 + err2^2)`.
- **Graph Clustering (Greedy Merge with Overlap):**
  - **Logic:** Iteratively merge clusters connected by high-probability matches.
  - **Constraint 1 (Dataset):** A cluster can contain at most one level from each Dataset.
  - **Constraint 2 (Consistency):** Merges are only allowed if *all* cross-dataset pairs in the resulting cluster are valid ML matches (present in the high-probability candidates list).
  - **Doublet Support:** If two clusters cannot merge due to a conflict (e.g., Dataset A vs Dataset A), but a level (e.g., from Dataset C) matches *both* clusters consistently, that level is assigned to **both** clusters. This allows handling "Doublets" where one experimental level corresponds to multiple physical states.
  - **Anchor Selection:** The member with the lowest energy uncertainty (`DE_level`) defines the cluster's physical properties.
- **XREF List:**
  - Output format: `ID(Prob%) + ID(Prob%)`.
  - Probabilities are calculated against the **Cluster Anchor**.

## Development Workflow
- **Training Data:** Modify `training_data_points` list in `Level_Matcher_Gemini.py` to adjust model sensitivity (e.g., changing Z=2.0 probability).
- **Execution:** Run `python Level_Matcher_Gemini.py`.
- **No External Data:** All training data is synthetic and embedded in the script.
