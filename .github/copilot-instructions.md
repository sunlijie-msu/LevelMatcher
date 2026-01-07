# LevelMatcher Project Instructions

## Project Overview
Nuclear structure analysis tool that matches energy levels across separate experimental datasets (A, B, C). It uses **XGBoost Regression** for soft probability matching and **Anchor-Based Hierarchical Clustering** for resolving assignments.

- **Simplicity First:** Avoid creating new functions for simple logic. Keep the logic inline and use clear comments to explain complex lines.

- **No Acronyms:** Do not use acronyms for variable names or documentation. Spell out long variable names completely (e.g., use `tentative` instead of `tent`, `error` instead of `err`). Long, descriptive naming for variables is preferred for clarity.

## Key Patterns & Conventions
- **Preserve Comments:** Do not summarize, shorten, or delete existing user's comments. Keep them exactly as they are.
- **Edit vs Replace:** Always use editing tools (`replace_string_in_file`, `edit_notebook_file`) to modify existing code. Do not use file creation/overwriting tools (`create_file`) on existing files unless explicitly asked to "rewrite" or "reset" the file.
**Strictly Forbidden:** Do not self-use `git restore` or `git checkout` to revert changes. Nuclear data coding requires high-precision work, not typical software development. The common LLM tendency to resort to git for error recovery is strictly prohibited. You must identify and fix errors carefully to maintain absolute rigor.

- **Soft Labels:** Use `XGBRegressor(objective='binary:logistic')` to output continuous probabilities (e.g., 0.79) rather than binary classes.
- **Prediction Syntax:** `model.predict([[z, veto]])[0]`
  - `[[...]]`: The model expects a 2D array (batch of inputs).
  - `[0]`: The model returns a list of results; we take the first one for our single pair.

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

