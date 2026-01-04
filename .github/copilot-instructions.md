# LevelMatcher Project Instructions

## Project Overview
Nuclear structure analysis tool that matches energy levels across disparate experimental datasets (e.g., Adopted vs. Reaction data). It uses **XGBoost** for ranking/classification and **Graph Theory/Optimization** for resolving global assignments.

## Core Architecture
- **Standalone Scripts:** Logic is contained in single-file scripts (e.g., `Level_Matcher_Gemini.py`, `Level_Matcher_XGBoost.py`).
- **Pipeline:**
  1.  **Data Ingestion:** `pandas` DataFrames with columns `E_level`, `DE_level`, `Spin`, `Parity`, `DS`.
  2.  **Physics-Informed Training:** XGBoost models are trained on **synthetic, hardcoded datasets** (`X_train`) that encode physical rules (e.g., "Spin mismatch = No Match").
  3.  **Inference:** Pairwise probability prediction between all levels of different datasets.
  4.  **Clustering/Assignment:** Greedy constraint solving or Hungarian algorithm to form "Adopted Levels".

## Key Patterns & Conventions
- **Physics Veto:** Critical logic. If Spin/Parity mismatch, `Veto=1`.
- **XGBoost Constraints:** Always use `monotone_constraints='(-1, -1)'` (or similar) to enforce that increasing Energy Difference (Z-Score) or Veto flag *decreases* match probability.
- **ID Generation:** Use readable IDs: `f"{row['DS']}_{int(row['E_level'])}"` (e.g., `A_3005`).
- **Data Columns:**
  - `E_level`: Energy (keV).
  - `DE_level`: Energy Uncertainty.
  - `Spin` / `Parity`: Quantum numbers (nullable).
  - `DS`: Dataset identifier (A, B, C...).

## Algorithms
- **Z-Score:** `abs(E1 - E2) / sqrt(err1^2 + err2^2)`.
- **Greedy Clustering:** Used to handle multiplets. A cluster (Adopted Level) cannot contain >1 level from the same Dataset (`DS`).

## Development Workflow
- **Execution:** Run scripts directly: `python Level_Matcher_Gemini.py`.
- **No External Training Data:** Do not look for `.csv` training files. The model "learns" physics from the `X_train` list defined in the code.
