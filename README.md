# Level Matcher - Physics-aware matching and clustering for nuclear levels

An automated system for matching nuclear energy levels across separate experimental datasets using Gradient Boosted Decision Trees (GBDT) and optimal assignment algorithms.

## Overview
This tool solves the correspondence problem in nuclear structure analysis by ranking candidate matches between datasets (e.g., Adopted Levels vs. Reaction Data) and finding a globally optimal one-to-one mapping.

## Methodology
The system employs a multi-stage approach:
1.  **Feature Engineering:** Extracts physical features including energy differences ($\Delta E$), spin-parity ($J^\pi$) compatibility, and angular momentum transfer ($L$).
2.  **Ranker (Boosting):** Utilizes a Gradient Boosting strategy to rank candidates. Boosting is preferred over Bagging (Random Forest) for its ability to strictly enforce physics constraints (e.g., spin vetoes) through sequential error correction.
3.  **Global Optimization:** Applies the Hungarian algorithm (Linear Sum Assignment) to the ranked scores to ensure a unique, physically consistent mapping across the entire level scheme.

## Implementation Strategy
Based on the [Machine Learning Hierarchy](Technology_Hierarchy.md), the project prioritizes **XGBoost** for its:
*   **Sparsity Awareness:** Native handling of missing experimental values (NaNs) without biased imputation.
*   **Small Data Stability:** Level-wise tree growth and $L1/L2$ regularization to prevent overfitting on typical nuclear datasets.
*   **Physics Compliance:** Effective modeling of hard constraints and selection rules.

## Project Structure
*   `Level_Matcher_XGBoost.py`: (Recommended) Implementation optimized for stability and sparsity.
*   `Level_Matcher_GPT.py`: Reference implementation using LightGBM.
*   `Level_Matcher_Gemini.py`: Reference implementation using Scikit-learn.
