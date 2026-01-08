"""
Feature Engineering For Nuclear Level Matching
======================================

Explanation of Code Structure:
-------------------------
1.  **Data Loading** (`load_levels_from_json`):
    - Ingests nuclear level data from JSON files (ENSDF schema).
    - Standardizes attributes into flat dictionaries: `energy_value`, `energy_uncertainty`, `spin_parity_list`, `spin_parity_string`.
    - FRIBND: The standardized key `spin_parity_string` stores the JSON `spinParity.evaluatorInput` text (the human-readable Jπ string).

2.  **Physics Feature Extraction** (`calculate_*_similarity`):
    - Mathematical comparison of level attributes using `Scoring_Config` weights.
    - **Energy**: Gaussian similarity based on Z-score overlap (0.0 to 1.0).
    - **Spin (J)**: Physics-informed scoring (1.0 for match, 0.0 for prohibition). Handles ranges and tentative assignments.
    - **Parity (Pi)**: Strict parity checking (Match vs Mismatch).

3.  **Feature Vector Construction** (`extract_features`):
    - Aggregates individual scores into a numerical vector for the ML model.
    - Vector Format: `[Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]`
    - Metrics:
        - *Similarity*: How well values match (Physics inputs).
        - *Certainty*: Penalty for unsure/tentative data (1.0=Firm, 0.0=Tentative).
        - *Specificity*: Penalty for ambiguous multiple options (1.0=Specific, <1.0=Ambiguous).

4.  **Training Data Generation** (`get_training_data`):
    - Generates synthetic "Gold Standard" pairs to teach the XGBoost model core physics constraints.
"""

import re
import numpy as np
import json
import os

# ==========================================
# Configuration: Scoring Parameters
# ==========================================
# This section defines the weights used to calculate similarity between levels.
# Range: 0.0 (No Match) to 1.0 (Perfect Match).
# Modify these values to adjust the physics logic sensitivity.

Scoring_Config = {
    'Energy': {
        # Controls how strictly energy values must match. 
        # - Higher value (e.g. 1.0) = stricter (score drops fast if energy differs).
        # - Lower value (e.g. 0.1) = looser (score stays high even with differences).
        'Sigma_Scale': 0.1,
        
        # Default energy uncertainty (in keV) used when input data lacks uncertainty.
        'Default_Uncertainty': 10.0
    },
    'Spin': {
        # Similarity scores for Spin (J) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # e.g.: 2+ vs 2+ (both firm)
        'Match_Tentative': 0.9,    # e.g.: tentative 2+ vs firm 2+
        'Mismatch_Weak': 0.2,      # e.g.: firm 2 vs tentative 3 (ΔJ=1, any tentative)
        'Mismatch_Strong': 0.0,    # e.g.: 2 vs 3 (ΔJ=1, both firm)
        'Veto': 0.0               # e.g.: 2 vs 4 (ΔJ>1)
    },
    'Parity': {
        # Similarity scores for Parity (Pi) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # e.g.: + vs + (both firm)
        'Match_Tentative': 0.9,    # e.g.: tentative + vs firm +
        'Mismatch_Tentative': 0.2, # e.g.: firm + vs tentative -
        'Mismatch_Firm': 0.0       # e.g.: + vs - (both firm)
    },
    'General': {
        # Score used when data is missing (e.g. unknown).
        'Neutral_Score': 0.5
    }
}

def load_levels_from_json(dataset_codes):
    """
    Parses JSON files (modern schema) for the given dataset codes and returns a list of standardized level dictionaries.
    Input files: test_dataset_{code}.json
        Standardized Keys: dataset_code, level_id, energy_value, energy_uncertainty, spin_parity_list, spin_parity_string.
        FRIBND: `spin_parity_string` is the string taken from `spinParity.evaluatorInput`.
    Example Output (includes an ENSDF-style (3/2,5/2)+ case):
      [
        {
            'dataset_code': 'A',
            'level_id': 'A_1000',
            'energy_value': 1000.0,
            'energy_uncertainty': 1.0,
            'spin_parity_list': [
                {'twoTimesSpin': 3, 'isTentativeSpin': True, 'parity': '+', 'isTentativeParity': False},
                {'twoTimesSpin': 5, 'isTentativeSpin': True, 'parity': '+', 'isTentativeParity': False}
            ],
            'spin_parity_string': '(3/2,5/2)+'
        }
      ]
    """
    levels = []
    for dataset_code in dataset_codes:
        # Strictly use the test_dataset_{code}.json files with the new schema
        filename = f"test_dataset_{dataset_code}.json"

        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Format: New ENSDF JSON schema (levelsTable -> levels)
                if isinstance(data, dict) and 'levelsTable' in data:
                    raw_levels = data['levelsTable'].get('levels', [])
                    for item in raw_levels:
                        energy_value = item.get('energy', {}).get('value')
                        energy_uncertainty = item.get('energy', {}).get('uncertainty', {}).get('value')
                        spin_parity_list = item.get('spinParity', {}).get('values', [])
                        spin_parity_string = item.get('spinParity', {}).get('evaluatorInput') 
                        
                        if energy_value is not None:
                            levels.append({
                                'dataset_code': dataset_code,
                                'level_id': f"{dataset_code}_{int(energy_value)}",
                                'energy_value': float(energy_value),
                                'energy_uncertainty': float(energy_uncertainty) if energy_uncertainty is not None else 10.0,
                                'spin_parity_list': spin_parity_list,
                                'spin_parity_string': spin_parity_string
                            })
    return levels


def calculate_energy_similarity(energy_1, energy_uncertainty_1, energy_2, energy_uncertainty_2):
    """
    # FRIBND: Energy similarity (Gaussian kernel)
    # Definitions:
    #   ΔE = |E1 − E2|
    #   σ_c = sqrt(σ1^2 + σ2^2)  (combined uncertainty)
    #   z   = ΔE / σ_c
    # Formula:
    #   similarity = exp( - Sigma_Scale * z^2 )
    #   where Sigma_Scale = Scoring_Config['Energy']['Sigma_Scale'] (default 0.1)
    # Range and behavior (Sigma_Scale = 0.1): z=0 → 1.00; z=1 → ~0.90; z=2 → ~0.67; z=3 → ~0.41
    # Rationale: measure energy separation in units of combined experimental uncertainty and map smoothly to (0, 1].
    """
    if energy_1 is None or energy_2 is None: return 0.0
    combined_uncertainty = np.sqrt(energy_uncertainty_1**2 + energy_uncertainty_2**2)
    if combined_uncertainty == 0: combined_uncertainty = 1.0
    
    z_score = abs(energy_1 - energy_2) / combined_uncertainty
    return np.exp(-Scoring_Config['Energy']['Sigma_Scale'] * z_score**2)


def calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates spin similarity (0 to 1) for matching levels across experimental datasets.
    FRIBND: Empirical Physics Logic - Two levels from different experiments are likely the same level if spins match:
            - ΔJ = 0 with both firm, e.g., 2 vs 2: Strongest match → score 1.0
            - ΔJ = 0 with any tentative, e.g., 2 vs (2): Good match with slightly reduced → score 0.9
            - ΔJ = 1 with both firm, e.g., 2 vs 3: Strong mismatch → score 0.0
            - ΔJ = 1 with any tentative, e.g., 2 vs (3): Weak mismatch → score 0.2
            - ΔJ ≥ 2: Strong mismatch regardless of tentativeness →  score 0.0
    """
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # FRIBND: Extract spins as J (convert twoTimesSpin → J by dividing by 2)
    # FRIBND: JSON stores 2J as integers (e.g., 3 → J=3/2). We transform to J here
    # FRIBND: so downstream logic compares ΔJ = |J1 − J2| directly.
    spins_1 = [
        (state.get('twoTimesSpin') / 2.0, state.get('isTentativeSpin', False))
        for state in spin_parity_list_1 if state.get('twoTimesSpin') is not None
    ]
    spins_2 = [
        (state.get('twoTimesSpin') / 2.0, state.get('isTentativeSpin', False))
        for state in spin_parity_list_2 if state.get('twoTimesSpin') is not None
    ]

    if not spins_1 or not spins_2:
        return Scoring_Config['General']['Neutral_Score']

    max_similarity_score = 0.0
    
    for j1, spin1_is_tentative in spins_1:
        for j2, spin2_is_tentative in spins_2:
            # FRIBND: Calculate spin difference and determine if both measurements are firm
            spin_difference = abs(j1 - j2)
            both_firm = (not spin1_is_tentative) and (not spin2_is_tentative)
            
            # FRIBND: Scoring logic (decision table)
            # ΔJ = 0: Match           → 1.0 (both firm) or 0.9 (any tentative)
            # ΔJ = 1: Adjacent        → 0.0 (both firm) or 0.2 (any tentative)
            # ΔJ = 2: Large mismatch  → 0.0 (both firm) or 0.1 (any tentative)
            # ΔJ > 2: Veto            → 0.0 always
            if spin_difference == 0.0:
                pair_similarity = Scoring_Config['Spin']['Match_Firm'] if both_firm else Scoring_Config['Spin']['Match_Tentative']
            elif spin_difference == 1.0:
                pair_similarity = Scoring_Config['Spin']['Veto'] if both_firm else Scoring_Config['Spin']['Mismatch_Weak']
            elif spin_difference == 2.0:
                pair_similarity = Scoring_Config['Spin']['Veto'] if both_firm else 0.1
            else:
                pair_similarity = Scoring_Config['Spin']['Veto']

            if pair_similarity > max_similarity_score:
                max_similarity_score = pair_similarity
    
    return max_similarity_score

def calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates parity similarity (0 to 1) for matching levels across experimental datasets.
    FRIBND: Empirical Physics Logic - Two levels from different experiments are likely the same level if parities match:
      - Same parity with both firm, e.g., + vs + or - vs -: Strongest match → score 1.0
      - Same parity with any tentative, e.g., + vs (+): Good match but reduced → score 0.9
      - Different parity with both firm, e.g., + vs -: Strong mismatch → score 0.0
      - Different parity with any tentative, e.g., + vs (-): Weak mismatch → score 0.2
    """
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract parities (+1, -1) and tentative flags
    parities_1 = [(state.get('parity'), state.get('isTentativeParity', False)) 
                  for state in spin_parity_list_1 if state.get('parity') is not None]
    parities_2 = [(state.get('parity'), state.get('isTentativeParity', False)) 
                  for state in spin_parity_list_2 if state.get('parity') is not None]

    if not parities_1 or not parities_2:
        return Scoring_Config['General']['Neutral_Score']

    max_parity_similarity_score = 0.0
    
    for parity_signal_1, is_tentative_1 in parities_1:
        for parity_signal_2, is_tentative_2 in parities_2:
            # FRIBND: Determine match/mismatch and firmness
            parities_match = (parity_signal_1 == parity_signal_2)
            both_firm = (not is_tentative_1) and (not is_tentative_2)
            
            # FRIBND: Scoring logic (decision table)
            # Match:    → 1.0 (both firm) or 0.9 (any tentative)
            # Mismatch: → 0.0 (both firm) or 0.2 (any tentative)
            if parities_match:
                pair_similarity = Scoring_Config['Parity']['Match_Firm'] if both_firm else Scoring_Config['Parity']['Match_Tentative']
            else:
                pair_similarity = Scoring_Config['Parity']['Mismatch_Firm'] if both_firm else Scoring_Config['Parity']['Mismatch_Tentative']
            
            if pair_similarity > max_parity_similarity_score:
                max_parity_similarity_score = pair_similarity
                
    return max_parity_similarity_score

def extract_features(level_1, level_2):
    """
    Constructs six-dimensional feature vector from two level dictionaries.
    FRIBND: Feature Design - All scores are monotonic increasing (higher value → better match)
    FRIBND: Output: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]
    """

    energy_similarity = calculate_energy_similarity(
        level_1.get('energy_value'), level_1.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty']),
        level_2.get('energy_value'), level_2.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty'])
    )
    
    spin_parity_list_1 = level_1.get('spin_parity_list', [])
    spin_parity_list_2 = level_2.get('spin_parity_list', [])

    spin_similarity = calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2)
    parity_similarity = calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2)
    
    # FRIBND: Features 4-5 - Certainty scores (penalize tentative assignments)
    # FRIBND: Firm assignment = 1.0, Tentative assignment = 0.0
    spin_is_tentative_1 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_1)
    spin_is_tentative_2 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_2)
    spin_certainty = 0.0 if (spin_is_tentative_1 or spin_is_tentative_2) else 1.0

    parity_is_tentative_1 = any(state.get('isTentativeParity', False) for state in spin_parity_list_1)
    parity_is_tentative_2 = any(state.get('isTentativeParity', False) for state in spin_parity_list_2)
    parity_certainty = 0.0 if (parity_is_tentative_1 or parity_is_tentative_2) else 1.0

    # FRIBND: Feature 6 - Specificity score (penalize ambiguous multiple options)
    # FRIBND: Formula: 1/(1+log10(multiplicity)) → 1 option=1.0, 10 options=0.5, 100 options=0.33
    options_count_1 = len(spin_parity_list_1) if spin_parity_list_1 else 1
    options_count_2 = len(spin_parity_list_2) if spin_parity_list_2 else 1
    multiplicity = max(1, options_count_1 * options_count_2)
    specificity = 1.0 / (1.0 + np.log10(multiplicity))
    
    return np.array([energy_similarity, spin_similarity, parity_similarity, spin_certainty, parity_certainty, specificity])

def get_training_data():
    """
    Generates synthetic training data encoding nuclear physics constraints.
    FRIBND: Strategy - Create labeled examples covering key physics scenarios:
      1. Perfect matches (high energy + good physics → high probability)
      2. Physics vetoes (spin/parity violation → zero probability)
      3. Energy mismatches (poor energy overlap → low probability)
      4. Ambiguous cases (unknown physics → energy-dependent probability)
      5. Weak physics matches (marginal compatibility → reduced probability)
      6. Random background (fill feature space with rule-based labels)
    FRIBND: Output - (training_features, training_labels) for XGBoost model
    FRIBND: Feature Order: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]
    """
    training_points = []
    
    # FRIBND: Case 1 - Perfect match zone (high energy + good physics → high probability)
    # FRIBND: Energy > 0.8, Spin/Parity = 1.0 or 0.9 → Probability > 0.9
    for e in np.linspace(0.8, 1.0, 5):
        for s in [1.0, 0.9]: # Firm, Tentative
            for p in [1.0, 0.9]:
                prob = 0.9 + (e-0.8)/2 * (s * p) 
                training_points.append(([e, s, p, 1.0, 1.0, 1.0], min(prob, 0.99)))

    # FRIBND: Case 2 - Physics veto zone (spin or parity incompatible → zero probability)
    # FRIBND: Any energy, but Spin=0.0 or Parity=0.0 (strong incompatibility) → Probability=0.0
    for e in np.linspace(0.0, 1.0, 10):
        # Spin Veto
        training_points.append(([e, 0.0, 1.0, 1.0, 1.0, 1.0], 0.0))
        # Parity Veto
        training_points.append(([e, 1.0, 0.0, 1.0, 1.0, 1.0], 0.0))
        # Both Veto
        training_points.append(([e, 0.0, 0.0, 1.0, 1.0, 1.0], 0.0))

    # FRIBND: Case 3 - Energy mismatch zone (poor energy overlap → low probability)
    # FRIBND: Perfect physics but Energy < 0.2 → Probability decays to zero
    for e in np.linspace(0.0, 0.2, 5):
        training_points.append(([e, 1.0, 1.0, 1.0, 1.0, 1.0], e * 0.5))

    # FRIBND: Case 4 - Ambiguous physics zone (unknown spin/parity → energy-dependent)
    # FRIBND: Energy good, but Spin=Parity=0.5 (neutral/unknown) → Probability relies on energy
    for e in np.linspace(0.0, 1.0, 20):
        prob = e * 0.85  # Maximum 0.85 when physics unknown
        training_points.append(([e, 0.5, 0.5, 1.0, 1.0, 1.0], prob))

    # FRIBND: Case 5 - Weak physics match (marginal compatibility → reduced probability)
    # FRIBND: Energy good, Spin/Parity weak (0.2) → Low-to-mid probability
    for e in np.linspace(0.5, 1.0, 5):
        training_points.append(([e, 0.2, 1.0, 1.0, 1.0, 1.0], e * 0.4))
        training_points.append(([e, 1.0, 0.2, 1.0, 1.0, 1.0], e * 0.4))

    # FRIBND: Case 6 - Random background (fill feature space with rule-based labels)
    # FRIBND: Generate 500 random points, apply physics rules to determine labels
    np.random.seed(42)
    for _ in range(500):
        e = np.random.uniform(0, 1)
        s = np.random.choice([0.0, 0.2, 0.5, 0.9, 1.0])
        p = np.random.choice([0.0, 0.2, 0.5, 0.9, 1.0])
        cert_s = np.random.choice([0.0, 1.0])
        cert_p = np.random.choice([0.0, 1.0])
        spec = np.random.uniform(0.5, 1.0)
        
        # FRIBND: Apply rule-based labeling using physics constraints
        if s == 0.0 or p == 0.0:
            # FRIBND: Physics veto → zero probability
            label = 0.0
        elif e < 0.1:
            # FRIBND: Poor energy overlap → zero probability
            label = 0.0
        else:
            # FRIBND: Calculate base probability from energy and physics quality
            phys_factor = (s + p) / 2.0
            
            if s == 0.5 and p == 0.5:
                # FRIBND: Unknown physics → rely on energy only
                label = e * 0.8
            else:
                # FRIBND: Active physics → modulate energy-based probability
                label = e * phys_factor
                
            # FRIBND: Certainty penalty → reduce probability if tentative
            if cert_s == 0.0 or cert_p == 0.0:
                label *= 0.9
                
        training_points.append(([e, s, p, cert_s, cert_p, spec], label))

    training_features = np.array([record[0] for record in training_points])
    training_labels = np.array([record[1] for record in training_points])
    return training_features, training_labels


