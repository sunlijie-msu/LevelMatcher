"""
DATA PARSER & PHYSICS FEATURE ENGINEERING
======================================

Explanation of Code Structure:
-------------------------
1.  **Data Loading** (`load_levels_from_json`):
    - Ingests nuclear level data from JSON files (ENSDF schema).
    - Standardizes attributes into flat dictionaries: `energy_value`, `energy_uncertainty`, `spin_parity_list`.

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
        # - Higher Value (e.g. 1.0) = Stricter (Score drops fast if energy differs).
        # - Lower Value (e.g. 0.1) = Looser (Score stays high even with differences).
        'Sigma_Scale': 0.5,
        
        # Default energy uncertainty (in keV) used when input data lacks uncertainty.
        'Default_Uncertainty': 10.0
    },
    'Spin': {
        # Similarity scores for Spin (J) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # Strongest Match: 2+ vs 2+
        'Match_Tentative': 0.9,    # Good Match: (2+) vs 2+
        'Mismatch_Weak': 0.25,     # Weak Conflict: 2 vs (3) (Differs by 1, tentative)
        'Mismatch_Strong': 0.0,    # Strong Conflict: 2 vs 3 (Differs by 1, firm)
        'Veto': 0.0               # Impossible: 2 vs 4 (Differs by >1)
    },
    'Parity': {
        # Similarity scores for Parity (Pi) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # Strongest Match: + vs +
        'Match_Tentative': 0.9,    # Good Match: (+) vs +
        'Mismatch_Tentative': 0.2, # Weak Conflict: + vs (-)
        'Mismatch_Firm': 0.0       # Strong Conflict: + vs -
    },
    'General': {
        # Score used when data is missing (e.g. unknown).
        'Neutral_Score': 0.5
    }
}

def load_levels_from_json(dataset_codes):
    """
    Parses JSON files for the given dataset codes and returns a list of standardized level dictionaries.
    """
    levels = []
    for dataset_code in dataset_codes:
        filename = f"test_dataset_{dataset_code}.json"
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle new ENSDF JSON schema (levelsTable -> levels)
                if isinstance(data, dict) and 'levelsTable' in data:
                    raw_levels = data['levelsTable'].get('levels', [])
                    for item in raw_levels:
                        # Extract Energy
                        energy_value = item.get('energy', {}).get('value')
                        
                        # Extract Energy Uncertainty
                        energy_uncertainty = item.get('energy', {}).get('uncertainty', {}).get('value')
                        
                        # Extract Spin_Parity Values
                        spin_parity_list = item.get('spinParity', {}).get('values', [])
                        spin_parity_string = item.get('spinParity', {}).get('evaluatorInput') 
                        
                        if energy_value is not None:
                            levels.append({
                                'dataset_code': dataset_code,
                                'energy_value': float(energy_value),
                                'energy_uncertainty': float(energy_uncertainty) if energy_uncertainty is not None else 10.0,
                                'spin_parity_list': spin_parity_list,
                                'spin_parity': spin_parity_string
                            })
                elif isinstance(data, list):
                    # Fallback for flat list format
                    for item in data:
                        item['dataset_code'] = dataset_code
                        levels.append(item)
    return levels


# Feature engineering is one of the most critical steps in building high-performance machine learning models.

# Feature engineering is the process of transforming raw data into meaningful features that make it easier for the machine learning model to understand patterns.

# Well-engineered features can significantly boost XGBoost’s performance, leading to improved accuracy and predictive power.

def calculate_energy_similarity(energy_1, energy_uncertainty_1, energy_2, energy_uncertainty_2):
    """
    Calculates energy similarity based on Gaussian kernel of Z-score.
    Returns 1.0 for perfect match, decays to 0.0 for large differences.
    """
    if energy_1 is None or energy_2 is None: return 0.0
    combined_uncertainty = np.sqrt(energy_uncertainty_1**2 + energy_uncertainty_2**2)
    if combined_uncertainty == 0: combined_uncertainty = 1.0
    
    z_score = abs(energy_1 - energy_2) / combined_uncertainty
    # Gaussian similarity: exp(-0.5 * z^2). 
    return np.exp(-1.0 * Scoring_Config['Energy']['Sigma_Scale'] * z_score**2)

def calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates a similarity score (0.0 to 1.0) based on nuclear physics logic.
    """
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract spins (2*J) and tentative status
    spins_1 = [(state.get('twoTimesSpin'), state.get('isTentativeSpin', False)) 
               for state in spin_parity_list_1 if state.get('twoTimesSpin') is not None]
    spins_2 = [(state.get('twoTimesSpin'), state.get('isTentativeSpin', False)) 
               for state in spin_parity_list_2 if state.get('twoTimesSpin') is not None]

    if not spins_1 or not spins_2:
        return Scoring_Config['General']['Neutral_Score']

    max_similarity_score = 0.0
    
    for two_times_j1, spin1_is_tentative in spins_1:
        for two_times_j2, spin2_is_tentative in spins_2:
            spin_distance = abs(two_times_j1 - two_times_j2) / 2.0 
            pair_similarity = 0.0
            
            if spin_distance == 0.0:
                pair_similarity = Scoring_Config['Spin']['Match_Firm'] if (not spin1_is_tentative and not spin2_is_tentative) else Scoring_Config['Spin']['Match_Tentative']
            elif spin_distance == 1.0:
                pair_similarity = Scoring_Config['Spin']['Veto'] if (not spin1_is_tentative and not spin2_is_tentative) else Scoring_Config['Spin']['Mismatch_Weak']
            elif spin_distance == 2.0:
                pair_similarity = Scoring_Config['Spin']['Veto'] if (not spin1_is_tentative and not spin2_is_tentative) else 0.1 # Very weak match
            else:
                pair_similarity = Scoring_Config['Spin']['Veto']

            if pair_similarity > max_similarity_score:
                max_similarity_score = pair_similarity
    
    return max_similarity_score

def calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates a parity similarity score.
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
            is_match = (parity_signal_1 == parity_signal_2)
            pair_similarity = 0.0
            
            if is_match:
                pair_similarity = Scoring_Config['Parity']['Match_Firm'] if (not is_tentative_1 and not is_tentative_2) else Scoring_Config['Parity']['Match_Tentative']
            else:
                pair_similarity = Scoring_Config['Parity']['Mismatch_Firm'] if (not is_tentative_1 and not is_tentative_2) else Scoring_Config['Parity']['Mismatch_Tentative']
            
            if pair_similarity > max_parity_similarity_score:
                max_parity_similarity_score = pair_similarity
                
    return max_parity_similarity_score

def extract_features(level_1, level_2):
    """
    Calculates similarity scores between two nuclear level dictionaries.
    Inputs: level_1, level_2 (dictionaries with keys energy_value, energy_uncertainty, spin_parity_list)
    Output: numpy array containing physics-informed similarity scores (Higher is Better).
    """
    # 1. Energy Similarity
    energy_similarity = calculate_energy_similarity(
        level_1.get('energy_value'), level_1.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty']),
        level_2.get('energy_value'), level_2.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty'])
    )
    
    spin_parity_list_1 = level_1.get('spin_parity_list', [])
    spin_parity_list_2 = level_2.get('spin_parity_list', [])

    # 2. Comparison of Physics Properties
    spin_similarity = calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2)
    parity_similarity = calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2)
    
    # 3. Certainty Score (Inverse of Tentativeness)
    # 1.0 = Firm, 0.0 = Tentative
    spin_is_tentative_1 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_1)
    spin_is_tentative_2 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_2)
    spin_certainty = 0.0 if (spin_is_tentative_1 or spin_is_tentative_2) else 1.0

    parity_is_tentative_1 = any(state.get('isTentativeParity', False) for state in spin_parity_list_1)
    parity_is_tentative_2 = any(state.get('isTentativeParity', False) for state in spin_parity_list_2)
    parity_certainty = 0.0 if (parity_is_tentative_1 or parity_is_tentative_2) else 1.0

    # 4. Speficicity Score (Inverse of Multiplicity/Ambiguity)
    # 1.0 = Single Option, decays as log10(options) increases.
    options_count_1 = len(spin_parity_list_1) if spin_parity_list_1 else 1
    options_count_2 = len(spin_parity_list_2) if spin_parity_list_2 else 1
    multiplicity = max(1, options_count_1 * options_count_2)
    specificity = 1.0 / (1.0 + np.log10(multiplicity)) # 1 option -> 1.0. 10 options -> 0.5.
    
    # Feature Vector ordering: All scores are "Higher is Better"
    # [Energy, Spin, Parity, SpinCertainty, ParityCertainty, Specificity]
    return np.array([energy_similarity, spin_similarity, parity_similarity, spin_certainty, parity_certainty, specificity])

def get_training_data():
    """
    Returns (training_features, training_labels) for training the XGBoost Model.
    Feature Order: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]
    All features are Monotonic Increasing (High Score == Better Match).
    """
    training_records = []

    # 1. Gold Standard Match
    # Perfect Energy(1.0), Firm Spin(1.0), Firm Parity(1.0), Firm Certainty(1.0), Specific(1.0)
    training_records.append(([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.00))
    # Good Energy (0.8 ~ Z=0.6)
    training_records.append(([0.8, 1.0, 1.0, 1.0, 1.0, 1.0], 0.95))

    # 2. Spin Veto
    # Hard conflict in spin (0.0). Match is impossible.
    training_records.append(([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], 0.00))

    # 3. Parity Veto
    # Hard conflict in parity (0.0). Match is impossible.
    training_records.append(([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 0.00))

    # 4. Tentative Sensitivity
    # (2+) vs 2+ is a good match (0.9 physics score) but less certain (Certainty=0.0).
    # We penalize slightly for lack of certainty.
    training_records.append(([1.0, 0.9, 1.0, 0.0, 1.0, 1.0], 0.90))

    # 5. Energy Degradation
    # Good physics, but Energy is far away (Energy_Similarity=0.1)
    training_records.append(([0.1, 1.0, 1.0, 1.0, 1.0, 1.0], 0.10))
    # Very far (Energy_Similarity=0.0)
    training_records.append(([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.00))

    # 6. High Ambiguity (Low Specificity)
    # Search space is large (Specificity ~0.4), lowering probability.
    training_records.append(([1.0, 1.0, 1.0, 1.0, 1.0, 0.4], 0.80))
    
    # 7. Information Void
    # No J or Pi information (0.5 scores). Match relies entirely on energy.
    training_records.append(([1.0, 0.5, 0.5, 1.0, 1.0, 1.0], 0.85))
    # Bad energy with no physics info
    training_records.append(([0.1, 0.5, 0.5, 1.0, 1.0, 1.0], 0.10))


    training_features = np.array([record[0] for record in training_records])
    training_labels = np.array([record[1] for record in training_records])
    return training_features, training_labels


