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
        'Sigma_Scale': 0.1,
        
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
    Parses JSON files (modern schema) for the given dataset codes and returns a list of standardized level dictionaries.
    Target files: test_dataset_{code}.json
    Standardized Keys: energy_value, energy_uncertainty, spin_parity_list, spin_parity.
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
                                'spin_parity': spin_parity_string
                            })
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
    training_points = []
    
    # Generate synthetic data covering the feature space
    # Simulating: Energy (0-1), Spin (0-1), Parity (0-1), Certainties (0/1), Specificity (0-1)
    
    # Case 1: The "Perfect Match" Zone
    # High Energy (>0.8), Good Physics (>0.9)
    # Result: High Probability (>0.9)
    for e in np.linspace(0.8, 1.0, 5):
        for s in [1.0, 0.9]: # Firm, Tentative
            for p in [1.0, 0.9]:
                prob = 0.9 + (e-0.8)/2 * (s * p) 
                training_points.append(([e, s, p, 1.0, 1.0, 1.0], min(prob, 0.99)))

    # Case 2: The "Physics Veto" Zone
    # Any Energy, but Spin or Parity == 0.0 (Mismatch)
    # Result: 0.0 Probability
    for e in np.linspace(0.0, 1.0, 10):
        # Spin Veto
        training_points.append(([e, 0.0, 1.0, 1.0, 1.0, 1.0], 0.0))
        # Parity Veto
        training_points.append(([e, 1.0, 0.0, 1.0, 1.0, 1.0], 0.0))
        # Both Veto
        training_points.append(([e, 0.0, 0.0, 1.0, 1.0, 1.0], 0.0))

    # Case 3: The "Energy Mismatch" Zone
    # Perfect Physics, but Energy is far off (<0.1)
    # Result: Low Probability (~0.0)
    for e in np.linspace(0.0, 0.2, 5):
        training_points.append(([e, 1.0, 1.0, 1.0, 1.0, 1.0], e * 0.5)) # Decays to 0

    # Case 4: The "Grey Zone" (Ambiguous Physics)
    # Energy is Good, but Physics is Neutral (0.5) (e.g. "unknown" spins)
    # Result: Probability relies on Energy
    for e in np.linspace(0.0, 1.0, 20):
        # Neutral Physics
        prob = e * 0.85 # Max 0.85 if physics is unknown
        training_points.append(([e, 0.5, 0.5, 1.0, 1.0, 1.0], prob))

    # Case 5: "Weak Physics Match"
    # Energy Good, Spin/Parity Weak (0.2 - 0.25)
    # Result: Low to Mid Probability
    for e in np.linspace(0.5, 1.0, 5):
        training_points.append(([e, 0.25, 1.0, 1.0, 1.0, 1.0], e * 0.4))
        training_points.append(([e, 1.0, 0.2, 1.0, 1.0, 1.0], e * 0.4))

    # Case 6: Random Noise / Background
    # Generate random points to fill the space
    np.random.seed(42)
    for _ in range(500):
        e = np.random.uniform(0, 1)
        s = np.random.choice([0.0, 0.25, 0.5, 0.9, 1.0])
        p = np.random.choice([0.0, 0.2, 0.5, 0.9, 1.0])
        cert_s = np.random.choice([0.0, 1.0])
        cert_p = np.random.choice([0.0, 1.0])
        spec = np.random.uniform(0.5, 1.0)
        
        # Rule Based Labeling
        if s == 0.0 or p == 0.0:
            label = 0.0
        elif e < 0.1:
            label = 0.0
        else:
            # Base probability is Energy * Physics Quality
            # Physics Factor: Average of Spin/Parity, penalized if < 0.5 is rare
            phys_factor = (s + p) / 2.0
            
            # If Physics is neutral (0.5), we rely on Energy
            if s == 0.5 and p == 0.5:
                label = e * 0.8
            else:
                # If Physics is active, it boosts or suppresses
                label = e * phys_factor
                
            # Certainty Penalty
            if cert_s == 0.0 or cert_p == 0.0:
                label *= 0.9
                
        training_points.append(([e, s, p, cert_s, cert_p, spec], label))

    training_features = np.array([record[0] for record in training_points])
    training_labels = np.array([record[1] for record in training_points])
    return training_features, training_labels


