"""
Feature Engineering For Nuclear Level Matching
======================================

Explanation of Code Structure:
-------------------------
1.  **Data Loading** (`load_levels_from_json`):
    - Ingests nuclear level data from JSON files (ENSDF schema).
    - Standardizes attributes into flat dictionaries: `energy_value`, `energy_uncertainty`, `spin_parity_list`, `spin_parity_string`.
    - The standardized key `spin_parity_string` stores the JSON `spinParity.evaluatorInput` text (the 80-col style Jπ string).

2.  **Physics Feature Extraction** (`calculate_*_similarity`):
    - Mathematical comparison of level attributes using `Scoring_Config` weights.
    - **Energy**: Gaussian similarity based on Z-score overlap (0.0 to 1.0).
    - **Spin (J)**: Physics-informed scoring (1.0 for match, 0.0 for mismatch). Handles ranges and tentative assignments.
    - **Parity (Pi)**: Physics-informed scoring (1.0 for match, 0.0 for mismatch). Handles tentative assignments.

3.  **Feature Vector Construction** (`extract_features`):
    - Aggregates individual scores into a numerical vector for the ML model.
    - Vector Format: `[Energy_Similarity, Spin_Similarity, Parity_Similarity, Specificity]`
    - Metrics:
        - *Similarity*: How well values match (Physics inputs). Tentativeness encoded within similarity scores.
        - *Specificity*: Penalty for multiple options (1.0=Specific, <1.0=Ambiguous).

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
# This section quantifies the similarity scoring for energy, spin, parity, and level property sensitivity/quality/specificity/certainty.
# Range: 0.0 (No Match) to 1.0 (Perfect Match).
# Modify these values to adjust the empirical physics scale.

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
        'Match_Firm': 1.0,         # both firm, e.g., 2 vs 2
        'Match_Strong': 0.9,    # any tentative, e.g., 2 vs (2)
        'Mismatch_Weak': 0.2,      # any tentative, e.g., 2 vs (3) with ΔJ=1
        'Mismatch_Strong': 0.05,    # both firm, e.g., 2 vs 3 with ΔJ=1
        'Mismatch_Firm': 0.0               # any ΔJ ≥ 2
    },
    'Parity': {
        # Similarity scores for Parity (Pi) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # both firm, e.g., + vs + or - vs -
        'Match_Strong': 0.9,    # any tentative, e.g., + vs (+)
        'Mismatch_Weak': 0.1, # any tentative, e.g., + vs (-)
        'Mismatch_Firm': 0.0       # both firm, e.g., + vs -
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
    # Energy similarity: exp(-Sigma_Scale × z²)
    # where z = ΔE/σ_c = energy difference in units of combined uncertainty
    #
    # Example: Level A (E=1000±5 keV) vs Level B (E=1010±3 keV)
    #   ΔE=10 keV, σ_c=√(25+9)=5.83 keV, z=1.72 → similarity=exp(-0.1×1.72²)=0.744 (74.4%)
    #
    # Similarity table vs energy separation (z in σ units):
    # ┌─────────────┬────────┬────────┬────────┬────────┬────────┐
    # │ Sigma_Scale │   1σ   │   2σ   │   3σ   │   4σ   │   5σ   │
    # ├─────────────┼────────┼────────┼────────┼────────┼────────┤
    # │ 0.1 loose │  90.5% │  67.0% │  40.7% │  20.2% │   8.2% │ lenient: tolerates large separations
    # │ 0.2 moderate│  81.9% │  44.9% │  16.5% │   4.1% │   0.7% │ standard: penalizes >2σ strongly
    # │ 0.5 strict  │  60.7% │  13.5% │   1.1% │   0.0% │   0.0% │ aggressive: rejects >2σ
    # │ 1.0 extreme │  36.8% │   1.8% │   0.0% │   0.0% │   0.0% │ ultra-strict: even 1σ penalized
    # └─────────────┴────────┴────────┴────────┴────────┴────────┘
    """
    if energy_1 is None or energy_2 is None: return 0.0
    combined_uncertainty = np.sqrt(energy_uncertainty_1**2 + energy_uncertainty_2**2)
    if combined_uncertainty == 0: combined_uncertainty = 1.0
    
    z_score = abs(energy_1 - energy_2) / combined_uncertainty
    return np.exp(-Scoring_Config['Energy']['Sigma_Scale'] * z_score**2)


def calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates spin similarity (0 to 1) for matching levels across experimental datasets.
    
    ENSDF Notation Distinctions:
      - Firm multiple:    2,3      = J is definitely 2 or 3 (limited to 2 options)
      - Tentative multiple: (2,3)  = J is tentatively 2 or 3 (and possibly others)
      - Firm single:      2        = J is definitely 2
      - Tentative single: (2)      = J is tentatively 2
    
    Matching Strategy (Optimistic):
      - Compare ALL spin combinations between two levels using nested loop
      - Keep the BEST (maximum) similarity score across all pairs
      - Rationale: If any option matches well, levels likely identical
      - Example: Level A has J=2, Level B has J=(2,3)
        → Compare 2 vs 2 (score 0.9, tentative penalty)
        → Compare 2 vs 3 (score 0.2, adjacent mismatch)
        → Return max(0.9, 0.2) = 0.9 (optimistic: assume match)
    
    Scoring Rules:
      - ΔJ = 0 with both firm, e.g., 2 vs 2: Strongest match → score 1.0
      - ΔJ = 0 with any tentative, e.g., 2 vs (2): Good match with penalty → score 0.9
      - ΔJ = 1 with both firm, e.g., 2 vs 3: Strong mismatch → score 0.05
      - ΔJ = 1 with any tentative, e.g., 2 vs (3): Weak mismatch → score 0.2
      - ΔJ ≥ 2: Strong mismatch regardless of tentativeness → score 0.0
    """
    # Guard: If either level has no spin data, return neutral score (cannot compare)
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract spin values (J) from both levels
    # Input: twoTimesSpin (2J stored as integer in JSON, e.g., 3 means J=3/2)
    # Output: list of (J, is_tentative) tuples
    spins_1 = [
        (state['twoTimesSpin'] / 2.0, state['isTentativeSpin'])
        for state in spin_parity_list_1 if state.get('twoTimesSpin') is not None
    ]
    spins_2 = [
        (state['twoTimesSpin'] / 2.0, state['isTentativeSpin'])
        for state in spin_parity_list_2 if state.get('twoTimesSpin') is not None
    ]

    # Guard: If filtering removed all spin values (all were None), return neutral score
    if not spins_1 or not spins_2:
        return Scoring_Config['General']['Neutral_Score']

    # Optimistic matching: Compare all spin pairs and keep the best (maximum) similarity score
    # If ANY spin combination matches, accept two levels match compatibility
    max_similarity_score = 0.0
    
    for j1, spin1_is_tentative in spins_1:
        for j2, spin2_is_tentative in spins_2:
            # Calculate spin difference ΔJ and determine if both measurements are firm
            spin_difference = abs(j1 - j2)
            both_firm = (not spin1_is_tentative) and (not spin2_is_tentative)
            
            # Apply scoring rules based on ΔJ and firmness
            # ΔJ = 0: Match      → 1.0 (both firm) or 0.9 (any tentative)
            # ΔJ = 1: Adjacent   → 0.05 (both firm) or 0.2 (any tentative)
            # ΔJ ≥ 2: Mismatch  → 0.0 always
            
            if spin_difference == 0.0:
                # Identical spins → evidence for same level
                if both_firm:
                    pair_similarity = Scoring_Config['Spin']['Match_Firm']
                else:
                    pair_similarity = Scoring_Config['Spin']['Match_Strong']
                    
            elif spin_difference == 1.0:
                # Adjacent spins (ΔJ=1) → evidence for different levels
                if both_firm:
                    pair_similarity = Scoring_Config['Spin']['Mismatch_Strong']
                else:
                    pair_similarity = Scoring_Config['Spin']['Mismatch_Weak']
                    
            else:
                # Large spin difference (ΔJ≥2) → definitely different levels
                pair_similarity = Scoring_Config['Spin']['Mismatch_Firm']

            # Keep the best score across all spin pairs
            if pair_similarity > max_similarity_score:
                max_similarity_score = pair_similarity
    
    return max_similarity_score

def calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates parity similarity (0 to 1) for matching levels across experimental datasets.
    
    ENSDF Notation Distinctions (Parity is binary: + or -, never both):
      - Firm:      +        = Parity is definitely positive
      - Firm:      -        = Parity is definitely negative
      - Tentative: (+)      = Parity is tentatively positive
      - Tentative: (-)      = Parity is tentatively negative
    
    Note: Unlike spin, parity CANNOT have multiple values like "+,-" because parity is a 
    discrete binary quantum number. A level has either positive OR negative parity, never both.
    Multiple parity entries in spin_parity_list occur when BOTH spin AND parity are uncertain
    (e.g., "2+,3-" means J=2 with π=+ OR J=3 with π=-, correlated uncertainty).
    
    Matching Strategy (Optimistic):
      - Compare ALL parity combinations between two levels using nested loop
      - Keep the BEST (maximum) similarity score across all pairs
      - Rationale: If any option matches well, levels likely identical
      - Example: Level A has +, Level B has correlated options (2+, 3-)
        → Compare + vs + (score 1.0, both firm)
        → Compare + vs - (score 0.0, firm mismatch)
        → Return max(1.0, 0.0) = 1.0 (optimistic: assume J=2,π=+ match)
    
    Scoring Rules:
      - Same parity with both firm, e.g., + vs +: Strongest match → score 1.0
      - Same parity with any tentative, e.g., + vs (+): Good match with penalty → score 0.9
      - Different parity with both firm, e.g., + vs -: Strong mismatch → score 0.0
      - Different parity with any tentative, e.g., + vs (-): Weak mismatch → score 0.1
    """
    # Guard: If either level has no spin/parity data, return neutral score (cannot compare)
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract parity values (+/-) from both levels
    # Input: parity ('+' or '-' string in JSON)
    # Output: list of (parity, is_tentative) tuples
    parities_1 = [
        (state['parity'], state['isTentativeParity'])
        for state in spin_parity_list_1 if state.get('parity') is not None
    ]
    parities_2 = [
        (state['parity'], state['isTentativeParity'])
        for state in spin_parity_list_2 if state.get('parity') is not None
    ]

    # Guard: If filtering removed all parity values (all were None), return neutral score
    if not parities_1 or not parities_2:
        return Scoring_Config['General']['Neutral_Score']

    # Optimistic matching: Compare all parity pairs and keep the best (maximum) similarity score
    # If ANY parity combination matches, accept two levels match compatibility
    max_parity_similarity_score = 0.0
    
    for p1, is_tentative_1 in parities_1:
        for p2, is_tentative_2 in parities_2:
            # Determine if parities match and if both measurements are firm
            parities_match = (p1 == p2)
            both_firm = (not is_tentative_1) and (not is_tentative_2)
            
            # Apply scoring rules based on match/mismatch and firmness
            # Match:    → 1.0 (both firm) or 0.9 (any tentative)
            # Mismatch: → 0.0 (both firm) or 0.1 (any tentative)
            
            if parities_match:
                # Same parity → evidence for same level
                if both_firm:
                    pair_similarity = Scoring_Config['Parity']['Match_Firm']
                else:
                    pair_similarity = Scoring_Config['Parity']['Match_Strong']
            else:
                # Different parity → evidence for different levels
                if both_firm:
                    pair_similarity = Scoring_Config['Parity']['Mismatch_Firm']
                else:
                    pair_similarity = Scoring_Config['Parity']['Mismatch_Weak']
            
            # Keep the best score across all parity pairs
            if pair_similarity > max_parity_similarity_score:
                max_parity_similarity_score = pair_similarity
                
    return max_parity_similarity_score

def extract_features(level_1, level_2):
    """
    Constructs four-dimensional feature vector from two level dictionaries.
    Feature Design - All scores are monotonic increasing (higher value → better match)
    Output: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Specificity]
    Note: Tentativeness is encoded within similarity scores (1.0=firm match, 0.9=tentative match)
    """

    energy_similarity = calculate_energy_similarity(
        level_1.get('energy_value'), level_1.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty']),
        level_2.get('energy_value'), level_2.get('energy_uncertainty', Scoring_Config['Energy']['Default_Uncertainty'])
    )
    
    spin_parity_list_1 = level_1.get('spin_parity_list', [])
    spin_parity_list_2 = level_2.get('spin_parity_list', [])

    spin_similarity = calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2)
    parity_similarity = calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2)

    # Feature 4: Specificity score (penalize ambiguous multiple options)
    # Formula: specificity = 1/(1+log10(multiplicity))
    #   where multiplicity = options_count_1 × options_count_2
    #
    # ENSDF spin option counts (typical real data):
    #   Single:   "2+" → 1 option
    #   Double:   "1/2+,3/2+" or "(1,2)+" → 2 options
    #   Multiple: "1/2,3/2,5/2" → 3 options
    #   Multiple:  "1/2:11/2" expand to "1/2,3/2,5/2,7/2,9/2,11/2" 6 options
    #
    # Specificity table for realistic ENSDF scenarios:
    # ┌──────────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    # │ Multiplicity │    1   │    2   │    3   │    4   │    6   │    9   │
    # ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────┤
    # │ Specificity  │  1.000 │  0.769 │  0.677 │  0.624 │  0.563 │  0.512 │
    # └──────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    #   mult=1: both levels have single definite Jπ (most specific)
    #   mult=2: one level has 2 options (e.g., "1,2" vs single)
    #   mult=3: one level has 3 options (e.g., "1/2,3/2,5/2" vs single)
    #   mult=4: both have 2 options or one has 4 (e.g., "1:4")
    #   mult=6: one has 2, other has 3 options (moderate ambiguity)
    #   mult=9: both have 3 options (e.g., "1,2,3" vs "2,3,4" - high ambiguity)
    #
    # Note: multiplicity>10 extremely rare in ENSDF (would require 4+ options on both sides)
    options_count_1 = len(spin_parity_list_1) if spin_parity_list_1 else 1
    options_count_2 = len(spin_parity_list_2) if spin_parity_list_2 else 1
    multiplicity = max(1, options_count_1 * options_count_2)
    specificity = 1.0 / (1.0 + np.log10(multiplicity))

    # ===== DISABLED CERTAINTY FEATURES (easily re-enable if needed) =====
    # Rationale: Certainty is redundant - already encoded in similarity scores:
    #   - Match: both firm → 1.0, any tentative → 0.9 (certainty fully predictable)
    #   - Mismatch: certainty adds independent signal but creates feature redundancy
    # To re-enable: uncomment lines below, change return to 6D vector, update training data
    #
    # spin_is_tentative_1 = any(state['isTentativeSpin'] for state in spin_parity_list_1 if 'isTentativeSpin' in state)
    # spin_is_tentative_2 = any(state['isTentativeSpin'] for state in spin_parity_list_2 if 'isTentativeSpin' in state)
    # spin_certainty = 0.0 if (spin_is_tentative_1 or spin_is_tentative_2) else 1.0
    #
    # parity_is_tentative_1 = any(state['isTentativeParity'] for state in spin_parity_list_1 if 'isTentativeParity' in state)
    # parity_is_tentative_2 = any(state['isTentativeParity'] for state in spin_parity_list_2 if 'isTentativeParity' in state)
    # parity_certainty = 0.0 if (parity_is_tentative_1 or parity_is_tentative_2) else 1.0
    # ===================================================================
    
    return np.array([energy_similarity, spin_similarity, parity_similarity, specificity])

def get_training_data():
    """
    Generates synthetic training data encoding nuclear physics constraints.
    Strategy - Create labeled examples covering key physics scenarios:
      1. Perfect matches (high energy + good physics → high probability)
      2. Physics vetoes (spin/parity violation → zero probability)
      3. Energy mismatches (poor energy overlap → low probability)
      4. Ambiguous cases (unknown physics → energy-dependent probability)
      5. Weak physics matches (marginal compatibility → reduced probability)
      6. Random background (fill feature space with rule-based labels)
    Output - (training_features, training_labels) for XGBoost model
    Feature Order: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Specificity]
    """
    training_points = []
    
    # Case 1 - Perfect match zone (high energy + good physics → high probability)
    # Energy > 0.8, Spin/Parity = 1.0 or 0.9 → Probability > 0.9
    for e in np.linspace(0.8, 1.0, 5):
        for s in [1.0, 0.9]: # Firm, Tentative
            for p in [1.0, 0.9]:
                prob = 0.9 + (e-0.8)/2 * (s * p) 
                training_points.append(([e, s, p, 1.0], min(prob, 0.99)))

    # Case 2 - Physics veto zone (spin or parity incompatible → zero probability)
    # Any energy, but Spin=0.0 or Parity=0.0 (strong incompatibility) → Probability=0.0
    for e in np.linspace(0.0, 1.0, 10):
        # Spin Veto
        training_points.append(([e, 0.0, 1.0, 1.0], 0.0))
        # Parity Veto
        training_points.append(([e, 1.0, 0.0, 1.0], 0.0))
        # Both Veto
        training_points.append(([e, 0.0, 0.0, 1.0], 0.0))

    # Case 3 - Energy mismatch zone (poor energy overlap → low probability)
    # Perfect physics but Energy < 0.2 → Probability decays to zero
    for e in np.linspace(0.0, 0.2, 5):
        training_points.append(([e, 1.0, 1.0, 1.0], e * 0.5))

    # Case 4 - Ambiguous physics zone (unknown spin/parity → energy-dependent)
    # Energy good, but Spin=Parity=0.5 (neutral/unknown) → Probability relies on energy
    for e in np.linspace(0.0, 1.0, 20):
        prob = e * 0.85  # Maximum 0.85 when physics unknown
        training_points.append(([e, 0.5, 0.5, 1.0], prob))

    # Case 5 - Weak physics match (marginal compatibility → reduced probability)
    # Energy good, Spin/Parity weak (0.2 or 0.1) → Low-to-mid probability
    for e in np.linspace(0.5, 1.0, 5):
        training_points.append(([e, 0.2, 1.0, 1.0], e * 0.4))
        training_points.append(([e, 1.0, 0.1, 1.0], e * 0.3))
        training_points.append(([e, 0.05, 1.0, 1.0], e * 0.2))

    # Case 6 - Random background (fill feature space with rule-based labels)
    # Generate 500 random points, apply physics rules to determine labels
    np.random.seed(42)
    for _ in range(500):
        e = np.random.uniform(0, 1)
        s = np.random.choice([0.0, 0.05, 0.2, 0.5, 0.9, 1.0])
        p = np.random.choice([0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
        spec = np.random.uniform(0.5, 1.0)
        
        # Apply rule-based labeling using physics constraints
        if s == 0.0 or p == 0.0:
            # Physics veto → zero probability
            label = 0.0
        elif e < 0.1:
            # Poor energy overlap → zero probability
            label = 0.0
        else:
            # Calculate base probability from energy and physics quality
            phys_factor = (s + p) / 2.0
            
            if s == 0.5 and p == 0.5:
                # Unknown physics → rely on energy only
                label = e * 0.8
            else:
                # Active physics → modulate energy-based probability
                label = e * phys_factor
                
        training_points.append(([e, s, p, spec], label))

    training_features = np.array([record[0] for record in training_points])
    training_labels = np.array([record[1] for record in training_points])
    return training_features, training_labels


