"""
Feature_Engineer.py (Gamma Section Update)
=========================================================
This module handles the physics-informed feature extraction for the Machine Learning model.

Specific Focus: Gamma Decay Pattern Similarity
----------------------------------------------
The goal is to quantify how similar two nuclear levels are based on their gamma ray fingerprints.
This is difficult because:
1. Detectors have different efficiencies (intensity fluctuations).
2. Old experiments may only report energies (missing intensities).
3. One dataset may be high-resolution (20 gammas) while another is low-resolution (2 gammas).

Solution: Dual-Mode Similarity
1. If Intensities exist: Use 'Bray-Curtis Similarity' (Shared Intensity).
2. If Intensities missing: Use 'Overlap Coefficient' (Subset Matching).
"""

import numpy as np

# Assuming Scoring_Config is defined elsewhere in your file
# If not, ensure this key exists or pass it as a parameter
# Scoring_Config = { ... } 

def calculate_gamma_decay_pattern_similarity(gamma_decays_1, gamma_decays_2):
    """
    Calculates the similarity score (0.0 to 1.0) between two sets of gamma decays.

    Logic & Workflow:
    -----------------
    1. **Pre-processing**: 
       - Extracts Energy (E) and Intensity (I) from input dictionaries.
       - Checks if valid intensity data exists for BOTH datasets.
       - If intensities exist, normalizes the strongest gamma in each set to 100.0.

    2. **Mode Selection**:
       - **Mode A (Intensity Mode)**: Used when both datasets have branching ratios.
         - Metric: Bray-Curtis Similarity.
         - Formula: 2 * Sum(Min(Ia, Ib)) / (Sum(Ia) + Sum(Ib)).
         - Physics: Calculates the "Shared Area" under the spectrum. Robust to efficiency 
           fluctuations because it ignores the non-overlapping excess.

       - **Mode B (Binary Mode)**: Used when either dataset lacks intensities.
         - Metric: Szymkiewicz–Simpson (Overlap) Coefficient.
         - Formula: Count(Matches) / Min(Count_A, Count_B).
         - Physics: Checks for "Subset Behavior". If Dataset A has 2 gammas and both 
           are found in Dataset B (which has 20), score is 1.0. This handles 
           high-res vs low-res comparisons perfectly.

    Parameters:
    -----------
    gamma_decays_1 : list of dicts
        [{'energy': 1000.0, 'branching_ratio': 100.0}, ...]
    gamma_decays_2 : list of dicts
        Format matches above.

    Returns:
    --------
    float
        0.0 (No similarity) to 1.0 (Perfect identity).
    """
    
    # --- Guard Clause: Missing Data ---
    # If either level has no reported gammas, we cannot assess similarity.
    # Return a "Neutral" score (usually 0.5) to let other features (Energy/Spin) decide.
    if not gamma_decays_1 or not gamma_decays_2:
        # Default to 0.5 if Scoring_Config not available in scope
        return Scoring_Config['General']['Neutral_Score'] if 'Scoring_Config' in globals() else 0.5

    # =========================================================================
    # PHASE 1: Data Standardization & Normalization
    # =========================================================================
    def process_spectrum(raw_decays):
        """
        Parses raw list into clean list of dicts {'E': val, 'I': val}.
        detects if valid intensities exist, and normalizes strongest to 100.0.
        """
        clean_list = []
        has_intensity = False
        max_intensity = 0.0
        
        for g in raw_decays:
            # Safely extract values, defaulting to 0.0 if missing
            e = float(g.get('energy', 0))
            i = float(g.get('branching_ratio', 0))
            
            # Valid gamma requires positive Energy
            if e > 0:
                # Logic: If ANY gamma has intensity > 0, we consider the dataset 
                # to have intensity data.
                if i > 0: 
                    has_intensity = True
                    if i > max_intensity: 
                        max_intensity = i
                
                clean_list.append({'E': e, 'I': i})
        
        # Normalize: Scale strongest peak to 100.0 (Standard ENSDF convention)
        # This aligns the two datasets to the same relative scale.
        if has_intensity and max_intensity > 0:
            for g in clean_list: 
                g['I'] = (g['I'] / max_intensity) * 100.0
            
        return clean_list, has_intensity

    # Process both inputs
    list_A, has_int_A = process_spectrum(gamma_decays_1)
    list_B, has_int_B = process_spectrum(gamma_decays_2)
    
    # If normalization resulted in empty lists (e.g. all energies were 0), return neutral
    if not list_A or not list_B: 
        return Scoring_Config['General']['Neutral_Score'] if 'Scoring_Config' in globals() else 0.5

    # Physics Constant: Energy Tolerance
    # 3.0 keV allows for calibration shifts (e.g. 1000 vs 1002.5) common in old data.
    TOLERANCE_KEV = 3.0 

    # =========================================================================
    # PHASE 2: Mode Selection & Calculation
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # MODE A: INTENSITY MODE (Bray-Curtis Similarity)
    # Triggered only if BOTH datasets have intensity data.
    # -------------------------------------------------------------------------
    if has_int_A and has_int_B:
        intersection_sum = 0.0
        matched_indices_B = set() # Track usage to prevent double-counting matches

        # Compare every gamma in A against B
        for gA in list_A:
            best_match_intensity = 0.0
            best_idx = -1
            min_energy_diff = 999.0
            
            # Greedy Search: Find the closest energy match in B
            for idx, gB in enumerate(list_B):
                if idx in matched_indices_B: continue # Don't reuse gammas
                
                diff = abs(gA['E'] - gB['E'])
                if diff <= TOLERANCE_KEV:
                    # We pick the closest energy match, not necessarily intensity match
                    if diff < min_energy_diff:
                        min_energy_diff = diff
                        best_match_intensity = gB['I']
                        best_idx = idx
            
            # If a valid match was found within tolerance:
            if best_idx != -1:
                # KEY PHYSICS LOGIC: Use MINIMUM intensity.
                # If A has 100 and B has 20, they only definitively "share" 20.
                # Using average (60) would hallucinate data B never saw.
                intersection_sum += min(gA['I'], best_match_intensity)
                matched_indices_B.add(best_idx)

        # Calculate Totals
        sum_A = sum(g['I'] for g in list_A)
        sum_B = sum(g['I'] for g in list_B)
        
        # Avoid division by zero
        if (sum_A + sum_B) == 0: return 0.0
        
        # Formula: 2 * Intersection / (Total_A + Total_B)
        return (2.0 * intersection_sum) / (sum_A + sum_B)

    # -------------------------------------------------------------------------
    # MODE B: BINARY MODE (Overlap Coefficient)
    # Triggered if EITHER dataset is missing intensities (Energies Only).
    # -------------------------------------------------------------------------
    else:
        matches_count = 0
        matched_indices_B = set()
        
        for gA in list_A:
            # Search for existence of this energy in B
            for idx, gB in enumerate(list_B):
                if idx in matched_indices_B: continue
                
                if abs(gA['E'] - gB['E']) <= TOLERANCE_KEV:
                    matches_count += 1
                    matched_indices_B.add(idx)
                    break # Stop looking after first match for this gA
        
        # KEY PHYSICS LOGIC: Subset Matching
        # We divide by the MINIMUM set size.
        # Case: A has 2 gammas, B has 20 gammas.
        # If A's 2 gammas are found in B -> Matches=2, Min_Len=2 -> Score = 1.0 (Perfect)
        # This correctly identifies that A is just a "low-stat" version of B.
        min_len = min(len(list_A), len(list_B))
        
        if min_len == 0: return 0.0
        
        return float(matches_count) / float(min_len)