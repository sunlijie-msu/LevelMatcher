import pandas as pd
import numpy as np
import json
import os
import re
from xgboost import XGBRegressor
# support Soft Labels (continuous probabilities like 0.3, 0.5) instead of just binary 0/1

# ==========================================
# 1. DATA INGESTION (From JSON files)
# ==========================================
levels = []
for ds_code in ['A', 'B', 'C']:
    filename = f"test_dataset_{ds_code}.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            levels.extend(json.load(f))

df = pd.DataFrame(levels)

# Generate unique IDs (e.g., A_1000) for tracking
df['ID'] = df.apply(lambda x: f"{x['DS']}_{int(x['E_level'])}", axis=1)

# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Input Features:
# 1. Z-Score: Energy difference in standard deviations (abs(E1-E2)/sigma).
# 2. Physics Veto: Binary flag (1=Mismatch in Spin/Parity, 0=Match/Potential Overlap).
#
# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Input Features:
# 1. Z-Score: Energy difference in standard deviations (abs(E1-E2)/sigma).
# 2. Physics Veto: Binary flag (1=Mismatch in Spin/Parity, 0=Match/Potential Overlap).
#
# Helper Function for J-Pi consistency

def parse_val_to_float(val_str):
    try:
        val_str = val_str.strip()
        if '/' in val_str:
            n, d = val_str.split('/')
            return float(n) / float(d)
        return float(val_str)
    except:
        return None

def extract_parity(s):
    # Returns (cleaned_string, parity_char or None)
    if not s: return "", None
    if s.endswith('+'): return s[:-1], '+'
    if s.endswith('-'): return s[:-1], '-'
    return s, None

class QuantumState:
    def __init__(self, j, p, j_tentative=False, p_firm=True):
        self.j = j          # Float or None
        self.p = p          # '+' or '-' or None
        self.j_tentative = j_tentative # Bool
        self.p_firm = p_firm           # Bool
        
    def __repr__(self):
        return f"(J={self.j}, P={self.p}, J~={self.j_tentative}, P!={self.p_firm})"

def expand_j_pi(text, default_parity=None, is_parent_tentative=False):
    """
    Parses a J-Pi string into a list of QuantumState objects.
    Handles ENSDF notation:
    - (1, 2)- : Spin Tentative (allows +/- 1), Parity Firm (-).
    - 0-, 1-  : Spin Firm (Exact), Parity Firm (-).
    - (1, 2)  : Spin Tentative, Parity Not Firm (Allows match with anything).
    - (2+, 3) : Spin Tentative, Inner Parities Not Firm.
    """
    states = []
    text = text.strip()
    if not text: return states

    # SPECIAL CASE: Pure Parity (e.g. "+", "(-)")
    # Treat as unknown spin
    if text in ['+', '-', '(+)', '(-)']:
        # If wrapped in parens, effectively tentative parity? 
        # User says (-) means Parity - is firm in (1,2)-. 
        # But isolated (-)? Likely firm parity, unknown spin.
        p_char = '+' if '+' in text else '-'
        p_firm = '(' not in text # ((-)) -> Not firm? Assume (-) matches user (1,2)- logic? 
        # Actually user example (1,2)- says firm. 
        # Let's assume isolated parity string implies pure parity constraint.
        # But if it is '(-)', is it firm? Usually parens mean tentative.
        # Given user logic on (1,2)-, suffixes are firm. Logic for isolated (-) is ambiguous.
        # Safer to make it Firm if no spin is attached, or follow parens.
        # Let's assume (-) is Tentative Parity, - is Firm.
        p_firm = not (text.startswith('(') and text.endswith(')'))
        states.append(QuantumState(None, p_char, j_tentative=True, p_firm=p_firm))
        return states

    # 1. Handle Grouped Parens e.g. "(1/2,3/2)+", "(1,2)"
    # Pattern: ^ \s* \( (content) \) (suffix) \s* $
    paren_match = re.match(r'^\s*\((.*)\)([\+\-]?)?\s*$', text)
    
    if paren_match:
        content = paren_match.group(1)
        suffix = paren_match.group(2) # Can be + or - or None/Empty
        
        # Logic: 
        # 1. Outer parens -> J is Tentative. 
        # 2. Suffix Parity -> Overrides inner, P is FIRM.
        # 3. No Suffix -> P is NOT FIRM (inner parities become tentative).
        
        current_j_tentative = True
        
        if suffix:
            # Suffix Parity -> Firm for the whole group
            recurse_states = expand_j_pi(content, default_parity=None, is_parent_tentative=True)
            for s in recurse_states:
                s.p = suffix
                s.p_firm = True # Suffix makes it firm
                s.j_tentative = True # Inherited from wrapper
            return recurse_states
        else:
            # No Suffix -> Parity is loose/tentative
            recurse_states = expand_j_pi(content, default_parity=default_parity, is_parent_tentative=True)
            for s in recurse_states:
                s.p_firm = False # Downgrade firmness because of wrapper
                s.j_tentative = True # Inherited from wrapper
            return recurse_states

    # 2. Handle Commas (Splitter) at this level
    # Simple comma split if no parens here (inner parens handled by recursion in step 1 if we failed match)
    # But wait, expand_j_pi might be called on "1/2, (3, 4)". 
    # We need a comma splitter that respects parens.
    if ',' in text:
        # Simple paren-aware split
        parts = []
        depth = 0
        curr = []
        for char in text:
            if char == '(': depth += 1
            if char == ')': depth -= 1
            if char == ',' and depth == 0:
                parts.append("".join(curr))
                curr = []
            else:
                curr.append(char)
        parts.append("".join(curr))
        
        if len(parts) > 1:
            for p in parts:
                states.extend(expand_j_pi(p, default_parity, is_parent_tentative))
            return states

    # 3. Handle Range (:)
    if ':' in text:
        r_parts = text.split(':')
        if len(r_parts) == 2:
            s_str, e_str = r_parts
            s_clean, s_par = extract_parity(s_str.strip())
            e_clean, e_par = extract_parity(e_str.strip())
            
            # Resolve Parity
            # If default_parity provided (e.g. global column), use it if local missing
            # However, range usually implies same parity or mixed?
            # Standard: 1/2+ : 5/2+
            # If no parity on ends, use default
            r_parity = s_par if s_par else (e_par if e_par else default_parity)
            r_p_firm = True # Defaults to firm unless parent says otherwise
            
            j_start = parse_val_to_float(s_clean)
            j_end = parse_val_to_float(e_clean)
            
            if j_start is not None and j_end is not None:
                curr = j_start
                while curr <= j_end + 0.001:
                    states.append(QuantumState(curr, r_parity, j_tentative=is_parent_tentative, p_firm=r_p_firm))
                    curr += 1.0
                return states

    # 4. Handle Single Item
    clean_txt, item_par = extract_parity(text)
    final_par = item_par if item_par else default_parity
    
    j_val = parse_val_to_float(clean_txt)
    
    if j_val is not None:
        # It's a specific spin
        # Firmness: 
        # J is firm unless is_parent_tentative is True
        # P is firm unless is_parent_tentative is True (downgraded by wrapper)
        # Note: If we had a firm suffix wrapper, we logic-ed that in Step 1.
        # This step is reached for un-suffixed items (inside parens without suffix, or top level).
        
        # If we are at top level (is_parent_tentative=False):
        # "2+" -> J Firm, P Firm.
        # "2" -> J Firm, P None.
        
        # If we are inside parens (is_parent_tentative=True):
        # "2+" -> J Tentative (from parent), P Tentative (from parent).
        
        # If existing P is None, P_firm is irrelevant (checks as match)
        
        p_firm = not is_parent_tentative
        if final_par is None: p_firm = False
        
        states.append(QuantumState(j_val, final_par, j_tentative=is_parent_tentative, p_firm=p_firm))
    else:
        # Fallback for just Parity or Garbage
        if final_par:
            # "3000+" -> Spin 3000? No parse_val handles strings.
            # If parse failed, assume explicit "None" spin
             states.append(QuantumState(None, final_par, j_tentative=True, p_firm=not is_parent_tentative))
            
    return states

def parse_j_pi(spin_raw, parity_raw):
    """
    Entry point for parsing. 
    """
    s_str = str(spin_raw).strip() if pd.notna(spin_raw) else ""
    p_str = str(parity_raw).strip() if pd.notna(parity_raw) else ""
    
    # Global Parity Column acts as a Firm Default if Spin doesn't override?
    # Actually standard ENSDF: Parity column is just extra info.
    # We pass it as default.
    
    global_p = None
    if p_str == '+': global_p = '+'
    if p_str == '-': global_p = '-'
    
    if s_str and s_str.lower() not in ["", "nan", "none"]:
        return expand_j_pi(s_str, default_parity=global_p)
    elif global_p:
        # Only parity known
        # (-) in parity col? usually just + or -
        return [QuantumState(None, global_p, j_tentative=True, p_firm=True)]
    
    return [] # Unknown

def check_physics_veto(r1, r2):
    """
    Veto Logic with Tentative Rules:
    - Spin Match: 
      - Exact match (Firm).
      - +/- 1.0 match (if either is Tentative).
    - Parity Match:
      - Exact match (Firm vs Firm).
      - Any match (if either is Tentative or Unknown).
    """
    states1 = parse_j_pi(r1.get('Spin'), r1.get('Parity'))
    states2 = parse_j_pi(r2.get('Spin'), r2.get('Parity'))
    
    if not states1 or not states2:
        return 0 # No Info -> Consistent
        
    # Check for ANY valid pair
    for s1 in states1:
        for s2 in states2:
            
            # --- SPIN CHECK ---
            spin_consistent = False
            if s1.j is None or s2.j is None:
                spin_consistent = True
            else:
                diff = abs(s1.j - s2.j)
                if diff < 0.001:
                    spin_consistent = True # Exact match always good
                else:
                    # Check Tentative Tolerance (+/- 1 spin allowed)
                    # "possibly allow other spins like 0 or 3" for (1,2)
                    if (s1.j_tentative or s2.j_tentative) and diff <= 1.001:
                        spin_consistent = True
            
            # --- PARITY CHECK ---
            parity_consistent = False
            if s1.p is None or s2.p is None:
                parity_consistent = True
            elif not s1.p_firm or not s2.p_firm:
                parity_consistent = True # Tentative parity matches anything
            elif s1.p == s2.p:
                parity_consistent = True # Firm match
                
            if spin_consistent and parity_consistent:
                return 0 # Found a consistent path
                
    return 1 # Veto

# Objective: Predict probability of two levels being the same physical state.

# Training Data: (Z_Score, Veto, Probability)
# Synthetic data encoding physics logic: Low Z + No Veto = High Prob.

MATCH_EXCELLENT = [
    (0.0, 0, 1.00), (0.5, 0, 0.99), (1.0, 0, 0.98), (1.5, 0, 0.95)
]

MATCH_TRANSITION = [
    (1.8, 0, 0.90), (2.0, 0, 0.80), (2.2, 0, 0.70), (2.5, 0, 0.60), (2.8, 0, 0.55), (3.0, 0, 0.50)
]

if __name__ == "__main__":
    # ==========================================
    # 2. MODEL TRAINING (Physics-Informed XGBoost)
    # ==========================================
    MATCH_WEAK = [
        (3.2, 0, 0.40), (3.5, 0, 0.20), (4.0, 0, 0.10), (5.0, 0, 0.01), (10.0, 0, 0.00), (20.0, 0, 0.00), (100.0, 0, 0.00)
    ]

    # Veto=1 forces Probability to 0.0 (Strict Physics Rules)
    MATCH_VETOED = [
        (0.0, 1, 0.00), (0.1, 1, 0.00), (0.2, 1, 0.00), (0.3, 1, 0.00),
        (0.4, 1, 0.00), (0.5, 1, 0.00), (0.6, 1, 0.00), (0.7, 1, 0.00),
        (0.8, 1, 0.00), (0.9, 1, 0.00), (1.0, 1, 0.00), (1.5, 1, 0.00),
        (2.0, 1, 0.00), (3.0, 1, 0.00), (5.0, 1, 0.00), (100.0, 1, 0.00)
    ]

    training_data_points = MATCH_EXCELLENT + MATCH_TRANSITION + MATCH_WEAK + MATCH_VETOED

    # Split into X (Input Features) and y (Target Labels)
    X_train = np.array([[pt[0], pt[1]] for pt in training_data_points])
    y_train = np.array([pt[2] for pt in training_data_points])

    # Monotonic constraints: 
    # Feature 0 (Z-Score): Increasing Z reduces prob (-1)
    # Feature 1 (Veto): Increasing Veto (0->1) reduces prob (-1)
    level_matcher_model = XGBRegressor(objective='binary:logistic', # Learning Objective Function: Training Loss + Regularization. Optimizes the log loss function to predict probability of an instance belonging to a class.
                                       monotone_constraints='(-1, -1)', # Enforce a decreasing constraint on both predictors. In some cases, where there is a very strong prior belief that the true relationship has some quality, constraints can be used to improve the predictive performance of the model.
                                       n_estimators=200, # Number of gradient boosted trees
                                       max_depth=4, # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
                                       random_state=42) # Random number seed
    # Train the model on the training data
    level_matcher_model.fit(X_train, y_train)

    # Print all matching candidates with probabilities
    # for pt in training_data_points:
    #     z, veto, true_prob = pt
    #     pred_prob = level_matcher_model.predict([[z, veto]])[0]
    #     print(f"Z: {z:<6} Veto: {veto} | True Prob: {true_prob:<5} Predicted Prob: {pred_prob:.4f}")


    # ==========================================
    # 3. PAIRWISE INFERENCE
    # ==========================================
    # Calculate match probability for every pair of levels across different datasets.
    candidates = []

    for i, r1 in df.iterrows():
        for j, r2 in df.iterrows():
            # Rule: Only compare levels from different datasets (e.g., A vs B).
            # We use 'r1['DS'] < r2['DS']' to avoid:
            # 1. Self-matches (A_1000 vs A_1000)
            # 2. Internal matches (A_1000 vs A_2000)
            # 3. Duplicate pairs (Checking A vs B AND B vs A)
            if not (r1['DS'] < r2['DS']):
                continue

            # 1. Calculate Z-Score (Energy difference in sigma units)
            # If DE_level is missing, we assume a default uncertainty of 10 keV.
            sig1 = r1['DE_level'] if pd.notna(r1['DE_level']) else 10
            sig2 = r2['DE_level'] if pd.notna(r2['DE_level']) else 10
            combined_err = np.sqrt(sig1**2 + sig2**2)
            z_score = abs(r1['E_level'] - r2['E_level']) / combined_err

            # 2. Check Physics Veto (Spin/Parity Mismatch)
            # Old strict logic replaced by sets intersection logic
            veto = check_physics_veto(r1, r2)

            # Calculate the match probability (0.0 to 1.0) using the trained XGBoost model.
            # The model interpolates between our training rules (Z-Score and Physics Veto).
            # .predict() requires a 2D list [[Z, Veto]] and returns a list of results; 
            # we use [0] to get the specific probability for this pair.
            prob = level_matcher_model.predict([[z_score, veto]])[0]
            
            # WORKFLOW: This is a "First Pass" (Pairwise Analysis).
            # We look at every possible pair of levels across different datasets (A vs B, A vs C, B vs C).
            # If the model says there is a >50% chance they match, we save them to the candidates list.
            if prob > 0.1:
                candidates.append({
                    'ID1': r1['ID'], 'DS1': r1['DS'],
                    'ID2': r2['ID'], 'DS2': r2['DS'],
                    'prob': prob
                })

    candidates.sort(key=lambda x: x['prob'], reverse=True)

    print("\n=== MATCHING CANDIDATES (>10% Probability) ===")
    for cand in candidates:
        print(f"{cand['ID1']} <-> {cand['ID2']} | Probability: {cand['prob']:.2%}")

    # ==========================================
    # 4. GRAPH CLUSTERING (Greedy Merge with Overlap Support)
    # ==========================================
    # Algorithm:
    # 1. Initialize every level as a unique cluster.
    # 2. Iterate through high-probability candidates.
    # 3. Merge clusters if they don't contain conflicting levels.
    # 4. Support "Doublets": If two clusters cannot merge due to conflict (e.g. A_3000 vs A_3005),
    #    but a level (C_3005) matches both, allow C_3005 to belong to BOTH clusters.

    level_lookup = {row['ID']: row for _, row in df.iterrows()}

    # Map ID -> List of Cluster Objects (A level can belong to multiple clusters)
    # Each cluster is a dict {DS: ID}
    initial_clusters = [{row['DS']: row['ID']} for _, row in df.iterrows()]
    id_to_clusters = {}
    for c in initial_clusters:
        mid = list(c.values())[0]
        id_to_clusters[mid] = [c]

    # Create a set of valid ML-approved pairs for O(1) lookup
    valid_ml_matches = set()
    for cand in candidates:
        valid_ml_matches.add((cand['ID1'], cand['ID2']))
        valid_ml_matches.add((cand['ID2'], cand['ID1']))

    for cand in candidates:
        if cand['prob'] < 0.5: break
        
        id1 = cand['ID1']
        id2 = cand['ID2']
        
        # Work on snapshots of the cluster lists since we might modify them
        clusters_1 = list(id_to_clusters[id1])
        clusters_2 = list(id_to_clusters[id2])
        
        for c1 in clusters_1:
            for c2 in clusters_2:
                if c1 is c2: continue # Already in same cluster
                
                # Check Dataset Conflict between the two clusters
                ds_c1 = set(c1.keys())
                ds_c2 = set(c2.keys())
                
                # If datasets overlap (e.g. both have 'A'), we cannot merge the clusters.
                if not ds_c1.isdisjoint(ds_c2):
                    # CONFLICT DETECTED.
                    # Try "Soft Assignment" (Doublet Logic).
                    # If we can't merge the clusters, can we add the individual level to the other cluster?
                    
                    # Case A: Add id1 to c2?
                    # Req 1: c2 must not have a level from id1's dataset.
                    # Req 2: id1 must be consistent with all members of c2.
                    if cand['DS1'] not in ds_c2:
                        consistent = True
                        for m_id in c2.values():
                            if (id1, m_id) not in valid_ml_matches:
                                consistent = False
                                break
                        if consistent:
                            # Add id1 to c2
                            c2[cand['DS1']] = id1
                            if c2 not in id_to_clusters[id1]:
                                id_to_clusters[id1].append(c2)

                    # Case B: Add id2 to c1?
                    if cand['DS2'] not in ds_c1:
                        consistent = True
                        for m_id in c1.values():
                            if (id2, m_id) not in valid_ml_matches:
                                consistent = False
                                break
                        if consistent:
                            # Add id2 to c1
                            c1[cand['DS2']] = id2
                            if c1 not in id_to_clusters[id2]:
                                id_to_clusters[id2].append(c1)
                    
                    continue # Done with this pair (Merge rejected, but overlap maybe added)

                # No Dataset Conflict. Check Consistency for Merge.
                # All members of c1 must match all members of c2
                consistent_merge = True
                for m1 in c1.values():
                    for m2 in c2.values():
                        if (m1, m2) not in valid_ml_matches:
                            consistent_merge = False
                            break
                    if not consistent_merge: break
                
                if consistent_merge:
                    # Perform Merge: c2 into c1
                    c1.update(c2)
                    
                    # Update references for all members of c2
                    for m in c2.values():
                        # Remove c2 from their list
                        if c2 in id_to_clusters[m]:
                            id_to_clusters[m].remove(c2)
                        # Add c1 if not present
                        if c1 not in id_to_clusters[m]:
                            id_to_clusters[m].append(c1)
                    
                    # Since c2 is merged (consumed), we shouldn't process it further in this loop
                    # But we are iterating a snapshot 'clusters_2', so it's fine.
                    # The check 'if c2 in id_to_clusters[m]' handles cleanup.

    # Extract unique active clusters
    unique_clusters = []
    seen_ids = set()
    for clist in id_to_clusters.values():
        for c in clist:
            cid = id(c)
            if cid not in seen_ids:
                unique_clusters.append(c)
                seen_ids.add(cid)

    clusters = []
    for members in unique_clusters:
        # Anchor Selection: The member with the lowest uncertainty (DE) defines the cluster physics.
        best_mid = None
        min_de = float('inf')
        
        for mid in members.values():
            row = level_lookup[mid]
            de = row['DE_level'] if pd.notna(row['DE_level']) else 10.0
            if de < min_de:
                min_de = de
                best_mid = mid
                
        clusters.append({'Anchor_ID': best_mid, 'Members': members})

    print("\n=== CLUSTERING RESULTS ===")
    for i, cluster in enumerate(clusters):
        members_str = ", ".join([f"{ds}:{mid}" for ds, mid in sorted(cluster['Members'].items())])
        print(f"Cluster {i+1}: Anchor={cluster['Anchor_ID']} | Members=[{members_str}]")

    # ==========================================
    # 5. ADOPTED LEVEL GENERATION
    # ==========================================
    # Calculate final properties for each cluster.
    adopted_levels = []

    for cluster in clusters:
        members = cluster['Members']
        anchor = level_lookup[cluster['Anchor_ID']]
        
        # 1. Adopted Energy: Weighted average of all members (Weight = 1/sigma^2)
        vals = []
        for mid in members.values():
            row = level_lookup[mid]
            err = row['DE_level'] if pd.notna(row['DE_level']) else 10.0
            vals.append((row['E_level'], err))
            
        weights = [1/err**2 for _, err in vals]
        adopted_e = sum(e*w for (e, _), w in zip(vals, weights)) / sum(weights)
        
        # 2. Adopted Spin/Parity: Taken from the Anchor (most precise measurement)
        sp = ""
        if pd.notna(anchor['Spin']):
            sp = f"{anchor['Spin']}{anchor['Parity'] if pd.notna(anchor['Parity']) else ''}"
        
        # 3. XREF (Cross-Reference): List of datasets contributing to this level.
        # Derived strictly from the ML clustering results.
        xref_entries = []
        for mid in sorted(members.values()):
            if mid == cluster['Anchor_ID']:
                xref_entries.append(f"{mid}(100%)")
            else:
                # Calculate probability against Anchor
                r1 = level_lookup[mid]
                r2 = anchor # Anchor row
                
                # Recalculate Z-Score and Veto
                sig1 = r1['DE_level'] if pd.notna(r1['DE_level']) else 10
                sig2 = r2['DE_level'] if pd.notna(r2['DE_level']) else 10
                combined_err = np.sqrt(sig1**2 + sig2**2)
                z_score = abs(r1['E_level'] - r2['E_level']) / combined_err
                
                veto = check_physics_veto(r1, r2)
                
                prob = level_matcher_model.predict([[z_score, veto]])[0]
                xref_entries.append(f"{mid}({prob:.0%})")

        adopted_levels.append({
            'Adopted_E': round(adopted_e, 1),
            'XREF': " + ".join(xref_entries),
            'Spin_Parity': sp
        })

    final_df = pd.DataFrame(adopted_levels).sort_values('Adopted_E')
    print("\n=== ML-Predicted Adopted Dataset ===")
    print(final_df.to_string(index=False))
