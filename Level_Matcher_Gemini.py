import pandas as pd
import numpy as np
import json
import os
import re
from xgboost import XGBRegressor
from physics_parser import calculate_physics_distance

# support Soft Labels (continuous probabilities like 0.3, 0.5) instead of just binary 0/1

# ==========================================
# 1. DATA INGESTION (From JSON files)
# ==========================================
levels = []
for ds_code in ['A', 'B', 'C']:
    filename = f"test_dataset_{ds_code}.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle new ENSDF JSON schema (levelsTable -> levels)
            if isinstance(data, dict) and 'levelsTable' in data:
                raw_levels = data['levelsTable'].get('levels', [])
                for item in raw_levels:
                    # Extract Energy
                    e_val = item.get('energy', {}).get('value')
                    
                    # Extract Uncertainty (DE)
                    de_val = item.get('energy', {}).get('uncertainty', {}).get('value')
                    # If uncertainty is symmetric, it might be in 'value'. If asymmetric, might be different.
                    # Assuming simple value for now based on attachment.
                    
                    # Extract Spin_Parity
                    sp_str = item.get('spinParity', {}).get('evaluatorInput')
                    
                    if e_val is not None:
                        levels.append({
                            'DS': ds_code,
                            'E_level': float(e_val),
                            'DE_level': float(de_val) if de_val is not None else 10.0, # Default 10 if missing
                            'Spin_Parity': sp_str
                        })
            elif isinstance(data, list):
                # Fallback for flat list if mixed (though we updated files)
                for item in data:
                    item['DS'] = ds_code
                    levels.append(item)

df = pd.DataFrame(levels)
# Generate unique IDs (e.g., A_1000) for tracking
df['ID'] = df.apply(lambda x: f"{x['DS']}_{int(x['E_level'])}", axis=1)

# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Input Features:
# 1. Z-Score: Energy difference in standard deviations (abs(E1-E2)/sigma).
# 2. Physics Distance: Continuous metric (0.0=Match, 0.25=Tentative, 1.0=Mismatch).
#    (Replaces previous Binary Veto)
#
# Physics logic is now handled by calculate_physics_distance from physics_parser.py

# Objective: Predict probability of two levels being the same physical state.

# Training Data: (Z_Score, Physics_Distance, Probability)
# Synthetic data encoding physics logic: Low Z + Low Distance = High Prob.

MATCH_EXCELLENT = [
    (0.0, 0.0, 1.00), (0.5, 0.0, 0.99), (1.0, 0.0, 0.98), (1.5, 0.0, 0.95)
]

MATCH_TRANSITION = [
    (1.8, 0.0, 0.90), (2.0, 0.0, 0.80), (2.5, 0.0, 0.60), (3.0, 0.0, 0.50)
]

MATCH_WEAK = [
    (3.5, 0.0, 0.20), (4.0, 0.0, 0.10), (5.0, 0.0, 0.01)
]

# TENTATIVE PHYSICS (Low Z, Tentative/Soft Physics = 0.25)
MATCH_TENTATIVE = [
    (0.0, 0.25, 0.95), (1.0, 0.25, 0.90), (2.0, 0.25, 0.70)
]

# UNKNOWN PHYSICS (Low Z, No Info = 0.5)
MATCH_UNKNOWN = [
    (0.0, 0.5, 0.98), (1.0, 0.5, 0.95), (2.0, 0.5, 0.75)
]

# VETOED / MISMATCH (Any Z, Distance = 1.0)
MATCH_VETOED = [
    (0.0, 1.0, 0.00), (0.5, 1.0, 0.00), (1.0, 1.0, 0.00), (10.0, 1.0, 0.00)
]

if __name__ == "__main__":
    training_data_points = MATCH_EXCELLENT + MATCH_TRANSITION + MATCH_WEAK + MATCH_TENTATIVE + MATCH_UNKNOWN + MATCH_VETOED

    # Split into X (Input Features) and y (Target Labels)
    X_train = np.array([[pt[0], pt[1]] for pt in training_data_points])
    y_train = np.array([pt[2] for pt in training_data_points])

    # Monotonic constraints:
    # Feature 0 (Z-Score): Increasing Z reduces prob (-1)
    # Feature 1 (Distance): Increasing Distance (0->1) reduces prob (-1)
    level_matcher_model = XGBRegressor(objective='binary:logistic', # Learning Objective Function: Training Loss + Regularization. Optimizes the log loss function to predict probability of an instance belonging to a class.
                                       monotone_constraints='(-1, -1)', # Enforce a decreasing constraint on both predictors. In some cases, where there is a very strong prior belief that the true relationship has some quality, constraints can be used to improve the predictive performance of the model.
                                       n_estimators=200, # Number of gradient boosted trees
                                       max_depth=4, # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
                                       random_state=42) # Random number seed
    
    # Train the model on the training data
    level_matcher_model.fit(X_train, y_train)

    # ==========================================
    # 3. PAIRWISE INFERENCE
    # ==========================================
    # Calculate match probability for every pair of levels across different datasets.
    candidates = []
    
    # Pre-calculate distances could be optimized, but easy to loop for small datasets
    for i, r1 in df.iterrows():
        for j, r2 in df.iterrows():
            # Rule: Only compare levels from different datasets (e.g., A vs B).
            # We use 'r1['DS'] < r2['DS']' to avoid:
            # 1. Self-matches (A_1000 vs A_1000)
            # 2. Internal matches (A_1000 vs A_2000)
            # 3. Duplicate pairs (Checking A vs B AND B vs A)
            if not (r1['DS'] < r2['DS']): continue 

            # 1. Calculate Z-Score (Energy difference in sigma units)
            # If DE_level is missing, we assume a default uncertainty of 10 keV.
            sig1 = r1['DE_level'] if pd.notna(r1['DE_level']) else 10
            sig2 = r2['DE_level'] if pd.notna(r2['DE_level']) else 10
            combined_err = np.sqrt(sig1**2 + sig2**2)
            z_score = abs(r1['E_level'] - r2['E_level']) / combined_err

            # 2. Physics Distance Calculation (replaces Veto)
            sp1 = r1.get('Spin_Parity')
            sp2 = r2.get('Spin_Parity')
            phys_dist = calculate_physics_distance(sp1, sp2)

            # Calculate the match probability (0.0 to 1.0) using the trained XGBoost model.
            # The model interpolates between our training rules (Z-Score and Physics Distance).
            prob = level_matcher_model.predict([[z_score, phys_dist]])[0]

            # WORKFLOW: This is a "First Pass" (Pairwise Analysis).
            # We look at every possible pair of levels across different datasets (A vs B, A vs C, B vs C).
            # If the model says there is a >10% chance they match, we save them to the candidates list.
            if prob > 0.1:
                candidates.append({
                    'ID1': r1['ID'], 'DS1': r1['DS'],
                    'ID2': r2['ID'], 'DS2': r2['DS'],
                    'prob': prob
                })

    candidates.sort(key=lambda x: x['prob'], reverse=True)

    print("\n=== MATCHING CANDIDATES (>10%) ===")
    for c in candidates:
        r1 = df[df['ID']==c['ID1']].iloc[0]
        r2 = df[df['ID']==c['ID2']].iloc[0]
        dist = calculate_physics_distance(r1.get('Spin_Parity'), r2.get('Spin_Parity'))
        print(f"{c['ID1']} <-> {c['ID2']} | P: {c['prob']:.1%} (Z={abs(r1['E_level']-r2['E_level'])/np.sqrt((r1['DE_level']or 10)**2 + (r2['DE_level']or 10)**2):.2f}, Dist={dist})")

    # ==========================================
    # 4. CLUSTERING (Graph Clustering with Overlap Support)
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

    # Valid High-Prob Limit for Consistency Checks
    # We define "Valid" as present in the candidates list
    valid_pairs = set((c['ID1'], c['ID2']) for c in candidates)
    valid_pairs.update((c['ID2'], c['ID1']) for c in candidates)

    for cand in candidates:
        if cand['prob'] < 0.5: break # Only merge strong candidates
        id1, id2 = cand['ID1'], cand['ID2']
        
        # Work on snapshots of the cluster lists since we might modify them
        c1_list = list(id_to_clusters[id1])
        c2_list = list(id_to_clusters[id2])

        for c1 in c1_list:
            for c2 in c2_list:
                if c1 is c2: continue # Already same cluster
                
                ds1 = set(c1.keys())
                ds2 = set(c2.keys())
                
                # Check 1: Dataset Conflict
                if not ds1.isdisjoint(ds2):
                    # They overlap in datasets (e.g. both have a 'Dataset A' level)
                    # CONFLICT DETECTED.
                    # Try "Soft Assignment" (Doublet Logic).
                    
                    # Try Adding id1 to c2? (Only if c2 lacks DS1)
                    if cand['DS1'] not in ds2:
                        # Consistency check: id1 must match ALL members of c2
                        if all((id1, m) in valid_pairs for m in c2.values()):
                            if id1 not in c2.values(): # avoid dupe
                                c2[cand['DS1']] = id1
                                if c2 not in id_to_clusters[id1]: id_to_clusters[id1].append(c2)
                    
                    # Try Adding id2 to c1? (Only if c1 lacks DS2)
                    if cand['DS2'] not in ds1:
                        # Consistency check: id2 must match ALL members of c1
                        if all((id2, m) in valid_pairs for m in c1.values()):
                            if id2 not in c1.values():
                                c1[cand['DS2']] = id2
                                if c1 not in id_to_clusters[id2]: id_to_clusters[id2].append(c1)
                    
                    continue

                # Check 2: Consistency (All-to-All)
                # If we merge c1 and c2, every member of c1 must match every member of c2
                consistent = True
                for m1 in c1.values():
                    for m2 in c2.values():
                        if (m1, m2) not in valid_pairs:
                            consistent = False; break
                    if not consistent: break
                
                if consistent:
                    # Perform Merge: c2 into c1
                    c1.update(c2)
                    # Update references for all members of c2
                    for m in c2.values():
                        # Remove c2 from their list
                        if c2 in id_to_clusters[m]: id_to_clusters[m].remove(c2)
                        # Add c1 if not present
                        if c1 not in id_to_clusters[m]: id_to_clusters[m].append(c1)
                    # c2 is effectively dissolved into c1

    # Extract unique active clusters
    unique_clusters = []
    seen = set()
    for clist in id_to_clusters.values():
        for c in clist:
            cid = id(c)
            if cid not in seen:
                unique_clusters.append(c)
                seen.add(cid)

    # ==========================================
    # 5. REPORTING / ADOPTED LEVEL GENERATION
    # ==========================================
    adopted = []
    for c in unique_clusters:
        members = c
        if not members: continue
        
        # Anchor Selection: Lowest DE
        # If DE is None/NaN, treat as high error
        anchor_id = min(members.values(), key=lambda x: level_lookup[x]['DE_level'] if pd.notna(level_lookup[x]['DE_level']) else 999)
        anchor_row = level_lookup[anchor_id]
        
        # Weighted Average Energy calculation
        vals = []
        for mid in members.values():
            r = level_lookup[mid]
            err = r['DE_level'] if pd.notna(r['DE_level']) else 10
            vals.append((r['E_level'], err))
        
        weights = [1/(v[1]**2) for v in vals]
        sum_w = sum(weights)
        e_adopt = sum(v[0]*w for v,w in zip(vals, weights)) / sum_w
        
        # XREF String Generation
        xref_parts = []
        sorted_mids = sorted(members.values(), key=lambda x: (level_lookup[x]['DS'], level_lookup[x]['E_level']))
        
        for mid in sorted_mids:
            if mid == anchor_id:
                xref_parts.append(f"{mid}(Anchor)")
            else:
                # Recalc prob against Anchor for display
                r1 = level_lookup[mid]
                r2 = anchor_row
                sig1 = r1['DE_level'] if pd.notna(r1['DE_level']) else 10
                sig2 = r2['DE_level'] if pd.notna(r2['DE_level']) else 10
                z = abs(r1['E_level'] - r2['E_level']) / np.sqrt(sig1**2 + sig2**2)
                
                pd_val = calculate_physics_distance(r1.get('Spin_Parity'), r2.get('Spin_Parity'))
                prob = level_matcher_model.predict([[z, pd_val]])[0]
                xref_parts.append(f"{mid}({prob:.0%})")
        
        adopted.append({
            'Adopted_E': round(e_adopt, 1),
            'XREF': " + ".join(xref_parts),
            'Spin_Parity': anchor_row.get('Spin_Parity', '')
        })

    adf = pd.DataFrame(adopted).sort_values('Adopted_E')
    print("\n=== ADOPTED LEVELS ===")
    print(adf.to_string(index=False))
