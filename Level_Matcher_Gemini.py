import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# support Soft Labels (continuous probabilities like 0.3, 0.5) instead of just binary 0/1

# ==========================================
# 1. SETUP DATASETS
# ==========================================
levels = []

# Dataset A:
levels.append({'E_level': 1000, 'DE_level': 3, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 2000, 'DE_level': 4, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 3000, 'DE_level': 2, 'Spin': 1,  'Parity': '+',  'DS': 'A'})
levels.append({'E_level': 3005, 'DE_level': 2, 'Spin': 2,  'Parity': '-',  'DS': 'A'})
levels.append({'E_level': 4000, 'DE_level': 4, 'Spin': 2,  'Parity': '+',  'DS': 'A'})
levels.append({'E_level': 6000, 'DE_level': 5, 'Spin': None, 'Parity': None, 'DS': 'A'})

# Dataset B:
levels.append({'E_level': 1005, 'DE_level': 3, 'Spin': None, 'Parity': None, 'DS': 'B'})
levels.append({'E_level': 2008, 'DE_level': 2, 'Spin': None, 'Parity': None, 'DS': 'B'})
levels.append({'E_level': 3000, 'DE_level': 1, 'Spin': 2,  'Parity': '-',  'DS': 'B'}) 
levels.append({'E_level': 5000, 'DE_level': 6, 'Spin': 2,  'Parity': '+',  'DS': 'B'})

# Dataset C:
levels.append({'E_level': 1010, 'DE_level': 50, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 2010, 'DE_level': 60, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 3005, 'DE_level': 40, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 5020, 'DE_level': 60, 'Spin': None, 'Parity': None, 'DS': 'C'})

df = pd.DataFrame(levels)

# Generate readable IDs (e.g., A_1000) for better interpretability
# Note: In real scenarios, handle duplicates if multiple levels have same integer energy
df['ID'] = df.apply(lambda x: f"{x['DS']}_{int(x['E_level'])}", axis=1)

# ==========================================
# 2. TRAIN XGBOOST (Physics-Informed)
# ==========================================
# Features: [Z_Score, Physics_Veto]
# Feature 1: Z-Score = abs(E1 - E2) / sqrt(err1^2 + err2^2)
#            This represents the energy difference in units of "standard deviations".
#            Z < 2.0 is typically a good match. Z > 3.0 is typically a non-match.
# Feature 2: Physics_Veto = 1 if Spin / Parity mismatch. 0 otherwise.

# ==========================================
# TRAINING DATA DEFINITION
# ==========================================
# We define the training data as a list of tuples for readability.
# Format: (Z_Score, Physics_Veto, Target_Probability)
#
# X_train (Questions): The first two columns [Z_Score, Veto]
# y_train (Answers):   The last column [Probability]
#
# Logic:
# 1. Z-Score: Energy difference in sigma units. Lower is better.
# 2. Physics_Veto: 1 means Spin/Parity mismatch. This kills the match (Prob=0).
# 3. Probability: 1.0 (Certain) -> 0.0 (Impossible).

training_data_points = [
    # --- Excellent Matches (Z < 1.5) ---
    (0.0, 0, 1.00),  # Perfect match
    (0.5, 0, 0.99),  # Slight penalty to distinguish from perfect
    (1.0, 0, 0.98),  # 1-sigma is still excellent
    (1.5, 0, 0.95),
    # --- Transition Zone (1.5 < Z < 3.0) ---
    (1.8, 0, 0.90),
    (2.0, 0, 0.80),  # 2-sigma: Good candidate
    (2.2, 0, 0.70),
    (2.5, 0, 0.60),
    (2.8, 0, 0.55),
    (3.0, 0, 0.50),  # 3-sigma: The "Maybe" threshold (50/50)
    # --- Weak Matches (Z > 3.0) ---
    (3.2, 0, 0.40),
    (3.5, 0, 0.20),  # Rapid drop-off after 3-sigma
    (4.0, 0, 0.10),  # 4-sigma: Very unlikely
    (5.0, 0, 0.01),  # 5-sigma: Effectively zero
    (10.0, 0, 0.00), # Far away
    (20.0, 0, 0.00), # Very Far
    (100.0, 0, 0.00), # Extremely Far
    # --- Physics Veto Cases ---
    # If Veto=1, Probability is likely 0.0, regardless of Z-score.
    (0.0, 1, 0.00), 
    (1.0, 1, 0.00), 
    (2.0, 1, 0.00), 
    (3.0, 1, 0.00), 
    (5.0, 1, 0.00),
    (100.0, 1, 0.00)
]

# Split into X (Input Features) and y (Labels) for XGBoost
X_train = np.array([[pt[0], pt[1]] for pt in training_data_points])
y_train = np.array([pt[2] for pt in training_data_points])

# Monotonic constraints: 
# Feature 0 (Z-Score): Increasing Z reduces prob (-1)
# Feature 1 (Veto): Increasing Veto (0->1) reduces prob (-1)
level_matcher_model = XGBRegressor(objective='binary:logistic', # Objective Function: Training Loss + Regularization. Optimizes the log loss function to predict probability of an instance belonging to a class.
                                   monotone_constraints='(-1, -1)', # Enforce a decreasing constraint on both predictors
                                   n_estimators=100, # Number of gradient boosted trees
                                   max_depth=2, # Maximum tree depth for base learners
                                   random_state=42) # Random number seed
# Train the model on the training data
level_matcher_model.fit(X_train, y_train)

# ==========================================
# 3. PREDICT MATCHES (Pairwise Analysis)
# ==========================================
# This section identifies all potential matches between DIFFERENT datasets.
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

        # 2. Physics Veto
        veto = 0
        # Spin Mismatch
        if pd.notna(r1['Spin']) and pd.notna(r2['Spin']) and r1['Spin'] != r2['Spin']:
            # pd.notna detects non-missing (existing) values in a dataset
            veto = 1
        # Parity Mismatch
        if pd.notna(r1['Parity']) and pd.notna(r2['Parity']) and r1['Parity'] != r2['Parity']:
            veto = 1

        # Calculate the match probability (0.0 to 1.0) using the trained XGBoost model.
        # The model interpolates between our training rules (Z-Score and Physics Veto).
        # .predict() requires a 2D list [[Z, Veto]] and returns a list of results; 
        # we use [0] to get the specific probability for this pair.
        prob = level_matcher_model.predict([[z_score, veto]])[0]
        
        # WORKFLOW: This is a "First Pass" (Pairwise Analysis).
        # We look at every possible pair of levels across different datasets (A vs B, A vs C, B vs C).
        # If the model says there is a >50% chance they match, we save them to the candidates list.
        if prob > 0.5:
            candidates.append({
                'ID1': r1['ID'], 'DS1': r1['DS'],
                'ID2': r2['ID'], 'DS2': r2['DS'],
                'prob': prob
            })

# Sort candidates by probability (Highest confidence first)
candidates.sort(key=lambda x: x['prob'], reverse=True)

# ==========================================
# 4. HIERARCHICAL CLUSTERING (Anchor-Based)
# ==========================================
# Logic:
# 1. Prioritize Datasets: A > B > C
# 2. "A" levels are the primary Anchors.
# 3. "B" levels match to "A" if possible, else form new anchors.
# 4. "C" levels match to existing clusters, else form new anchors.
# 5. Constraint: A cluster can contain AT MOST one level from each Dataset.

# Priority Order
ds_priority = ['A', 'B', 'C']

# Fast Lookup for Level Data (ID -> Row Series)
level_lookup = {row['ID']: row for _, row in df.iterrows()}

# Initialize Clusters with Dataset A
# Structure: [ {'Anchor_ID': 'A_1000', 'Members': {'A': 'A_1000'} } ]
clusters = []

# 1. Start with all levels from the highest priority dataset (A)
primary_ds = ds_priority[0]
for _, row in df[df['DS'] == primary_ds].iterrows():
    clusters.append({
        'Anchor_ID': row['ID'],
        'Members': {primary_ds: row['ID']}
    })

# 2. Iterate through remaining datasets
for current_ds in ds_priority[1:]:
    # Get all levels for this dataset
    current_levels = df[df['DS'] == current_ds]
    
    for _, row in current_levels.iterrows():
        level_id = row['ID']
        best_cluster_idx = -1
        best_prob = 0.0
        
        # Try to match with existing clusters
        for idx, cluster in enumerate(clusters):
            # CONSTRAINT: Cluster must not already have a level from current_ds
            if current_ds in cluster['Members']:
                continue
                
            # Compare against the ANCHOR of the cluster
            anchor_id = cluster['Anchor_ID']
            anchor_row = level_lookup[anchor_id]
            
            # Calc Z-Score
            err = np.sqrt(row['DE_level']**2 + anchor_row['DE_level']**2)
            z = abs(row['E_level'] - anchor_row['E_level']) / err
            
            # Calc Veto
            veto = 0
            if pd.notna(row['Spin']) and pd.notna(anchor_row['Spin']) and row['Spin'] != anchor_row['Spin']: veto = 1
            if pd.notna(row['Parity']) and pd.notna(anchor_row['Parity']) and row['Parity'] != anchor_row['Parity']: veto = 1
            
            # Predict Match Probability
            p = level_matcher_model.predict([[z, veto]])[0]

            # Find the best match (highest probability)
            if p > best_prob:
                best_prob = p
                best_cluster_idx = idx
        
        # Decision: Match or New Cluster?
        if best_prob > 0.5 and best_cluster_idx != -1:
            # Add to cluster
            clusters[best_cluster_idx]['Members'][current_ds] = level_id
        else:
            # Create New Cluster
            clusters.append({
                'Anchor_ID': level_id,
                'Members': {current_ds: level_id}
            })

# ==========================================
# 5. GENERATE ADOPTED LEVELS & SOFT LISTS
# ==========================================
adopted_levels = []

for cluster in clusters:
    # Calculate Adopted Energy (Weighted Average of Members)
    sum_e_wt = 0
    sum_wt = 0
    
    for ds, mem_id in cluster['Members'].items():
        mem_row = level_lookup[mem_id]
        wt = 1 / (mem_row['DE_level']**2)
        sum_e_wt += mem_row['E_level'] * wt
        sum_wt += wt
        
    adopted_e = sum_e_wt / sum_wt
    
    # Determine Spin/Parity (Naive: Take from Anchor or first available)
    # Better: Take from highest priority member that has info
    spin_parity = ""
    for ds in ds_priority:
        if ds in cluster['Members']:
            mem_id = cluster['Members'][ds]
            mem_row = level_lookup[mem_id]
            if pd.notna(mem_row['Spin']):
                spin_parity = f"{mem_row['Spin']}{mem_row['Parity'] if pd.notna(mem_row['Parity']) else ''}"
                break
    
    # Generate Soft Source List
    # Rule: Show probability of ALL levels matching this cluster, 
    # BUT exclude levels from datasets that are already IN the cluster (unless it IS the member).
    
    source_entries = []
    
    # We iterate through ALL levels in the dataframe
    for _, cand_row in df.iterrows():
        cand_id = cand_row['ID']
        cand_ds = cand_row['DS']
        
        # 1. If this candidate IS the member for this DS, it's 100%
        if cand_ds in cluster['Members'] and cluster['Members'][cand_ds] == cand_id:
            source_entries.append(f"{cand_id}(100%)")
            continue
            
        # 2. If this candidate is from a DS that is ALREADY in the cluster, SKIP IT.
        # This prevents A_3005 from appearing in A_3000's cluster list.
        if cand_ds in cluster['Members']:
            continue
            
        # 3. Otherwise, calculate match probability to the Cluster ANCHOR
        # This prevents "chaining" through weak links and enforces the Anchor's physics.
        anchor_id = cluster['Anchor_ID']
        anchor_row = level_lookup[anchor_id]
        
        err = np.sqrt(cand_row['DE_level']**2 + anchor_row['DE_level']**2)
        z = abs(cand_row['E_level'] - anchor_row['E_level']) / err
        
        veto = 0
        if pd.notna(cand_row['Spin']) and pd.notna(anchor_row['Spin']) and cand_row['Spin'] != anchor_row['Spin']: veto = 1
        if pd.notna(cand_row['Parity']) and pd.notna(anchor_row['Parity']) and cand_row['Parity'] != anchor_row['Parity']: veto = 1
        
        # WORKFLOW: This is the "Soft Source List" generation.
        # We calculate the probability of EVERY level matching the Cluster Anchor.
        # [[z, veto]] (2D input) and [0] (first result) are used for the prediction.
        p = level_matcher_model.predict([[z, veto]])[0]
            
        if p > 0.2: # Display threshold (20%)
            source_entries.append(f"{cand_id}({int(p*100)}%)")

    adopted_levels.append({
        'Adopted_E': round(adopted_e, 1),
        'Sources': " + ".join(sorted(source_entries)),
        'Spin_Parity': spin_parity
    })

final_df = pd.DataFrame(adopted_levels).sort_values('Adopted_E')
print("\n=== FINAL ADOPTED DATASET ===")
print(final_df.to_string(index=False))
