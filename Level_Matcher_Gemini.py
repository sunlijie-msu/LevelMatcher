import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# support Soft Labels (continuous probabilities like 0.3, 0.5) instead of just binary 0/1

# ==========================================
# 1. SETUP DATASETS
# ==========================================
levels = []

# Dataset A:
levels.append({'E_level': 1000.0, 'DE_level': 3.0, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 2000.0, 'DE_level': 4.0, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 3000.0, 'DE_level': 2.0, 'Spin': 1.0,  'Parity': '+',  'DS': 'A'})
levels.append({'E_level': 3005.0, 'DE_level': 2.0, 'Spin': 2.0,  'Parity': '-',  'DS': 'A'})
levels.append({'E_level': 4000.0, 'DE_level': 4.0, 'Spin': 2.0,  'Parity': '+',  'DS': 'A'})

# Dataset B:
levels.append({'E_level': 1005.0, 'DE_level': 3.0, 'Spin': None, 'Parity': None, 'DS': 'B'})
levels.append({'E_level': 2008.0, 'DE_level': 2.0, 'Spin': None, 'Parity': None, 'DS': 'B'})
levels.append({'E_level': 3000.0, 'DE_level': 1.0, 'Spin': 2.0,  'Parity': '-',  'DS': 'B'}) 
levels.append({'E_level': 5000.0, 'DE_level': 6.0, 'Spin': 2.0,  'Parity': '+',  'DS': 'B'})

# Dataset C:
levels.append({'E_level': 1010.0, 'DE_level': 30.0, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 2010.0, 'DE_level': 40.0, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 3005.0, 'DE_level': 30.0, 'Spin': None, 'Parity': None, 'DS': 'C'})
levels.append({'E_level': 5020.0, 'DE_level': 20.0, 'Spin': None, 'Parity': None, 'DS': 'C'})

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
    # We define a smooth curve here
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

    # --- Physics Veto Cases (CRITICAL) ---
    # If Veto=1, Probability is ALWAYS 0.0, regardless of Z-score.
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
level_matcher_model = XGBRegressor(objective='binary:logistic', # Regression for probabilities
                                   monotone_constraints='(-1, -1)', # Enforce a decreasing constraint on both predictors
                                   n_estimators=100, # Number of gradient boosted trees
                                   max_depth=2, # Maximum tree depth for base learners
                                   random_state=42) # Random number seed
# Train the model
level_matcher_model.fit(X_train, y_train)

# ==========================================
# 3. PREDICT MATCHES
# ==========================================
candidates = []

for i, r1 in df.iterrows():
    for j, r2 in df.iterrows():
        if r1['DS'] >= r2['DS']: continue # Avoid duplicates and self-matches

        # 1. Z-Score
        err = np.sqrt(r1.get('DE_level', 10)**2 + r2.get('DE_level', 10)**2)
        z_score = abs(r1['E_level'] - r2['E_level']) / err

        # 2. Physics Veto
        veto = 0
        # Spin Mismatch
        if pd.notna(r1['Spin']) and pd.notna(r2['Spin']) and r1['Spin'] != r2['Spin']:
            # pd.notna detects non-missing (existing) values in a dataset
            veto = 1
        # Parity Mismatch
        if pd.notna(r1['Parity']) and pd.notna(r2['Parity']) and r1['Parity'] != r2['Parity']:
            veto = 1

        # Predict
        prob = level_matcher_model.predict([[z_score, veto]])[0]
        
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

# Initialize Clusters with Dataset A
# Structure: [ {'Adopted_E': ..., 'Members': {'A': 'A_1000', 'B': 'B_1005'} } ]
clusters = []

# Helper to get level data by ID
def get_level_data(id):
    return df[df['ID'] == id].iloc[0]

# 1. Start with all levels from the highest priority dataset (A)
primary_ds = ds_priority[0]
for _, row in df[df['DS'] == primary_ds].iterrows():
    clusters.append({
        'Anchor_ID': row['ID'],
        'Members': {primary_ds: row['ID']}, # Map DS -> ID
        'Sum_E_Wt': row['E_level'] / (row['DE_level']**2),
        'Sum_Wt': 1 / (row['DE_level']**2)
    })

# 2. Iterate through remaining datasets
for current_ds in ds_priority[1:]:
    # Get all levels for this dataset
    current_levels = df[df['DS'] == current_ds]
    
    for _, row in current_levels.iterrows():
        level_id = row['ID']
        best_cluster_idx = -1
        best_prob = 0.0
        best_z = 9999.0 # Tie-breaker
        
        # Try to match with existing clusters
        for idx, cluster in enumerate(clusters):
            # CONSTRAINT: Cluster must not already have a level from current_ds
            if current_ds in cluster['Members']:
                continue
                
            # Compare against the ANCHOR of the cluster
            # This ensures we respect the primary definition of the cluster
            anchor_id = cluster['Anchor_ID']
            anchor_row = get_level_data(anchor_id)
            
            # Calc Z-Score
            err = np.sqrt(row['DE_level']**2 + anchor_row['DE_level']**2)
            z = abs(row['E_level'] - anchor_row['E_level']) / err
            
            # Calc Veto
            veto = 0
            if pd.notna(row['Spin']) and pd.notna(anchor_row['Spin']) and row['Spin'] != anchor_row['Spin']: veto = 1
            if pd.notna(row['Parity']) and pd.notna(anchor_row['Parity']) and row['Parity'] != anchor_row['Parity']: veto = 1
            
            p = level_matcher_model.predict([[z, veto]])[0]
            
            # Logic: Prefer higher probability. If tied, prefer lower Z-score.
            if p > best_prob:
                best_prob = p
                best_cluster_idx = idx
                best_z = z
            elif abs(p - best_prob) < 0.01 and p > 0.1: # Tie (within 1%)
                if z < best_z:
                    best_prob = p
                    best_cluster_idx = idx
                    best_z = z
        
        # Decision: Match or New Cluster?
        # Threshold can be from training data (e.g., 0.5 for 3-sigma)
        if best_prob > 0.5 and best_cluster_idx != -1:
            # Add to cluster
            target_cluster = clusters[best_cluster_idx]
            target_cluster['Members'][current_ds] = level_id
            
            # Update Weighted Average Stats
            wt = 1 / (row['DE_level']**2)
            target_cluster['Sum_E_Wt'] += row['E_level'] * wt
            target_cluster['Sum_Wt'] += wt
        else:
            # Create New Cluster
            clusters.append({
                'Anchor_ID': level_id,
                'Members': {current_ds: level_id},
                'Sum_E_Wt': row['E_level'] / (row['DE_level']**2),
                'Sum_Wt': 1 / (row['DE_level']**2)
            })

# ==========================================
# 5. GENERATE ADOPTED LEVELS & SOFT LISTS
# ==========================================
adopted_levels = []

for cluster in clusters:
    # Calculate Adopted Energy
    adopted_e = cluster['Sum_E_Wt'] / cluster['Sum_Wt']
    
    # Determine Spin/Parity (Naive: Take from Anchor or first available)
    # Better: Take from highest priority member that has info
    spin_parity = ""
    for ds in ds_priority:
        if ds in cluster['Members']:
            mem_id = cluster['Members'][ds]
            mem_row = get_level_data(mem_id)
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
        anchor_row = get_level_data(anchor_id)
        
        err = np.sqrt(cand_row['DE_level']**2 + anchor_row['DE_level']**2)
        z = abs(cand_row['E_level'] - anchor_row['E_level']) / err
        
        veto = 0
        if pd.notna(cand_row['Spin']) and pd.notna(anchor_row['Spin']) and cand_row['Spin'] != anchor_row['Spin']: veto = 1
        if pd.notna(cand_row['Parity']) and pd.notna(anchor_row['Parity']) and cand_row['Parity'] != anchor_row['Parity']: veto = 1
        
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
