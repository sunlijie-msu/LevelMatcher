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

# Generate unique IDs (e.g., A_1000) for tracking
df['ID'] = df.apply(lambda x: f"{x['DS']}_{int(x['E_level'])}", axis=1)

# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Input Features:
# 1. Z-Score: Energy difference in standard deviations (abs(E1-E2)/sigma).
# 2. Physics Veto: Binary flag (1=Mismatch in Spin/Parity, 0=Match/Unknown).
#
# Objective: Predict probability of two levels being the same physical state.

# Training Data: (Z_Score, Veto, Probability)
# Synthetic data encoding physics logic: Low Z + No Veto = High Prob.
training_data_points = [
    # --- Excellent Matches (Z < 1.5) ---
    (0.0, 0, 1.00),
    (0.5, 0, 0.99),
    (1.0, 0, 0.98),
    (1.5, 0, 0.95),
    # --- Transition Zone (1.5 < Z < 3.0) ---
    (1.8, 0, 0.90),
    (2.0, 0, 0.80),
    (2.2, 0, 0.70),
    (2.5, 0, 0.60),
    (2.8, 0, 0.55),
    (3.0, 0, 0.50),
    # --- Weak Matches (Z > 3.0) ---
    (3.2, 0, 0.40),
    (3.5, 0, 0.20),
    (4.0, 0, 0.10),
    (5.0, 0, 0.01),
    (10.0, 0, 0.00),
    (20.0, 0, 0.00),
    (100.0, 0, 0.00),
    # --- Physics Veto Cases ---
    # Veto=1 forces Probability to 0.0
    (0.0, 1, 0.00), 
    (1.0, 1, 0.00), 
    (2.0, 1, 0.00), 
    (3.0, 1, 0.00), 
    (5.0, 1, 0.00),
    (100.0, 1, 0.00)
]
# Split into X (Input Features) and y (Target Labels)
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
        veto = 0
        if pd.notna(r1['Spin']) and pd.notna(r2['Spin']) and r1['Spin'] != r2['Spin']: veto = 1
        if pd.notna(r1['Parity']) and pd.notna(r2['Parity']) and r1['Parity'] != r2['Parity']: veto = 1

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

candidates.sort(key=lambda x: x['prob'], reverse=True)

print("\n=== MATCHING CANDIDATES (>50% Probability) ===")
for cand in candidates:
    print(f"{cand['ID1']} <-> {cand['ID2']} | Probability: {cand['prob']:.2%}")

# ==========================================
# 4. GRAPH CLUSTERING (Greedy Merge)
# ==========================================
# Algorithm:
# 1. Initialize every level as a unique cluster.
# 2. Iterate through high-probability candidates.
# 3. Merge clusters if they don't contain conflicting levels (same Dataset).

level_lookup = {row['ID']: row for _, row in df.iterrows()}

# Map ID -> Cluster Object (Set of Members)
id_to_cluster = {row['ID']: {row['DS']: row['ID']} for _, row in df.iterrows()}

for cand in candidates:
    if cand['prob'] < 0.5: break
    
    c1 = id_to_cluster[cand['ID1']]
    c2 = id_to_cluster[cand['ID2']]
    
    # Constraint: A cluster cannot contain two levels from the same Dataset.
    # If intersection of dataset keys is non-empty, skip merge.
    if c1 is c2 or (set(c1.keys()) & set(c2.keys())): 
        continue
        
    # Merge c2 into c1
    c1.update(c2)
    
    # Update references
    for member_id in c2.values():
        id_to_cluster[member_id] = c1

# Extract unique clusters
clusters = []
unique_clusters = {id(c): c for c in id_to_cluster.values()}.values()

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
    xref_entries = sorted(members.values())

    adopted_levels.append({
        'Adopted_E': round(adopted_e, 1),
        'XREF': " + ".join(xref_entries),
        'Spin_Parity': sp
    })

final_df = pd.DataFrame(adopted_levels).sort_values('Adopted_E')
print("\n=== FINAL ADOPTED DATASET ===")
print(final_df.to_string(index=False))
