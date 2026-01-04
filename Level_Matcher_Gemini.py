import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# ==========================================
# 1. SETUP DATASETS
# ==========================================
levels = []

# Dataset A: High resolution, good spin/parity
levels.append({'E_level': 1000.0, 'DE_level': 3.0, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 2000.0, 'DE_level': 4.0, 'Spin': None, 'Parity': None, 'DS': 'A'})
levels.append({'E_level': 3000.0, 'DE_level': 2.0, 'Spin': 1.0,  'Parity': '+',  'DS': 'A'})
levels.append({'E_level': 3005.0, 'DE_level': 3.0, 'Spin': 2.0,  'Parity': '-',  'DS': 'A'})
levels.append({'E_level': 4000.0, 'DE_level': 4.0, 'Spin': 2.0,  'Parity': '+',  'DS': 'A'})

# Dataset B: Reaction data (good selection rules, energy might vary)
levels.append({'E_level': 1005.0, 'DE_level': 3.0, 'Spin': None, 'Parity': None, 'DS': 'B'})
levels.append({'E_level': 2008.0, 'DE_level': 2.0, 'Spin': None, 'Parity': None, 'DS': 'B'})
# Note: B_3000 (2-) should match A_3005 (2-), NOT A_3000 (1+)
levels.append({'E_level': 3000.0, 'DE_level': 1.0, 'Spin': 2.0,  'Parity': '-',  'DS': 'B'}) 
levels.append({'E_level': 5000.0, 'DE_level': 6.0, 'Spin': 2.0,  'Parity': '+',  'DS': 'B'})

# Dataset C: Low resolution (Multiplets possible)
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
# Veto = 1 if Spin OR Parity mismatch. 0 otherwise.

X_train = [
    # --- Clear Matches ---
    [0.0, 0], [0.5, 0], [1.0, 0], [1.5, 0], # Perfect/Good Energy, No Veto -> Match
    
    # --- Energy Mismatch but Acceptable ---
    [2.0, 0], [2.5, 0], [3.0, 0],           # Marginal Energy, No Veto -> Match (Lower prob, but > 0.5)
    
    # --- Clear Non-Matches (Energy) ---
    [4.0, 0], [5.0, 0], [10.0, 0],          # Too far away -> No Match
    
    # --- Physics Veto (CRITICAL) ---
    # Even if Energy is perfect (Z=0), a Physics Veto MUST kill the match.
    [0.0, 1], [0.1, 1], [0.5, 1], [1.0, 1], # Perfect Energy but Wrong Spin/Parity -> No Match
    [2.0, 1], [5.0, 1]                      # Bad Energy AND Wrong Spin -> No Match
]

y_train = [
    1, 1, 1, 1,   # Good
    1, 1, 0,      # Marginal (Z=3.0 is usually cutoff)
    0, 0, 0,      # Bad Energy
    0, 0, 0, 0,   # Veto kills good energy (Strict enforcement)
    0, 0          # Veto kills bad energy
]

# Monotonic constraints: 
# Feature 0 (Z-Score): Increasing Z reduces prob (-1)
# Feature 1 (Veto): Increasing Veto (0->1) reduces prob (-1)
model = XGBClassifier(monotone_constraints='(-1, -1)', n_estimators=100, max_depth=2, random_state=42)
model.fit(X_train, y_train)

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
            veto = 1
        # Parity Mismatch
        if pd.notna(r1['Parity']) and pd.notna(r2['Parity']) and r1['Parity'] != r2['Parity']:
            veto = 1

        # Predict
        prob = model.predict_proba([[z_score, veto]])[0][1]
        
        if prob > 0.5:
            candidates.append({
                'ID1': r1['ID'], 'DS1': r1['DS'],
                'ID2': r2['ID'], 'DS2': r2['DS'],
                'prob': prob
            })

# Sort candidates by probability (Highest confidence first)
candidates.sort(key=lambda x: x['prob'], reverse=True)

# ==========================================
# 4. GREEDY CLUSTERING (Constraint Solver)
# ==========================================
# Goal: Create clusters where NO dataset appears more than once.
# This handles multiplets by forcing the "best" match to take the slot.

# Initialize each level as its own cluster
clusters = {id: {id} for id in df['ID']} # Map ID -> Set of IDs in its cluster
dataset_map = {id: ds for id, ds in zip(df['ID'], df['DS'])}

# Store all pairwise probabilities for lookup
prob_map = {} 
for cand in candidates:
    id1, id2 = cand['ID1'], cand['ID2']
    prob = cand['prob']
    prob_map[(id1, id2)] = prob
    prob_map[(id2, id1)] = prob # Symmetric

    # Greedy Clustering Logic
    c1 = clusters[id1]
    c2 = clusters[id2]
    
    if c1 is c2: continue # Already merged
    
    # Check Constraint: Do c1 and c2 have any conflicting datasets?
    ds_in_c1 = {dataset_map[x] for x in c1}
    ds_in_c2 = {dataset_map[x] for x in c2}
    
    if ds_in_c1.isdisjoint(ds_in_c2):
        # Merge c2 into c1
        new_cluster = c1.union(c2)
        for member in new_cluster:
            clusters[member] = new_cluster

# Extract unique clusters
unique_clusters = []
seen = set()
for c in clusters.values():
    c_frozen = frozenset(c)
    if c_frozen not in seen:
        unique_clusters.append(c)
        seen.add(c_frozen)

# ==========================================
# 5. GENERATE ADOPTED LEVELS
# ==========================================
adopted_levels = []

for cluster in unique_clusters:
    sub = df[df['ID'].isin(cluster)]
    
    # Weighted Average Energy
    errs = sub['DE_level'].fillna(20.0)
    weights = 1 / errs**2
    mean_e = (sub['E_level'] * weights).sum() / weights.sum()
    
    # Spin/Parity Extraction
    spin_val = ""
    if not sub['Spin'].dropna().empty:
        spin_val = str(sub['Spin'].dropna().iloc[0])
        
    parity_val = ""
    if not sub['Parity'].dropna().empty:
        parity_val = str(sub['Parity'].dropna().iloc[0])
    
    # --- Generate "Soft" Source List with Probabilities ---
    # 1. Identify Representative (Anchor) for 100% reference
    # Hierarchy: A > B > C
    cluster_ds = sub['DS'].tolist()
    cluster_ids = sub['ID'].tolist()
    
    if 'A' in cluster_ds:
        rep_idx = cluster_ds.index('A')
    elif 'B' in cluster_ds:
        rep_idx = cluster_ds.index('B')
    else:
        rep_idx = 0
    representative = cluster_ids[rep_idx]

    # 2. Check every level in the dataframe for a match to this cluster
    source_entries = []
    all_ids = df['ID'].tolist()
    
    for candidate_id in all_ids:
        # If it's the representative, it's 100%
        if candidate_id == representative:
            source_entries.append(f"{candidate_id}(100%)")
            continue
            
        # Otherwise, find max probability to ANY member of the cluster
        max_prob = 0.0
        for member in cluster:
            if candidate_id == member:
                # It is in the cluster, but not the rep. 
                # If we have a direct link prob, use it. If not (e.g. merged via third party), might be tricky.
                # For now, check prob_map. If missing (self?), assume 1.0? 
                # No, user wants relative confidence.
                pass
            
            p = prob_map.get((candidate_id, member), 0.0)
            if p > max_prob: max_prob = p
            
        # Threshold for display (e.g., 10%)
        if max_prob > 0.1:
            source_entries.append(f"{candidate_id}({int(max_prob*100)}%)")

    adopted_levels.append({
        'Adopted_E': round(mean_e, 1),
        'Sources': " + ".join(sorted(source_entries)), # Sort for consistency
        'Spin_Parity': spin_val + parity_val
    })

final_df = pd.DataFrame(adopted_levels).sort_values('Adopted_E')
print("\n=== FINAL ADOPTED DATASET ===")
print(final_df.to_string(index=False))
