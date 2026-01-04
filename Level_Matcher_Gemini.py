import pandas as pd
import numpy as np
import networkx as nx
from xgboost import XGBClassifier

# ==========================================
# 1. SETUP DATASETS
# ==========================================
# Define levels from datasets A, B, and C.
# Use None or np.nan for missing values.
levels = []

# Dataset A: Gamma Data
levels.append({'E': 4059.0, 'DE': 3.0, 'J': None, 'DS': 'A', 'ID': 'A_4059'})
levels.append({'E': 4112.0, 'DE': 4.0, 'J': None, 'DS': 'A', 'ID': 'A_4112'})
levels.append({'E': 4173.0, 'DE': 2.0, 'J': 3.0,  'DS': 'A', 'ID': 'A_4173'}) # J=3
levels.append({'E': 4178.0, 'DE': 2.0, 'J': 2.0,  'DS': 'A', 'ID': 'A_4178'}) # J=2
levels.append({'E': 4347.0, 'DE': 4.0, 'J': None, 'DS': 'A', 'ID': 'A_4347'})

# Dataset B: Reaction Data
levels.append({'E': 4055.0, 'DE': 3.0, 'J_excl': None, 'DS': 'B', 'ID': 'B_4055'})
levels.append({'E': 4108.0, 'DE': 2.0, 'J_excl': None, 'DS': 'B', 'ID': 'B_4108'})
# B_4174 Excludes L=3 (cannot match A_4173)
levels.append({'E': 4174.0, 'DE': 1.0, 'J_excl': 3.0,  'DS': 'B', 'ID': 'B_4174'}) 
levels.append({'E': 4343.0, 'DE': 2.0, 'J_excl': None, 'DS': 'B', 'ID': 'B_4343'})

# Dataset C: Low Res Data
levels.append({'E': 2065.0, 'DE': 20.0, 'DS': 'C', 'ID': 'C_2065'})
levels.append({'E': 4170.0, 'DE': 20.0, 'DS': 'C', 'ID': 'C_4170'})
levels.append({'E': 4350.0, 'DE': 20.0, 'DS': 'C', 'ID': 'C_4350'})

df = pd.DataFrame(levels)

# ==========================================
# 2. TRAIN XGBOOST (Teaching it Physics)
# ==========================================
# We create a simple synthetic dataset to teach the model rules:
# Rule 1: Low Z-Score (Energy Diff / Uncertainty) -> Match
# Rule 2: Physics Veto (e.g. Spin Mismatch) -> No Match
# Features: [Z_Score, Veto_Flag]

X_train = [
    [0.0, 0], [0.5, 0], [1.0, 0], [2.0, 0], # Good Matches (Z <= 2)
    [3.0, 0], [5.0, 0], [10.0, 0],          # Bad Energy Matches (Z > 2.5)
    [0.0, 1], [1.0, 1], [5.0, 1]            # Veto=1 is ALWAYS No Match
]
y_train = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

# monotone_constraints='(-1, -1)' forces the model to learn that:
# Increasing Z-Score reduces probability (-1)
# Increasing Veto Flag reduces probability (-1)
model = XGBClassifier(monotone_constraints='(-1, -1)', n_estimators=50, max_depth=2, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. RUN MATCHING
# ==========================================
G = nx.Graph()
for id in df['ID']: G.add_node(id)

# Iterate through every pair of levels from different datasets
import itertools
for i, r1 in df.iterrows():
    for j, r2 in df.iterrows():
        if i >= j or r1['DS'] == r2['DS']: continue

        # --- Feature Calculation ---
        # 1. Z-Score (Energy Difference relative to errors)
        err1 = r1.get('DE', 10.0)
        err2 = r2.get('DE', 10.0)
        sigma = np.sqrt(err1**2 + err2**2)
        z_score = abs(r1['E'] - r2['E']) / sigma

        # 2. Physics Veto
        veto = 0
        # Check if r1 has J and r2 excludes it
        if pd.notna(r1.get('J')) and pd.notna(r2.get('J_excl')):
            if r1['J'] == r2['J_excl']: veto = 1
        # Symmetric check
        if pd.notna(r2.get('J')) and pd.notna(r1.get('J_excl')):
            if r2['J'] == r1['J_excl']: veto = 1

        # --- Prediction ---
        # Returns probability of being a match (Class 1)
        prob = model.predict_proba([[z_score, veto]])[0][1]

        # If probability > 50%, link them
        if prob > 0.5:
            G.add_edge(r1['ID'], r2['ID'], weight=prob)

# ==========================================
# 4. CREATE ADOPTED DATASET
# ==========================================
adopted_levels = []

for comp in nx.connected_components(G):
    sub = df[df['ID'].isin(comp)]
    
    # Calculate Weighted Average Energy
    # Weights = 1 / err^2
    errs = sub['DE'].fillna(20.0) # Use 20 keV if error missing
    weights = 1 / errs**2
    mean_e = (sub['E'] * weights).sum() / weights.sum()
    
    # Calculate Confidence (Average probability of links in this group)
    # If single level, confidence is N/A (or 1.0 for "existence")
    if len(comp) > 1:
        sub_graph = G.subgraph(comp)
        probs = [d['weight'] for u, v, d in sub_graph.edges(data=True)]
        confidence = f"{np.mean(probs):.2f}"
    else:
        confidence = "-"

    adopted_levels.append({
        'Adopted_E': round(mean_e, 1),
        'Confidence': confidence,
        'Sources': " + ".join(sorted(sub['ID']))
    })

# Output
final_df = pd.DataFrame(adopted_levels).sort_values('Adopted_E')
print("\n=== FINAL ADOPTED DATASET ===")
print(final_df.to_string(index=False))