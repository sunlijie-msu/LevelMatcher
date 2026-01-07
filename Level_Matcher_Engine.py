import pandas as pd
import numpy as np
import json
import os
import re
from xgboost import XGBRegressor
from physics_parser import PhysicsFeatureEngine

# ==========================================
# 1. DATA INGESTION (From JSON files)
# ==========================================
levels = []
for dataset_code in ['A', 'B', 'C']:
    filename = f"test_dataset_{dataset_code}.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle new ENSDF JSON schema (levelsTable -> levels)
            if isinstance(data, dict) and 'levelsTable' in data:
                raw_levels = data['levelsTable'].get('levels', [])
                for item in raw_levels:
                    # Extract Energy
                    energy_value = item.get('energy', {}).get('value')
                    
                    # Extract Energy Uncertainty
                    energy_uncertainty = item.get('energy', {}).get('uncertainty', {}).get('value')
                    
                    # Extract Spin_Parity Values
                    spin_parity_list = item.get('spinParity', {}).get('values', [])
                    spin_parity_string = item.get('spinParity', {}).get('evaluatorInput') 
                    
                    if energy_value is not None:
                        levels.append({
                            'dataset_code': dataset_code,
                            'energy_value': float(energy_value),
                            'energy_uncertainty': float(energy_uncertainty) if energy_uncertainty is not None else 10.0,
                            'Spin_Parity_List': spin_parity_list,
                            'Spin_Parity': spin_parity_string
                        })
            elif isinstance(data, list):
                # Fallback for flat list format
                for item in data:
                    item['dataset_code'] = dataset_code
                    levels.append(item)

dataframe = pd.DataFrame(levels)
# Generate unique level IDs for tracking (e.g., A_1000)
dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Features: [Z_Score, J_Score (Spin), Pi_Score (Parity), J_Tentative, Pi_Tentative, Ambiguity]
# Constraints:
# 1. Z-Score (-1): Higher Z-Score -> Lower Probability
# 2. J_Score (+1): Higher Score (1.0=Match) -> Higher Probability
# 3. Pi_Score (+1): Higher Score (1.0=Match) -> Higher Probability
# 4. J_Tentative (-1): Higher Tentativeness -> Lower Probability
# 5. Pi_Tentative (-1): Higher Tentativeness -> Lower Probability
# 6. Ambiguity (-1): Higher Ambiguity (more candidates) -> Lower Probability

if __name__ == "__main__":
    # Get Training Data from Physics Parser
    training_features, training_labels = PhysicsFeatureEngine.get_training_data()

    # Monotonic constraints strictly enforce physics rules in the model:
    # Feature 0 (Z-Score): -1
    # Feature 1 (J_Score): +1
    # Feature 2 (Pi_Score): +1
    # Feature 3 (J_Tentative): -1
    # Feature 4 (Pi_Tentative): -1
    # Feature 5 (Ambiguity): -1
    level_matcher_model = XGBRegressor(objective='binary:logistic', 
                                       monotone_constraints='(-1, 1, 1, -1, -1, -1)', 
                                       n_estimators=500, 
                                       max_depth=6, 
                                       learning_rate=0.05,
                                       random_state=42)
    
    # Train the model on the generated training data
    level_matcher_model.fit(training_features, training_labels)

    # ==========================================
    # 3. PAIRWISE INFERENCE
    # ==========================================
    candidates = []
    
    # Extract row list for efficient iteration
    rows_list = [row for _, row in dataframe.iterrows()]
    
    for i in range(len(rows_list)):
        for j in range(i + 1, len(rows_list)):
            level_1 = rows_list[i]
            level_2 = rows_list[j]
            
            # Cross-Dataset matching only
            if level_1['dataset_code'] == level_2['dataset_code']: continue

            # Physics-Grounded Feature Extraction
            input_vector = PhysicsFeatureEngine.extract_features(level_1, level_2)
            
            # Predict match probability
            probability = level_matcher_model.predict([input_vector])[0]
            
            if probability > 0.1:
                # Store matching candidate
                candidates.append({
                    'ID1': level_1['level_id'], 'dataset_1': level_1['dataset_code'],
                    'ID2': level_2['level_id'], 'dataset_2': level_2['dataset_code'],
                    'probability': probability,
                    'features': input_vector
                })

    candidates.sort(key=lambda x: x['probability'], reverse=True)

    print("\n=== MATCHING CANDIDATES (>10%) ===")
    for candidate in candidates:
        # Unpack features for display: [Z_Score, J_Score, Pi_Score, J_Tentative, Pi_Tentative, Ambiguity]
        z_score, j_score, parity_score, j_tentative, parity_tentative, ambiguity = candidate['features']
        print(f"{candidate['ID1']} <-> {candidate['ID2']} | Probability: {candidate['probability']:.1%} "
              f"(Z={z_score:.2f}, SpinScore={j_score:.2f}, ParityScore={parity_score:.2f}, "
              f"SpinTentative={j_tentative:.0f}, ParityTentative={parity_tentative:.0f}, Ambiguity={ambiguity:.2f})")

    # ==========================================
    # 4. CLUSTERING (Graph Clustering with Overlap Support)
    # ==========================================
    # Algorithm:
    # 1. Initialize every level as a unique cluster.
    # 2. Iterate through high-probability candidates.
    # 3. Merge clusters if they don't contain conflicting levels.
    # 4. Support "Doublets": If two clusters cannot merge due to conflict (e.g. A_3000 vs A_3005),
    #    but a level (C_3005) matches both, allow C_3005 to belong to BOTH clusters.

    level_lookup = {row['level_id']: row for _, row in dataframe.iterrows()}
    
    # Map ID -> List of Cluster Objects (A level can belong to multiple clusters)
    # Each cluster is a dictionary mapping dataset_code to level_id
    initial_clusters = [{row['dataset_code']: row['level_id']} for _, row in dataframe.iterrows()]
    id_to_clusters = {}
    for cluster in initial_clusters:
        member_id = list(cluster.values())[0]
        id_to_clusters[member_id] = [cluster]

    # Valid High-Probability pairs for consistency checks
    valid_pairs = set((candidate['ID1'], candidate['ID2']) for candidate in candidates)
    valid_pairs.update((candidate['ID2'], candidate['ID1']) for candidate in candidates)

    for candidate in candidates:
        if candidate['probability'] < 0.5: break # Only merge strong candidates
        id_1, id_2 = candidate['ID1'], candidate['ID2']
        
        # Work on snapshots of the cluster lists since we might modify them
        cluster_list_1 = list(id_to_clusters[id_1])
        cluster_list_2 = list(id_to_clusters[id_2])

        for cluster_1 in cluster_list_1:
            for cluster_2 in cluster_list_2:
                if cluster_1 is cluster_2: continue # Already same cluster
                
                datasets_1 = set(cluster_1.keys())
                datasets_2 = set(cluster_2.keys())
                
                # Check 1: Dataset Conflict
                if not datasets_1.isdisjoint(datasets_2):
                    # They overlap in datasets (e.g. both have a 'Dataset A' level)
                    # CONFLICT DETECTED.
                    # Try "Soft Assignment" (Doublet Logic).
                    
                    # Try Adding id_1 to cluster_2? (Only if cluster_2 lacks dataset_1)
                    if candidate['dataset_1'] not in datasets_2:
                        # Consistency check: id_1 must match ALL members of cluster_2
                        if all((id_1, member_id) in valid_pairs for member_id in cluster_2.values()):
                            if id_1 not in cluster_2.values(): # avoid duplication
                                cluster_2[candidate['dataset_1']] = id_1
                                if cluster_2 not in id_to_clusters[id_1]: id_to_clusters[id_1].append(cluster_2)
                    
                    # Try Adding id_2 to cluster_1? (Only if cluster_1 lacks dataset_2)
                    if candidate['dataset_2'] not in datasets_1:
                        # Consistency check: id_2 must match ALL members of cluster_1
                        if all((id_2, member_id) in valid_pairs for member_id in cluster_1.values()):
                            if id_2 not in cluster_1.values():
                                cluster_1[candidate['dataset_2']] = id_2
                                if cluster_1 not in id_to_clusters[id_2]: id_to_clusters[id_2].append(cluster_1)
                    
                    continue

                # Check 2: Consistency (All-to-All)
                # If we merge cluster_1 and cluster_2, every member of cluster_1 must match every member of cluster_2
                consistent = True
                for member_1 in cluster_1.values():
                    for member_2 in cluster_2.values():
                        if (member_1, member_2) not in valid_pairs:
                            consistent = False; break
                    if not consistent: break
                
                if consistent:
                    # Perform Merge: cluster_2 into cluster_1
                    cluster_1.update(cluster_2)
                    # Update references for all members of cluster_2
                    for member_id in cluster_2.values():
                        # Remove cluster_2 from their list
                        if cluster_2 in id_to_clusters[member_id]: id_to_clusters[member_id].remove(cluster_2)
                        # Add cluster_1 if not present
                        if cluster_1 not in id_to_clusters[member_id]: id_to_clusters[member_id].append(cluster_1)
                    # cluster_2 is effectively dissolved into cluster_1

    # Extract unique active clusters
    unique_clusters = []
    seen = set()
    for cluster_list in id_to_clusters.values():
        for cluster in cluster_list:
            cluster_id = id(cluster)
            if cluster_id not in seen:
                unique_clusters.append(cluster)
                seen.add(cluster_id)

    # ==========================================
    # 5. REPORTING / ADOPTED LEVEL GENERATION
    # ==========================================
    adopted_levels = []
    for cluster in unique_clusters:
        if not cluster: continue
        
        # Anchor Selection: Lowest energy uncertainty (DE_level)
        # If DE_level is missing, treat as high error (999 keV)
        anchor_level_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        anchor_level_data = level_lookup[anchor_level_id]
        
        # Weighted Average Energy calculation
        energy_and_error_pairs = []
        for member_id in cluster.values():
            level_data = level_lookup[member_id]
            error = level_data['energy_uncertainty'] if pd.notna(level_data['energy_uncertainty']) else 10 # Default to 10 keV if unknown
            energy_and_error_pairs.append((level_data['energy_value'], error))
        
        weights = [1.0 / (error**2) for energy, error in energy_and_error_pairs]
        sum_of_weights = sum(weights)
        adopted_energy = sum(pair[0] * weight for pair, weight in zip(energy_and_error_pairs, weights)) / sum_of_weights
        
        # Cross-Reference (XREF) String Generation
        xref_parts = []
        sorted_member_ids = sorted(cluster.values(), key=lambda x: (level_lookup[x]['dataset_code'], level_lookup[x]['energy_value']))
        
        for member_id in sorted_member_ids:
            if member_id == anchor_level_id:
                xref_parts.append(f"{member_id}(Anchor)")
            else:
                # Recalculate probability against Anchor for display clarity
                level_data_1 = level_lookup[member_id]
                level_data_2 = anchor_level_data
                
                # Use updated feature engine with long names
                input_vector = PhysicsFeatureEngine.extract_features(level_data_1, level_data_2)
                probability = level_matcher_model.predict([input_vector])[0]
                xref_parts.append(f"{member_id}({probability:.0%})")
        
        adopted_levels.append({
            'Adopted_Energy': round(adopted_energy, 1),
            'XREF': " + ".join(xref_parts),
            'Spin_Parity': anchor_level_data.get('Spin_Parity', '') # Preserving case for standard nuclear labels
        })

    adopted_dataframe = pd.DataFrame(adopted_levels).sort_values('Adopted_Energy')
    print("\n=== ADOPTED LEVELS ===")
    print(adopted_dataframe.to_string(index=False))
