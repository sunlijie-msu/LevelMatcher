import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from Feature_Engineer import extract_features, get_training_data, load_levels_from_json

"""
Explanation of Code Structure:
1.  **Data Ingestion**: Loads standardized nuclear level data using `Feature_Engineer`.
2.  **Model Training**: Trains an XGBoost model using physics-informed monotonic constraints.
3.  **Inference**: Calculates match probabilities for all cross-dataset pairs.
4.  **Clustering**: Groups matching levels into unique clusters, handling conflicts via logic.
5.  **Reporting**: Generates "Adopted Levels" with weighted energy averages and cross-references.
"""

# ==========================================
# 1. Test Data Ingestion (From JSON files)
# ==========================================
levels = load_levels_from_json(['A', 'B', 'C'])

dataframe = pd.DataFrame(levels)
# Generate unique level IDs for tracking (e.g., A_1000)
dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

# ==========================================
# 2. MODEL TRAINING (Physics-Informed XGBoost)
# ==========================================
# Features: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]
# Constraints (All Monotonic Increasing):
# 1. Energy_Similarity (+1): Higher Score (1.0=Match) -> Higher Probability
# 2. Spin_Similarity (+1): Higher Score (1.0=Match) -> Higher Probability
# 3. Parity_Similarity (+1): Higher Score (1.0=Match) -> Higher Probability
# 4. Spin_Certainty (+1): Higher Certainty (1.0=Firm) -> Higher Probability
# 5. Parity_Certainty (+1): Higher Certainty (1.0=Firm) -> Higher Probability
# 6. Specificity (+1): Higher Specificity (1.0=Single Option) -> Higher Probability

if __name__ == "__main__":
    # Get Training Data from Physics Parser
    training_features, training_labels = get_training_data()

    # Monotonic constraints strictly enforce physics rules in the model:
    # All features are designed so that Higher Value == Better Match
    # 1. Energy_Similarity (+1)
    # 2. Spin_Similarity (+1)
    # 3. Parity_Similarity (+1)
    # 4. Spin_Certainty (+1)
    # 5. Parity_Certainty (+1)
    # 6. Specificity (+1)
    level_matcher_model = XGBRegressor(objective='binary:logistic', # Sun: Learning Objective Function: Training Loss + Regularization.
                                       # Sun: The loss function computes the difference between the true y value and the predicted y value.
                                       # Sun: The regularization term discourages overly complex trees.
                                       monotone_constraints='(1, 1, 1, 1, 1, 1)', # Enforce an increasing constraint on all predictors. In some cases, where there is a very strong prior belief that the true relationship has some quality, constraints can be used to improve the predictive performance of the model.
                                       n_estimators=100, # Number of gradient boosted trees
                                       max_depth=3, # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
                                       learning_rate=0.05,
                                       random_state=42) # Random number seed
    
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
            input_vector = extract_features(level_1, level_2)
            
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
        # Unpack features for display: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Spin_Certainty, Parity_Certainty, Specificity]
        energy_similarity, spin_similarity, parity_similarity, spin_certainty, parity_certainty, specificity = candidate['features']
        print(f"{candidate['ID1']} <-> {candidate['ID2']} | Probability: {candidate['probability']:.1%} "
              f"(Energy_Similarity={energy_similarity:.2f}, Spin_Similarity={spin_similarity:.2f}, Parity_Similarity={parity_similarity:.2f}, "
              f"Spin_Certainty={spin_certainty:.0f}, Parity_Certainty={parity_certainty:.0f}, Specificity={specificity:.2f})")

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

    # Valid High-Probability pairs for consistency checks (Prob > 0.5)
    # We enforce that all members of a cluster must be strongly compatible.
    # A low-probability "Technical Match" (e.g. 20%) is not enough to sustain a cluster merge.
    valid_pairs = set((candidate['ID1'], candidate['ID2']) for candidate in candidates if candidate['probability'] >= 0.5)
    valid_pairs.update((candidate['ID2'], candidate['ID1']) for candidate in candidates if candidate['probability'] >= 0.5)

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
    
    # Sort clusters by average energy for consistent display
    def get_cluster_avg_energy(c):
        energies = [level_lookup[mid]['energy_value'] for mid in c.values()]
        return sum(energies) / len(energies)
    unique_clusters.sort(key=get_cluster_avg_energy)

    print("\n=== CLUSTERING RESULTS ===")
    for i, cluster in enumerate(unique_clusters):
        # Find Anchor
        anchor_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        members_str = ", ".join([f"{ds}:{mid}" for ds, mid in sorted(cluster.items())])
        print(f"Cluster {i+1}: Anchor={anchor_id} | Members=[{members_str}]")

    # ==========================================
    # 5. REPORTING / ADOPTED LEVEL GENERATION
    # ==========================================
    adopted_levels = []
    
    # Define columns for the output table
    dataset_columns = sorted(list(set(row['dataset_code'] for _, row in dataframe.iterrows())))
    
    for cluster in unique_clusters:
        if not cluster: continue
        
        # Anchor Selection
        anchor_level_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        anchor_level_data = level_lookup[anchor_level_id]
        
        # Weighted Average Energy calculation
        energy_and_error_pairs = []
        for member_id in cluster.values():
            level_data = level_lookup[member_id]
            error = level_data['energy_uncertainty'] if pd.notna(level_data['energy_uncertainty']) else 10
            energy_and_error_pairs.append((level_data['energy_value'], error))
        
        weights = [1.0 / (error**2) for energy, error in energy_and_error_pairs]
        sum_of_weights = sum(weights)
        adopted_energy = sum(pair[0] * weight for pair, weight in zip(energy_and_error_pairs, weights)) / sum_of_weights
        
        # Row Construction
        jpi_opt = anchor_level_data.get('spin_parity', '')
        if jpi_opt == "unknown": jpi_opt = ""
        
        row_data = {
            'E_Adopted': f"{adopted_energy:.1f}",
            'Jpi_Adopted': jpi_opt
        }
        
        # Fill Dataset Columns
        for ds in dataset_columns:
            if ds in cluster:
                member_id = cluster[ds]
                level_data = level_lookup[member_id]
                e_val = level_data['energy_value']
                jpi = level_data.get('spin_parity', '')
                if not jpi or jpi == "unknown": jpi = ""
                
                # Probability
                if member_id == anchor_level_id:
                    prob_str = "Anchor"
                else:
                    input_vector = extract_features(level_data, anchor_level_data)
                    prob = level_matcher_model.predict([input_vector])[0]
                    prob_str = f"{prob:.0%}"
                
                # Format: "Energy Jpi (Prob)"
                if jpi:
                    cell_str = f"{e_val:.1f} {jpi} ({prob_str})"
                else:
                    cell_str = f"{e_val:.1f} ({prob_str})"
                    
                row_data[ds] = cell_str
            else:
                row_data[ds] = "-"
        
        adopted_levels.append(row_data)

    adopted_dataframe = pd.DataFrame(adopted_levels)
    # Reorder columns: E_Adopted, Jpi_Adopted, then Datasets
    cols = ['E_Adopted', 'Jpi_Adopted'] + dataset_columns
    adopted_dataframe = adopted_dataframe[cols]
    
    print("\n=== ADOPTED LEVELS TABLE ===")
    # Adjust pandas display settings for alignment
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    # Calculate a dynamic col_space based on content or stick to a reasonable default
    # 18 chars is usually enough for "8888.8 88(+) (100%)"
    print(adopted_dataframe.to_string(index=False, col_space=18, justify='left'))
