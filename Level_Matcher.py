import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from Feature_Engineer import extract_features, get_training_data, load_levels_from_json

"""
# Sun: Explanation of Code Structure:
1.  **Data Ingestion**: Loads standardized nuclear level data using `Feature_Engineer`.
2.  **Model Training**: Trains an XGBoost model using physics-informed monotonic constraints.
3.  **Inference**: Calculates match probabilities for all cross-dataset pairs.
4.  **Clustering**: Groups matching levels into unique clusters with anchor-based probability reporting.
"""

# ==========================================
# Sun: 1. Test Data Ingestion (From JSON files)
# ==========================================
levels = load_levels_from_json(['A', 'B', 'C'])

dataframe = pd.DataFrame(levels)
# Generate unique level IDs for tracking (e.g., A_1000)
dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

# ==========================================
# Sun: 2. Model Training (Physics-Informed XGBoost)
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
    # Sun: 3. Pairwise Inference
    # ==========================================
    # Extract match probabilities for all cross-dataset level pairs using the trained XGBoost model
    candidates = []
    rows_list = list(dataframe.iterrows())
    
    for i in range(len(rows_list)):
        for j in range(i + 1, len(rows_list)):
            _, level_1 = rows_list[i]
            _, level_2 = rows_list[j]
            
            # Skip same-dataset pairs (only match across different datasets)
            if level_1['dataset_code'] == level_2['dataset_code']:
                continue

            # Extract physics-informed features for this level pair
            feature_vector = extract_features(level_1, level_2)
            
            # Predict match probability using trained XGBoost model
            match_probability = level_matcher_model.predict([feature_vector])[0]
            
            # Store candidates with probability > 10%
            if match_probability > 0.1:
                candidates.append({
                    'ID1': level_1['level_id'],
                    'ID2': level_2['level_id'],
                    'dataset_1': level_1['dataset_code'],
                    'dataset_2': level_2['dataset_code'],
                    'probability': match_probability,
                    'features': feature_vector
                })

    # Sort by probability (highest first)
    candidates.sort(key=lambda x: x['probability'], reverse=True)

    # Write matching candidates to file
    with open('matching_candidates.txt', 'w', encoding='utf-8') as output_file:
        output_file.write("=== MATCHING CANDIDATES (>10%) ===\n\n")
        output_file.write(f"Total Candidates Found: {len(candidates)}\n\n")
        
        for candidate in candidates:
            energy_sim, spin_sim, parity_sim, spin_cert, parity_cert, specificity = candidate['features']
            output_file.write(
                f"{candidate['ID1']} <-> {candidate['ID2']} | "
                f"Probability: {candidate['probability']:.1%}\n"
                f"  Features: Energy_Sim={energy_sim:.2f}, Spin_Sim={spin_sim:.2f}, "
                f"Parity_Sim={parity_sim:.2f}, Spin_Cert={spin_cert:.0f}, "
                f"Parity_Cert={parity_cert:.0f}, Specificity={specificity:.2f}\n\n"
            )
    
    print(f"\n[INFO] Pairwise Inference Complete: {len(candidates)} matching candidates (>10%) written to 'matching_candidates.txt'")

    # ==========================================
    # Sun: 4. Graph Clustering with Overlap Support
    # ==========================================
    # Sun: Core Strategy: Build clusters by merging compatible cross-dataset levels
    # Sun: Key Feature: Doublet support allows one level to belong to multiple clusters when conflicts exist
    
    # Step 1: Initialize data structures
    level_lookup = {row['level_id']: row for _, row in dataframe.iterrows()}
    
    # Sun: Each level starts as its own cluster (dictionary: dataset_code -> level_id)
    initial_clusters = [{row['dataset_code']: row['level_id']} for _, row in dataframe.iterrows()]
    
    # Sun: Track which clusters each level belongs to (one level can be in multiple clusters)
    id_to_clusters = {}
    for cluster in initial_clusters:
        member_id = list(cluster.values())[0]
        id_to_clusters[member_id] = [cluster]

    # Step 2: Build compatibility set (only strong matches with probability > 30%)
    # Sun: We only merge clusters if all members are mutually compatible
    valid_pairs = set()
    for candidate in candidates:
        if candidate['probability'] >= 0.3:
            valid_pairs.add((candidate['ID1'], candidate['ID2']))
            valid_pairs.add((candidate['ID2'], candidate['ID1']))

    # Step 3: Merge compatible clusters by iterating through strong candidates
    for candidate in candidates:
        if candidate['probability'] < 0.3:
            break  # Sun: Skip weak matches
        
        id_1, id_2 = candidate['ID1'], candidate['ID2']
        
        # Sun: Snapshot current cluster memberships before modification
        cluster_list_1 = list(id_to_clusters[id_1])
        cluster_list_2 = list(id_to_clusters[id_2])

        # Sun: Try to merge each cluster pair that contains id_1 and id_2
        for cluster_1 in cluster_list_1:
            for cluster_2 in cluster_list_2:
                if cluster_1 is cluster_2:
                    continue  # Sun: Already in same cluster
                
                datasets_1 = set(cluster_1.keys())
                datasets_2 = set(cluster_2.keys())
                
                # Scenario A: Dataset conflict exists (both clusters have levels from same dataset)
                if not datasets_1.isdisjoint(datasets_2):
                    # Sun: Example: cluster_1 has A_3000, cluster_2 has A_3005
                    # Sun: Cannot merge these clusters, but try "doublet assignment"
                    
                    # Sun: Can we add id_1 to cluster_2?
                    if candidate['dataset_1'] not in datasets_2:
                        # Sun: Check if id_1 is compatible with ALL existing members of cluster_2
                        if all((id_1, member_id) in valid_pairs for member_id in cluster_2.values()):
                            if id_1 not in cluster_2.values():
                                cluster_2[candidate['dataset_1']] = id_1
                                if cluster_2 not in id_to_clusters[id_1]:
                                    id_to_clusters[id_1].append(cluster_2)
                    
                    # Sun: Can we add id_2 to cluster_1?
                    if candidate['dataset_2'] not in datasets_1:
                        # Sun: Check if id_2 is compatible with ALL existing members of cluster_1
                        if all((id_2, member_id) in valid_pairs for member_id in cluster_1.values()):
                            if id_2 not in cluster_1.values():
                                cluster_1[candidate['dataset_2']] = id_2
                                if cluster_1 not in id_to_clusters[id_2]:
                                    id_to_clusters[id_2].append(cluster_1)
                    
                    continue  # Sun: Move to next cluster pair

                # Scenario B: No dataset conflict (can potentially merge)
                # Sun: Verify all-to-all compatibility before merging
                consistent = True
                for member_1 in cluster_1.values():
                    for member_2 in cluster_2.values():
                        if (member_1, member_2) not in valid_pairs:
                            consistent = False
                            break
                    if not consistent:
                        break
                
                if consistent:
                    # Sun: Perform merge: absorb cluster_2 into cluster_1
                    cluster_1.update(cluster_2)
                    
                    # Sun: Update all members of cluster_2 to point to cluster_1
                    for member_id in cluster_2.values():
                        if cluster_2 in id_to_clusters[member_id]:
                            id_to_clusters[member_id].remove(cluster_2)
                        if cluster_1 not in id_to_clusters[member_id]:
                            id_to_clusters[member_id].append(cluster_1)

    # Step 4: Extract unique active clusters
    unique_clusters = []
    seen = set()
    for cluster_list in id_to_clusters.values():
        for cluster in cluster_list:
            cluster_id = id(cluster)
            if cluster_id not in seen:
                unique_clusters.append(cluster)
                seen.add(cluster_id)
    
    # Step 5: Sort clusters by average energy for consistent output
    def calculate_cluster_average_energy(cluster):
        energies = [level_lookup[member_id]['energy_value'] for member_id in cluster.values()]
        return sum(energies) / len(energies)
    
    unique_clusters.sort(key=calculate_cluster_average_energy)

    print("\n=== FINAL CLUSTERING RESULTS ===")
    for i, cluster in enumerate(unique_clusters):
        # Anchor Selection (Level with smallest uncertainty)
        anchor_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        anchor_data = level_lookup[anchor_id]
        anchor_energy = anchor_data['energy_value']
        anchor_jpi = anchor_data.get('spin_parity', '')
        if anchor_jpi == "unknown": anchor_jpi = "N/A"
        
        print(f"\nCluster {i+1}:")
        print(f"  Anchor: {anchor_id} | E={anchor_energy:.1f} keV | Jπ={anchor_jpi}")
        print(f"  Members:")
        
        # Display each member with probability
        for dataset_code in sorted(cluster.keys()):
            member_id = cluster[dataset_code]
            member_data = level_lookup[member_id]
            member_energy = member_data['energy_value']
            member_jpi = member_data.get('spin_parity', '')
            if member_jpi == "unknown": member_jpi = "N/A"
            
            if member_id == anchor_id:
                print(f"    [{dataset_code}] {member_id}: E={member_energy:.1f} keV, Jπ={member_jpi} (Anchor)")
            else:
                input_vector = extract_features(member_data, anchor_data)
                probability = level_matcher_model.predict([input_vector])[0]
                print(f"    [{dataset_code}] {member_id}: E={member_energy:.1f} keV, Jπ={member_jpi} (Match Prob: {probability:.1%})")
