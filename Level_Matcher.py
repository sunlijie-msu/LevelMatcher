import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from Feature_Engineer import extract_features, get_training_data, load_levels_from_json

"""
# FRIBND: High-level Strategy Explanation:
1.  **Data Ingestion**: Loads standardized nuclear level data using `Feature_Engineer`.
2.  **Model Training**: Trains XGBoost regressor using physics-informed monotonic constraints on 6 features.
3.  **Pairwise Inference**: Calculates match probabilities for all cross-dataset level pairs using trained ML model.
4.  **Graph Clustering**: Groups matching levels using rule-based graph algorithm (no ML, only logic-based merging).
"""

# Configuration Parameters
pairwise_output_threshold = 0.01  # Minimum probability for outputting level pairs (1%)
clustering_merge_threshold = 0.30  # Minimum probability for cluster merging (30%)

# ==========================================
# FRIBND: 1. Test Data Ingestion (From JSON files)
# ==========================================
levels = load_levels_from_json(['A', 'B', 'C'])

dataframe = pd.DataFrame(levels)
# Generate unique level IDs for tracking (e.g., A_1000)
dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

# ==========================================
# FRIBND: 2. Model Training (Physics-Informed XGBoost)
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
    # Get training data from physics parser
    training_features, training_labels = get_training_data()

    # FRIBND: Train XGBoost regressor with monotonic constraints enforcing physics rules
    # FRIBND: All six features designed so higher value → better match probability
    level_matcher_model = XGBRegressor(objective='binary:logistic',
                                       # FRIBND: Learning Objective Function: Training Loss + Regularization.
                                       # FRIBND: The loss function computes the difference between the true y value and the predicted y value.
                                       # FRIBND: The regularization term discourages overly complex trees.
                                       monotone_constraints='(1, 1, 1, 1, 1, 1)',
                                       # FRIBND: Enforce increasing constraint on all six features.
                                       # FRIBND: Physics prior: higher feature values always indicate better matches.
                                       # FRIBND: Monotonic constraints improve predictive performance when strong prior beliefs exist.
                                       n_estimators=100,
                                       # FRIBND: Number of gradient boosted trees (ensemble size).
                                       max_depth=3,
                                       # FRIBND: Maximum tree depth. Lower values prevent overfitting but may underfit.
                                       # FRIBND: Value of 3 balances model complexity with generalization.
                                       learning_rate=0.05,
                                       # FRIBND: Step size shrinkage to prevent overfitting. Lower values require more trees.
                                       random_state=42)
                                       # FRIBND: Random number seed for reproducibility.
    
    # Train model on synthetic training data
    level_matcher_model.fit(training_features, training_labels)

    # ==========================================
    # FRIBND: 3. Pairwise Inference
    # ==========================================
    # FRIBND: Extract match probabilities for all cross-dataset level pairs using the trained XGBoost model
    matching_level_pairs = []
    rows_list = list(dataframe.iterrows())
    
    for i in range(len(rows_list)):
        for j in range(i + 1, len(rows_list)):
            _, level_1 = rows_list[i]
            _, level_2 = rows_list[j]
            
            # FRIBND: Skip same-dataset pairs (only match across different datasets)
            if level_1['dataset_code'] == level_2['dataset_code']:
                continue

            # FRIBND: Extract physics-informed features for this level pair
            feature_vector = extract_features(level_1, level_2)
            
            # FRIBND: Predict match probability using trained XGBoost model
            match_probability = level_matcher_model.predict([feature_vector])[0]
            
            # Record level pairs above output threshold
            if match_probability > pairwise_output_threshold:
                matching_level_pairs.append({
                    'ID1': level_1['level_id'],
                    'ID2': level_2['level_id'],
                    'dataset_1': level_1['dataset_code'],
                    'dataset_2': level_2['dataset_code'],
                    'probability': match_probability,
                    'features': feature_vector
                })

    # Sort by probability descending
    matching_level_pairs.sort(key=lambda x: x['probability'], reverse=True)

    # Write pairwise inference results to file
    threshold_percent = int(pairwise_output_threshold * 100)
    with open('level_pairs_inference.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== PAIRWISE INFERENCE RESULTS (>{threshold_percent}%) ===\n\n")
        output_file.write(f"Total Level Pairs Found: {len(matching_level_pairs)}\n\n")
        
        for matching_level_pair in matching_level_pairs:
            energy_sim, spin_sim, parity_sim, spin_cert, parity_cert, specificity = matching_level_pair['features']
            output_file.write(
                f"{matching_level_pair['ID1']} <-> {matching_level_pair['ID2']} | "
                f"Probability: {matching_level_pair['probability']:.1%}\n"
                f"  Features: Energy_Sim={energy_sim:.2f}, Spin_Sim={spin_sim:.2f}, "
                f"Parity_Sim={parity_sim:.2f}, Spin_Cert={spin_cert:.0f}, "
                f"Parity_Cert={parity_cert:.0f}, Specificity={specificity:.2f}\n\n"
            )
    
    print(f"\n[INFO] Pairwise Inference Complete: {len(matching_level_pairs)} level pairs (>{threshold_percent}%) written to 'level_pairs_inference.txt'")

    # ==========================================
    # FRIBND: 4. Graph Clustering with Overlap Support
    # ==========================================
    # FRIBND: Algorithm: Greedy cluster merging based on match probability ranking
    # FRIBND: Key Constraints:
    #   - Dataset Uniqueness: Each cluster contains at most one level per dataset
    #   - Mutual Consistency: All cluster members must be pairwise compatible
    #   - Ambiguity Resolution: Levels with poor resolution (large uncertainty) can belong to multiple clusters when they are compatible with multiple well-resolved levels
    # FRIBND: This section uses NO ML, only rule-based graph algorithms for logical cluster merging
    
    # FRIBND: Step 1 - Initialize singleton clusters and lookup table
    level_lookup = {row['level_id']: row for _, row in dataframe.iterrows()}
    initial_clusters = [{row['dataset_code']: row['level_id']} for _, row in dataframe.iterrows()]
    
    # FRIBND: Track which clusters each level belongs to (one level can be in multiple clusters)
    id_to_clusters = {}
    for cluster in initial_clusters:
        # FRIBND: Extract the single level_id from this single-member cluster
        # FRIBND: cluster.values() returns dict_values(['A_1000']), list() converts to ['A_1000'], [0] gets 'A_1000'
        member_id = list(cluster.values())[0]
        id_to_clusters[member_id] = [cluster]

    # FRIBND: Step 2 - Extract strong candidate pairs for merging
    # FRIBND: Only level pairs exceeding clustering_merge_threshold qualify for cluster operations
    valid_pairs = set()
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] >= clustering_merge_threshold:
            valid_pairs.add((matching_level_pair['ID1'], matching_level_pair['ID2']))
            valid_pairs.add((matching_level_pair['ID2'], matching_level_pair['ID1']))

    # FRIBND: Step 3 - Greedy cluster merging
    # FRIBND: Process candidates in descending probability order, attempting merges or multi-cluster assignment for ambiguous levels
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] < clustering_merge_threshold:
            break  # FRIBND: Skip weak matches (list is sorted by probability)
        
        id_1, id_2 = matching_level_pair['ID1'], matching_level_pair['ID2']
        
        # FRIBND: Snapshot current cluster memberships before modification
        cluster_list_1 = list(id_to_clusters[id_1])
        cluster_list_2 = list(id_to_clusters[id_2])

        # FRIBND: Try to merge each cluster pair that contains id_1 and id_2
        for cluster_1 in cluster_list_1:
            for cluster_2 in cluster_list_2:
                if cluster_1 is cluster_2:
                    continue  # Already in same cluster
                
                datasets_1 = set(cluster_1.keys())
                datasets_2 = set(cluster_2.keys())
                
                # Scenario A: Dataset overlap (both clusters have levels from same dataset)
                if not datasets_1.isdisjoint(datasets_2):
                    # FRIBND: Example: cluster_1 has A_3000, cluster_2 has A_3005 (both resolved), C_3002 (poorly resolved with large uncertainty) matches both
                    # FRIBND: Cannot merge these clusters (violates dataset uniqueness), but try multi-cluster assignment for ambiguous levels
                    
                    # FRIBND: Can we add id_1 to cluster_2?
                    if matching_level_pair['dataset_1'] not in datasets_2:
                        # FRIBND: Check if id_1 is compatible with ALL existing members of cluster_2
                        if all((id_1, member_id) in valid_pairs for member_id in cluster_2.values()):
                            if id_1 not in cluster_2.values():
                                cluster_2[matching_level_pair['dataset_1']] = id_1
                                if cluster_2 not in id_to_clusters[id_1]:
                                    id_to_clusters[id_1].append(cluster_2)
                    
                    # FRIBND: Can we add id_2 to cluster_1?
                    if matching_level_pair['dataset_2'] not in datasets_1:
                        # FRIBND: Check if id_2 is compatible with ALL existing members of cluster_1
                        if all((id_2, member_id) in valid_pairs for member_id in cluster_1.values()):
                            if id_2 not in cluster_1.values():
                                cluster_1[matching_level_pair['dataset_2']] = id_2
                                if cluster_1 not in id_to_clusters[id_2]:
                                    id_to_clusters[id_2].append(cluster_1)
                    
                    continue  # Move to next cluster pair

                # Scenario B: No dataset overlap (can potentially merge)
                # FRIBND: Verify all-to-all compatibility before merging
                consistent = True
                for member_1 in cluster_1.values():
                    for member_2 in cluster_2.values():
                        if (member_1, member_2) not in valid_pairs:
                            consistent = False
                            break
                    if not consistent:
                        break
                
                if consistent:
                    # FRIBND: Perform merge: absorb cluster_2 into cluster_1
                    cluster_1.update(cluster_2)
                    
                    # FRIBND: Update all members of cluster_2 to point to cluster_1
                    for member_id in cluster_2.values():
                        if cluster_2 in id_to_clusters[member_id]:
                            id_to_clusters[member_id].remove(cluster_2)
                        if cluster_1 not in id_to_clusters[member_id]:
                            id_to_clusters[member_id].append(cluster_1)

    # FRIBND: Step 4 - Extract unique active clusters (remove duplicates)
    unique_clusters = []
    seen = set()
    for cluster_list in id_to_clusters.values():
        for cluster in cluster_list:
            cluster_id = id(cluster)
            if cluster_id not in seen:
                unique_clusters.append(cluster)
                seen.add(cluster_id)
    
    # FRIBND: Step 5 - Sort clusters by average energy for consistent output
    def calculate_cluster_average_energy(cluster):
        energies = [level_lookup[member_id]['energy_value'] for member_id in cluster.values()]
        return sum(energies) / len(energies)
    
    unique_clusters.sort(key=calculate_cluster_average_energy)

    # FRIBND: Write clustering results to both console and file
    clustering_output_lines = []
    clustering_output_lines.append("=== FINAL CLUSTERING RESULTS ===\n")
    
    print("\n=== FINAL CLUSTERING RESULTS ===")
    for i, cluster in enumerate(unique_clusters):
        # FRIBND: Select anchor as the level with smallest energy uncertainty (most precise measurement)
        anchor_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        anchor_data = level_lookup[anchor_id]
        anchor_energy = anchor_data['energy_value']
        anchor_uncertainty = anchor_data['energy_uncertainty'] if pd.notna(anchor_data['energy_uncertainty']) else 0
        anchor_jpi = anchor_data.get('spin_parity', '')
        if anchor_jpi == "unknown": anchor_jpi = "N/A"
        
        cluster_header = f"\nCluster {i+1}:"
        anchor_line = f"  Anchor: {anchor_id} | E={anchor_energy:.1f}±{anchor_uncertainty:.1f} keV | Jπ={anchor_jpi}"
        members_header = "  Members:"
        
        print(cluster_header)
        print(anchor_line)
        print(members_header)
        
        clustering_output_lines.append(cluster_header + "\n")
        clustering_output_lines.append(anchor_line + "\n")
        clustering_output_lines.append(members_header + "\n")
        
        # FRIBND: Display each member with match probability relative to anchor
        # FRIBND: Anchor member shows no probability (identity match). Other members show ML-predicted probability.
        for dataset_code in sorted(cluster.keys()):
            member_id = cluster[dataset_code]
            member_data = level_lookup[member_id]
            member_energy = member_data['energy_value']
            member_uncertainty = member_data['energy_uncertainty'] if pd.notna(member_data['energy_uncertainty']) else 0
            member_jpi = member_data.get('spin_parity', '')
            if member_jpi == "unknown": member_jpi = "N/A"
            
            if member_id == anchor_id:
                member_line = f"    [{dataset_code}] {member_id}: E={member_energy:.1f}±{member_uncertainty:.1f} keV, Jπ={member_jpi} (Anchor)"
                print(member_line)
                clustering_output_lines.append(member_line + "\n")
            else:
                input_vector = extract_features(member_data, anchor_data)
                probability = level_matcher_model.predict([input_vector])[0]
                member_line = f"    [{dataset_code}] {member_id}: E={member_energy:.1f}±{member_uncertainty:.1f} keV, Jπ={member_jpi} (Match Prob: {probability:.1%})"
                print(member_line)
                clustering_output_lines.append(member_line + "\n")
    
    # FRIBND: Write clustering results to file
    clustering_threshold_percent = int(clustering_merge_threshold * 100)
    with open('clustering_results.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== FINAL CLUSTERING RESULTS (Merge Threshold: >{clustering_threshold_percent}%) ===\n")
        output_file.write(f"Total Clusters: {len(unique_clusters)}\n")
        output_file.writelines(clustering_output_lines)
    
    print(f"\n[INFO] Clustering Complete: {len(unique_clusters)} clusters written to 'clustering_results.txt'")
