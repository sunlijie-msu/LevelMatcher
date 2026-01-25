import pandas as pd
from Feature_Engineer import extract_features

"""
# High-level Structure and Workflow Explanation:
======================================

Module Purpose:
This module implements the Graph-Based Clustering algorithm used to reconcile matched level pairs into coherent physical levels. It operates independently of the training/inference pipeline, taking probability-weighted edges and grouping them into clusters.

Workflow Diagram:
[Input: Matched Pairs (List)] --> [Filter: Probability > 15%] --> [Sort: Descending Probability]
                                         |
                                         v
                                  [Greedy Merging Loop]
                                         |
                                         v
                  +----------------------------------------------+
                  |  For each candidate pair (Node A, Node B):   |
                  |  1. Check existing clusters for A & B        |
                  |  2. If disjoint -> Check Compatibility       |
                  |     (Must actally match ALL members)         |
                  |  3. Merge Clusters or Add to Singleton       |
                  |     (Enforce Dataset Uniqueness Constraint)  |
                  +----------------------------------------------+
                                         |
                                         v
           [Post-Processing: Sort Clusters by Energy] --> [Output: Clustering Report]

Technical Steps:
1. **initialization**: Create a singleton cluster for every level in the dataset.
2. **Filtering**: Select only level pairs with match probability > clustering_merge_threshold (15%).
3. **Greedy Merging**: Iterate through pairs from highest to lowest probability.
   - Algorithm: Agglomerative clustering with domain-specific constraints.
   - Constraint 1 (Dataset Uniqueness): A cluster cannot contain two levels from the same dataset (e.g., two 'A' levels).
   - Constraint 2 (Mutual Consistency): To merge two clusters, every member of Cluster 1 must be compatible with every member of Cluster 2.
4. **Result Formatting**: Identify "Anchor" levels (lowest uncertainty) and format output for human review.
"""

def perform_clustering_and_output(matching_level_pairs, model_instance, output_filename, model_name, level_dataframe):
    """
    Executes greedy graph clustering and writes results to file.
    Refactored to allow independent execution for different models (XGB vs LGBM).
    Preserves all original logic and educational comments.
    """
    
    # Feature names for explicit labeling (prevents sklearn warnings)
    feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                     'Specificity', 'Gamma_Decay_Pattern_Similarity']
    
    # Configuration Parameters
    clustering_merge_threshold = 0.15  # Minimum probability for cluster merging (15%)

    # ==========================================
    # Logic: Greedy Cluster Merging
    # ==========================================
    # Algorithm: Greedy cluster merging based on match probability ranking
    # Key Constraints:
    #   1. Dataset Uniqueness: Each cluster contains at most one level per dataset (no duplicate sources)
    #   2. Mutual Consistency: All cluster members must be pairwise compatible (all pairs > clustering_merge_threshold)
    #   3. Ambiguity Support: Poorly resolved levels can belong to multiple clusters when compatible with multiple anchors
    
    print(f"\n--- Starting Clustering for {model_name} ---")

    # Initialize singleton clusters and lookup table
    # Note: Using level_dataframe passed as argument
    level_lookup = {row['level_id']: row for _, row in level_dataframe.iterrows()}
    initial_clusters = [{row['dataset_code']: row['level_id']} for _, row in level_dataframe.iterrows()]
    
    # Track which clusters each level belongs to (one level can be in multiple clusters)
    id_to_clusters = {}
    for cluster in initial_clusters:
        # Extract the single level_id from this single-member cluster
        # cluster.values() returns dict_values(['A_1000']), list() converts to ['A_1000'], [0] gets 'A_1000'
        member_id = list(cluster.values())[0]
        id_to_clusters[member_id] = [cluster]

    # Extract strong candidate pairs for merging
    # Only level pairs exceeding clustering_merge_threshold qualify for cluster operations
    valid_pairs = set()
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] >= clustering_merge_threshold:
            valid_pairs.add((matching_level_pair['ID1'], matching_level_pair['ID2']))
            valid_pairs.add((matching_level_pair['ID2'], matching_level_pair['ID1']))

    # Greedy cluster merging
    # Process candidates in descending probability order, attempting merges or multi-cluster assignment for ambiguous levels
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] < clustering_merge_threshold:
            break  # Skip weak matches (list is sorted by probability)
        
        id_1, id_2 = matching_level_pair['ID1'], matching_level_pair['ID2']
        
        # Snapshot current cluster memberships before modification
        cluster_list_1 = list(id_to_clusters[id_1])
        cluster_list_2 = list(id_to_clusters[id_2])

        # Track if this pair successfully joined any existing cluster
        pair_processed = False

        # Try to merge each cluster pair that contains id_1 and id_2
        for cluster_1 in cluster_list_1:
            for cluster_2 in cluster_list_2:
                if cluster_1 is cluster_2:
                    pair_processed = True  # Already in same cluster
                    continue
                
                datasets_1 = set(cluster_1.keys())
                datasets_2 = set(cluster_2.keys())
                
                # Scenario A: Dataset overlap (both clusters have levels from same dataset)
                if not datasets_1.isdisjoint(datasets_2):
                    # Example: cluster_1 has A_3000, cluster_2 has A_3005 (both resolved), C_3002 (poorly resolved with large uncertainty) matches both
                    # Cannot merge these clusters (violates dataset uniqueness), but try multi-cluster assignment for ambiguous levels
                    
                    # Can we add id_1 to cluster_2?
                    if matching_level_pair['dataset_1'] not in datasets_2:
                        # Check if id_1 is compatible with ALL existing members of cluster_2
                        if all((id_1, member_id) in valid_pairs for member_id in cluster_2.values()):
                            if id_1 not in cluster_2.values():
                                cluster_2[matching_level_pair['dataset_1']] = id_1
                                if cluster_2 not in id_to_clusters[id_1]:
                                    id_to_clusters[id_1].append(cluster_2)
                                pair_processed = True
                    
                    # Can we add id_2 to cluster_1?
                    if matching_level_pair['dataset_2'] not in datasets_1:
                        # Check if id_2 is compatible with ALL existing members of cluster_1
                        if all((id_2, member_id) in valid_pairs for member_id in cluster_1.values()):
                            if id_2 not in cluster_1.values():
                                cluster_1[matching_level_pair['dataset_2']] = id_2
                                if cluster_1 not in id_to_clusters[id_2]:
                                    id_to_clusters[id_2].append(cluster_1)
                                pair_processed = True
                    
                    continue  # Move to next cluster pair

                # Scenario B: No dataset overlap (can potentially merge)
                # Verify all-to-all compatibility before merging
                consistent = True
                for member_1 in cluster_1.values():
                    for member_2 in cluster_2.values():
                        if (member_1, member_2) not in valid_pairs:
                            consistent = False
                            break
                    if not consistent:
                        break
                
                if consistent:
                    # Perform merge: absorb cluster_2 into cluster_1
                    cluster_1.update(cluster_2)
                    
                    # Update all members of cluster_2 to point to cluster_1
                    for member_id in cluster_2.values():
                        if cluster_2 in id_to_clusters[member_id]:
                            id_to_clusters[member_id].remove(cluster_2)
                        if cluster_1 not in id_to_clusters[member_id]:
                            id_to_clusters[member_id].append(cluster_1)
                    
                    pair_processed = True
        
        # If this valid pair was not absorbed into any existing cluster, try adding to singleton clusters first
        if not pair_processed:
            # Check if id_1 is in a singleton cluster that we can expand
            for cluster in id_to_clusters[id_1]:
                if len(cluster) == 1 and matching_level_pair['dataset_2'] not in cluster:
                    # Expand singleton cluster by adding id_2
                    cluster[matching_level_pair['dataset_2']] = id_2
                    if cluster not in id_to_clusters[id_2]:
                        id_to_clusters[id_2].append(cluster)
                    pair_processed = True
                    break
            
            # If still not processed, check if id_2 is in a singleton cluster
            if not pair_processed:
                for cluster in id_to_clusters[id_2]:
                    if len(cluster) == 1 and matching_level_pair['dataset_1'] not in cluster:
                        # Expand singleton cluster by adding id_1
                        cluster[matching_level_pair['dataset_1']] = id_1
                        if cluster not in id_to_clusters[id_1]:
                            id_to_clusters[id_1].append(cluster)
                        pair_processed = True
                        break
            
            # Only create new cluster if neither member has a singleton to expand
            if not pair_processed:
                new_cluster = {
                    matching_level_pair['dataset_1']: id_1,
                    matching_level_pair['dataset_2']: id_2
                }
                id_to_clusters[id_1].append(new_cluster)
                id_to_clusters[id_2].append(new_cluster)

    # Extract unique active clusters (remove duplicates)
    unique_clusters = []
    seen = set()
    for cluster_list in id_to_clusters.values():
        for cluster in cluster_list:
            cluster_id = id(cluster)
            if cluster_id not in seen:
                unique_clusters.append(cluster)
                seen.add(cluster_id)
    
    # Sort clusters by average energy for consistent output
    def calculate_cluster_average_energy(cluster):
        energies = [level_lookup[member_id]['energy_value'] for member_id in cluster.values()]
        return sum(energies) / len(energies)
    
    unique_clusters.sort(key=calculate_cluster_average_energy)

    # Write clustering results to both console and file
    clustering_output_lines = []
    clustering_output_lines.append(f"=== FINAL CLUSTERING RESULTS ({model_name}) ===\n")
    
    print(f"\n=== FINAL CLUSTERING RESULTS ({model_name}) ===")
    for i, cluster in enumerate(unique_clusters):
        # Select anchor as the level with smallest energy uncertainty (most precise measurement)
        anchor_id = min(cluster.values(), key=lambda x: level_lookup[x]['energy_uncertainty'] if pd.notna(level_lookup[x]['energy_uncertainty']) else 999)
        anchor_data = level_lookup[anchor_id]
        anchor_energy = anchor_data['energy_value']
        anchor_uncertainty = anchor_data['energy_uncertainty'] if pd.notna(anchor_data['energy_uncertainty']) else 0
        anchor_jpi = anchor_data.get('spin_parity_string', '')
        if anchor_jpi == "unknown": anchor_jpi = "N/A"
        
        cluster_header = f"\nCluster {i+1}:"
        anchor_line = f"  Anchor: {anchor_id} | E={anchor_energy:.1f}±{anchor_uncertainty:.1f} keV | Jπ={anchor_jpi}"
        members_header = "  Members:"
        
        # Only print details to console for the primary model (XGBoost) to avoid cluttered output
        if model_name == "XGBoost":
            print(cluster_header)
            print(anchor_line)
            print(members_header)
        
        clustering_output_lines.append(cluster_header + "\n")
        clustering_output_lines.append(anchor_line + "\n")
        clustering_output_lines.append(members_header + "\n")
        
        # Display each member with match probability relative to anchor
        # Anchor member shows no probability (identity match). Other members show ML-predicted probability.
        for dataset_code in sorted(cluster.keys()):
            member_id = cluster[dataset_code]
            member_data = level_lookup[member_id]
            member_energy = member_data['energy_value']
            member_uncertainty = member_data['energy_uncertainty'] if pd.notna(member_data['energy_uncertainty']) else 0
            member_jpi = member_data.get('spin_parity_string', '')
            if member_jpi == "unknown": member_jpi = "N/A"
            
            if member_id == anchor_id:
                member_line = f"    [{dataset_code}] {member_id}: E={member_energy:.1f}±{member_uncertainty:.1f} keV, Jπ={member_jpi} (Anchor)"
                if model_name == "XGBoost": print(member_line)
                clustering_output_lines.append(member_line + "\n")
            else:
                input_vector = extract_features(member_data, anchor_data)
                # Use the specific model instance passed to the function
                # Convert to DataFrame with explicit feature names
                input_dataframe = pd.DataFrame([input_vector], columns=feature_names)
                probability = model_instance.predict(input_dataframe)[0]
                member_line = f"    [{dataset_code}] {member_id}: E={member_energy:.1f}±{member_uncertainty:.1f} keV, Jπ={member_jpi} (Match Prob: {probability:.1%})"
                if model_name == "XGBoost": print(member_line)
                clustering_output_lines.append(member_line + "\n")
    
    # Write clustering results to file
    clustering_threshold_percent = int(clustering_merge_threshold * 100)
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== FINAL CLUSTERING RESULTS: {model_name} (Merge Threshold: >{clustering_threshold_percent}%) ===\n")
        output_file.write(f"Total Clusters: {len(unique_clusters)}\n")
        output_file.writelines(clustering_output_lines)
    
    print(f"\n[INFO] Clustering Complete ({model_name}): {len(unique_clusters)} clusters written to '{output_filename}'")
