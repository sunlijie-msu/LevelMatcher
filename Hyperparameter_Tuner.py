"""
Hyperparameter Tuning for Nuclear Level Matcher
================================================

Strategy:
1. Split synthetic training data for validation (holdout method)
2. Train multiple model configurations
3. Validate with MSE on held-out synthetic data
4. Run full pipeline (pairwise + clustering) for each config on real datasets
5. Compare clustering results against expert knowledge
6. Select best configuration based on validation metrics + expert review

Note: Real datasets (A, B, C) are NEVER used for training, only inference validation
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Feature_Engineer import extract_features, generate_synthetic_training_data, load_levels_from_json

# ==========================================
# Configuration: Parameter Grid
# ==========================================
# Define candidate configurations to test
# Baseline is current: n_estimators=1000, max_depth=10, learning_rate=0.05
# Hyperparameter meanings (concise):
# - n_estimators: number of boosting trees; higher can fit harder patterns but increases time
# - max_depth: maximum tree depth; higher depth captures complex interactions but can overfit
# - learning_rate: shrinkage per tree; lower rate needs more trees but can improve stability
# - reg_lambda: L2 regularization weight on leaf scores; higher values discourage overfitting

parameter_configurations = [
    {
        'name': 'Baseline (Current)',
        'n_estimators': 1000,
        'max_depth': 10,
        'learning_rate': 0.05,
        'reg_lambda': 1.0
    },
    {
        'name': 'Conservative (Shallow)',
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'reg_lambda': 1.0
    },
    {
        'name': 'Aggressive (Deep)',
        'n_estimators': 1000,
        'max_depth': 15,
        'learning_rate': 0.05,
        'reg_lambda': 1.0
    },
    {
        'name': 'Slow Learner (High Regularization)',
        'n_estimators': 2000,
        'max_depth': 10,
        'learning_rate': 0.01,
        'reg_lambda': 5.0
    },
    {
        'name': 'Fast Learner (Low Regularization)',
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.1,
        'reg_lambda': 0.5
    }
]

# Thresholds for inference and clustering (same as Level_Matcher.py)
pairwise_output_threshold = 0.01
clustering_merge_threshold = 0.15


def train_and_validate_model(config):
    """
    Train model with given configuration and compute validation metrics.
    Returns trained model and MSE validation score.
    """
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config['name']}")
    print(f"{'='*60}")
    print(f"Parameters: n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, "
          f"learning_rate={config['learning_rate']}, reg_lambda={config['reg_lambda']}")
    
    # Generate synthetic training data
    training_features, training_labels = generate_synthetic_training_data()
    
    # Split into train and validation sets (80/20 split)
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        training_features, training_labels, test_size=0.2, random_state=42
    )
    
    # Train model with current configuration
    model = XGBRegressor(
        objective='binary:logistic',
        monotone_constraints='(1, 1, 1, 1)',  # CRITICAL: Never remove physics constraint
        random_state=42,
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['learning_rate'],
        reg_lambda=config['reg_lambda']
    )
    
    model.fit(train_features, train_labels)
    
    # Compute validation metrics
    validation_predictions = model.predict(validation_features)
    
    # Note: training_labels are probabilities (0.0-1.0), not binary classes
    # MSE is appropriate for regression, skip log_loss for continuous targets
    mse = mean_squared_error(validation_labels, validation_predictions)
    
    print(f"  Validation MSE: {mse:.4f}")
    
    return model, mse


def run_inference_and_clustering(model, config_name):
    """
    Run full pairwise inference and clustering pipeline with trained model.
    Returns clustering results and matching pairs for expert validation.
    """
    print(f"\n  Running inference and clustering for: {config_name}")
    
    # Load real test datasets
    levels = load_levels_from_json(['A', 'B', 'C'])
    dataframe = pd.DataFrame(levels)
    dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)
    
    # Pairwise inference
    matching_level_pairs = []
    rows_list = list(dataframe.iterrows())
    
    for i in range(len(rows_list)):
        for j in range(i + 1, len(rows_list)):
            _, level_1 = rows_list[i]
            _, level_2 = rows_list[j]
            
            if level_1['dataset_code'] == level_2['dataset_code']:
                continue
            
            feature_vector = extract_features(level_1, level_2)
            match_probability = model.predict([feature_vector])[0]
            
            if match_probability > pairwise_output_threshold:
                matching_level_pairs.append({
                    'ID1': level_1['level_id'],
                    'ID2': level_2['level_id'],
                    'dataset_1': level_1['dataset_code'],
                    'dataset_2': level_2['dataset_code'],
                    'probability': match_probability,
                    'features': feature_vector
                })
    
    matching_level_pairs.sort(key=lambda x: x['probability'], reverse=True)
    
    # Clustering (simplified version - just count clusters)
    level_lookup = {row['level_id']: row for _, row in dataframe.iterrows()}
    initial_clusters = [{row['dataset_code']: row['level_id']} for _, row in dataframe.iterrows()]
    
    id_to_clusters = {}
    for cluster in initial_clusters:
        member_id = list(cluster.values())[0]
        id_to_clusters[member_id] = [cluster]
    
    valid_pairs = set()
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] >= clustering_merge_threshold:
            valid_pairs.add((matching_level_pair['ID1'], matching_level_pair['ID2']))
            valid_pairs.add((matching_level_pair['ID2'], matching_level_pair['ID1']))
    
    # Greedy merging (same logic as Level_Matcher.py)
    for matching_level_pair in matching_level_pairs:
        if matching_level_pair['probability'] < clustering_merge_threshold:
            break
        
        id_1, id_2 = matching_level_pair['ID1'], matching_level_pair['ID2']
        cluster_list_1 = list(id_to_clusters[id_1])
        cluster_list_2 = list(id_to_clusters[id_2])
        pair_processed = False
        
        for cluster_1 in cluster_list_1:
            for cluster_2 in cluster_list_2:
                if cluster_1 is cluster_2:
                    pair_processed = True
                    continue
                
                datasets_1 = set(cluster_1.keys())
                datasets_2 = set(cluster_2.keys())
                
                if not datasets_1.isdisjoint(datasets_2):
                    if matching_level_pair['dataset_1'] not in datasets_2:
                        if all((id_1, member_id) in valid_pairs for member_id in cluster_2.values()):
                            if id_1 not in cluster_2.values():
                                cluster_2[matching_level_pair['dataset_1']] = id_1
                                if cluster_2 not in id_to_clusters[id_1]:
                                    id_to_clusters[id_1].append(cluster_2)
                                pair_processed = True
                    
                    if matching_level_pair['dataset_2'] not in datasets_1:
                        if all((id_2, member_id) in valid_pairs for member_id in cluster_1.values()):
                            if id_2 not in cluster_1.values():
                                cluster_1[matching_level_pair['dataset_2']] = id_2
                                if cluster_1 not in id_to_clusters[id_2]:
                                    id_to_clusters[id_2].append(cluster_1)
                                pair_processed = True
                    continue
                
                consistent = True
                for member_1 in cluster_1.values():
                    for member_2 in cluster_2.values():
                        if (member_1, member_2) not in valid_pairs:
                            consistent = False
                            break
                    if not consistent:
                        break
                
                if consistent:
                    cluster_1.update(cluster_2)
                    for member_id in cluster_2.values():
                        if cluster_2 in id_to_clusters[member_id]:
                            id_to_clusters[member_id].remove(cluster_2)
                        if cluster_1 not in id_to_clusters[member_id]:
                            id_to_clusters[member_id].append(cluster_1)
                    pair_processed = True
        
        if not pair_processed:
            for cluster in id_to_clusters[id_1]:
                if len(cluster) == 1 and matching_level_pair['dataset_2'] not in cluster:
                    cluster[matching_level_pair['dataset_2']] = id_2
                    if cluster not in id_to_clusters[id_2]:
                        id_to_clusters[id_2].append(cluster)
                    pair_processed = True
                    break
            
            if not pair_processed:
                for cluster in id_to_clusters[id_2]:
                    if len(cluster) == 1 and matching_level_pair['dataset_1'] not in cluster:
                        cluster[matching_level_pair['dataset_1']] = id_1
                        if cluster not in id_to_clusters[id_1]:
                            id_to_clusters[id_1].append(cluster)
                        pair_processed = True
                        break
            
            if not pair_processed:
                new_cluster = {
                    matching_level_pair['dataset_1']: id_1,
                    matching_level_pair['dataset_2']: id_2
                }
                id_to_clusters[id_1].append(new_cluster)
                id_to_clusters[id_2].append(new_cluster)
    
    # Extract unique clusters
    unique_clusters = []
    seen = set()
    for cluster_list in id_to_clusters.values():
        for cluster in cluster_list:
            cluster_id = id(cluster)
            if cluster_id not in seen:
                unique_clusters.append(cluster)
                seen.add(cluster_id)
    
    # Calculate average cluster size
    cluster_sizes = [len(cluster) for cluster in unique_clusters]
    average_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
    multi_member_clusters = [c for c in unique_clusters if len(c) >= 2]
    
    print(f"    Total Clusters: {len(unique_clusters)}")
    print(f"    Multi-member Clusters (size >= 2): {len(multi_member_clusters)}")
    print(f"    Average Cluster Size: {average_cluster_size:.2f}")
    print(f"    Total High-confidence Pairs (>= {clustering_merge_threshold}): {len(valid_pairs)//2}")
    
    return {
        'total_clusters': len(unique_clusters),
        'multi_member_clusters': len(multi_member_clusters),
        'average_cluster_size': average_cluster_size,
        'high_confidence_pairs': len(valid_pairs)//2,
        'clusters': unique_clusters,
        'level_lookup': level_lookup,
        'matching_pairs': matching_level_pairs
    }


def save_clustering_results(config_name, clustering_results, model):
    """
    Save detailed clustering results to file for expert review.
    Includes match probabilities relative to anchor (like Level_Matcher.py).
    """
    filename = f"Output_Hyperparameter_Test_{config_name.replace(' ', '_')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== CLUSTERING RESULTS: {config_name} ===\n")
        output_file.write(f"Total Clusters: {clustering_results['total_clusters']}\n")
        output_file.write(f"Multi-member Clusters: {clustering_results['multi_member_clusters']}\n\n")
        
        for i, cluster in enumerate(clustering_results['clusters']):
            if len(cluster) >= 2:  # Only output multi-member clusters
                # Select anchor as the level with smallest energy uncertainty (most precise measurement)
                anchor_id = min(cluster.values(), 
                               key=lambda x: clustering_results['level_lookup'][x].get('energy_uncertainty', 999))
                anchor_data = clustering_results['level_lookup'][anchor_id]
                anchor_energy = anchor_data['energy_value']
                anchor_uncertainty = anchor_data.get('energy_uncertainty', 0)
                anchor_jpi = anchor_data.get('spin_parity_string', 'N/A')
                if anchor_jpi == "unknown": anchor_jpi = "N/A"
                
                output_file.write(f"\nCluster {i+1}:\n")
                output_file.write(f"  Anchor: {anchor_id} | E={anchor_energy:.1f}±{anchor_uncertainty:.1f} keV | Jπ={anchor_jpi}\n")
                output_file.write(f"  Members:\n")
                
                for dataset_code in sorted(cluster.keys()):
                    member_id = cluster[dataset_code]
                    member_data = clustering_results['level_lookup'][member_id]
                    energy = member_data['energy_value']
                    uncertainty = member_data.get('energy_uncertainty', 0)
                    jpi = member_data.get('spin_parity_string', 'N/A')
                    if jpi == "unknown": jpi = "N/A"
                    
                    if member_id == anchor_id:
                        output_file.write(f"    [{dataset_code}] {member_id}: E={energy:.1f}±{uncertainty:.1f} keV, Jπ={jpi} (Anchor)\n")
                    else:
                        # Calculate match probability relative to anchor
                        input_vector = extract_features(member_data, anchor_data)
                        probability = model.predict([input_vector])[0]
                        output_file.write(f"    [{dataset_code}] {member_id}: E={energy:.1f}±{uncertainty:.1f} keV, Jπ={jpi} (Match Prob: {probability:.1%})\n")
    
    print(f"    Saved detailed results to: {filename}")


def save_pairwise_results(config_name, matching_level_pairs):
    """
    Save pairwise inference results with match probabilities to file for expert review.
    """
    filename = f"Output_Hyperparameter_Pairwise_{config_name.replace(' ', '_')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== PAIRWISE INFERENCE RESULTS: {config_name} ===\n")
        output_file.write(f"Total Level Pairs: {len(matching_level_pairs)}\n\n")
        
        for matching_level_pair in matching_level_pairs:
            energy_sim, spin_sim, parity_sim, specificity = matching_level_pair['features']
            output_file.write(
                f"{matching_level_pair['ID1']} <-> {matching_level_pair['ID2']} | "
                f"Probability: {matching_level_pair['probability']:.1%}\n"
                f"  Features: Energy_Sim={energy_sim:.2f}, Spin_Sim={spin_sim:.2f}, "
                f"Parity_Sim={parity_sim:.2f}, Specificity={specificity:.2f}\n\n"
            )
    
    print(f"    Saved pairwise results to: {filename}")


# ==========================================
# Main Tuning Loop
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING FOR NUCLEAR LEVEL MATCHER")
    print("="*60)
    print("\nStrategy:")
    print("  1. Validate on synthetic data split (MSE regression metric)")
    print("  2. Run full inference + clustering on real datasets A, B, C")
    print("  3. Compare clustering results with your expert knowledge")
    print("  4. Select best configuration based on metrics + expert validation")
    
    results_summary = []
    
    for config in parameter_configurations:
        # Train and validate on synthetic data
        model, mse = train_and_validate_model(config)
        
        # Run inference and clustering on real datasets
        clustering_results = run_inference_and_clustering(model, config['name'])
        
        # Save detailed clustering for expert review (with probabilities)
        save_clustering_results(config['name'], clustering_results, model)
        
        # Save pairwise inference results with match probabilities
        save_pairwise_results(config['name'], clustering_results['matching_pairs'])
        
        # Store summary results
        results_summary.append({
            'name': config['name'],
            'mse': mse,
            'total_clusters': clustering_results['total_clusters'],
            'multi_member_clusters': clustering_results['multi_member_clusters'],
            'average_cluster_size': clustering_results['average_cluster_size'],
            'high_confidence_pairs': clustering_results['high_confidence_pairs']
        })
    
    # Print comparison table
    print("\n" + "="*60)
    print("SUMMARY: HYPERPARAMETER COMPARISON")
    print("="*60)
    print(f"\n{'Configuration':<40} {'MSE':<10} {'Clusters':<10} {'Multi':<10} {'AvgSize':<10}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['name']:<40} {result['mse']:<10.4f} "
              f"{result['total_clusters']:<10} {result['multi_member_clusters']:<10} {result['average_cluster_size']:<10.2f}")
    
    # Identify best by validation metric
    best_by_mse = min(results_summary, key=lambda x: x['mse'])
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"\nBest by Validation MSE: {best_by_mse['name']} (MSE: {best_by_mse['mse']:.4f})")
    print("\nNext Steps:")
    print("  1. Review detailed clustering files: Output_Hyperparameter_Test_*.txt")
    print("  2. Compare with your expert knowledge of correct clusters")
    print("  3. Select configuration that best matches your domain expertise")
    print("  4. Update Level_Matcher.py parameters with chosen configuration")
