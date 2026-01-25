import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from Feature_Engineer import extract_features, generate_synthetic_training_data, parse_json_datasets
from Level_Clusterer import perform_clustering_and_output
import warnings

"""
# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Start]
   |
   v
[Step 1: Synthetic Data Generation] --> [Physics Rules]
   |                                  |
   v                                  v
[Step 2: Model Training] <---- [Synthetic Labels]
   |
   | (Models Ready: XGBoost & LightGBM)
   v
[Step 3: Test Data Ingestion] --> [Data Standardization (A, B, C)]
   |
   v
[Step 4: Pairwise Inference] --> [Feature Extraction] --> [Model Prediction]
   |
   v
[Step 5: Clustering] --> [Level_Clusterer Module] --> [Final Reconciled Scheme]
   |
   v
[End]

Workflow Steps (5-Step Pipeline):

Step 1: Synthetic Training Data Generation
  Input: Physics rules and configuration from Feature_Engineer.py
  Output: 580+ synthetic balanced labeled examples covering physics scenarios
  Function: generate_synthetic_training_data()

Step 2: Model Training
  Input: Training features and labels from Step 1
  Process: Train two independent models with monotonic constraints (Physics Prior)
    - XGBoost: Baseline model
    - LightGBM: Validation model with heavier regularization
  Output: Trained binary logistic regressors

Step 3: Test Data Ingestion
  Source: Test datasets (A, B, C) in data/raw/
  Input: Local JSON files (test_dataset_A.json, etc.) containing levels and gamma transitions
  Process: Standardizes energy, uncertainty, Spin-Parity strings, and gamma decay records into a unified DataFrame
  Constraint: Strictly for inference; these levels are never seen during the training phase.
  Function: parse_json_datasets()

Step 4: Pairwise Inference
  Input: Standardized test data (Step 3) and trained models (Step 2)
  Process: 
    - Generate all cross-dataset pairs
    - Extract feature vectors (Feature_Engineer.extract_features)
    - Apply both Machine Learning models to calculate matching probabilities
  Output: Ranked list of candidate matches

Step 5: Constrained Clustering (Graph Partitioning)
  Input: Candidate match probabilities from Step 4
  Process: Enforce clique consistency and dataset uniqueness constraints
  Function: Level_Clusterer.perform_clustering_and_output()
  Output: Final unified level scheme reconciling datasets A, B, and C
"""

# Configuration Parameters
pairwise_output_threshold = 0.001  # Minimum probability for outputting level pairs (0.1%)
clustering_merge_threshold = 0.15  # Minimum probability for cluster merging (15%)

# Feature names for explicit labeling (prevents sklearn warnings)
feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                 'Specificity', 'Gamma_Decay_Pattern_Similarity']

if __name__ == "__main__":
    # ==========================================
    # Main Execution Pipeline:
    # 1. Synthetic Training Data Generation
    # 2. Model Training
    # 3. Test Data Ingestion
    # 4. Pairwise Inference
    # 5. Graph-Based Clustering
    # ==========================================

    # ==========================================
    # Step 1: Synthetic Training Data Generation
    # ==========================================
    # Get training data from physics parser
    training_features, training_labels = generate_synthetic_training_data()
    
    # Convert to DataFrame with explicit feature names to prevent sklearn warnings
    training_dataframe = pd.DataFrame(training_features, columns=feature_names)

    # Train XGBoost regressor with monotonic constraints enforcing physics rules
    # All five features designed as higher value → better match probability
    # ==========================================
    # Step 2: Model Training
    # ==========================================
    
    # 2a: Train XGBoost (Primary Model)
    print("Training XGBoost Model...")
    model_xgboost = XGBRegressor(objective='binary:logistic',
                                       # Learning Objective Function: Training Loss + Regularization.
                                       # The loss function computes the difference between the true y value and the predicted y value.
                                       # The regularization term discourages overly complex trees.
                                       monotone_constraints='(1, 1, 1, 1, 1)',
                                       # Enforce increasing constraint on all five features.
                                       # Physics prior: higher feature values always indicate better matches.
                                       # Monotonic constraints improve predictive performance when strong prior beliefs exist.
                                       n_estimators=1000,
                                       # Number of gradient boosted trees or boosting rounds. Typical ranges for n_estimators are between 50 and 1000, with higher values generally leading to better performance but longer training times.
                                       max_depth=10,
                                       # Maximum tree depth. Lower values prevent overfitting but may underfit, while larger values allow the model to capture more complex relationships but may lead to overfitting.
                                       # Value of 3 balances model complexity with generalization.
                                       learning_rate=0.05,
                                       # Step size; determines the contribution of each tree to the final outcome by scaling the weights of the features.
                                       # Impact on model performance:
                                       # Lower values slow down learning but can improve generalization
                                       # Higher values speed up learning but may lead to suboptimal results
                                       # Interaction with the number of trees:
                                       # Lower learning rates typically require more trees
                                       # Higher learning rates may converge faster but require careful tuning of other parameters
                                       random_state=42)
                                       # Random number seed for reproducibility.
    
    # Train model on synthetic training data
    model_xgboost.fit(training_dataframe, training_labels)

    # 2b: Train LightGBM (Secondary Model)
    print("Training LightGBM Model...")
    # Note: 'objective="binary"' in LightGBM is equivalent to 'binary:logistic' in XGBoost (outputs probability 0-1).
    # Heavy regularization prevents extreme confidence (100%/0%) and overfitting on small datasets.
    # Configuration Logic:
    # 1. reg_alpha=1.0 (L1 regularization), reg_lambda=10.0 (L2 regularization): Penalize extreme weights to ensure soft probability outputs (e.g. 99.5% instead of 100%).
    # 2. num_leaves=7, max_depth=5: Constrain model complexity to prevent memorization of synthetic samples.
    # 3. min_child_samples=20: Force generalization by requiring more data per leaf.
    # Observation: This configuration causes LightGBM to weight perfect Gamma Patterns higher than XGBoost (e.g., A_2000<->B_2006 case).
    model_lightgbm = LGBMRegressor(objective='binary',
                                             monotone_constraints="1,1,1,1,1",  # Increasing constraints
                                             n_estimators=500,
                                             max_depth=5,
                                             num_leaves=7,
                                             min_child_samples=20,
                                             learning_rate=0.02,
                                             reg_alpha=1.0,       # L1 regularization to encourage sparse solutions
                                             reg_lambda=10.0,     # L2 regularization to prevent extreme leaf weights (100% confidence)
                                             verbose=-1,
                                             random_state=42)
    
    # Train LightGBM model
    model_lightgbm.fit(training_dataframe, training_labels)

    # ==========================================
    # Step 3: Test Data Ingestion
    # ==========================================
    # Load level and gamma test data from JSON files for matching.
    # Logic: These datasets (A, B, C) are used solely for inference and were not seen during training.
    print("Ingesting Test Data...")
    levels = parse_json_datasets(['A', 'B', 'C'])
    dataframe = pd.DataFrame(levels)
    dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

    # ==========================================
    # Step 4: Pairwise Inference
    # ==========================================
    # Predict match probabilities for all cross-dataset level pairs using trained models
    matching_level_pairs_xgboost = []
    matching_level_pairs_lightgbm = []
    all_pairwise_results_for_display = []

    level_dataframe_rows_list = list(dataframe.iterrows())
    
    print("\nRunning Pairwise Inference (XGBoost & LightGBM)...")
    for i in range(len(level_dataframe_rows_list)):
        for j in range(i + 1, len(level_dataframe_rows_list)):
            _, level_1 = level_dataframe_rows_list[i]
            _, level_2 = level_dataframe_rows_list[j]
            
            # Skip same-dataset pairs (only match across different datasets)
            if level_1['dataset_code'] == level_2['dataset_code']:
                continue

            # Extract the input features for each level pair
            feature_vector = extract_features(level_1, level_2)
            
            # Prediction: Use the trained models to predict match probability for this input feature vector
            # Convert to DataFrame with explicit feature names
            feature_dataframe = pd.DataFrame([feature_vector], columns=feature_names)
            xgboost_probability = model_xgboost.predict(feature_dataframe)[0]
            lightgbm_probability = model_lightgbm.predict(feature_dataframe)[0]
            
            # Separate logic: Each model is independent.
            
            # Record level pairs above output threshold (Dual Report)
            if xgboost_probability > pairwise_output_threshold or lightgbm_probability > pairwise_output_threshold:
                 all_pairwise_results_for_display.append({
                    'level_1_id': level_1['level_id'], 'level_2_id': level_2['level_id'],
                    'dataset_code_1': level_1['dataset_code'], 'dataset_code_2': level_2['dataset_code'],
                    'xgboost_probability': xgboost_probability, 'lightgbm_probability': lightgbm_probability,
                    'features': feature_vector
                })

            # Prepare separate lists for clustering
            if xgboost_probability > pairwise_output_threshold:
                matching_level_pairs_xgboost.append({
                    'level_1_id': level_1['level_id'],
                    'level_2_id': level_2['level_id'],
                    'dataset_code_1': level_1['dataset_code'],
                    'dataset_code_2': level_2['dataset_code'],
                    'probability': xgboost_probability,
                    'features': feature_vector
                })
            
            if lightgbm_probability > pairwise_output_threshold:
                matching_level_pairs_lightgbm.append({
                    'level_1_id': level_1['level_id'],
                    'level_2_id': level_2['level_id'],
                    'dataset_code_1': level_1['dataset_code'],
                    'dataset_code_2': level_2['dataset_code'],
                    'probability': lightgbm_probability,
                    'features': feature_vector
                })

    # Sort by probability descending (using max of both for the combined list)
    all_pairwise_results_for_display.sort(key=lambda x: max(x['xgboost_probability'], x['lightgbm_probability']), reverse=True)
    matching_level_pairs_xgboost.sort(key=lambda x: x['probability'], reverse=True)
    matching_level_pairs_lightgbm.sort(key=lambda x: x['probability'], reverse=True)

    # Write pairwise inference results to file (Side-by-Side Comparison)
    threshold_percent = pairwise_output_threshold * 100
    with open('outputs/pairwise/Output_Level_Pairwise_Inference.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== PAIRWISE INFERENCE RESULTS (>{threshold_percent:.1f}%) ===\n")
        output_file.write(f"Model Comparison: XGBoost vs LightGBM\n")
        output_file.write(f"Total Level Pairs Found: {len(all_pairwise_results_for_display)}\n\n")
        
        for item in all_pairwise_results_for_display:
            energy_similarity, spin_similarity, parity_similarity, specificity, gamma_decay_pattern_similarity = item['features']
            output_file.write(
                f"{item['level_1_id']} <-> {item['level_2_id']} | "
                f"XGBoost: {item['xgboost_probability']:.1%} | LightGBM: {item['lightgbm_probability']:.1%}\n"
                f"  Features: Energy_Similarity={energy_similarity:.2f}, Spin_Similarity={spin_similarity:.2f}, "
                f"Parity_Similarity={parity_similarity:.2f}, Specificity={specificity:.2f}, Gamma_Pattern_Similarity={gamma_decay_pattern_similarity:.2f}\n\n"
            )
    
    print(f"\n[INFO] Pairwise Inference Complete: {len(all_pairwise_results_for_display)} level pairs (>{threshold_percent:.1f}%) written to 'outputs/pairwise/Output_Level_Pairwise_Inference.txt'")

    # ==========================================
    # Step 5: Graph-Based Clustering
    # ==========================================
    
    # Execute clustering independently for each model (XGBoost primary, LightGBM secondary for comparison only)
    perform_clustering_and_output(matching_level_pairs_xgboost, model_xgboost, "outputs/clustering/Output_Clustering_Results_XGBoost.txt", "XGBoost", dataframe)
    perform_clustering_and_output(matching_level_pairs_lightgbm, model_lightgbm, "outputs/clustering/Output_Clustering_Results_LightGBM.txt", "LightGBM", dataframe)
