import matplotlib.pyplot as plt
import os
import json

"""
# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Input: Metrics Dictionary]
   |
   v
[Plot Generation] --> [Subplot 1: RMSE]
   |             --> [Subplot 2: MAE]
   v
[Save Figure] --> [outputs/figures/Training_Validation_Metrics.png]

Numbered Technical Steps:
1. **Input Parsing**: Receive metric dictionaries for XGBoost and LightGBM.
2. **Setup Canvas**: Initialize Matplotlib figure with two subplots (RMSE, MAE).
3. **Bar Plotting**: Compare Train vs Validation errors for both models side-by-side.
4. **Annotation**: Label each bar with its exact value for precision.
5. **Output**: Save high-resolution PNG to the standardized figure directory.
"""

def plot_training_metrics(xgboost_metrics, lightgbm_metrics, output_path="outputs/figures/Training_Validation_Metrics.png"):
    """
    Visualizes training vs validation metrics for XGBoost and LightGBM models.
    
    Args:
        xgboost_metrics (dict): {'Train RMSE': float, 'Validation RMSE': float, ...}
        lightgbm_metrics (dict): {'Train RMSE': float, 'Validation RMSE': float, ...}
        output_path (str): Path to save the generated figure.
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define Metric Groups
    models = ['XGBoost', 'LightGBM']
    metrics = ['RMSE', 'MAE']
    
    # Prepare Data Structure for Plotting
    # Structure: {Metric: {Model: (Train, Validation)}}
    data = {
        'RMSE': {
            'XGBoost': (xgboost_metrics['train_rmse'], xgboost_metrics['validation_rmse']),
            'LightGBM': (lightgbm_metrics['train_rmse'], lightgbm_metrics['validation_rmse'])
        },
        'MAE': {
            'XGBoost': (xgboost_metrics['train_mae'], xgboost_metrics['validation_mae']),
            'LightGBM': (lightgbm_metrics['train_mae'], lightgbm_metrics['validation_mae'])
        }
    }
    
    # Plot configuration
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.35
    opacity = 0.8
    
    # Color Scheme (Blue for Train, Orange for Validation)
    color_train = '#1f77b4'
    color_validation = '#ff7f0e'
    
    # Iterate through metrics (RMSE, MAE) and creating subplots
    for index, metric in enumerate(metrics):
        axis = axes[index]
        x_indices = np.arange(len(models))
        
        train_values = [data[metric][model][0] for model in models]
        validation_values = [data[metric][model][1] for model in models]
        
        # Create grouped bars
        bars_train = axis.bar(x_indices - bar_width/2, train_values, bar_width,
                             alpha=opacity, color=color_train, label='Train')
        bars_validation = axis.bar(x_indices + bar_width/2, validation_values, bar_width,
                                  alpha=opacity, color=color_validation, label='Validation')
        
        # Styling
        axis.set_xlabel('Model', fontsize=12)
        axis.set_ylabel(f'{metric} Value', fontsize=12)
        axis.set_title(f'{metric} Comparison: Train vs Validation', fontsize=14)
        axis.set_xticks(x_indices)
        axis.set_xticklabels(models, fontsize=12)
        axis.legend()
        axis.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                axis.annotate(f'{height:.4f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)
                             
        add_labels(bars_train)
        add_labels(bars_validation)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Diagnostics visualization saved to {output_path}")

# Note: Importing numpy here to avoid dependency issues if file runs standalone
import numpy as np
