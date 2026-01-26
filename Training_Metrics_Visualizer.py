"""
Training Metrics Visualizer for Machine Learning Model Diagnostics
======================================

# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Input: Training Metrics]
   |
   v
[Step 1: Data Collection] --> [RMSE, MAE, LogLoss, AUC, Feature Importance]
   |
   v
[Step 2: Figure Layout] --> [5-Panel Comparison Grid]
   |                              |
   |                              v
   |                [Panel 1: Train vs Validation RMSE]
   |                [Panel 2: Train vs Validation MAE]
   |                [Panel 3: Train vs Validation LogLoss]
   |                [Panel 4: Feature Importance (Gain)]
   |                [Panel 5: Overfitting Gap Analysis]
   v
[Step 3: Rendering] --> [High-Quality Matplotlib Output]
   |
   v
[End: Output Figure File]

Numbered Technical Steps:
1. **Data Ingestion**: Collects training and validation metrics (RMSE, MAE, LogLoss) and feature importance for both models.
2. **Layout Design**: Creates 5-panel grid layout (2 rows × 3 columns) for comprehensive visualization.
3. **Panel 1 - RMSE Comparison**: Bar chart showing training vs validation RMSE for XGBoost and LightGBM.
4. **Panel 2 - MAE Comparison**: Bar chart showing training vs validation MAE for XGBoost and LightGBM.
5. **Panel 3 - LogLoss Comparison**: Binary classification calibration metric (primary metric for probability predictions).
6. **Panel 4 - Feature Importance**: Horizontal bar chart showing Gain (average loss reduction) for each feature.
7. **Panel 5 - Overfitting Analysis**: Calculates and visualizes train/validation gap (overfitting indicator). Sets model-specific colors (XGBoost: Bluish, LightGBM: Redish).
8. **Quality Assurance**: Adds reference lines, annotations, and standardized color coding for instant cross-model comparison.

Architecture:
- Single function `visualize_training_metrics()` for simplicity
- Uses matplotlib subplots for multi-panel layout
- Standardized Hex Color Palette: XGBoost (`#1f77b4`), LightGBM (`#d62728`)
- Automatic scaling and formatting for publication-quality output

Purpose:
Provides instant visual assessment of model training quality, enabling rapid detection of overfitting,
poor fit, or configuration issues without manual inspection of numerical logs.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_training_metrics(xgboost_metrics, lightgbm_metrics, output_path='outputs/figures/Training_Metrics_Diagnostic.png'):
    """
    Generates comprehensive diagnostic plots for model training metrics.
    
    Parameters:
        xgboost_metrics (dict): {'train_rmse', 'validation_rmse', 'train_mae', 'validation_mae', 'train_logloss', 'validation_logloss', 'feature_importance', 'best_iteration', 'maximum_iterations'}
        lightgbm_metrics (dict): {'train_rmse', 'validation_rmse', 'train_mae', 'validation_mae', 'train_logloss', 'validation_logloss', 'feature_importance', 'best_iteration', 'maximum_iterations'}
        output_path (str): File path for saving the output figure
    
    Output:
        5-panel diagnostic figure saved to specified path
    """
    
    # Create figure with 5 panels in 2x3 grid
    figure, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Font configuration
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    annotation_fontsize = 11
    
    # Feature names for importance plots
    feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                     'Specificity', 'Gamma_Decay_Pattern_Similarity']
    
    # Model Color Palette (Standardized)
    color_xgboost = '#1f77b4'  # Professional Bluish
    color_lightgbm = '#d62728' # Professional Redish
    alpha_train = 0.9
    alpha_val = 0.5
    
    # ==========================================
    # Panel 1: RMSE Comparison
    # ==========================================
    axis_rmse = axes[0]
    
    model_labels = ['XGBoost', 'LightGBM']
    training_rmse_values = [xgboost_metrics['train_rmse'], lightgbm_metrics['train_rmse']]
    validation_rmse_values = [xgboost_metrics['validation_rmse'], lightgbm_metrics['validation_rmse']]
    
    x_positions = np.arange(len(model_labels))
    bar_width = 0.35
    
    # XGBoost bars
    axis_rmse.bar(x_positions[0] - bar_width/2, training_rmse_values[0], bar_width, 
                 label='XGB Training', color=color_xgboost, alpha=alpha_train)
    axis_rmse.bar(x_positions[0] + bar_width/2, validation_rmse_values[0], bar_width,
                 label='XGB Validation', color=color_xgboost, alpha=alpha_val)
    
    # LightGBM bars
    axis_rmse.bar(x_positions[1] - bar_width/2, training_rmse_values[1], bar_width, 
                 label='LGBM Training', color=color_lightgbm, alpha=alpha_train)
    axis_rmse.bar(x_positions[1] + bar_width/2, validation_rmse_values[1], bar_width,
                 label='LGBM Validation', color=color_lightgbm, alpha=alpha_val)
    
    # Add reference line for excellent performance threshold (RMSE < 0.05)
    axis_rmse.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent Threshold')
    axis_rmse.axhline(y=0.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Poor Threshold')
    
    # Add value labels
    for i, (t, v) in enumerate(zip(training_rmse_values, validation_rmse_values)):
        axis_rmse.text(x_positions[i] - bar_width/2, t, f'{t:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
        axis_rmse.text(x_positions[i] + bar_width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
    
    axis_rmse.set_xlabel('Model', fontsize=label_fontsize, fontweight='bold')
    axis_rmse.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=label_fontsize, fontweight='bold')
    axis_rmse.set_title('RMSE: Training vs Validation', fontsize=title_fontsize, fontweight='bold', pad=15)
    axis_rmse.set_xticks(x_positions)
    axis_rmse.set_xticklabels(model_labels, fontsize=tick_fontsize)
    axis_rmse.tick_params(axis='y', labelsize=tick_fontsize)
    axis_rmse.legend(fontsize=annotation_fontsize, loc='upper left')
    axis_rmse.grid(axis='y', alpha=0.3, linestyle=':')
    
    # ==========================================
    # Panel 2: MAE Comparison
    # ==========================================
    axis_mae = axes[1]
    
    training_mae_values = [xgboost_metrics['train_mae'], lightgbm_metrics['train_mae']]
    validation_mae_values = [xgboost_metrics['validation_mae'], lightgbm_metrics['validation_mae']]
    
    # XGBoost bars
    axis_mae.bar(x_positions[0] - bar_width/2, training_mae_values[0], bar_width,
                label='XGB Training', color=color_xgboost, alpha=alpha_train)
    axis_mae.bar(x_positions[0] + bar_width/2, validation_mae_values[0], bar_width,
                label='XGB Validation', color=color_xgboost, alpha=alpha_val)
    
    # LightGBM bars
    axis_mae.bar(x_positions[1] - bar_width/2, training_mae_values[1], bar_width,
                label='LGBM Training', color=color_lightgbm, alpha=alpha_train)
    axis_mae.bar(x_positions[1] + bar_width/2, validation_mae_values[1], bar_width,
                label='LGBM Validation', color=color_lightgbm, alpha=alpha_val)
    
    # Add reference line for excellent performance threshold (MAE < 0.02)
    axis_mae.axhline(y=0.02, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent Threshold')
    axis_mae.axhline(y=0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Poor Threshold')
    
    # Add value labels
    for i, (t, v) in enumerate(zip(training_mae_values, validation_mae_values)):
        axis_mae.text(x_positions[i] - bar_width/2, t, f'{t:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
        axis_mae.text(x_positions[i] + bar_width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
    
    axis_mae.set_xlabel('Model', fontsize=label_fontsize, fontweight='bold')
    axis_mae.set_ylabel('MAE (Mean Absolute Error)', fontsize=label_fontsize, fontweight='bold')
    axis_mae.set_title('MAE: Training vs Validation', fontsize=title_fontsize, fontweight='bold', pad=15)
    axis_mae.set_xticks(x_positions)
    axis_mae.set_xticklabels(model_labels, fontsize=tick_fontsize)
    axis_mae.tick_params(axis='y', labelsize=tick_fontsize)
    axis_mae.legend(fontsize=annotation_fontsize, loc='upper left')
    axis_mae.grid(axis='y', alpha=0.3, linestyle=':')
    
    # ==========================================
    # Panel 3: LogLoss Comparison
    # ==========================================
    axis_logloss = axes[2]
    
    training_logloss_values = [xgboost_metrics['train_logloss'], lightgbm_metrics['train_logloss']]
    validation_logloss_values = [xgboost_metrics['validation_logloss'], lightgbm_metrics['validation_logloss']]
    
    # XGBoost bars
    axis_logloss.bar(x_positions[0] - bar_width/2, training_logloss_values[0], bar_width,
                    label='XGB Training', color=color_xgboost, alpha=alpha_train)
    axis_logloss.bar(x_positions[0] + bar_width/2, validation_logloss_values[0], bar_width,
                    label='XGB Validation', color=color_xgboost, alpha=alpha_val)
    
    # LightGBM bars
    axis_logloss.bar(x_positions[1] - bar_width/2, training_logloss_values[1], bar_width,
                    label='LGBM Training', color=color_lightgbm, alpha=alpha_train)
    axis_logloss.bar(x_positions[1] + bar_width/2, validation_logloss_values[1], bar_width,
                    label='LGBM Validation', color=color_lightgbm, alpha=alpha_val)
    
    # Add reference line for excellent performance threshold (LogLoss < 0.1)
    axis_logloss.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent Threshold')
    axis_logloss.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Poor Threshold')
    
    # Add value labels
    for i, (t, v) in enumerate(zip(training_logloss_values, validation_logloss_values)):
        axis_logloss.text(x_positions[i] - bar_width/2, t, f'{t:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
        axis_logloss.text(x_positions[i] + bar_width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize)
    
    axis_logloss.set_xlabel('Model', fontsize=label_fontsize, fontweight='bold')
    axis_logloss.set_ylabel('LogLoss (Cross-Entropy)', fontsize=label_fontsize, fontweight='bold')
    axis_logloss.set_title('LogLoss: Training vs Validation', fontsize=title_fontsize, fontweight='bold', pad=15)
    axis_logloss.set_xticks(x_positions)
    axis_logloss.set_xticklabels(model_labels, fontsize=tick_fontsize)
    axis_logloss.tick_params(axis='y', labelsize=tick_fontsize)
    axis_logloss.legend(fontsize=annotation_fontsize, loc='upper left')
    axis_logloss.grid(axis='y', alpha=0.3, linestyle=':')
    
    # ==========================================
    # Panel 4: Feature Importance (XGBoost Only)
    # ==========================================
    axis_importance = axes[3]
    
    # Extract importance values for all 5 features
    # XGBoost returns dict using column names from the training DataFrame
    # If a feature is never used in split, it might be missing from the dict, so we default to 0.0
    importance_values = []
    
    # Get importance dictionary. If empty or None, handle gracefully.
    raw_importance = xgboost_metrics.get('feature_importance', {})
    
    for name in feature_names:
        # Try exact name match first
        val = raw_importance.get(name, 0.0)
        
        # Fallback: if keys are f0, f1 etc. map by index
        if val == 0.0 and 'f0' in raw_importance:
            index = feature_names.index(name)
            val = raw_importance.get(f'f{index}', 0.0)
            
        importance_values.append(val)
    
    # Horizontal bar chart for better label readability
    y_positions_importance = np.arange(len(feature_names))
    # Sort for visual hierarchy
    
    bars_importance = axis_importance.barh(y_positions_importance, importance_values, color=color_xgboost, alpha=0.8)
    
    # Add value labels
    for index, bar in enumerate(bars_importance):
        width = bar.get_width()
        axis_importance.text(width, bar.get_y() + bar.get_height()/2.,
                            f'{width:.1f}', ha='left', va='center', fontsize=annotation_fontsize, fontweight='bold')
    
    axis_importance.set_xlabel('Gain (Average Loss Reduction)', fontsize=label_fontsize, fontweight='bold')
    axis_importance.set_ylabel('Feature', fontsize=label_fontsize, fontweight='bold')
    axis_importance.set_title('Feature Importance (XGBoost)', fontsize=title_fontsize, fontweight='bold', pad=15)
    axis_importance.set_yticks(y_positions_importance)
    axis_importance.set_yticklabels(feature_names, fontsize=tick_fontsize)
    axis_importance.tick_params(axis='x', labelsize=tick_fontsize)
    axis_importance.grid(axis='x', alpha=0.3, linestyle=':')
    
    # ==========================================
    # Panel 5: Overfitting Gap Analysis
    # ==========================================
    axis_gap = axes[4]
    
    # Calculate train/validation gaps (overfitting indicator)
    xgboost_gap = xgboost_metrics['validation_rmse'] - xgboost_metrics['train_rmse']
    lightgbm_gap = lightgbm_metrics['validation_rmse'] - lightgbm_metrics['train_rmse']
    gap_values = [xgboost_gap, lightgbm_gap]
    
    # Use consistent model colors
    bars_gap = axis_gap.bar(x_positions, gap_values, color=[color_xgboost, color_lightgbm], alpha=0.8)
    
    # Add reference line for acceptable gap threshold
    axis_gap.axhline(y=0.01, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Excellent (<0.01)')
    axis_gap.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Warning (>0.05)')
    
    # Add value labels on bars
    for bar in bars_gap:
        height = bar.get_height()
        axis_gap.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=annotation_fontsize, fontweight='bold')
    
    # Add iteration count annotations below bars
    xgboost_iteration_percent = (xgboost_metrics['best_iteration'] / xgboost_metrics['maximum_iterations']) * 100
    lightgbm_iteration_percent = (lightgbm_metrics['best_iteration'] / lightgbm_metrics['maximum_iterations']) * 100
    
    axis_gap.text(x_positions[0], -0.002, 
                 f"Stopped: {xgboost_metrics['best_iteration']}/{xgboost_metrics['maximum_iterations']} ({xgboost_iteration_percent:.0f}%)",
                 ha='center', va='top', fontsize=annotation_fontsize-1, style='italic')
    axis_gap.text(x_positions[1], -0.002,
                 f"Stopped: {lightgbm_metrics['best_iteration']}/{lightgbm_metrics['maximum_iterations']} ({lightgbm_iteration_percent:.0f}%)",
                 ha='center', va='top', fontsize=annotation_fontsize-1, style='italic')
    
    axis_gap.set_xlabel('Model', fontsize=label_fontsize, fontweight='bold')
    axis_gap.set_ylabel('Overfitting Gap (Validation - Training RMSE)', fontsize=label_fontsize, fontweight='bold')
    axis_gap.set_title('Overfitting Analysis', fontsize=title_fontsize, fontweight='bold', pad=15)
    axis_gap.set_xticks(x_positions)
    axis_gap.set_xticklabels(model_labels, fontsize=tick_fontsize)
    axis_gap.tick_params(axis='y', labelsize=tick_fontsize)
    axis_gap.legend(fontsize=annotation_fontsize, loc='upper left')
    axis_gap.grid(axis='y', alpha=0.3, linestyle=':')
    
    # ==========================================
    # Panel 6: Hide unused subplot
    # ==========================================
    axes[5].axis('off')
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Training metrics diagnostic plot saved to {output_path}")
