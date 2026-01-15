"""Combined Level Scheme and Clustering Visualizer

This script generates two visualizations:
1. Input_Level_Scheme.png: Input datasets showing all levels.
2. Output_Cluster_Scheme.png: Output clustering results with cluster and probability labels.

Explanation of Code Structure
1) Parse clustering results into clusters and members.
2) Build per-dataset level lists and apply collision resolution for energy/Jπ text labels.
3) Compute a single global y-position per cluster (collision-resolved using cluster anchor energies).
4) Draw level bars at true energies (no vertical bar offsets).
5) Place cluster/probability labels at the cluster y-position for vertical alignment across datasets,
   and draw a short connector if the label y differs from the bar energy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import re

# Tunable clustering layout knobs for quick manual adjustments
clustering_box_pad = 1.0                # Box padding; increase for wider/taller boxes
clustering_min_distance = 500           # Minimum vertical spacing between rows
clustering_fig_height_multiplier = 1.0  # Height scaling; raise to add vertical breathing room
clustering_fig_width_inches = 10.0      # Overall figure width in inches; shrink to reduce horizontal scale
clustering_x_spacing = 0.45             # Horizontal distance between dataset columns (reduce for tighter columns)
clustering_x_margin = 0.3               # Extra blank space padding on left/right of the outer columns

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def spread_text_positions(energies, min_distance=250):
    """
    Adjusts text positions to ensure they are at least min_distance apart vertically.
    Collision resolution algorithm using iterative relaxation.
    """
    if not energies:
        return []
    
    number_of_positions = len(energies)
    positions = np.array(energies, dtype=float)
    
    # Relaxation iterations to push apart overlapping labels
    for iteration in range(200):
        changed = False
        for index in range(number_of_positions - 1):
            distance = positions[index + 1] - positions[index]
            if distance < min_distance:
                # Overlap detected: push levels apart
                center = (positions[index + 1] + positions[index]) / 2.0
                overlap = min_distance - distance
                positions[index] -= overlap / 2.0
                positions[index + 1] += overlap / 2.0
                changed = True
        if not changed:
            break
    
    return positions

# ============================================================================
# LEVEL SCHEME VISUALIZER (INPUT DATA)
# ============================================================================

def load_dataset(dataset_code):
    """Load level data from JSON file for a given dataset code."""
    filename = f"test_dataset_{dataset_code}.json"
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)
        if isinstance(data, dict) and 'levelsTable' in data:
            return data['levelsTable'].get('levels', [])
        elif isinstance(data, list):
            return data
    return []

def plot_level_schemes():
    """Generate visualization of input level schemes from all datasets."""
    datasets = ['A', 'B', 'C', 'D', 'E', 'F']
    figure, axis = plt.subplots(figsize=(20, 10))
    
    # X-axis positions for each dataset column
    x_positions = {'A': 0, 'B': 2.5, 'C': 5.0, 'D': 7.5, 'E': 10.0, 'F': 12.5}
    line_width = 0.8
    maximum_energy = 0
    
    for dataset_code in datasets:
        raw_levels = load_dataset(dataset_code)
        
        # Extract level data
        levels_data = []
        for level in raw_levels:
            # Energy
            if isinstance(level.get('energy'), dict):
                energy_value = level.get('energy', {}).get('value')
                uncertainty = level.get('energy', {}).get('uncertainty', {}).get('value')
            else:
                energy_value = level.get('energy_value')
                uncertainty = level.get('energy_uncertainty')
                
            if energy_value is None:
                continue
            energy_value = float(energy_value)
            
            # Spin/Parity
            spin_parity_string = ""
            if isinstance(level.get('spinParity'), dict):
                spin_parity_string = level.get('spinParity', {}).get('evaluatorInput', '')
            else:
                spin_parity_string = level.get('spin_parity_string', '')
            
            uncertainty_string = f"({int(uncertainty)})" if uncertainty is not None else ""
            label_left = f"{int(energy_value)}{uncertainty_string}"
            label_right = spin_parity_string
            
            levels_data.append({
                'energy': energy_value,
                'label_left': label_left,
                'label_right': label_right
            })
            
            if energy_value > maximum_energy:
                maximum_energy = energy_value

        # Sort by energy
        levels_data.sort(key=lambda x: x['energy'])
        
        # Calculate text positions with collision resolution
        energies = [x['energy'] for x in levels_data]
        text_y_positions = spread_text_positions(energies, min_distance=250)
        
        # Calculate vertical offsets for level bars when too close (within 150 keV)
        bar_offsets = [0.0] * len(energies)
        for index in range(len(energies) - 1):
            if energies[index + 1] - energies[index] < 150:
                # Push bars apart slightly for visual separation
                bar_offsets[index] -= 30
                bar_offsets[index + 1] += 30
        
        # Plot levels
        x_center = x_positions[dataset_code]
        x_start = x_center - line_width / 2
        x_end = x_center + line_width / 2
        
        for index, item in enumerate(levels_data):
            energy = item['energy']
            y_text = text_y_positions[index]
            bar_offset = bar_offsets[index]
            
            # Draw level line with vertical offset for close-by levels
            bar_y_position = energy + bar_offset
            axis.hlines(y=bar_y_position, xmin=x_start, xmax=x_end, colors='black', linewidth=1.5)
            
            # Check if text is displaced significantly
            is_displaced = abs(bar_y_position - y_text) > 50
            
            if is_displaced:
                # Energy label with connector
                axis.annotate(item['label_left'], 
                            xy=(x_start, bar_y_position), xycoords='data',
                            xytext=(x_start - 0.2, y_text), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                            va='center', ha='right', fontsize=20, family='Times New Roman')
                
                # Spin/Parity label with connector
                if item['label_right']:
                    axis.annotate(item['label_right'], 
                                xy=(x_end, bar_y_position), xycoords='data',
                                xytext=(x_end + 0.2, y_text), textcoords='data',
                                arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                                va='center', ha='left', fontsize=20, family='Times New Roman')
            else:
                # Standard text placement
                axis.text(x_start - 0.1, y_text, item['label_left'], va='center', ha='right', fontsize=20, family='Times New Roman')
                if item['label_right']:
                    axis.text(x_end + 0.1, y_text, item['label_right'], va='center', ha='left', fontsize=20, family='Times New Roman')

    # Styling
    axis.set_xlim(-1.5, 14.0)
    axis.set_ylim(0, maximum_energy * 1.15)
    axis.set_xticks([0, 2.5, 5.0, 7.5, 10.0, 12.5])
    axis.set_xticklabels(['Dataset A', 'Dataset B', 'Dataset C', 'Dataset D', 'Dataset E', 'Dataset F'], 
                         fontsize=20, fontweight='bold', family='Times New Roman')
    
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_linewidth(1.5)
    
    axis.set_ylabel("Energy (keV)", fontsize=20, family='Times New Roman')
    axis.tick_params(axis='x', length=0)
    axis.tick_params(axis='y', labelsize=16)
    for label in axis.get_yticklabels():
        label.set_family('Times New Roman')
    
    axis.set_title("Input Level Schemes", fontsize=20, fontweight='bold', pad=20, family='Times New Roman')
    
    plt.tight_layout()
    output_file = 'Input_Level_Scheme.png'
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Level scheme visualization saved to {output_file}")
    plt.close()

# ============================================================================
# CLUSTERING VISUALIZER (OUTPUT RESULTS)
# ============================================================================

def parse_clustering_results(clustering_file_path):
    """Parse Output_Clustering_Results.txt to extract cluster information."""
    clusters = []
    current_cluster = None
    
    with open(clustering_file_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            line = line.strip()
            
            # Match cluster header
            cluster_match = re.match(r'^Cluster (\d+):$', line)
            if cluster_match:
                if current_cluster is not None:
                    clusters.append(current_cluster)
                current_cluster = {
                    'cluster_number': int(cluster_match.group(1)),
                    'members': []
                }
                continue
            
            # Match anchor line
            anchor_match = re.match(r'^Anchor:\s+(\S+)\s+\|\s+E=([\d.]+)±([\d.]+)\s+keV\s+\|\s+Jπ=(.+)$', line)
            if anchor_match and current_cluster is not None:
                current_cluster['anchor_id'] = anchor_match.group(1)
                current_cluster['anchor_energy'] = float(anchor_match.group(2))
                current_cluster['anchor_uncertainty'] = float(anchor_match.group(3))
                current_cluster['anchor_jpi'] = anchor_match.group(4)
                continue
            
            # Match member line
            member_match = re.match(r'^\[(\w+)\]\s+(\S+):\s+E=([\d.]+)±([\d.]+)\s+keV,\s+Jπ=(.+?)\s+\((.+)\)$', line)
            if member_match and current_cluster is not None:
                dataset = member_match.group(1)
                level_id = member_match.group(2)
                energy = float(member_match.group(3))
                uncertainty = float(member_match.group(4))
                jpi = member_match.group(5).strip()
                status_info = member_match.group(6).strip()
                
                is_anchor = status_info == "Anchor"
                match_probability = None
                if not is_anchor:
                    probability_match = re.search(r'Match Prob:\s+([\d.]+)%', status_info)
                    if probability_match:
                        match_probability = float(probability_match.group(1)) / 100.0
                
                current_cluster['members'].append({
                    'dataset': dataset,
                    'level_id': level_id,
                    'energy': energy,
                    'uncertainty': uncertainty,
                    'jpi': jpi,
                    'is_anchor': is_anchor,
                    'match_probability': match_probability
                })
                continue
    
    # Add final cluster
    if current_cluster is not None:
        clusters.append(current_cluster)
    
    return clusters

def plot_clustering_results():
    """
    Generates a textual table-like visualization of clustering results.
    Aligned horizontally by cluster, sorted vertically by energy.
    Reads directly from Output_Clustering_Results.txt.
    """
    clustering_file_path = 'Output_Clustering_Results.txt'
    clusters = parse_clustering_results(clustering_file_path)
    
    if not clusters:
        print("[WARNING] No clusters found in Output_Clustering_Results.txt")
        return
    
    # Sort clusters by anchor energy (Low Energy at Bottom)
    clusters.sort(key=lambda x: x.get('anchor_energy', 0))
    
    datasets = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Calculate figure size based on density (tunable multiplier)
    fig_height = max(8, len(clusters) * clustering_fig_height_multiplier)
    figure, axis = plt.subplots(figsize=(clustering_fig_width_inches + 5, fig_height))
    
    # X-axis positions (tunable spacing)
    x_positions = {ds: index * clustering_x_spacing for index, ds in enumerate(datasets)}
    
    # Y-Position Calculation
    # Use anchor energies as base, then spread them to prevent text overlap
    anchor_energies = [c.get('anchor_energy', 0) for c in clusters]
    y_positions = spread_text_positions(anchor_energies, min_distance=clustering_min_distance)
    
    if len(y_positions) > 0:
        y_max_limit = y_positions[-1] + 300
    else: 
        y_max_limit = 1000

    # Plot Text for each cluster
    for i, cluster in enumerate(clusters):
        y_pos = y_positions[i]
        cluster_id = cluster['cluster_number']
        
        # Plot members
        for member in cluster['members']:
            ds = member['dataset']
            if ds not in x_positions: continue
            
            x_pos = x_positions[ds]
            
            # Text Content
            # Format: 'Energy(Unc)', 'Jpi', 'Prob', 'ClusterID'
            e_str = f"{member['energy']:.0f}({int(member['uncertainty'])})"
            jpi_str = member['jpi']
            
            if member['is_anchor']:
                prob_str = "Anchor"
            elif member.get('match_probability') is not None:
                prob_str = f"{member['match_probability']:.1%}"
            else:
                prob_str = "N/A"
            
            # Combine into block - 2 lines for wider, shorter box
            text_block = (
                f"C{cluster_id} | {e_str}\n"
                f"{jpi_str} | {prob_str}"
            )
            
            axis.text(x_pos, y_pos, text_block, 
                     ha='center', va='center', 
                     fontsize=12, family='Times New Roman',
                     bbox=dict(boxstyle=f"round,pad={clustering_box_pad}", fc="white", ec="gray", alpha=0.9))

    # Determine Y-Axis Limits
    axis.set_ylim(-200, y_max_limit)
    right_limit = (len(datasets) - 1) * clustering_x_spacing + clustering_x_margin
    left_limit = -clustering_x_margin
    axis.set_xlim(left_limit, right_limit)
    
    # X-Axis Labels
    axis.set_xticks([index * clustering_x_spacing for index in range(len(datasets))])
    axis.set_xticklabels(['Dataset A', 'Dataset B', 'Dataset C', 'Dataset D', 'Dataset E', 'Dataset F'], 
                         fontsize=16, fontweight='bold', family='Times New Roman')
    
    # Y-Axis Label
    axis.set_ylabel("Energy / Cluster Index (Spread)", fontsize=16, family='Times New Roman')
    
    # Styling: Remove spines for clean table look
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    
    axis.tick_params(left=True, bottom=False, labelsize=14)
    for label in axis.get_yticklabels():
        label.set_family('Times New Roman')
    
    axis.set_title("Clustering Results (Aligned by Cluster)", fontsize=20, fontweight='bold', pad=20, family='Times New Roman')
    
    plt.tight_layout()
    output_file = 'Output_Cluster_Scheme.png'
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Clustering visualization saved to {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Combined Visualizer: Generating both plots...")
    print("=" * 60)
    
    # Generate both visualizations
    plot_level_schemes()
    plot_clustering_results()
    
    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
