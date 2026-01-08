"""Combined Level Scheme and Clustering Visualizer

This script generates two visualizations:
1. Level_Scheme_Visualization.png: Input datasets showing all levels.
2. Clustering_Visualization.png: Output clustering results with cluster and probability labels.

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
    datasets = ['A', 'B', 'C']
    figure, axis = plt.subplots(figsize=(15, 10))
    
    # X-axis positions for each dataset column
    x_positions = {'A': 0, 'B': 2.5, 'C': 5.0}
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
    axis.set_xlim(-1.5, 6.5)
    axis.set_ylim(0, maximum_energy * 1.15)
    axis.set_xticks([0, 2.5, 5.0])
    axis.set_xticklabels(['Dataset A', 'Dataset B', 'Dataset C'], fontsize=20, fontweight='bold', family='Times New Roman')
    
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
    output_file = 'Level_Scheme_Visualization.png'
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Level scheme visualization saved to {output_file}")
    plt.close()

# ============================================================================
# CLUSTERING VISUALIZER (OUTPUT RESULTS)
# ============================================================================

def parse_clustering_results(clustering_file_path):
    """Parse clustering_results.txt to extract cluster information."""
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
    """Generate clean hierarchical visualization of clustering results organized by cluster."""
    clustering_file_path = 'clustering_results.txt'
    clusters = parse_clustering_results(clustering_file_path)
    
    if not clusters:
        print("[WARNING] No clusters found in clustering_results.txt")
        return
    
    # Determine which datasets are present
    all_datasets = set()
    for cluster in clusters:
        for member in cluster['members']:
            all_datasets.add(member['dataset'])
    datasets = sorted(all_datasets)
    
    # Create figure
    figure, axis = plt.subplots(figsize=(15, 10))
    
    # X-axis positions for each dataset column
    # Use wider spacing to create a dedicated gutter for cluster/probability labels.
    horizontal_spacing = 4.0
    x_positions = {dataset: index * horizontal_spacing for index, dataset in enumerate(datasets)}
    line_width = 0.8
    
    # Track maximum energy
    maximum_energy = 0
    
    # Collect ALL levels for each dataset column for global collision resolution
    all_levels_by_dataset = {dataset: {} for dataset in datasets}  # Use dict to avoid duplicates
    for cluster in clusters:
        for member in cluster['members']:
            level_key = (member['dataset'], member['level_id'], member['energy'])
            if level_key not in all_levels_by_dataset[member['dataset']]:
                all_levels_by_dataset[member['dataset']][level_key] = {
                    'unique_level_key': level_key,
                    'cluster_numbers': [],  # Track all clusters this level belongs to
                    'energy': member['energy'],
                    'level_id': member['level_id'],
                    'uncertainty': member['uncertainty'],
                    'jpi': member['jpi'],
                    'is_anchor': member['is_anchor'],
                    'match_probability': member['match_probability']
                }
            # Add this cluster to the level's cluster list
            all_levels_by_dataset[member['dataset']][level_key]['cluster_numbers'].append(cluster['cluster_number'])
            if member['energy'] > maximum_energy:
                maximum_energy = member['energy']
    
    # Apply global collision resolution for each dataset column
    text_position_lookup = {}
    bar_offset_lookup = {}
    levels_by_dataset_list = {}  # Store the sorted lists for reuse
    
    for dataset in datasets:
        levels_dict = all_levels_by_dataset[dataset]
        if not levels_dict:
            continue
        
        # Convert dict to list and sort by energy ONCE
        levels_in_column = list(levels_dict.values())
        levels_in_column.sort(key=lambda x: x['energy'])
        levels_by_dataset_list[dataset] = levels_in_column  # Store for reuse
        
        # Extract energies and apply collision resolution
        energies = [level['energy'] for level in levels_in_column]
        text_y_positions = spread_text_positions(energies, min_distance=250)
        
        # No bar offsets - bars stay at their true energy positions
        bar_offsets = [0.0] * len(energies)
        
        # Store adjusted positions and offsets
        for index, level in enumerate(levels_in_column):
            unique_level_key = level['unique_level_key']
            text_position_lookup[unique_level_key] = text_y_positions[index]
            bar_offset_lookup[unique_level_key] = bar_offsets[index]

    # === Calculate global cluster label y-positions (shared across datasets) ===
    # Use cluster anchor energies as the canonical y values, then apply collision resolution ONCE.
    cluster_anchor_energy_by_cluster_number = {}
    for cluster in clusters:
        if 'cluster_number' in cluster and 'anchor_energy' in cluster:
            cluster_anchor_energy_by_cluster_number[cluster['cluster_number']] = float(cluster['anchor_energy'])

    sorted_cluster_numbers = sorted(cluster_anchor_energy_by_cluster_number.keys())
    sorted_cluster_anchor_energies = [cluster_anchor_energy_by_cluster_number[number] for number in sorted_cluster_numbers]
    sorted_cluster_label_y_positions = spread_text_positions(sorted_cluster_anchor_energies, min_distance=250)

    cluster_label_y_by_cluster_number = {
        number: float(sorted_cluster_label_y_positions[index])
        for index, number in enumerate(sorted_cluster_numbers)
    }
    
    # Draw all levels with cluster and probability labels
    for dataset in datasets:
        if dataset not in levels_by_dataset_list:
            continue
        
        levels_in_column = levels_by_dataset_list[dataset]  # Reuse the SAME sorted list
        
        x_center = x_positions[dataset]
        x_start = x_center - line_width / 2
        x_end = x_center + line_width / 2
        
        for level in levels_in_column:
            energy = level['energy']
            unique_level_key = level['unique_level_key']

            # Look up pre-calculated text position and bar offset
            y_text = text_position_lookup[unique_level_key]
            bar_offset = bar_offset_lookup[unique_level_key]
            
            # Draw level line with vertical offset (all black, no colors)
            bar_y_position = energy + bar_offset
            axis.hlines(y=bar_y_position, xmin=x_start, xmax=x_end, 
                       colors='black', linewidth=2.0)
            
            # Construct label text
            energy_label = f"{int(level['energy'])}({int(level['uncertainty'])})"
            jpi_label = level['jpi'] if level['jpi'] != 'N/A' else ''
            
            # Check if text is displaced significantly
            is_displaced = abs(bar_y_position - y_text) > 50
            
            if is_displaced:
                # Draw connector line and text
                axis.annotate(energy_label,
                            xy=(x_start, bar_y_position), xycoords='data',
                            xytext=(x_start - 0.2, y_text), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                            va='center', ha='right', fontsize=20, family='Times New Roman')
                if jpi_label:
                    axis.annotate(jpi_label,
                                xy=(x_end, bar_y_position), xycoords='data',
                                xytext=(x_end + 0.2, y_text), textcoords='data',
                                arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                                va='center', ha='left', fontsize=20, family='Times New Roman')
            else:
                # Standard text placement
                axis.text(x_start - 0.1, y_text, energy_label, va='center', ha='right', fontsize=20, family='Times New Roman')
                if jpi_label:
                    axis.text(x_end + 0.1, y_text, jpi_label, va='center', ha='left', fontsize=20, family='Times New Roman')
    
    # Add combined cluster and probability labels with smart positioning
    for dataset in datasets:
        if dataset not in levels_by_dataset_list:
            continue
        
        levels_in_column = levels_by_dataset_list[dataset]  # Reuse the SAME sorted list
        x_center = x_positions[dataset]
        
        # NO sorting needed - use the list as-is (already sorted by energy)
        # Each level corresponds to its text_position_lookup entry by index
        for index, level in enumerate(levels_in_column):
            unique_level_key = level['unique_level_key']
            text_y = text_position_lookup[unique_level_key]
            
            # Construct combined label with ALL cluster numbers
            cluster_numbers = level['cluster_numbers']
            sorted_cluster_numbers_for_label = sorted(cluster_numbers)
            primary_cluster = sorted_cluster_numbers_for_label[0]
            
            if len(cluster_numbers) > 1:
                cluster_label = ','.join([f"C{c}" for c in sorted_cluster_numbers_for_label])
            else:
                cluster_label = f"C{primary_cluster}"
            
            # Add probability if not anchor
            if level['is_anchor']:
                combined_label = cluster_label
            elif level['match_probability'] is not None:
                combined_label = f"{cluster_label}, {level['match_probability']:.0%}"
            else:
                combined_label = cluster_label
            
            # Place cluster label at FIXED global position (same y for all datasets)
            # Use collision-resolved cluster anchor y positions so close clusters do not overlap.
            label_y_values = []
            for cluster_number in sorted_cluster_numbers_for_label:
                if cluster_number in cluster_label_y_by_cluster_number:
                    label_y_values.append(cluster_label_y_by_cluster_number[cluster_number])

            if label_y_values:
                label_y = sum(label_y_values) / float(len(label_y_values))
            else:
                # Fallback: use the collision-resolved text position for this level
                label_y = float(text_y)

            # Place the label in a dedicated right-side gutter so it does not overlap
            # energy/Jπ labels or connector arrows.
            cluster_label_horizontal_offset = 2.2
            label_x = x_center + line_width / 2 + cluster_label_horizontal_offset
            axis.text(label_x, label_y, combined_label,
                     va='center', ha='left', fontsize=20, style='italic', color='black', family='Times New Roman', zorder=5)

            # Draw a connector only when the label y differs from the actual energy bar.
            # Route it as an L-shape inside the gutter to avoid crossing Jπ text.
            bar_y_position = float(level['energy'])
            if abs(label_y - bar_y_position) > 50:
                bar_end_x = x_center + line_width / 2
                gutter_elbow_x = label_x - 0.25
                axis.plot([bar_end_x, gutter_elbow_x], [bar_y_position, bar_y_position], color='gray', lw=2.0, zorder=1)
                axis.plot([gutter_elbow_x, gutter_elbow_x], [bar_y_position, label_y], color='gray', lw=2.0, zorder=1)
                axis.plot([gutter_elbow_x, label_x - 0.08], [label_y, label_y], color='gray', lw=2.0, zorder=1)
    
    # Styling
    x_limit_min = -1.5
    last_dataset = datasets[-1]
    x_limit_max = x_positions[last_dataset] + line_width / 2 + 3.0
    axis.set_xlim(x_limit_min, x_limit_max)
    axis.set_ylim(0, maximum_energy * 1.15)
    
    x_tick_positions = [x_positions[dataset] for dataset in datasets]
    x_tick_labels = [f'Dataset {dataset}' for dataset in datasets]
    axis.set_xticks(x_tick_positions)
    axis.set_xticklabels(x_tick_labels, fontsize=20, fontweight='bold', family='Times New Roman')
    
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_linewidth(1.5)
    
    axis.set_ylabel("Energy (keV)", fontsize=20, family='Times New Roman')
    axis.tick_params(axis='x', length=0)
    axis.tick_params(axis='y', labelsize=16)
    for label in axis.get_yticklabels():
        label.set_family('Times New Roman')
    

    plt.tight_layout()
    output_file = 'Clustering_Visualization.png'
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
