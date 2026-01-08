"""
Explanation of Code Structure:

This script visualizes the final clustering results from Level_Matcher.py by generating
a multi-column level scheme diagram showing which levels from different datasets have
been matched into clusters.

Workflow:
1. Parse clustering_results.txt to extract cluster information
2. For each cluster, extract member levels with their energy, uncertainty, Jπ, and dataset
3. Position clusters vertically by anchor energy
4. Draw horizontal level lines at true energy positions
5. Apply collision resolution for text labels (similar to Level_Scheme_Visualizer.py)
6. Use color coding to distinguish clusters
7. Draw connecting lines between members of the same cluster across datasets
8. Output high-quality PNG file with no overlapping text

Key Features:
- Each cluster shown in a different color
- Levels grouped by dataset (A, B, C columns)
- Connecting lines show which levels belong to the same cluster
- Anchor levels highlighted with bold borders
- Match probabilities displayed for non-anchor members
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re

def parse_clustering_results(clustering_file_path):
    """
    Parse clustering_results.txt to extract cluster information.
    
    Returns:
        List of clusters, where each cluster is a dict with:
        {
            'cluster_number': int,
            'anchor_id': str,
            'anchor_energy': float,
            'anchor_uncertainty': float,
            'anchor_jpi': str,
            'members': [
                {
                    'dataset': str,
                    'level_id': str,
                    'energy': float,
                    'uncertainty': float,
                    'jpi': str,
                    'is_anchor': bool,
                    'match_probability': float or None
                },
                ...
            ]
        }
    """
    clusters = []
    current_cluster = None
    
    with open(clustering_file_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            line = line.strip()
            
            # Match cluster header: "Cluster 1:"
            cluster_match = re.match(r'^Cluster (\d+):$', line)
            if cluster_match:
                if current_cluster is not None:
                    clusters.append(current_cluster)
                current_cluster = {
                    'cluster_number': int(cluster_match.group(1)),
                    'members': []
                }
                continue
            
            # Match anchor line: "Anchor: A_1000 | E=1000.0±1.0 keV | Jπ=N/A"
            anchor_match = re.match(r'^Anchor:\s+(\S+)\s+\|\s+E=([\d.]+)±([\d.]+)\s+keV\s+\|\s+Jπ=(.+)$', line)
            if anchor_match and current_cluster is not None:
                current_cluster['anchor_id'] = anchor_match.group(1)
                current_cluster['anchor_energy'] = float(anchor_match.group(2))
                current_cluster['anchor_uncertainty'] = float(anchor_match.group(3))
                current_cluster['anchor_jpi'] = anchor_match.group(4)
                continue
            
            # Match member line: "[A] A_1000: E=1000.0±1.0 keV, Jπ=N/A (Anchor)"
            # or: "[B] B_1005: E=1005.0±3.0 keV, Jπ=N/A (Match Prob: 56.5%)"
            member_match = re.match(r'^\[(\w+)\]\s+(\S+):\s+E=([\d.]+)±([\d.]+)\s+keV,\s+Jπ=([^(]+)\s+\((.+)\)$', line)
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
                    # Extract probability: "Match Prob: 56.5%"
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

def spread_text_positions(energies, min_distance=250):
    """
    Adjusts text positions to ensure they are at least min_distance apart vertically.
    Similar to Level_Scheme_Visualizer.py collision resolution algorithm.
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

def plot_clustering_results():
    """
    Main plotting function: visualize clustering results with multi-column layout.
    """
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
    x_positions = {dataset: index * 2.5 for index, dataset in enumerate(datasets)}
    line_width = 0.8
    
    # Color palette for clusters (cycle through colors)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Track maximum energy for plot limits
    maximum_energy = 0
    
    # Process each cluster
    for cluster in clusters:
        cluster_number = cluster['cluster_number']
        cluster_color = colors[(cluster_number - 1) % len(colors)]
        
        # Organize members by dataset for this cluster
        members_by_dataset = {dataset: [] for dataset in datasets}
        for member in cluster['members']:
            members_by_dataset[member['dataset']].append(member)
            if member['energy'] > maximum_energy:
                maximum_energy = member['energy']
        
        # Draw levels and connecting lines for this cluster
        for dataset in datasets:
            dataset_members = members_by_dataset[dataset]
            if not dataset_members:
                continue
            
            x_center = x_positions[dataset]
            x_start = x_center - line_width / 2
            x_end = x_center + line_width / 2
            
            # Sort members by energy within this dataset
            dataset_members.sort(key=lambda m: m['energy'])
            
            # Calculate text positions with collision resolution
            energies = [m['energy'] for m in dataset_members]
            text_y_positions = spread_text_positions(energies, min_distance=250)
            
            # Draw each level
            for index, member in enumerate(dataset_members):
                energy = member['energy']
                y_text = text_y_positions[index]
                
                # Draw level line at true energy
                line_style = '-'
                line_thickness = 3.0 if member['is_anchor'] else 1.5
                axis.hlines(y=energy, xmin=x_start, xmax=x_end, 
                           colors=cluster_color, linewidth=line_thickness, linestyle=line_style)
                
                # Construct label text
                energy_label = f"{int(member['energy'])}({int(member['uncertainty'])})"
                jpi_label = member['jpi'] if member['jpi'] != 'N/A' else ''
                
                # Check if text is displaced significantly
                is_displaced = abs(energy - y_text) > 50
                
                if is_displaced:
                    # Draw connector line and text
                    axis.annotate(energy_label,
                                xy=(x_start, energy), xycoords='data',
                                xytext=(x_start - 0.2, y_text), textcoords='data',
                                arrowprops=dict(arrowstyle="-", color='gray', lw=0.8),
                                va='center', ha='right', fontsize=12)
                    if jpi_label:
                        axis.annotate(jpi_label,
                                    xy=(x_end, energy), xycoords='data',
                                    xytext=(x_end + 0.2, y_text), textcoords='data',
                                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.8),
                                    va='center', ha='left', fontsize=12)
                else:
                    # Standard text placement
                    axis.text(x_start - 0.1, y_text, energy_label, va='center', ha='right', fontsize=12)
                    if jpi_label:
                        axis.text(x_end + 0.1, y_text, jpi_label, va='center', ha='left', fontsize=12)
                
                # Add match probability for non-anchor members
                if not member['is_anchor'] and member['match_probability'] is not None:
                    probability_text = f"{member['match_probability']:.0%}"
                    axis.text(x_center, y_text, probability_text, va='center', ha='center',
                            fontsize=9, color='black', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
        
        # Draw connecting lines between cluster members across datasets
        if len(cluster['members']) > 1:
            # Get y-positions for each member
            member_positions = []
            for member in cluster['members']:
                x_center = x_positions[member['dataset']]
                member_positions.append((x_center, member['energy']))
            
            # Sort by x-position to draw connections left-to-right
            member_positions.sort(key=lambda p: p[0])
            
            # Draw dashed lines connecting members
            for index in range(len(member_positions) - 1):
                x1, y1 = member_positions[index]
                x2, y2 = member_positions[index + 1]
                axis.plot([x1 + line_width / 2, x2 - line_width / 2], [y1, y2],
                        color=cluster_color, linestyle='--', linewidth=1.0, alpha=0.5)
    
    # Styling
    x_limit_min = -1.5
    x_limit_max = (len(datasets) - 1) * 2.5 + 1.5
    axis.set_xlim(x_limit_min, x_limit_max)
    axis.set_ylim(0, maximum_energy * 1.15)
    
    x_tick_positions = [x_positions[dataset] for dataset in datasets]
    x_tick_labels = [f'Dataset {dataset}' for dataset in datasets]
    axis.set_xticks(x_tick_positions)
    axis.set_xticklabels(x_tick_labels, fontsize=14, fontweight='bold')
    
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_linewidth(1.5)
    
    axis.set_ylabel("Energy (keV)", fontsize=14)
    axis.tick_params(axis='x', length=0)
    axis.tick_params(axis='y', labelsize=12)
    
    axis.set_title(f"Clustering Results: {len(clusters)} Clusters", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = 'Clustering_Visualization.png'
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Clustering visualization saved to {output_file}")

if __name__ == "__main__":
    plot_clustering_results()
