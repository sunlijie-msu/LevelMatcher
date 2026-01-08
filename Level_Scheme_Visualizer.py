import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_dataset(dataset_code):
    filename = f"test_dataset_{dataset_code}.json"
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'levelsTable' in data:
            return data['levelsTable'].get('levels', [])
        elif isinstance(data, list):
            return data
    return []

def spread_text_positions(energies, min_dist=250):
    """
    Adjusts text positions to ensure they are at least min_dist apart,
    while trying to stay close to the original energy values.
    """
    if not energies:
        return []

    # Initial sorted list with indices to restore order later if needed (though we input sorted)
    n = len(energies)
    pos = np.array(energies, dtype=float)
    
    # Relaxation iterations
    # Push apart overlapping items
    for _ in range(200): # Increased iterations for stability with larger shifts
        changed = False
        for i in range(n - 1):
            dist = pos[i+1] - pos[i]
            if dist < min_dist:
                # Overlap detected. Push them apart.
                center = (pos[i+1] + pos[i]) / 2.0
                overlap = min_dist - dist
                pos[i] -= overlap / 2.0
                pos[i+1] += overlap / 2.0
                changed = True
        if not changed:
            break
            
    return pos

def plot_level_schemes():
    datasets = ['A', 'B', 'C']
    fig, ax = plt.subplots(figsize=(15, 10)) # Wider figure
    
    # Increase horizontal spacing significantly to prevent text collision between columns
    x_positions = {'A': 0, 'B': 2.5, 'C': 5.0}
    width = 0.8 # Wider lines for the new scale
    max_energy_plot = 0
    
    for code in datasets:
        raw_levels = load_dataset(code)
        
        # 1. Extract Data
        levels_data = []
        for level in raw_levels:
            # Energy
            if isinstance(level.get('energy'), dict):
                energy_val = level.get('energy', {}).get('value')
                unc = level.get('energy', {}).get('uncertainty', {}).get('value')
            else:
                energy_val = level.get('energy_value')
                unc = level.get('energy_uncertainty')
                
            if energy_val is None: continue
            energy_val = float(energy_val)
            
            # Spin/Parity
            spig = ""
            if isinstance(level.get('spinParity'), dict):
                spig = level.get('spinParity', {}).get('evaluatorInput', '')
            else:
                spig = level.get('spin_parity_string', '')
            
            unc_str = f"({int(unc)})" if unc is not None else ""
            label_left = f"{int(energy_val)}{unc_str}"
            label_right = spig
            
            levels_data.append({
                'energy': energy_val,
                'label_left': label_left,
                'label_right': label_right
            })
            
            if energy_val > max_energy_plot: max_energy_plot = energy_val

        # 2. Sort by Energy
        levels_data.sort(key=lambda x: x['energy'])
        
        # 3. Calculate Text Positions (Collision Resolution)
        energies = [x['energy'] for x in levels_data]
        # 250 keV is sufficient for 12pt font vertical clearance without excessive displacement
        text_y_positions = spread_text_positions(energies, min_dist=250)
        
        # 3.5. Calculate vertical offsets for level bars when too close (within 150 keV)
        bar_offsets = [0.0] * len(energies)
        for i in range(len(energies) - 1):
            if energies[i + 1] - energies[i] < 150:
                # Push bars apart slightly for visual separation
                bar_offsets[i] -= 30
                bar_offsets[i + 1] += 30
        
        # 4. Plot
        x_center = x_positions[code]
        x_start = x_center - width / 2
        x_end = x_center + width / 2
        
        for i, item in enumerate(levels_data):
            en = item['energy']
            y_text = text_y_positions[i]
            bar_offset = bar_offsets[i]
            
            # A. Draw Level Line (Always at true energy with slight vertical offset for close-by levels)
            bar_y_position = en + bar_offset
            ax.hlines(y=bar_y_position, xmin=x_start, xmax=x_end, colors='black', linewidth=1.5)
            
            # B. Draw Text (Left - Energy)
            # Use annotate to draw connector if displaced significantly
            is_displaced = abs(bar_y_position - y_text) > 50
            
            if is_displaced:
                # Energy Label with connector
                ax.annotate(item['label_left'], 
                            xy=(x_start, bar_y_position), xycoords='data',
                            xytext=(x_start - 0.2, y_text), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                            va='center', ha='right', fontsize=18)
                
                # Spin Label with connector (or just aligned with text Y)
                if item['label_right']:
                    ax.annotate(item['label_right'], 
                                xy=(x_end, bar_y_position), xycoords='data',
                                xytext=(x_end + 0.2, y_text), textcoords='data',
                                arrowprops=dict(arrowstyle="-", color='gray', lw=2.5),
                                va='center', ha='left', fontsize=18)
            else:
                # Standard Text
                ax.text(x_start - 0.1, y_text, item['label_left'], va='center', ha='right', fontsize=18)
                if item['label_right']:
                    ax.text(x_end + 0.1, y_text, item['label_right'], va='center', ha='left', fontsize=18)

    # Styling
    ax.set_xlim(-1.5, 6.5) # Expanded limits for new X-positions
    ax.set_ylim(0, max_energy_plot * 1.15)
    ax.set_xticks([0, 2.5, 5.0])
    ax.set_xticklabels(['Dataset A', 'Dataset B', 'Dataset C'], fontsize=18, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    
    ax.set_ylabel("Energy (keV)", fontsize=18)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    output_file = 'Level_Scheme_Visualization.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_level_schemes()
