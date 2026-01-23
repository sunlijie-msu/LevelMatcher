import matplotlib.pyplot as plt
import numpy as np

def plot_subset_robust_mechanics():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Gamma Decay Pattern Similarity: Three Test Cases', fontsize=18, fontweight='bold', y=1.02)

    # ==========================================
    # PLOT 1: Dataset A vs B (Good Match)
    # ==========================================
    ax1 = axes[0]
    
    # Dataset A: Level 2000±1 keV, 4 gammas [100, 80±10, 15±3, 35±5], total=230
    # Dataset B: Level 2005±2 keV, 2 gammas [85±10, 100], total=185
    # Matched pairs: (100 vs 85) → 92.5, (80 vs 100) → 90
    # Score = 182.5/185 = 0.986
    
    pairs_AB = ['γ1\n2000 vs 2005', 'γ2\n1400 vs 1405']
    A_vals = [100, 80]
    B_vals = [85, 100]
    overlaps_AB = [92.5, 90]
    
    x = np.arange(len(pairs_AB))
    width = 0.35
    
    ax1.bar(x - width/2, A_vals, width, label='Dataset A', color='#4472C4', alpha=0.9)
    ax1.bar(x + width/2, B_vals, width, label='Dataset B', color='#ED7D31', alpha=0.9)
    
    for i, overlap in enumerate(overlaps_AB):
        ax1.plot([i - width/2 - 0.05, i + width/2 + 0.05], [overlap, overlap], 
                'g-', linewidth=5, zorder=10)
        ax1.text(i, overlap + 7, f'{overlap:.1f}', ha='center', fontweight='bold', 
                fontsize=12, color='green')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs_AB, fontsize=11)
    ax1.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax1.set_title('A vs B: Good Match\nCompatible Intensities (Z<3σ)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', alpha=0.3)
    
    # Score box
    ax1.text(0.5, 95, 'Score = 182.5/185\n= 0.986', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=2))

    # ==========================================
    # PLOT 2: Dataset A vs C (Bad Match)
    # ==========================================
    ax2 = axes[1]
    
    # Dataset C: Level 1999±2 keV, 2 gammas [65±10, 100], total=165
    # Dataset A gammas matched: [15, 35]
    # Matched pairs: (15 vs 65) → 15, (35 vs 100) → 35
    # Score = 50/165 = 0.303
    
    pairs_AC = ['γ3\n900 vs 899', 'γ4\n600 vs 598']
    A_vals_2 = [15, 35]
    C_vals = [65, 100]
    overlaps_AC = [15, 35]
    
    x = np.arange(len(pairs_AC))
    
    ax2.bar(x - width/2, A_vals_2, width, label='Dataset A', color='#4472C4', alpha=0.9)
    ax2.bar(x + width/2, C_vals, width, label='Dataset C', color='#A5A5A5', alpha=0.9)
    
    for i, overlap in enumerate(overlaps_AC):
        ax2.plot([i - width/2 - 0.05, i + width/2 + 0.05], [overlap, overlap], 
                'r--', linewidth=4, zorder=10)
        ax2.text(i, overlap + 7, f'{overlap:.1f}', ha='center', fontweight='bold', 
                fontsize=12, color='red')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(pairs_AC, fontsize=11)
    ax2.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax2.set_title('A vs C: Bad Match\nInconsistent Intensities (Z≫3σ)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', alpha=0.3)
    
    # Score box
    ax2.text(0.5, 95, 'Score = 50/165\n= 0.303', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2))

    # ==========================================
    # PLOT 3: Dataset A vs D (Binary Mode)
    # ==========================================
    ax3 = axes[2]
    
    # Dataset D: Level 2002±2 keV, 2 gammas [899±2, 598±3], no intensity data
    # Dataset A gammas matched: [15, 35] (energy only)
    # Binary mode: 2 matched gammas / 2 total = 1.0
    
    pairs_AD = ['γ3\n900 vs 899', 'γ4\n600 vs 598']
    energy_matches = [1, 1]
    
    x = np.arange(len(pairs_AD))
    width_bin = 0.5
    
    ax3.bar(x, energy_matches, width_bin, color='#70AD47', alpha=0.9, label='Energy Match')
    
    for i in range(len(pairs_AD)):
        ax3.text(i, energy_matches[i] + 0.05, '✓', ha='center', fontsize=20, 
                fontweight='bold', color='green')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(pairs_AD, fontsize=11)
    ax3.set_ylabel('Match Status', fontsize=12, fontweight='bold')
    ax3.set_title('A vs D: Binary Mode\nEnergy Only (No Intensity Data)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.set_ylim(0, 1.3)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No Match', 'Match'], fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # Score box
    ax3.text(0.5, 1.15, 'Score = 2/2\n= 1.000', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=2))
    
    # Add note
    ax3.text(0.5, 0.5, 'Dataset D has no\nintensity data\n(energy-only matching)', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_subset_robust_mechanics()