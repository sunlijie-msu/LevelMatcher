import matplotlib.pyplot as plt
import numpy as np

def plot_subset_robust_mechanics():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Visualizing "Subset-Robust Statistical Similarity"', fontsize=16, fontweight='bold', y=1.05)

    # ==========================================
    # PLOT 1: Statistical Intensity Logic (The "Z-Score Bridge")
    # ==========================================
    ax1 = axes[0]
    
    # Data: Two peaks that differ physically (80 vs 100) but agree statistically
    # A: 80 +/- 10
    # B: 100 +/- 10
    # Z-Score = |100-80| / sqrt(10^2 + 10^2) = 20 / 14.1 = 1.41 (Consistent!)
    
    labels = ['Dataset A', 'Dataset B']
    means = [80, 100]
    errs = [10, 10]
    
    # Plot Error Bars
    ax1.errorbar(x=[0, 1], y=means, yerr=errs, fmt='o', capsize=10, elinewidth=3, markeredgewidth=2, color='black', markersize=8, label='Measurement ± Uncertainty')
    
    # 1. Standard Min Approach (The gray box)
    ax1.hlines(80, -0.5, 1.5, colors='gray', linestyles='--', label='Standard Min (80)')
    ax1.fill_between([-0.5, 1.5], 0, 80, color='gray', alpha=0.2)
    
    # 2. Statistical Average Approach (The green box)
    # Since Z < 2.0, the algorithm promotes the overlap to the Average (90)
    avg_val = 90
    ax1.hlines(avg_val, -0.5, 1.5, colors='green', linewidth=2, label='Statistical Match (Avg: 90)')
    ax1.fill_between([-0.5, 1.5], 80, 90, color='green', alpha=0.3, hatch='//', label='Recovered Signal')

    ax1.set_title("Logic 1: Statistical Intensity Consistency\n(80±10 vs 100±10)", fontweight='bold')
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(0, 130)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Gamma Intensity')
    ax1.legend(loc='upper left')
    
    ax1.text(0.5, 85, "Code sees Z < 2.0\nTreats as Fluctuation\nUses Average (90)", 
             ha='center', color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # ==========================================
    # PLOT 2: Subset Normalization (The "Denominator" Fix)
    # ==========================================
    ax2 = axes[1]
    
    # Scenario: "3 vs 20 Gammas"
    # Dataset A (Small): Total Intensity = 150
    # Dataset B (Large): Total Intensity = 1500 (Contains A + huge excess)
    # Intersection = 150 (Perfect overlap on the ones A has)
    
    bar_x = [0, 1]
    bar_labels = ['Standard Bray-Curtis', 'Subset-Robust (Ours)']
    
    # Math
    intersect = 150
    total_A = 150
    total_B = 1500
    
    # Standard: Denominator = Sum(A) + Sum(B) = 1650
    denom_standard = total_A + total_B
    score_standard = (2 * intersect) / denom_standard # ~0.18
    
    # Robust: Denominator = 2 * Min(A, B) = 300
    denom_robust = 2 * min(total_A, total_B)
    score_robust = (2 * intersect) / denom_robust # 1.0
    
    # Plotting the "Penalty" (Denominator size)
    ax2.bar(0, denom_standard, color='red', alpha=0.3, label='Denominator (Penalty)')
    ax2.bar(0, 2*intersect, color='green', label='Numerator (2 * Overlap)')
    
    ax2.bar(1, denom_robust, color='red', alpha=0.3)
    ax2.bar(1, 2*intersect, color='green')
    
    ax2.set_title("Logic 2: Subset Normalization\n(Handling Incomplete Data)", fontweight='bold')
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel('Score Components')
    
    # Text Annotations
    ax2.text(0, denom_standard + 50, f"Score: {score_standard:.2f}\n(Fail)", ha='center', color='red', fontweight='bold')
    ax2.text(1, denom_robust + 50, f"Score: {score_robust:.1f}\n(Perfect Match)", ha='center', color='green', fontweight='bold')
    
    ax2.annotate('Huge excess in B\ncounts against A', xy=(0, 1000), xytext=(0.4, 1200),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    ax2.annotate('Excess ignored\n(Only Min Total used)', xy=(1, 350), xytext=(1.4, 800),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_subset_robust_mechanics()