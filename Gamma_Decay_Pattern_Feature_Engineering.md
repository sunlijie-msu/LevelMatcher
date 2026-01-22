# Feature Engineering Technical Report: Gamma Decay Pattern Similarity

## 1. Executive Summary
This report documents the mathematical formulation and selection process for the **Gamma Decay Pattern Similarity** algorithm. The objective is to quantify the likelihood that two sets of gamma-ray data represent the same physical nuclear level, accounting for experimental variations common in ENSDF datasets.

The selected methodology, **Subset-Robust Statistical Similarity**, replaces traditional vector metrics to specifically address data sparsity (incomplete datasets) and statistical fluctuations in detector efficiency.

## 2. Mathematical Evaluation of Candidate Methods

### A. Vector Space Model (Cosine Similarity)
**Status:** Rejected

**Formulation:**
Treats gamma intensities as vectors $\vec{A}$ and $\vec{B}$ in $n$-dimensional energy space.
$$ S_{\cos} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i} I_{A,i} I_{B,i}}{\sqrt{\sum_{i} I_{A,i}^2} \sqrt{\sum_{i} I_{B,i}^2}} $$

**Failure Analysis:**
1.  **Vector Length Bias:** In a "3 vs 20 gammas" scenario, the magnitude $\|\vec{B}\|$ (20 gammas) is significantly larger than $\|\vec{A}\|$ (3 gammas). Even if $\vec{A}$ is a perfect subset of $\vec{B}$, the denominator dilutes the score significantly below 1.0.
2.  **Rank Intolerance:** If detector efficiency causes the strongest peak to swap rank (e.g., Peak 1: 100$\to$90, Peak 2: 90$\to$100), the vector direction changes, penalizing the match despite physical consistency.

### B. Standard Bray-Curtis Similarity
**Status:** Rejected

**Formulation:**
Quantifies the shared abundance between two sites (datasets).
$$ S_{BC} = \frac{2 \sum_{i} \min(I_{A,i}, I_{B,i})}{\sum_{i} I_{A,i} + \sum_{i} I_{B,i}} $$

**Failure Analysis:**
1.  **The Denominator Problem:** The term $(\sum I_A + \sum I_B)$ penalizes incomplete datasets. If Dataset A (Total $I=150$) matches a subset of Dataset B (Total $I=1500$), the maximum possible score is $\approx \frac{2(150)}{1650} = 0.18$. A correct physics engine should score this as 1.0 (Subset Match).
2.  **Statistical Rigidity:** It strictly takes the minimum overlap. If $I_A = 80 \pm 10$ and $I_B = 100 \pm 10$, these are statistically identical ($Z < 2$). Standard BC takes 80, penalizing the match due to noise.

---

## 3. The Selected Solution: Subset-Robust Statistical Similarity

**Status:** Accepted

This algorithm modifies the Bray-Curtis approach to incorporate **Z-score hypothesis testing** and **Subset logic**.

### Component 1: Energy Matching via Z-Score
Matches are not defined by a fixed energy tolerance (e.g., 3 keV), but by statistical significance.
$$ Z_E = \frac{|E_A - E_B|}{\sqrt{\sigma_{E_A}^2 + \sigma_{E_B}^2}} $$
*   **Criterion:** Match accepted if $Z_E \le 3.0$ ($99.7\%$ confidence interval).

### Component 2: Intensity Overlap via Hypothesis Testing
For matched gammas, we test if the intensity difference is statistically significant.
$$ Z_I = \frac{|I_A - I_B|}{\sqrt{\sigma_{I_A}^2 + \sigma_{I_B}^2}} $$

The shared intensity $I_{shared}$ is calculated conditionally:
$$
I_{shared} = 
\begin{cases} 
\frac{I_A + I_B}{2} & \text{if } Z_I \le 2.0 \quad (\text{Consistent: noise fluctuation}) \\
\min(I_A, I_B) & \text{if } Z_I > 2.0 \quad (\text{Inconsistent: physical discrepancy})
\end{cases}
$$

### Component 3: Subset-Robust Normalization
To handle the "3 vs 20 gammas" case, we normalize by the **minimum** total intensity, representing the information content of the smaller dataset.

$$ S_{Robust} = \frac{\sum I_{shared}}{\min\left(\sum I_A, \sum I_B\right)} $$

*   **Outcome:** If Dataset A is fully contained within Dataset B, $\sum I_{shared} \approx \sum I_A$. The score becomes $\frac{\sum I_A}{\sum I_A} = 1.0$.

---

## 4. Python Implementation

```python
import numpy as np

# Default fallback uncertainties
DEFAULT_ENERGY_UNC = 1.0
DEFAULT_INTENSITY_REL_UNC = 0.10

def calculate_gamma_decay_pattern_similarity(gamma_decays_1, gamma_decays_2):
    """
    Calculates gamma decay pattern similarity using Subset-Robust Statistical Compatibility.
    
    Logic:
    1. Standardize inputs (normalize strongest gamma to 100.0).
    2. Match energies using Z-scores (Z < 3.0).
    3. Calculate shared intensity using statistical consistency (Z < 2.0 uses Average, else Min).
    4. Normalize final score by the MINIMUM total intensity to allow perfect subset matching.
    """
    # Guard: Missing data
    if not gamma_decays_1 or not gamma_decays_2:
        return 0.5 

    # --- Phase 1: Standardization ---
    def process_spectrum(raw_decays):
        clean_list = []
        has_intensity = False
        max_intensity = 0.0
        
        for g in raw_decays:
            e = float(g.get('energy', 0))
            if e <= 0: continue
            
            # Default Energy Uncertainty
            dE = float(g.get('energy_uncertainty', 0))
            if dE <= 0: dE = DEFAULT_ENERGY_UNC
            
            i = float(g.get('branching_ratio', 0))
            dI = float(g.get('intensity_uncertainty', 0))
            
            if i > 0:
                has_intensity = True
                # Default Intensity Uncertainty (10%)
                if dI <= 0: dI = max(0.5, i * DEFAULT_INTENSITY_REL_UNC)
                if i > max_intensity: max_intensity = i
            
            clean_list.append({'E': e, 'dE': dE, 'I': i, 'dI': dI})
            
        # Normalize to Strongest = 100.0
        if has_intensity and max_intensity > 0:
            scale = 100.0 / max_intensity
            for g in clean_list:
                g['I'] *= scale
                g['dI'] *= scale
                
        return clean_list, has_intensity

    list_A, has_int_A = process_spectrum(gamma_decays_1)
    list_B, has_int_B = process_spectrum(gamma_decays_2)
    
    if not list_A or not list_B: return 0.5

    # --- Phase 2: Helper Math ---
    def get_z_score(val1, err1, val2, err2):
        sigma = np.sqrt(err1**2 + err2**2)
        if sigma == 0: sigma = 1.0
        return abs(val1 - val2) / sigma

    Z_ENERGY_MATCH = 3.0        # 99.7% CI
    Z_INTENSITY_CONSISTENT = 2.0 # 95% CI

    # --- Phase 3: Intensity Mode ---
    if has_int_A and has_int_B:
        intersection_sum = 0.0
        matched_indices_B = set()
        
        for gA in list_A:
            best_match_idx = -1
            best_z_energy = 999.0
            
            # Greedy Search by Energy Z-Score
            for idx, gB in enumerate(list_B):
                if idx in matched_indices_B: continue
                
                z_E = get_z_score(gA['E'], gA['dE'], gB['E'], gB['dE'])
                
                if z_E <= Z_ENERGY_MATCH:
                    if z_E < best_z_energy:
                        best_z_energy = z_E
                        best_match_idx = idx
            
            if best_match_idx != -1:
                gB = list_B[best_match_idx]
                matched_indices_B.add(best_match_idx)
                
                # Statistical Intensity Logic
                z_I = get_z_score(gA['I'], gA['dI'], gB['I'], gB['dI'])
                
                if z_I <= Z_INTENSITY_CONSISTENT:
                    # Consistent: Use Average (Treat as statistical fluctuation)
                    overlap = (gA['I'] + gB['I']) / 2.0
                else:
                    # Inconsistent: Use Minimum (Penalize discrepancy)
                    overlap = min(gA['I'], gB['I'])
                
                intersection_sum += overlap

        # Subset Normalization
        sum_A = sum(g['I'] for g in list_A)
        sum_B = sum(g['I'] for g in list_B)
        
        if min(sum_A, sum_B) == 0: return 0.0
        return min(intersection_sum / min(sum_A, sum_B), 1.0)

    # --- Phase 4: Binary Mode (Energies Only) ---
    else:
        matches = 0
        matched_indices_B = set()
        for gA in list_A:
            for idx, gB in enumerate(list_B):
                if idx in matched_indices_B: continue
                z_E = get_z_score(gA['E'], gA['dE'], gB['E'], gB['dE'])
                if z_E <= Z_ENERGY_MATCH:
                    matches += 1
                    matched_indices_B.add(idx)
                    break
        
        min_len = min(len(list_A), len(list_B))
        if min_len == 0: return 0.0
        return float(matches) / float(min_len)
```