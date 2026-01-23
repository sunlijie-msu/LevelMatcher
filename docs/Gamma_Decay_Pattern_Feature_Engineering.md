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
\frac{I_A + I_B}{2} & \text{if } Z_I \le 3.0 \quad (\text{Consistent: statistical fluctuation, averaging reduces error}) \\
\min(I_A, I_B) & \text{if } Z_I > 3.0 \quad (\text{Inconsistent: physical discrepancy, penalize via minimum})
\end{cases}
$$

### Component 3: Subset-Robust Normalization
To handle the "3 vs 20 gammas" case, we normalize by the **minimum** total intensity, representing the information content of the smaller dataset. This ensures that if a smaller dataset is a subset of a larger one, the similarity score can still reach 1.0.

$$ S_{Robust} = \frac{\sum I_{shared}}{\min\left(\sum I_A, \sum I_B\right)} $$

### Illustrative Examples

#### Example 1: A vs B (Partial Subset Match)
*   **Dataset A (Complete):** 5 gammas [2000, 1400, 900, 600, 300]. Total Intensity = 215.
*   **Dataset B (Partial):** 2 gammas [2005, 1405]. Total Intensity = 185.
*   **Outcome:** The two gammas in B match A's strongest lines within $Z < 3\sigma$. Intensities are consistent.
*   **Score:** $\approx 0.986$ (High). The algorithm correctly identifies B as a likely partial observation of A.

#### Example 2: A vs C (Intensity Mismatch)
*   **Dataset A (Complete):** 5 gammas [2000, 1400, 900, 600, 300].
*   **Dataset C (Candidate):** 2 gammas [899, 598] with high relative intensities [65, 100].
*   **Outcome:** Energy matches are valid (A_900↔C_899, A_600↔C_598), but intensities diverge ($Z \gg 3\sigma$). The algorithm falls back to taking the minimum intensity, severely penalizing the score.
*   **Score:** $\approx 0.182$ (Low). Correctly identifies that while energies align, the decay physics (branching ratios) do not match.

#### Example 3: A vs D (Binary Match - Energy Only)
*   **Dataset A (Complete):** 5 gammas.
*   **Dataset D (Candidate):** 2 gammas [902, 1400] with **NO** intensity data.
*   **Outcome:** Algorithm detects missing intensities and switches to Binary Mode. It counts energy matches only.
*   **Score:** 1.0 (2 matches out of 2 gammas in the smaller dataset).
