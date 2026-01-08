# Level Matcher

A physics-informed nuclear level matching tool employing XGBoost (eXtreme Gradient Boosting) regression with graph clustering techniques has been developed by the FRIB Nuclear Data Group.

This tool facilitates matching energy levels across experimental datasets, generating "Adopted Levels" and a cross-reference list with corresponding probabilistic confidence scores.

## Key Features

*   **Physics-Informed XGBoost Regressor:** Trained on synthetic physics constraints to predict match probabilities (0.0 to 1.0) for nuclear level pairs based on four engineered features: Energy Similarity, Spin Similarity, Parity Similarity, and Specificity.
*   **Monotonic Constraints:** All four features enforce monotonic increasing relationships (higher feature value → higher match probability), ensuring physics compliance.
*   **Rule-Based Graph Clustering:** Deterministic algorithm merges high-confidence matches into clusters while enforcing:
    *   **Dataset Uniqueness:** Maximum one level per dataset per cluster
    *   **Mutual Consistency:** All cluster members must be mutually compatible (clique-like structure)
    *   **Ambiguity Resolution:** Levels with poor resolution (large measurement uncertainty) can belong to multiple clusters when compatible with multiple well-resolved levels
*   **Anchor-Based Reporting:** The level with smallest energy uncertainty defines each cluster's reference physics properties.

## Architecture

*   **`Level_Matcher.py`**: Main application. Orchestrates model training, pairwise inference, graph clustering, and console output.
*   **`Feature_Engineer.py`**: Physics engine. Contains:
    *   `Scoring_Config`: Physics scoring parameters for energy, spin, and parity comparisons
    *   `load_levels_from_json()`: Data ingestion from ENSDF-format JSON files
    *   `calculate_energy_similarity()`: Gaussian kernel scoring based on Z-score
    *   `calculate_spin_similarity()`: Nuclear selection rule enforcement (0.0 veto for forbidden transitions)
    *   `calculate_parity_similarity()`: Parity conservation checking
    *   `extract_features()`: Constructs four-dimensional feature vectors for machine learning model
    *   `generate_synthetic_training_data()`: Generates 580+ synthetic training points encoding physics constraints across six scenarios
*   **`Dataset_Parser.py`**: Converts ENSDF evaluator log files to structured JSON format. Handles complex Jπ notation including ranges, lists, tentative assignments, and nested parentheses.
*   **`Level_Scheme_Visualizer.py`**: Generates publication-quality level scheme diagrams with automatic collision resolution for text labels.

## Configuration

Edit these parameters in `Level_Matcher.py`:

```python
pairwise_output_threshold = 0.01   # Minimum probability for writing level pairs to file (1%)
clustering_merge_threshold = 0.30  # Minimum probability for cluster merging (30%)
```

Adjust physics scoring in `Feature_Engineer.py` → `Scoring_Config` dictionary:

```python
Scoring_Config = {
    'Energy': {
        'Sigma_Scale': 0.1,              # Gaussian kernel width for energy similarity
        'Default_Uncertainty': 10.0      # Default energy uncertainty in keV
    },
    'Spin': {
        'Match_Firm': 1.0,               # Score for definite spin match
        'Match_Tentative': 0.9,          # Score for tentative spin match
        'Mismatch_Weak': 0.2,            # Score for weak spin mismatch (ΔJ = 1, any tentative)
        'Mismatch_Strong': 0.0,          # Score for strong spin mismatch (ΔJ = 1, both firm)
        'Veto': 0.0                      # Score for impossible transition (ΔJ > 1)
    },
    'Parity': {
        'Match_Firm': 1.0,               # Score for definite parity match
        'Match_Tentative': 0.9,          # Score for tentative parity match
        'Mismatch_Tentative': 0.2,       # Score for tentative parity mismatch
        'Mismatch_Firm': 0.0             # Score for definite parity mismatch
    },
    'General': {
        'Neutral_Score': 0.5             # Score when data is missing/unknown
    }
}
```

## Workflow

1.  **Data Ingestion:**
    *   Reads `test_dataset_A.json`, `test_dataset_B.json`, `test_dataset_C.json`
    *   Extracts: energy value, energy uncertainty, spin-parity list, spin-parity string
    *   Generates unique level identifiers (e.g., `A_1000`)

2.  **Model Training (Supervised Learning):**
    *   Generates 580+ synthetic training samples encoding physics rules across six scenarios (perfect matches, physics vetoes, energy mismatches, ambiguous physics, weak matches, random background)
    *   Trains XGBoost regressor with `objective='binary:logistic'`
    *   Enforces `monotone_constraints='(1, 1, 1, 1)'` on all four features
    *   Hyperparameters: `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`

3.  **Pairwise Inference:**
    *   Compares all cross-dataset level pairs (A vs B, A vs C, B vs C)
    *   Extracts four-dimensional feature vector for each pair
    *   Predicts match probability using trained model
    *   Writes results to `level_pairs_inference.txt` (pairs above `pairwise_output_threshold`)

4.  **Graph Clustering (Rule-Based Algorithm):**
    *   Initializes each level as singleton cluster
    *   Iterates through level pairs sorted by probability (highest first)
    *   Attempts cluster merging only if:
        *   Both levels' probabilities exceed `clustering_merge_threshold`
        *   No dataset overlap exists between clusters (enforces dataset uniqueness)
        *   All existing cluster members are mutually compatible
    *   Handles ambiguous matches via multi-cluster assignment (when dataset overlap prevents merging):
        *   If merge blocked by dataset uniqueness constraint, attempts to add individual level to other cluster
        *   Level can belong to multiple clusters simultaneously when compatible with all members
    *   Outputs final clustering results to console and file

5.  **Anchor Selection & Reporting:**
    *   Selects cluster member with smallest energy uncertainty as anchor
    *   Reports anchor energy, Jπ assignment, and member list with match probabilities

## Usage

### Basic Workflow

```bash
# 1. Run the level matcher
python Level_Matcher.py

# 2. View pairwise inference results
cat level_pairs_inference.txt

# 3. View clustering results
cat clustering_results.txt
```

### Data Preparation

Option A - Use existing test datasets:
```bash
# Files already present: test_dataset_{A,B,C}.json
python Level_Matcher.py
```

Option B - Convert evaluator log file:
```bash
# Create evaluatorInput.log with format:
# # Dataset A:
# E_level = 1000(3) keV; Jπ: unknown.
# E_level = 2000(5) keV; Jπ: 2+.

python Dataset_Parser.py evaluatorInput.log
python Level_Matcher.py
```

### Generate Level Scheme Visualization

```bash
python Level_Scheme_Visualizer.py
# Output: Level_Scheme_Visualization.png
```

## Physics Logic

### Energy Similarity
Gaussian kernel of Z-score:

$$\text{Energy Similarity} = \exp\left(-\sigma_{\text{scale}} \cdot Z^2\right)$$

where $Z = \frac{|E_1 - E_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}$

### Spin Similarity
Nuclear selection rules enforce:
- **Match (J₁ = J₂):** Score = 1.0 (firm) or 0.9 (tentative)
- **Adjacent (|J₁ - J₂| = 1):** Score = 0.0 (both firm, vetoed) or 0.2 (any tentative, weak)
- **Forbidden (|J₁ - J₂| > 1):** Score = 0.0 (absolute veto)

### Parity Similarity
Conservation rules:
- **Match (π₁ = π₂):** Score = 1.0 (firm) or 0.9 (tentative)
- **Mismatch (π₁ ≠ π₂):** Score = 0.0 (both firm, vetoed) or 0.2 (any tentative, weak)

### Specificity (Ambiguity Penalty)
Measures how unique a level's spin-parity assignment is:

$$\text{Specificity} = \frac{1}{\sqrt{\text{multiplicity}}}$$

The square root formula provides balanced penalization of ambiguous levels:
- multiplicity=1 (unique Jπ assignment) → Specificity = 1.0 (100%, no penalty)
- multiplicity=2 (two possible Jπ assignments) → Specificity = 0.71 (29% penalty)
- multiplicity=4 (four possible Jπ assignments) → Specificity = 0.50 (50% penalty)
- multiplicity=9 (nine possible Jπ assignments) → Specificity = 0.33 (67% penalty)

This naturally represents uncertainty growth in quantum measurements. Alternative formulas considered:
- Logarithmic 1/(1+log10(mult)): Too gentle (mult=9 → 51%, only 49% penalty)
- Reciprocal 1/mult: Too aggressive (mult=9 → 11%, 89% penalty)
- Linear tunable 1/(1+α*(mult-1)): Requires manual tuning of α parameter

### Feature Vector Structure
```python
[Energy_Similarity,      # 0.0 to 1.0 (Gaussian kernel)
 Spin_Similarity,        # 0.0 to 1.0 (physics-informed scoring)
 Parity_Similarity,      # 0.0 to 1.0 (physics-informed scoring)
 Specificity]            # 1.0 / sqrt(multiplicity), measures assignment uniqueness
```

All features are monotonic increasing: higher values → higher match probability.

**Note on removed features:** Earlier versions included Spin Certainty and Parity Certainty features, but these were found to be redundant. Tentativeness information is already encoded in the similarity scores (firm matches score 1.0, tentative matches score 0.9), making separate certainty features unnecessary.

## Output Files

*   **`level_pairs_inference.txt`**: All cross-dataset level pairs above `pairwise_output_threshold` with match probabilities and feature breakdowns
*   **`clustering_results.txt`**: Final clustering results with anchor information and member probabilities
*   **Console Output**: Real-time progress and summary statistics

## Technology Stack

*   **Machine Learning:** XGBoost 2.x (Gradient Boosting with monotonic constraints)
*   **Data Processing:** NumPy, Pandas
*   **Visualization:** Matplotlib (level schemes)
*   **Rationale for XGBoost:**
    *   Native handling of missing values (common in nuclear data)
    *   Level-wise tree growth (stable for small datasets, N < 500)
    *   Advanced regularization (L1/L2) prevents overfitting
    *   Monotonic constraint support enforces physics rules
    *   See `Technology_Hierarchy.md` for detailed justification

## Jπ Notation Support

`Dataset_Parser.py` handles complex spin-parity notation:

| Notation | Meaning | Example Output |
|----------|---------|----------------|
| `2+` | Definite J=2, positive parity | `twoTimesSpin=4, isTentativeSpin=False, parity='+', isTentativeParity=False` |
| `(2)+` | Tentative spin, definite parity | `twoTimesSpin=4, isTentativeSpin=True, parity='+', isTentativeParity=False` |
| `2(+)` | Definite spin, tentative parity | `twoTimesSpin=4, isTentativeSpin=False, parity='+', isTentativeParity=True` |
| `(2+)` | Both tentative | `twoTimesSpin=4, isTentativeSpin=True, parity='+', isTentativeParity=True` |
| `1:3` | Range J=1,2,3 | Three separate entries with `twoTimesSpin={2,4,6}` |
| `(1+,2+,3+)` | Multiple tentative options with shared parity | Three entries, all with `isTentativeSpin=True, parity='+', isTentativeParity=True` |
| `3/2,5/2(+)` | List where only last has parity | First entry no parity, second entry has `parity='+', isTentativeParity=True` |

## Future Work

Planned features for future versions (requires JSON schema extensions):

| Feature | Physics Value | Implementation Notes |
|---------|---------------|----------------------|
| **Gamma-Ray Branching Ratios** | Critical — Often the only way to distinguish close-lying states with identical Jπ | Compare decay intensity patterns using cosine similarity of branching ratio vectors |
| **Half-Life / Lifetime** | High — Orders of magnitude difference is a definitive veto (isomer vs ground state) | Gaussian similarity on log-scale lifetimes |
| **Band Assignment** | Medium — Useful for high-spin rotational states | Match levels belonging to the same rotational band structure |

## Dependencies

```bash
pip install xgboost numpy pandas matplotlib
```

Verify installation:
```bash
python Library_Verification.py
```

## Contact
For questions or support, contact the FRIB Nuclear Data Group at nucleardata@frib.msu.edu