# Level Matcher

A physics-informed nuclear level matching system developed by the FRIB Nuclear Data Group. This tool employs dual machine learning models (XGBoost + LightGBM) with physics-informed feature engineering to match energy levels across experimental datasets, generating "Adopted Levels" with probabilistic confidence scores.

## Overview

Level Matcher identifies corresponding nuclear energy levels across different test datasets by combining:
- **Physics-Informed Feature Engineering**: Five-dimensional feature vectors encoding nuclear physics constraints
- **Dual-Model Comparison**: XGBoost (flexible baseline) + LightGBM (regularization-constrained validation)
- **Graph-Based Clustering**: Deterministic algorithm enforcing dataset uniqueness and mutual consistency
- **Ambiguity Resolution**: Multi-cluster membership support for poorly-resolved levels

## Key Features

### Machine Learning Models

**XGBoost (Primary Model)**
- Configuration: `binary:logistic`, 1000 estimators, max_depth=10, learning_rate=0.05
- Regularization: Light (reg_alpha=0.0, reg_lambda=1.0)
- Behavior: Flexible feature weighting based on training data patterns
- Purpose: Standard baseline model

**LightGBM (Comparison Model)**
- Configuration: `binary`, 500 estimators, max_depth=5, num_leaves=7, learning_rate=0.02
- Regularization: Heavy (reg_alpha=1.0, reg_lambda=10.0) + shallow trees
- Behavior: Constrained feature weighting; heavy regularization amplifies gamma-pattern importance when training data shows strong gamma-energy correlations
- Purpose: Validation tool - reveals when gamma patterns dominate predictions
- Example: A_2000↔B_2006 yields XGBoost 19.4% vs LightGBM 99.5% when gamma patterns perfectly align despite 6 keV energy mismatch

### Feature Engineering (5D Vector)

All features explicitly named for sklearn compatibility:
1. **Energy_Similarity**: Gaussian kernel based on Z-score (experimental uncertainties)
2. **Spin_Similarity**: Nuclear selection rules with forbidden transition vetoes (0.0 for violations)
3. **Parity_Similarity**: Parity conservation checking
4. **Specificity**: Multi-dataset validation confidence metric
5. **Gamma_Decay_Pattern_Similarity**: Cosine similarity of Gaussian-broadened gamma spectra

### Physics Rescue Mechanism

Formula: `effective_energy = energy_similarity ** rescue_exponent` (default exponent=0.5, implements sqrt transformation)

**Trigger Conditions** (threshold=0.85):
- `(spin_similarity >= 0.85 AND parity_similarity >= 0.85) OR gamma_similarity >= 0.85`

**Behavior**: Replaces raw energy similarity in final probability calculation when quantum numbers or gamma patterns provide strong evidence despite moderate energy mismatch. This prevents rejection of valid matches where measurement uncertainties cause apparent energy discrepancies.

### Graph Clustering Algorithm

**Key Constraints**:
- **Dataset Uniqueness**: Maximum one level per dataset per cluster
- **Mutual Consistency**: All members must be pairwise compatible (clique structure)
- **Ambiguity Resolution**: Levels with large uncertainties can belong to multiple clusters

**Anchor Selection**: Level with smallest energy uncertainty defines cluster reference properties

## Project Structure

```
LevelMatcher/
├── data/
│   └── raw/                    # Input JSON datasets
│       ├── test_dataset_A.json
│       ├── test_dataset_B.json
│       └── test_dataset_C.json
│
├── outputs/
│   ├── clustering/             # Clustering results (text files)
│   │   ├── Output_Clustering_Results_XGB.txt
│   │   └── Output_Clustering_Results_LightGBM.txt
│   ├── figures/                # Visualization outputs (PNG files)
│   │   ├── Input_Level_Scheme.png
│   │   ├── Output_Cluster_Scheme_XGB.png
│   │   └── Output_Cluster_Scheme_LightGBM.png
│   └── pairwise/              # Pairwise inference results
│       └── Output_Level_Pairwise_Inference.txt
│
├── docs/                       # Documentation and reports
│   ├── reports/
│   │   └── Hyperparameter_Analysis_Report.md
│   ├── Gamma_Decay_Pattern_Feature_Engineering.md
│   ├── Gamma_Decay_Pattern_Feature_Engineering.html
│   ├── Technology_Hierarchy.md
│   └── Technology_Hierarchy.pdf
│
├── scripts/
│   ├── hyperparameter_tuning/ # Hyperparameter optimization utilities
│   │   ├── Hyperparameter_Tuner.py
│   │   ├── Hyperparameter_Visualizer.py
│   │   └── Output_Hyperparameter_*.txt
│   └── legacy/                # Experimental/deprecated code
│       ├── Level_Matcher_Legacy.py
│       ├── Shared_Area_Model.py
│       ├── Subset_Robust_Model.py
│       ├── Vector_Space_Model.py
│       ├── Library_Verification.py
│       └── ai_studio_code.py
│
├── Level_Matcher.py           # Main application (training, inference, clustering)
├── Feature_Engineer.py        # Physics engine (feature extraction, synthetic data)
├── Dataset_Parser.py          # ENSDF log to JSON converter
├── Combined_Visualizer.py     # Level scheme and clustering visualizations
├── copy_for_notebookLM.py     # Utility: Convert .py to .txt for NotebookLM
├── data/
│   └── raw/                   # Input datasets (dataset_A.json, etc.)
├── docs/                      # Documentation and reports
├── outputs/
│   ├── clustering/            # Clustering results (XGBoost, LightGBM)
│   ├── pairwise/              # Pairwise inference results
│   └── figures/               # Generated visualizations
├── scripts/
│   ├── hyperparameter_tuning/ # Hyperparameter optimization tools
│   └── legacy/                # Archive of deprecated scripts
├── README.md                  # This file
└── .gitignore                 # Git exclusions (outputs, cache)
```

## Core Modules

### Level_Matcher.py
**Purpose**: Main orchestration engine

**Workflow**:
1. **Data Loading**: Import levels from `data/raw/dataset_*.json`
2. **Training**: Generate 580+ synthetic training samples, train both models (XGBoost + LightGBM)
3. **Pairwise Inference**: Calculate match probabilities for all cross-dataset level pairs
4. **Clustering**: Execute graph-based clustering independently for each model
5. **Output Generation**:
   - `outputs/pairwise/Output_Level_Pairwise_Inference.txt`: Side-by-side model comparison
   - `outputs/clustering/Output_Clustering_Results_XGB.txt`: XGBoost clusters
   - `outputs/clustering/Output_Clustering_Results_LightGBM.txt`: LightGBM clusters

**Configuration Parameters**:
```python
pairwise_output_threshold = 0.001  # Minimum probability for pairwise output (0.1%)
clustering_merge_threshold = 0.15  # Minimum probability for cluster merging (15%)
```

**Feature Names** (explicitly defined for sklearn compatibility):
```python
feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                 'Specificity', 'Gamma_Decay_Pattern_Similarity']
```

**DataFrame Implementation**: All model training and predictions use `pd.DataFrame(features, columns=feature_names)` to eliminate sklearn feature name warnings.

### Feature_Engineer.py
**Purpose**: Physics-informed feature extraction and synthetic training data generation

**Key Functions**:
- `load_levels_from_json()`: Data ingestion from `data/raw/` directory
- `calculate_energy_similarity()`: Gaussian kernel scoring (Z-score based)
- `calculate_spin_similarity()`: Selection rule enforcement (0.0 veto for forbidden transitions)
- `calculate_parity_similarity()`: Parity conservation validation
- `calculate_gamma_decay_pattern_similarity()`: Cosine similarity of Gaussian-broadened spectra
- `extract_features()`: Constructs 5D feature vectors
- `generate_synthetic_training_data()`: Generates 580+ synthetic points across six physics scenarios

**Physics Rescue Implementation** (lines 771-786):
```python
if ((spin_similarity >= physics_rescue_threshold and parity_similarity >= physics_rescue_threshold) or 
    gamma_decay_pattern_similarity >= physics_rescue_threshold):
    effective_energy = energy_similarity ** physics_rescue_exponent  # Default: sqrt(energy_similarity)
    probability = effective_energy * physics_confidence * specificity
```

**Configuration**: `Scoring_Config` dictionary contains physics parameters
```python
Scoring_Config = {
    'Energy': {
        'Sigma_Scale': 0.1,              # Gaussian kernel width
        'Default_Uncertainty': 10.0      # Default uncertainty (keV)
    },
    'Spin': {
        'Match_Firm': 1.0,               # Definite spin match score
        'Match_Tentative': 0.85,         # Tentative match score
        # ... (see file for complete configuration)
    }
}
        'Match_Strong': 0.8,             # Score for tentative spin match
        'Mismatch_Weak': 0.2,            # Score for weak spin mismatch (ΔJ = 1, any tentative)
        'Mismatch_Strong': 0.05,         # Score for strong spin mismatch (ΔJ = 1, both firm)
        'Mismatch_Firm': 0.0             # Score for impossible transition (ΔJ ≥ 2)
    },
    'Parity': {
        'Match_Firm': 1.0,               # Score for definite parity match
        'Match_Strong': 0.8,             # Score for tentative parity match
        'Mismatch_Weak': 0.05,           # Score for tentative parity mismatch
        'Mismatch_Firm': 0.0             # Score for definite parity mismatch
    },
    'Specificity': {
        'Formula': 'sqrt',               # Options: 'sqrt', 'linear', 'log', 'tunable'
        'Alpha': 0.5                     # Only for 'tunable': penalty steepness parameter
    },
    'Feature_Correlation': {
        'Enabled': True,                 # Enable physics rescue for perfect spin+parity
        'Threshold': 0.85,               # Minimum similarity to trigger rescue
        'Rescue_Exponent': 0.5           # Energy boost: e → e^0.5 (sqrt transformation)
    },
    'General': {
        'Neutral_Score': 0.5             # Score when data is missing/unknown
    }
}
```

## Workflow

1.  **Data Ingestion:**
    *   Reads `data/raw/test_dataset_A.json`, `data/raw/test_dataset_B.json`, `data/raw/test_dataset_C.json`
    *   Extracts: energy value, energy uncertainty, spin-parity list, spin-parity string
    *   Generates unique level identifiers (e.g., `A_1000`)

2.  **Model Training (Supervised Learning):**
    *   Generates 580+ synthetic training samples encoding physics rules across six scenarios (perfect matches, physics vetoes, energy mismatches, ambiguous physics, weak matches, random background)
    *   Implements **Feature Correlation** ("Physics Rescue"):
        *   **Condition**: Spin+Parity ≥ 0.85 OR Gamma Pattern ≥ 0.85.
        *   **Action**: Boosts energy similarity using `energy^Rescue_Exponent` (soft rescue).
        *   **Rationale**: Validates matches where internal structure is identical but energy calibration differs.
        *   **Effect**: Rescue != Firm Match. It prevents invalidation (0 $\to$ 15-20%) rather than forcing a high probability match, preserving the energy disagreement penalty while keeping the candidate alive.
    *   Trains XGBoost regressor with `objective='binary:logistic'`
    *   Enforces `monotone_constraints='(1, 1, 1, 1, 1)'` on all five features
    *   Hyperparameters: `n_estimators=1000`, `max_depth=10`, `learning_rate=0.05`

3.  **Pairwise Inference:**
    *   Compares all cross-dataset level pairs (A vs B, A vs C, B vs C)
    *   Extracts five-dimensional feature vector for each pair
    *   Predicts match probability using trained model
    *   Writes results to `Output_Level_Pairwise_Inference.txt` (pairs above `pairwise_output_threshold`)

4.  **Graph Clustering (Rule-Based Algorithm):**
    *   Initializes each level as singleton cluster
    *   Iterates through level pairs sorted by probability (highest first)
    *   Attempts cluster merging with four-tier fallback logic:
        1. **Scenario A - No Dataset Overlap:** Verifies all-to-all compatibility, then merges clusters
        2. **Scenario B - Dataset Overlap:** Attempts multi-cluster assignment for ambiguous levels
        3. **Singleton Expansion:** Adds partner to existing singleton cluster when merge fails
        4. **New Cluster Creation:** Creates independent two-member cluster as last resort
    *   Merge constraints:
        *   Both levels' probabilities exceed `clustering_merge_threshold`
        *   No dataset overlap in merged cluster (enforces dataset uniqueness)
        *   All members mutually compatible (clique constraint)
    *   Ambiguity handling: Poorly-resolved levels can belong to multiple clusters when compatible with all members
    *   Outputs final clustering results to console and file

5.  **Anchor Selection & Reporting:**
    *   Selects cluster member with smallest energy uncertainty as anchor
    *   Reports anchor energy, Jπ assignment, and member list with match probabilities

## Usage

### Basic Workflow
```

### Dataset_Parser.py
**Purpose**: Convert ENSDF evaluator log files to structured JSON format

**Capabilities**:
- Handles complex Jπ notation: ranges (`1:3`), lists (`1,2,3`), tentative assignments (`(2)+`), nested parentheses
- Parses energy values with uncertainties
- Outputs to `data/raw/test_dataset_*.json`

**Usage**:
```bash
python Dataset_Parser.py evaluatorInput.log
```

### Combined_Visualizer.py
**Purpose**: Generate level scheme and clustering visualizations

**Outputs**:
- `outputs/figures/Input_Level_Scheme.png`: Input datasets (all three datasets side-by-side)
- `outputs/figures/Output_Cluster_Scheme_XGB.png`: XGBoost clustering results
- `outputs/figures/Output_Cluster_Scheme_LightGBM.png`: LightGBM clustering results

**Visual Quality**: High-resolution (300 DPI), generous margins, no overlapping text

## Usage

### Basic Workflow

```bash
# 1. Run the level matcher (training + inference + clustering)
python Level_Matcher.py

# 2. Generate visualizations
python Combined_Visualizer.py

# 3. View pairwise inference results (model comparison)
cat outputs/pairwise/Output_Level_Pairwise_Inference.txt

# 4. View XGBoost clustering results
cat outputs/clustering/Output_Clustering_Results_XGB.txt

# 5. View LightGBM clustering results  
cat outputs/clustering/Output_Clustering_Results_LightGBM.txt
```

### Data Preparation

**Option A** - Use existing test datasets:
```bash
# Files already present in data/raw/: test_dataset_{A,B,C}.json
python Level_Matcher.py
```

**Option B** - Convert evaluator log file:
```bash
# Create evaluatorInput.log with format:
# # Dataset A:
# E_level = 1000(3) keV; Jπ: unknown.
# E_level = 2000(2) keV; Jπ: 2+.
# # Dataset B:
# E_level = 1003(5) keV; Jπ: unknown.

python Dataset_Parser.py evaluatorInput.log
# Outputs: data/raw/test_dataset_A.json, data/raw/test_dataset_B.json, ...
```

### Hyperparameter Tuning

Located in `scripts/hyperparameter_tuning/`:
```bash
cd scripts/hyperparameter_tuning

# Run hyperparameter exploration
python Hyperparameter_Tuner.py

# Visualize results
python Hyperparameter_Visualizer.py

# View detailed analysis
cat ../docs/reports/Hyperparameter_Analysis_Report.md
```

## Output Files

### Pairwise Inference
**File**: `outputs/pairwise/Output_Level_Pairwise_Inference.txt`

**Format**:
```
A_1000 <-> B_1003 | XGB: 85.3% | LGBM: 87.2%
  Features: Energy_Sim=0.95, Spin_Sim=0.80, Parity_Sim=1.00, Specificity=0.75, Gamma_Pattern_Sim=0.92
```

**Content**: All level pairs exceeding `pairwise_output_threshold` (default 0.1%), sorted by XGBoost probability, showing side-by-side model comparison.

### Clustering Results
**Files**: 
- `outputs/clustering/Output_Clustering_Results_XGB.txt`
- `outputs/clustering/Output_Clustering_Results_LightGBM.txt`

**Format**:
```
Cluster 1:
  Anchor: A_0 | E=0.0±0.0 keV | Jπ=0+
  Members:
    [A] A_0: E=0.0±0.0 keV, Jπ=0+ (Anchor)
    [B] B_0: E=0.0±0.0 keV, Jπ=0+ (Match Prob: 94.7%)
    [C] C_0: E=0.0±0.0 keV, Jπ=0+ (Match Prob: 94.7%)
```

**Content**: Clusters sorted by average energy, anchor defined as level with smallest uncertainty, member probabilities calculated relative to anchor.

## Configuration

### Level_Matcher.py Thresholds

```python
pairwise_output_threshold = 0.001  # Minimum probability for pairwise output (0.1%)
clustering_merge_threshold = 0.15  # Minimum probability for cluster merging (15%)
```

**Recommendations**:
- `pairwise_output_threshold`: Lower values (0.001-0.01) capture weak candidates for manual review
- `clustering_merge_threshold`: Higher values (0.15-0.30) enforce stricter cluster quality

### Feature_Engineer.py Physics Parameters

```python
Scoring_Config = {
    'Energy': {
        'Sigma_Scale': 0.1,              # Gaussian kernel width
        'Default_Uncertainty': 10.0      # Default uncertainty (keV)
    },
    'Spin': {
        'Match_Firm': 1.0,               # Definite spin match
        'Match_Tentative': 0.85,         # Tentative match
        'Conflict_Firm': 0.0,            # Forbidden transition (veto)
        'Unknown_Penalty': 0.6           # Unknown spin handling
    },
    'Parity': {
        'Match': 1.0,                    # Parity conservation
        'Conflict': 0.0,                 # Parity violation (veto)
        'Unknown_Penalty': 0.8           # Unknown parity handling
    },
    'Physics_Rescue': {
        'Threshold': 0.85,               # Trigger threshold
        'Exponent': 0.5                  # Energy transformation (sqrt)
    }
}
```

### Model Hyperparameters

**XGBoost** (Level_Matcher.py, line ~336):
```python
level_matcher_model_xgb = xgb.XGBRegressor(
    objective='binary:logistic',
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist'
)
```

**LightGBM** (Level_Matcher.py, line ~360):
```python
level_matcher_model_lgbm = lgb.LGBMRegressor(
    objective='binary',
    n_estimators=1000,
    num_leaves=7,            # Heavily regularized
    max_depth=5,             # Shallow trees
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=1.0,           # L1 regularization
    reg_lambda=10.0,         # L2 regularization (strong)
    random_state=42,
    verbose=-1
)
```

## Dependencies

```
python >= 3.8
pandas
numpy
xgboost
lightgbm
matplotlib
scikit-learn
```

Install via:
```bash
pip install pandas numpy xgboost lightgbm matplotlib scikit-learn
```

## Testing and Validation

### Mandatory Testing After Code Changes
Per project guidelines, **all code modifications must be immediately tested**:

```bash
# Test full pipeline
python Level_Matcher.py

# Verify outputs exist
ls outputs/clustering/Output_Clustering_Results_XGB.txt
ls outputs/clustering/Output_Clustering_Results_LightGBM.txt
ls outputs/pairwise/Output_Level_Pairwise_Inference.txt

# Test visualizations
python Combined_Visualizer.py

# Verify figures generated
ls outputs/figures/Input_Level_Scheme.png
ls outputs/figures/Output_Cluster_Scheme_XGB.png
ls outputs/figures/Output_Cluster_Scheme_LightGBM.png
```

### Validation Checklist
- ✅ No Python errors or warnings during execution
- ✅ All output files generated in correct locations
- ✅ Clustering results contain expected number of clusters
- ✅ Visualizations display without overlapping text
- ✅ Feature names eliminate sklearn warnings

## Known Behavior and Model Differences

### XGBoost vs LightGBM Divergence

**Expected Behavior**: Models produce different match probabilities due to different optimization objectives:
- **XGBoost**: Balanced feature weighting, energy-dominant
- **LightGBM**: Heavily regularized, gamma-pattern-aware

**Example Case** (A_2000 ↔ B_2006):
- Energy mismatch: 6 keV
- Gamma decay pattern: Perfect alignment
- XGBoost probability: 19.4% (low due to energy mismatch)
- LightGBM probability: 99.5% (high due to perfect gamma pattern)

**Interpretation**: LightGBM's strong regularization and shallow trees favor perfect feature matches (gamma patterns) over energy proximity. This divergence is intentional - LightGBM serves as a validation tool to flag ambiguous cases where gamma patterns provide strong evidence despite energy discrepancies.

### Physics Rescue Activation

When physics rescue activates (`spin >= 0.85 AND parity >= 0.85` OR `gamma >= 0.85`):
- Energy similarity undergoes sqrt transformation: `effective_energy = energy_similarity ** 0.5`
- Example: `energy_similarity = 0.49` → `effective_energy = 0.70` (44% boost)
- This prevents rejection of valid matches where quantum numbers are certain but energy measurements have large uncertainties

## Development Practices

### Code Standards
Per `.github/copilot-instructions.md`:
- **No acronyms**: Use `tentative` not `tent`, `error` not `err`
- **No ALL CAPS**: Use `Title_Case` or `snake_case`, never `ALL_CAPS_VARIABLES`
- **Self-explanatory naming**: Use `calculate_` prefix for math/logic functions
- **Educational comments**: Preserve and update all high-level strategy explanations
- **Clean codebase**: Remove redundant/obsolete files immediately

### Testing Requirements
- **Zero tolerance for untested changes**: All modifications must be runtime-validated
- **Full pipeline testing**: After edits affecting dependent modules, test end-to-end workflow
- **Visual verification**: For plotting code, ensure no overlapping text or illegible fonts

### Documentation Requirements
- **Update docstrings immediately**: Code changes must reflect in documentation on same commit
- **Professional terminology**: Use consistent physics vocabulary across all comments
- **Compliance checklist**: Verify adherence to `.github/copilot-instructions.md` before committing

## Troubleshooting

### "X does not have valid feature names" Warning
**Fixed** in current version via explicit feature naming:
```python
feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                 'Specificity', 'Gamma_Decay_Pattern_Similarity']
training_dataframe = pd.DataFrame(training_features, columns=feature_names)
```
If warnings persist: Delete `__pycache__/` directory and re-run.

### File Not Found Errors
Ensure correct directory structure:
- Input data: `data/raw/test_dataset_*.json`
- Outputs created automatically in `outputs/` subdirectories

### Empty Clustering Results
Check `clustering_merge_threshold` - if set too high (>0.50), no level pairs may qualify for merging. Recommended range: 0.15-0.30.

### Model Probability Divergence > 50%
This is expected behavior for cases with:
- Perfect gamma pattern + moderate energy mismatch
- Strong quantum number evidence + poor energy resolution

Use LightGBM output as validation tool, not replacement for XGBoost baseline.

## References

- FRIB Nuclear Data Group: https://groups.nscl.msu.edu/frib_decay/
- ENSDF Format Specification: https://www.nndc.bnl.gov/ensdf/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- LightGBM Documentation: https://lightgbm.readthedocs.io/

## License

Developed by FRIB Nuclear Data Group. For research and educational purposes.

## Contact

For questions or collaboration: FRIB Nuclear Data Group @ Michigan State University

---

**Last Updated**: January 2026  
**Version**: 2.0 (Dual-Model Architecture with Physics Rescue)  
**Project Status**: Active Development
# E_level = 2000(5) keV; Jπ: 2+.

python Dataset_Parser.py evaluatorInput.log
python Level_Matcher.py
```

### Generate Visualizations

```bash
python Combined_Visualizer.py
# Outputs: 
# 1. Input_Level_Scheme.png
# 2. Output_Cluster_Scheme.png
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

*   **`Output_Level_Pairwise_Inference.txt`**: All cross-dataset level pairs above `pairwise_output_threshold` with match probabilities and feature breakdowns
*   **`Output_Clustering_Results.txt`**: Final clustering results with anchor information and member probabilities
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