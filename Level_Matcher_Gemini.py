import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import HistGradientBoostingClassifier

# =============================================================================
# 1. DATA INGESTION (Simulating ENSDF L-Records)
# =============================================================================
# Data is structured to mimic the information content of ENSDF "L-Records".
# Terminology:
#   NUCID: Nucleus ID
#   E:     Level Energy (keV)
#   DE:    Standard Uncertainty in Energy (keV)
#   J:     Spin-Parity Jpi (e.g., "3-", "5/2+") - corresponding to ENSDF Col 23-39
#   L:     Angular Momentum Transfer (ENSDF Col 56-64)
#   DS:    Dataset Source ID (e.g., "ADOPTED", "HI-SPIN", "P-TRANSFER")
# =============================================================================

data = []

# --- Dataset A: High-Resolution Gamma Spectroscopy (e.g., Adopted) ---
data.extend([
    # E=4059, DE=3
    {'NUCID': '35CL', 'E': 4059.0, 'DE': 3.0, 'J': np.nan, 'L': np.nan, 'DS': 'DS_A', 'ID': 'A_4059'},
    # E=4173, DE=2, J=3- (Definite assignment)
    {'NUCID': '35CL', 'E': 4173.0, 'DE': 2.0, 'J': 3.0,    'L': 3.0,    'DS': 'DS_A', 'ID': 'A_4173'},
    # E=4178, DE=2, J=2+ (Definite assignment)
    {'NUCID': '35CL', 'E': 4178.0, 'DE': 2.0, 'J': 2.0,    'L': 2.0,    'DS': 'DS_A', 'ID': 'A_4178'},
    {'NUCID': '35CL', 'E': 4347.0, 'DE': 4.0, 'J': np.nan, 'L': np.nan, 'DS': 'DS_A', 'ID': 'A_4347'}
])

# --- Dataset B: Particle Transfer Reaction (e.g., (p,t)) ---
# Note: Reaction selection rules may exclude certain L transfers.
data.extend([
    {'NUCID': '35CL', 'E': 4055.0, 'DE': 3.0, 'J': np.nan, 'L': np.nan, 'DS': 'DS_B', 'ID': 'B_4055'},
    # B_4174: This reaction explicitly excludes L=3 transfer (L_exclusion)
    # In a real parser, this might come from specific cross-section analysis.
    {'NUCID': '35CL', 'E': 4174.0, 'DE': 1.0, 'J': np.nan, 'L_excl': 3.0, 'DS': 'DS_B', 'ID': 'B_4174'},
    {'NUCID': '35CL', 'E': 4343.0, 'DE': 2.0, 'J': np.nan, 'L': np.nan,    'DS': 'DS_B', 'ID': 'B_4343'}
])

# --- Dataset C: Low-Resolution / Historical Data ---
# High uncertainties (DE=20), potentially containing unresolved multiplets.
data.extend([
    {'NUCID': '35CL', 'E': 2065.0, 'DE': 20.0, 'J': np.nan, 'L': np.nan, 'DS': 'DS_C', 'ID': 'C_2065'},
    {'NUCID': '35CL', 'E': 4170.0, 'DE': 20.0, 'J': np.nan, 'L': np.nan, 'DS': 'DS_C', 'ID': 'C_4170'},
    # Missing Uncertainty handled via np.nan
    {'NUCID': '35CL', 'E': 4350.0, 'DE': np.nan, 'J': np.nan, 'L': np.nan, 'DS': 'DS_C', 'ID': 'C_4350'}
])

df_levels = pd.DataFrame(data)

# =============================================================================
# 2. FEATURE ENGINEERING: Physics Compatibility Metrics
# =============================================================================

def calculate_compatibility_metrics(rec1, rec2):
    """
    Computes a feature vector representing the compatibility between two levels.
    
    Parameters:
    rec1, rec2: Series/Dict representing ENSDF L-records.
    
    Returns:
    list: [normalized_energy_diff, selection_rule_violation]
    """
    
    # --- Metric 1: Normalized Energy Residual (Z-Score) ---
    # Formula: |E1 - E2| / sqrt(DE1^2 + DE2^2)
    # If DE is missing (NaN), assume a conservative default (e.g., 10 keV) 
    # to prevent division by zero or loss of data.
    de1 = rec1['DE'] if pd.notna(rec1.get('DE')) else 10.0
    de2 = rec2['DE'] if pd.notna(rec2.get('DE')) else 10.0
    
    combined_uncertainty = np.sqrt(de1**2 + de2**2)
    energy_diff = abs(rec1['E'] - rec2['E'])
    
    # Avoid singularity if combined_uncertainty is 0 (unlikely in physical data)
    norm_diff = energy_diff / combined_uncertainty if combined_uncertainty > 0 else 100.0
    
    # --- Metric 2: Selection Rule Violation (Binary Flag) ---
    # Checks for hard physics incompatibilities (Spin J or Angular Momentum L).
    # 0 = Compatible / Insufficient Info
    # 1 = Incompatible (Violation)
    violation = 0
    
    # Check L-Transfer Exclusion (e.g., Target state cannot be populated by L=3)
    # Case: Record 1 has definite L, Record 2 excludes that L
    if pd.notna(rec1.get('L')) and pd.notna(rec2.get('L_excl')):
        if rec1['L'] == rec2['L_excl']: 
            violation = 1
            
    # Symmetric check
    if pd.notna(rec2.get('L')) and pd.notna(rec1.get('L_excl')):
        if rec2['L'] == rec1['L_excl']: 
            violation = 1
            
    # Further checks (e.g., Jpi matching) can be added here
            
    return [norm_diff, violation]

# =============================================================================
# 3. MODEL TRAINING: Synthetic Physics Supervision
# =============================================================================
# Since ground-truth matched catalogs are sparse, we train the classifier 
# using "Synthetic Supervision" based on known laws of physics and statistical probability.

X_synthetic = [] 
y_synthetic = [] # 1 = Association (Match), 0 = Non-Association

# -- Class 1: Valid Associations --
# Statistically consistent energies (Low Z-score) AND No Selection Rule Violations.
for z in [0.0, 0.5, 1.0, 2.0, 3.0]: # Up to 3 sigma is generally acceptable
    X_synthetic.append([z, 0])      # [norm_diff, violation]
    y_synthetic.append(1)

# -- Class 2: Energy Mismatches --
# Statistically inconsistent energies (> 4 sigma).
for z in [4.0, 6.0, 10.0, 50.0]:
    X_synthetic.append([z, 0])
    y_synthetic.append(0)

# -- Class 3: Physics Violations (The "Veto") --
# Even if energy match is perfect (Z=0), a selection rule violation 
# renders the match physically impossible.
for z in [0.0, 0.5, 1.0, 5.0]:
    X_synthetic.append([z, 1])
    y_synthetic.append(0)

# Initialize HistGradientBoostingClassifier
# Selected for:
# 1. Native handling of NaNs (missing spectroscopic data).
# 2. Ability to model non-linear decision boundaries (Violation flag overrides Energy).
classifier = HistGradientBoostingClassifier(
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
classifier.fit(X_synthetic, y_synthetic)

# =============================================================================
# 4. GRAPH TOPOLOGY CONSTRUCTION (The Reconciliation)
# =============================================================================
# We treat the Level Scheme as a Graph where:
# Nodes = Observed Levels in individual datasets
# Edges = Probabilistic Association based on ML score

level_graph = nx.Graph()
for uid in df_levels['ID']: 
    level_graph.add_node(uid)

# Pairwise comparison across datasets
# Note: In production, spatial indexing (KDTree) can optimize this for N > 1000
for i, rec_i in df_levels.iterrows():
    for j, rec_j in df_levels.iterrows():
        # Optimization: Only compare upper triangle and distinct datasets
        if i >= j: continue 
        if rec_i['DS'] == rec_j['DS']: continue 
        
        # Feature Extraction
        features = calculate_compatibility_metrics(rec_i, rec_j)
        
        # Inference: Probability of Physical Association
        # returns [prob_class_0, prob_class_1]
        association_prob = classifier.predict_proba([features])[0][1]
        
        # Thresholding: Establish Edge if probability exceeds confidence level
        # A threshold of 0.50 implies "More likely than not"
        if association_prob > 0.50:
            level_graph.add_edge(rec_i['ID'], rec_j['ID'], weight=association_prob)

# =============================================================================
# 5. UNIFIED LEVEL SCHEME GENERATION
# =============================================================================
# Connected components in the graph represent the consensus levels.
# This naturally handles:
# 1. One-to-One Matches (Ideal case)
# 2. Orphans (Unmatched levels)
# 3. Unresolved Multiplets (One level in low-res dataset matches two in high-res)

unified_levels = []

for component in nx.connected_components(level_graph):
    # Extract dataframe subset for this component
    comp_df = df_levels[df_levels['ID'].isin(component)]
    
    # --- Weighted Average Energy Calculation ---
    # Weight w_i = 1 / sigma_i^2
    # Handling NaNs in DE by assigning low weight (high uncertainty default)
    comp_uncertainties = comp_df['DE'].fillna(20.0) 
    weights = 1.0 / (comp_uncertainties**2)
    
    weighted_energy = (comp_df['E'] * weights).sum() / weights.sum()
    
    # --- Classification of the Group ---
    # If a single dataset contributes >1 level to this component, 
    # it indicates an "Unresolved Multiplet" or "Complex" structure 
    # relative to the lower-resolution datasets in the group.
    dataset_counts = comp_df['DS'].value_counts()
    
    if dataset_counts.max() > 1:
        structure_type = "Unresolved Multiplet / Complex"
    else:
        structure_type = "Single Level"
        
    unified_levels.append({
        'Unified_E (keV)': round(weighted_energy, 2),
        'Structure': structure_type,
        'N_Sources': len(component),
        'Constituent_IDs': list(comp_df['ID'])
    })

# =============================================================================
# 6. REPORTING
# =============================================================================
df_unified = pd.DataFrame(unified_levels).sort_values('Unified_E (keV)')

print("--- RECONCILED LEVEL SCHEME ---")
# Adjusting display options for clarity
pd.set_option('display.max_colwidth', None)
print(df_unified.to_string(index=False))

# Validation Logic Example for Spot-Checking (Manual Review Trigger)
# If Structure is Complex, flag for human evaluator review.
review_candidates = df_unified[df_unified['Structure'].str.contains("Multiplet")]
if not review_candidates.empty:
    print("\n[ATTENTION] The following levels require Evaluator review (Multiplets Detected):")
    print(review_candidates['Unified_E (keV)'].tolist())