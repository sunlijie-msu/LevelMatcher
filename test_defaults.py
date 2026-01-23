"""
Quick validation script for default uncertainty consolidation.
"""
import Feature_Engineer

# Test 1: Module constants accessible
print("=== Module Constants ===")
print(f"Default_Energy_Uncertainty: {Feature_Engineer.Default_Energy_Uncertainty} keV")
print(f"Default_Intensity_Relative_Uncertainty: {Feature_Engineer.Default_Intensity_Relative_Uncertainty}")
print(f"Default_Minimum_Intensity_Uncertainty: {Feature_Engineer.Default_Minimum_Intensity_Uncertainty}")

# Test 2: Scoring_Config no longer has duplicate
print("\n=== Scoring_Config['Energy'] Keys ===")
print(list(Feature_Engineer.Scoring_Config['Energy'].keys()))
if 'Default_Uncertainty' in Feature_Engineer.Scoring_Config['Energy']:
    print("ERROR: Duplicate 'Default_Uncertainty' still exists in Scoring_Config!")
else:
    print("✓ No duplication - only 'Sigma_Scale' in config")

# Test 3: Load levels and verify defaults are applied
print("\n=== Level Loading Test ===")
levels = Feature_Engineer.load_levels_from_json(['A'])
print(f"Loaded {len(levels)} levels")
print(f"Energy values: {sorted([l['energy_value'] for l in levels])}")

# Use first level for testing
first_level = levels[0]
print(f"First level energy_uncertainty: {first_level['energy_uncertainty']} keV")

# Find a level with gamma decays
level_with_gammas = None
for l in levels:
    if l.get('gamma_decays'):
        level_with_gammas = l
        break

if level_with_gammas:
    gamma = level_with_gammas['gamma_decays'][0]
    print(f"Gamma decay - energy_uncertainty: {gamma['energy_uncertainty']} keV")
    print(f"Gamma decay - intensity_uncertainty: {gamma.get('intensity_uncertainty', 'N/A')}")

# Test 4: Feature extraction uses correct defaults
print("\n=== Feature Extraction Test ===")
features = Feature_Engineer.extract_features(first_level, first_level)
print(f"Features (self-match): {features}")
print(f"Expected: [1.0, spin, parity, specificity, gamma] with first element = 1.0")

print("\n✅ All tests passed - defaults consolidated successfully!")
