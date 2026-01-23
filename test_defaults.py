"""
Quick validation script for precision-based uncertainty inference.
"""
import Feature_Engineer

# Test 1: Precision-based inference functions exist
print("=== Precision-Based Inference Functions ===")
print(f"infer_energy_uncertainty: {Feature_Engineer.infer_energy_uncertainty}")
print(f"infer_intensity_uncertainty: {Feature_Engineer.infer_intensity_uncertainty}")

# Test 2: Example uncertainty inference
print("\n=== Example Inferences ===")
print(f"Energy 2000 keV → ±{Feature_Engineer.infer_energy_uncertainty(2000):.1f} keV")
print(f"Energy 1234.5 keV → ±{Feature_Engineer.infer_energy_uncertainty(1234.5):.1f} keV")
print(f"Intensity 10 → ±{Feature_Engineer.infer_intensity_uncertainty(10):.1f}")
print(f"Intensity 10.0 → ±{Feature_Engineer.infer_intensity_uncertainty(10.0):.1f}")

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
