# Precision-Based Uncertainty Inference - Architecture Refactor

## Summary

Successfully moved precision-based uncertainty inference from **Feature_Engineer.py** (wrong module) to **Dataset_Parser.py** (correct module), following user's architectural critique.

## Rationale

**User's Core Insight:**
> "This is terrible bad idea. You should modify the parser code instead of the Feature Engineer code."

**Problem with Old Architecture:**
- ENSDF text → JSON (precision lost) → Feature_Engineer.py infers from evaluatorInput
- JSON float parsing converts all numbers to binary floats, losing trailing zeros
- Scientific notation "2.00E3" vs "2.0E3" becomes indistinguishable after JSON parsing

**Correct Architecture:**
- ENSDF text → Dataset_Parser.py infers from energy_str → JSON with complete uncertainties
- Parser has access to original ENSDF strings BEFORE precision loss
- Feature_Engineer.py trusts parser output (no re-inference)

## Changes Made

### 1. Dataset_Parser.py (NEW Inference Logic)

**Added `infer_uncertainty_from_precision()` function:**
```python
def infer_uncertainty_from_precision(value_str):
    """
    Infers uncertainty from reported precision in ENSDF strings.
    
    Nuclear physics convention: Uncertainty = 0.5 × least_significant_digit_place_value
    
    Examples:
    - "2000" → ±5 keV (integer, precision to 10s place)
    - "2.0E3" → ±500 keV (1 decimal in mantissa)
    - "2.00E3" → ±50 keV (2 decimals in mantissa)
    - "1234.5" → ±0.5 keV (1 decimal place)
    """
```

**Modified `parse_ensdf_line()` to use inference:**
- When `unc_str` is empty (no explicit uncertainty), calls `infer_uncertainty_from_precision(energy_str)`
- When `unc_str` exists, uses `calculate_absolute_uncertainty()` (explicit value)
- JSON output includes `"type": "inferred"` vs `"type": "symmetric"` to distinguish

**Updated uncertainty structure in JSON:**
```python
# OLD: {"type": "unreported"} when missing
# NEW: {"value": 50.0, "type": "inferred"} when inferred
"uncertainty": { 
    "value": unc_val, 
    "type": "symmetric" if unc_str else "inferred"
}
```

### 2. Feature_Engineer.py (REMOVED Complex Inference)

**Deleted functions:**
- `infer_from_evaluator_input()` (119 lines) - now in Dataset_Parser.py
- `infer_energy_uncertainty()` (47 lines) - replaced with simple fallback
- `infer_intensity_uncertainty()` (25 lines) - replaced with simple fallback

**Simplified fallbacks for edge cases:**
```python
def infer_energy_uncertainty_simple(energy_value):
    """Conservative fallback: 5 keV for edge cases"""
    return 5.0

def infer_intensity_uncertainty_simple(intensity_value):
    """Conservative fallback: 0.5 for edge cases"""
    return 0.5
```

**Updated documentation:**
- Removed detailed precision inference explanation (now in parser docs)
- Added note: "Dataset_Parser.py infers uncertainties during ENSDF parsing"
- Clarified: "Feature_Engineer trusts parser output; fallbacks only for edge cases"

**Updated all usages:**
- `load_levels_from_json()`: Changed `infer_energy_uncertainty(...)` → simple `5.0` fallback
- `normalize_gamma_intensities()`: Removed complex inference loops
- All locations now trust JSON data from parser

### 3. Tests

**Created `test_parser_inference.py`:**
- Tests all precision cases: integer, decimals, scientific notation
- Verifies: 2000→±5, 2000.0→±0.5, 2.00E3→±50, 2.0E3→±500
- **Result:** All 8 tests pass ✓

**Updated `test_fallback_verification.py`:**
- Part 1: Tests Dataset_Parser.infer_uncertainty_from_precision()
- Part 2: Verifies precision-level handling
- Part 3: Integration test (Feature_Engineer reads parser output)
- **Result:** All tests pass ✓

## Validation

### Test Results

```
============================================================
Testing Dataset_Parser Precision-Based Inference
============================================================
✓ Integer precision test passed: '2000' → ±5 keV
✓ One decimal test passed: '2000.0' → ±0.5 keV
✓ Two decimal test passed: '1234.56' → ±0.05 keV
✓ Scientific notation (1 decimal) test passed: '2.0E3' → ±500 keV
✓ Scientific notation (2 decimals) test passed: '2.00E3' → ±50 keV
✓ Scientific notation (3 decimals) test passed: '2.000E3' → ±5 keV
✓ Scientific notation (integer mantissa) test passed: '2E3' → ±5000 keV
✓ Lowercase scientific notation test passed: '1.5e+02' → ±50 keV
============================================================
All tests passed! ✓
```

### Key Findings

1. **Scientific notation properly distinguished:**
   - "2.0E3" → ±500 keV (1 decimal)
   - "2.00E3" → ±50 keV (2 decimals)
   - "2.000E3" → ±5 keV (3 decimals)

2. **Parser infers during ENSDF → JSON conversion:**
   - Happens when `unc_str` is empty (line 115-117 in parse_ensdf_line)
   - Uses original `energy_str` text before precision loss

3. **Feature_Engineer trusts parser output:**
   - No re-inference or second-guessing
   - Simple 5.0 keV fallback only for corrupted JSON/manual data

4. **Precision inference respects conventions:**
   - Integer → ±5 keV (precision to 10s place)
   - 1 decimal → ±0.5 keV (precision to 0.1 place)
   - 2 decimals → ±0.05 keV (precision to 0.01 place)

## Code Quality

### Lines Changed

- **Dataset_Parser.py:** +62 lines (inference function + integration)
- **Feature_Engineer.py:** -191 lines (removed complex inference)
- **Net change:** -129 lines (simpler codebase)

### Compliance Checklist

- ✓ **VS Code Diff Preservation:** Used `replace_string_in_file` (not `create_file`)
- ✓ **No Acronyms:** Full names (`uncertainty` not `unc`, `mantissa` not `mant`)
- ✓ **Self-Explanatory Naming:** `infer_uncertainty_from_precision` clearly states purpose
- ✓ **Documentation Updated:** High-level strategy in Feature_Engineer.py reflects new architecture
- ✓ **Mandatory Testing:** All code changes immediately validated with test execution
- ✓ **Clean Codebase:** Removed 191 lines of obsolete inference logic

## Files Modified

1. [Dataset_Parser.py](d:\\X\\ND\\LevelMatcher\\Dataset_Parser.py)
   - Lines 1-67: Added `infer_uncertainty_from_precision()`
   - Lines 148-172: Modified `parse_ensdf_line()` to use inference
   - Lines 175-183: Updated level uncertainty structure
   - Lines 200-211: Added gamma intensity inference

2. [Feature_Engineer.py](d:\\X\\ND\\LevelMatcher\\Feature_Engineer.py)
   - Lines 48-63: Replaced complex inference with simple fallbacks
   - Line 11: Updated documentation header
   - Lines 241-248: Simplified gamma energy uncertainty handling
   - Lines 265-272: Simplified level energy uncertainty handling
   - Lines 443-469: Simplified normalize_gamma_intensities()

3. [test_parser_inference.py](d:\\X\\ND\\LevelMatcher\\test_parser_inference.py) - NEW
   - Comprehensive test suite for precision inference
   - 8 test cases covering all precision scenarios

4. [test_fallback_verification.py](d:\\X\\ND\\LevelMatcher\\test_fallback_verification.py) - UPDATED
   - Refactored to test Dataset_Parser (not Feature_Engineer)
   - Added integration test verifying Feature_Engineer reads parser output

## Future Work

None required. Architecture is now correct:
- Dataset_Parser handles all inference during ENSDF → JSON conversion
- Feature_Engineer trusts clean JSON data
- Simple fallbacks handle edge cases (manual data, corrupted JSON)

## References

- User requirement: "if energy uncertainty is missing, but reported energy value is integer like 2000 keV, assign ±5 keV; if 2000.0 one decimal assign ±0.5 keV"
- User escalation: "2.00E3, 2.0E3 cases uncertainties are also precision-based, did you ignore what I said?"
- User architectural critique: "This is terrible bad idea. You should modify the parser code instead of the Feature Engineer code."
