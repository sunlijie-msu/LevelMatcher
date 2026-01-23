"""
Test Dataset_Parser precision-based uncertainty inference.

Verifies that parser correctly infers uncertainties from ENSDF text precision.
"""

from Dataset_Parser import infer_uncertainty_from_precision

def test_integer_precision():
    """Integer values like 2000 should infer ±5 keV"""
    result = infer_uncertainty_from_precision("2000")
    assert result == 5.0, f"Expected 5.0, got {result}"
    print("✓ Integer precision test passed: '2000' → ±5 keV")

def test_one_decimal():
    """One decimal like 2000.0 should infer ±0.5 keV"""
    result = infer_uncertainty_from_precision("2000.0")
    assert result == 0.5, f"Expected 0.5, got {result}"
    print("✓ One decimal test passed: '2000.0' → ±0.5 keV")

def test_two_decimals():
    """Two decimals like 1234.56 should infer ±0.05 keV"""
    result = infer_uncertainty_from_precision("1234.56")
    assert result == 0.05, f"Expected 0.05, got {result}"
    print("✓ Two decimal test passed: '1234.56' → ±0.05 keV")

def test_scientific_notation_one_decimal():
    """Scientific notation 2.0E3 should infer ±500 keV"""
    result = infer_uncertainty_from_precision("2.0E3")
    assert result == 500.0, f"Expected 500.0, got {result}"
    print("✓ Scientific notation (1 decimal) test passed: '2.0E3' → ±500 keV")

def test_scientific_notation_two_decimals():
    """Scientific notation 2.00E3 should infer ±50 keV"""
    result = infer_uncertainty_from_precision("2.00E3")
    assert result == 50.0, f"Expected 50.0, got {result}"
    print("✓ Scientific notation (2 decimals) test passed: '2.00E3' → ±50 keV")

def test_scientific_notation_three_decimals():
    """Scientific notation 2.000E3 should infer ±5 keV"""
    result = infer_uncertainty_from_precision("2.000E3")
    assert result == 5.0, f"Expected 5.0, got {result}"
    print("✓ Scientific notation (3 decimals) test passed: '2.000E3' → ±5 keV")

def test_scientific_notation_integer_mantissa():
    """Scientific notation 2E3 should infer ±5000 keV"""
    result = infer_uncertainty_from_precision("2E3")
    assert result == 5000.0, f"Expected 5000.0, got {result}"
    print("✓ Scientific notation (integer mantissa) test passed: '2E3' → ±5000 keV")

def test_lowercase_scientific():
    """Lowercase scientific notation 1.5e+02 should work"""
    result = infer_uncertainty_from_precision("1.5e+02")
    assert result == 50.0, f"Expected 50.0, got {result}"  # 5 × 10^(-1) × 10^2 = 50
    print("✓ Lowercase scientific notation test passed: '1.5e+02' → ±50 keV")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Dataset_Parser Precision-Based Inference")
    print("=" * 60)
    
    test_integer_precision()
    test_one_decimal()
    test_two_decimals()
    test_scientific_notation_one_decimal()
    test_scientific_notation_two_decimals()
    test_scientific_notation_three_decimals()
    test_scientific_notation_integer_mantissa()
    test_lowercase_scientific()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
