from physics_parser import expand_spin_parity_string, PhysicsFeatureEngine
import sys

# Helpers to mock nuclear level data for testing
def create_mock_level(spin_parity_string):
    """
    Simulates a level record by parsing the spin-parity string into the internal list format.
    """
    detected_states = expand_spin_parity_string(spin_parity_string)
    return {
        'Spin_Parity': spin_parity_string,
        'Spin_Parity_List': [
            {
                'twoTimesSpin': int(state.spin_value * 2) if state.spin_value is not None else None,
                'parity': state.parity_value,
                'isTentativeSpin': state.spin_is_tentative,
                'isTentativeParity': not state.parity_is_firm
            } for state in detected_states
        ]
    }

def run_physics_unit_tests():
    print("=== NUCLEAR PHYSICS LOGIC UNIT TESTS ===")
    
    # 1. Verification of String Parsing (Range)
    print("\n[TEST] Parsing Range: '1/2:7/2'")
    states = expand_spin_parity_string("1/2:7/2")
    numerical_spins = sorted([s.spin_value for s in states if s.spin_value is not None])
    expected_spins = [0.5, 1.5, 2.5, 3.5]
    if numerical_spins != expected_spins:
        print(f"FAILURE. Expected {expected_spins}, Got {numerical_spins}")
        sys.exit(1)
    print(f"SUCCESS. Extracted: {numerical_spins}")

    # 2. Verification of Spin Matching (Heuristics)
    print("\n[TEST] Spin Match Heuristics: '(1,2)-' vs '1-'")
    level_1 = create_mock_level("(1,2)-")
    level_2 = create_mock_level("1-")
    
    spin_similarity = PhysicsFeatureEngine.evaluate_spin_match(level_1['Spin_Parity_List'], level_2['Spin_Parity_List'])
    if spin_similarity < 0.8:
        print(f"FAILURE. High similarity expected. Got {spin_similarity}")
        sys.exit(1)
    print(f"SUCCESS. Spin Similarity: {spin_similarity}")

    print("\n[TEST] Spin Veto: '(1,2)-' vs '4-'")
    level_3 = create_mock_level("4-")
    spin_similarity_veto = PhysicsFeatureEngine.evaluate_spin_match(level_1['Spin_Parity_List'], level_3['Spin_Parity_List'])
    if spin_similarity_veto > 0.3:
        print(f"FAILURE. Veto expected. Got {spin_similarity_veto}")
        sys.exit(1)
    print(f"SUCCESS. Spin Similarity (Veto): {spin_similarity_veto}")

    # 3. Verification of Parity Matching
    print("\n[TEST] Parity Match: '1+' vs '1+'")
    level_p1 = create_mock_level("1+")
    level_p2 = create_mock_level("2+")
    parity_similarity = PhysicsFeatureEngine.evaluate_parity_match(level_p1['Spin_Parity_List'], level_p2['Spin_Parity_List'])
    if parity_similarity != 1.0:
        print(f"FAILURE. Same parity should match 1.0. Got {parity_similarity}")
        sys.exit(1)
    print("SUCCESS. Parity Match: 1.0")

    print("\n[TEST] Parity Veto: '1+' vs '1-'")
    level_p3 = create_mock_level("1-")
    parity_similarity_veto = PhysicsFeatureEngine.evaluate_parity_match(level_p1['Spin_Parity_List'], level_p3['Spin_Parity_List'])
    if parity_similarity_veto != 0.0:
        print(f"FAILURE. Conflicting parity should veto (0.0). Got {parity_similarity_veto}")
        sys.exit(1)
    print("SUCCESS. Parity Veto: 0.0")

if __name__ == "__main__":
    run_physics_unit_tests()

