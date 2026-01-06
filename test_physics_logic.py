from Level_Matcher_Gemini import expand_j_pi, check_physics_veto
import sys

# Helpers to mock DataFrame rows
def r(spin, parity):
    return {'Spin': spin, 'Parity': parity}

def test_parser():
    print("=== VERIFYING PARSER LOGIC & VETO ===")
    
    # 1. Test Range
    print("\n[TEST] Range '1/2:7/2'")
    res = expand_j_pi("1/2:7/2")
    # Sort by spin
    spins = sorted([x.j for x in res if x.j is not None])
    expected_spins = [0.5, 1.5, 2.5, 3.5]
    if spins != expected_spins:
        print(f"FAILED. Expected {expected_spins}, Got {spins}")
        sys.exit(1)
    
    # Range is firm? Logic: Top level range -> firm.
    if any(x.j_tentative for x in res):
        print("FAILED. Top level range should be firm.")
        sys.exit(1)
    print(f"PASS. Got {spins}")

    # 2. Test List
    print("\n[TEST] List '3/2, 5/2'")
    res = expand_j_pi("3/2, 5/2")
    spins = sorted([x.j for x in res if x.j is not None])
    expected_spins = [1.5, 2.5]
    if spins != expected_spins:
        print(f"FAILED. Expected {expected_spins}, Got {spins}")
        sys.exit(1)
    print(f"PASS. Got {spins}")

    # 3. Test Parity Inheritance (Suffix)
    print("\n[TEST] Parity Inheritance '(1/2, 3/2)+'")
    res = expand_j_pi("(1/2, 3/2)+")
    # Should get (0.5, +, TentativeJ, FirmP)
    for s in res:
        if not s.j_tentative: print(f"FAILED. S/B Tentative. {s}"); sys.exit(1)
        if not s.p_firm: print(f"FAILED. S/B Firm Parity. {s}"); sys.exit(1)
        if s.p != '+': print(f"FAILED. S/B +. {s}"); sys.exit(1)
    print(f"PASS. Got {res}")

    # 4. Test Parity Inheritance Negative (Firm Suffix)
    print("\n[TEST] Parity Inheritance Negative '(1, 2)-'")
    res = expand_j_pi("(1, 2)-")
    for s in res:
        if not s.j_tentative: print(f"FAILED. S/B Tentative J. {s}"); sys.exit(1)
        if not s.p_firm: print(f"FAILED. S/B Firm Parity. {s}"); sys.exit(1)
        if s.p != '-': print(f"FAILED. S/B -. {s}"); sys.exit(1)
    
    # VETO CHECK: (1,2)- should match 1- (Exact)
    if check_physics_veto(r("(1,2)-", None), r("1-", None)) != 0:
        print("FAILED VETO Check: (1,2)- should match 1-")
        sys.exit(1)
    # VETO CHECK: (1,2)- should match 0- (Tentative logic +/- 1)
    # 0 is close to 1.
    if check_physics_veto(r("(1,2)-", None), r("0-", None)) != 0:
        print("FAILED VETO Check: (1,2)- should match 0- (Tolerance)")
        sys.exit(1)
    # VETO CHECK: (1,2)- should NOT match 4-
    if check_physics_veto(r("(1,2)-", None), r("4-", None)) != 1:
        print("FAILED VETO Check: (1,2)- should VETO 4-")
        sys.exit(1)
    print("PASS Logic Checks.")

    # 5. Test List Explicit Parity (Firm No Parens)
    print("\n[TEST] List Explicit Parity '0-,1-,2-'")
    res = expand_j_pi("0-,1-,2-")
    for s in res:
        if s.j_tentative: print(f"FAILED. S/B Firm J. {s}"); sys.exit(1)
        if not s.p_firm: print(f"FAILED. S/B Firm P. {s}"); sys.exit(1)
    
    # VETO CHECK: 0-,1-,2- should NOT match 3- (Strict)
    if check_physics_veto(r("0-,1-,2-", None), r("3-", None)) != 1:
        print("FAILED VETO Check: Firm List should Veto neighbors.")
        sys.exit(1)
    print("PASS. Got {res}")

    # 6. Test No Parity (Tentative Group, No Suffix)
    print("\n[TEST] No Parity '(1,2)'")
    res = expand_j_pi("(1,2)")
    # Parity should be Not Firm (downstream can match + or -)
    for s in res:
        if s.p_firm: print(f"FAILED. S/B Not Firm P. {s}"); sys.exit(1)
        if not s.j_tentative: print(f"FAILED. S/B Tentative J. {s}"); sys.exit(1)
        
    # VETO CHECK: (1,2) matches 1+
    if check_physics_veto(r("(1,2)", None), r("1+", None)) != 0:
         print("FAILED VETO Check: (1,2) should match 1+")
         sys.exit(1)
    print(f"PASS. Got {res}")

    # 7. Test Mixed
    print("\n[TEST] Mixed '(2+,3,4,5+)'")
    res = expand_j_pi("(2+,3,4,5+)")
    # 2+: Spin Tentative, Parity Not Firm (inside parens)
    s2 = next(x for x in res if x.j == 2.0)
    if s2.p_firm: print("FAILED. Inner parity inside parens S/B Not Firm."); sys.exit(1)
    
    # VETO CHECK: (2+,...) matches 2-
    if check_physics_veto(r("(2+,3,4,5+)", None), r("2-", None)) != 0:
         print("FAILED VETO Check: (2+) matches 2- because inner parity is tentative.")
         sys.exit(1)
    print(f"PASS. Got {res}")

if __name__ == "__main__":
    try:
        test_parser()
        print("\nSUCCESS: The code obeys all rules.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
