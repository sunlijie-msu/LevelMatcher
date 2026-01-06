import re

class QuantumState:
    def __init__(self, j, p, j_tentative=False, p_firm=True):
        self.j = j          # Float or None
        self.p = p          # '+' or '-' or None
        self.j_tentative = j_tentative # Bool (True if enclosed in parens)
        self.p_firm = p_firm           # Bool (True if parity is explicit and firm)
        
    def __repr__(self):
        return f"(J={self.j}, P={self.p}, J~={self.j_tentative}, P!={self.p_firm})"
    
    def __eq__(self, other):
        return (self.j == other.j and self.p == other.p and 
                self.j_tentative == other.j_tentative and self.p_firm == other.p_firm)

    def __hash__(self):
        return hash((self.j, self.p, self.j_tentative, self.p_firm))

def parse_val_to_float(val_str):
    try:
        val_str = val_str.strip()
        if '/' in val_str:
            n, d = val_str.split('/')
            return float(n) / float(d)
        return float(val_str)
    except:
        return None

def extract_parity(s):
    # Returns (cleaned_string, parity_char or None)
    if not s: return "", None
    if s.endswith('+'): return s[:-1], '+'
    if s.endswith('-'): return s[:-1], '-'
    return s, None

def expand_j_pi(text, default_parity=None, is_parent_tentative=False):
    """
    Recursive parser for ENSDF J-Pi strings.
    Returns: List of QuantumState objects.
    """
    states = []
    text = text.strip()
    if not text: return states

    # SPECIAL CASE: Pure Parity (e.g. "+", "(-)")
    if text in ['+', '-', '(+)', '(-)']:
        p_char = '+' if '+' in text else '-'
        # Assume isolated parity without spin is firm unless wrapped in double parens?
        # User logic: (-) usually implies tentative parity for the level, 
        # but in context of (1,2)-, user said - is firm.
        # Let's assume standard: (...) is tentative.
        p_firm = not (text.startswith('(') and text.endswith(')'))
        states.append(QuantumState(None, p_char, j_tentative=True, p_firm=p_firm))
        return states

    # 1. Handle Grouped Parens e.g. "(1/2,3/2)+", "(1,2)"
    # Pattern: ^ \s* \( (content) \) (suffix) \s* $
    paren_match = re.match(r'^\s*\((.*)\)([\+\-]?)?\s*$', text)
    
    if paren_match:
        content = paren_match.group(1)
        suffix = paren_match.group(2) # Can be + or - or None/Empty
        
        if suffix:
            # Suffix Parity -> Firm for the whole group (e.g. (1,2)- )
            # Propagate firmness down
            recurse_states = expand_j_pi(content, default_parity=None, is_parent_tentative=True)
            for s in recurse_states:
                s.p = suffix
                s.p_firm = True
                s.j_tentative = True # Wrapper makes J tentative
            return recurse_states
        else:
            # No Suffix -> Parity is loose/tentative (e.g. (1,2))
            recurse_states = expand_j_pi(content, default_parity=default_parity, is_parent_tentative=True)
            for s in recurse_states:
                s.p_firm = False # Downgrade firmness
                s.j_tentative = True
            return recurse_states

    # 2. Handle Commas (Splitter) respecting parens
    if ',' in text:
        parts = []
        depth = 0
        curr = []
        for char in text:
            if char == '(': depth += 1
            if char == ')': depth -= 1
            if char == ',' and depth == 0:
                parts.append("".join(curr))
                curr = []
            else:
                curr.append(char)
        parts.append("".join(curr))
        
        if len(parts) > 1:
            for p in parts:
                states.extend(expand_j_pi(p, default_parity, is_parent_tentative))
            return states

    # 3. Handle Range (:)
    if ':' in text:
        r_parts = text.split(':')
        if len(r_parts) == 2:
            s_str, e_str = r_parts
            s_clean, s_par = extract_parity(s_str.strip())
            e_clean, e_par = extract_parity(e_str.strip())
            
            # Resolve Parity
            r_parity = s_par if s_par else (e_par if e_par else default_parity)
            r_p_firm = True # Defaults to firm unless parent tentative
            if is_parent_tentative: r_p_firm = False # Inherit tentative nature
            
            # Note: "1/2+:5/2+" -> Both firm. "1/2:5/2" -> both No Parity.
            # Use 'r_parity' as the parity. Firmness depends on parens and explicit markings.
            # If explicit parity exists on range ends, it's firm for that item, 
            # but usually range implies shared property.
            
            j_start = parse_val_to_float(s_clean)
            j_end = parse_val_to_float(e_clean)
            
            if j_start is not None and j_end is not None:
                curr = j_start
                # Ensure we don't loop forever
                if j_end < curr: curr, j_end = j_end, curr 
                
                while curr <= j_end + 0.001:
                    states.append(QuantumState(curr, r_parity, j_tentative=is_parent_tentative, p_firm=r_p_firm))
                    curr += 1.0
                return states

    # 4. Handle Single Item
    clean_txt, item_par = extract_parity(text)
    final_par = item_par if item_par else default_parity
    
    j_val = parse_val_to_float(clean_txt)
    
    if j_val is not None:
        p_firm = not is_parent_tentative
        if final_par is None: p_firm = False # No parity info is never firm
        
        states.append(QuantumState(j_val, final_par, j_tentative=is_parent_tentative, p_firm=p_firm))
    else:
        # Fallback
        if final_par:
             states.append(QuantumState(None, final_par, j_tentative=True, p_firm=not is_parent_tentative))
            
    return states

def calculate_physics_distance(sp1_str, sp2_str):
    """
    Calculates a 'Consistency Distance' feature for XGBoost.
    
    Returns float:
    0.0 : Exact/Strict Match (Firm overlap)
    0.25: Tentative Match (Overlap exists, but via tentative assignments)
    0.5 : Information Missing (One side Unknown, consistent by default)
    1.0 : Mismatch (Strict Veto)
    """
    
    # Normalize inputs
    sp1_str = str(sp1_str).strip() if sp1_str and str(sp1_str).lower() not in ['nan','none','null'] else ""
    sp2_str = str(sp2_str).strip() if sp2_str and str(sp2_str).lower() not in ['nan','none','null'] else ""
    
    if not sp1_str or not sp2_str:
        return 0.5 # Missing Info = Neutral/Unknown
        
    states1 = expand_j_pi(sp1_str)
    states2 = expand_j_pi(sp2_str)
    
    if not states1 or not states2:
        return 0.5
        
    # Check for consistency
    best_match_category = 1.0 # Start with Mismatch
    
    for s1 in states1:
        for s2 in states2:
            match_type = compare_states(s1, s2)
            if match_type < best_match_category:
                best_match_category = match_type
                
    return best_match_category

def compare_states(s1, s2):
    """
    Compares two QuantumState objects.
    Returns:
    0.0: Exact Match
    0.25: Tentative Match
    1.0: Mismatch
    """
    
    # 1. SPIN Check
    spin_match = False
    is_spin_tentative_match = False
    
    if s1.j is None or s2.j is None:
        spin_match = True
        is_spin_tentative_match = True # Treated as weak match
    else:
        diff = abs(s1.j - s2.j)
        if diff < 0.001:
            spin_match = True
        elif (s1.j_tentative or s2.j_tentative) and diff <= 1.001:
            spin_match = True
            is_spin_tentative_match = True
            
    if not spin_match: return 1.0
    
    # 2. PARITY Check
    parity_match = False
    is_parity_tentative_match = False
    
    if s1.p is None or s2.p is None:
        parity_match = True
        is_parity_tentative_match = True
    elif not s1.p_firm or not s2.p_firm:
        parity_match = True
        is_parity_tentative_match = True
    elif s1.p == s2.p:
        parity_match = True
        
    if not parity_match: return 1.0
    
    # 3. Determine Quality
    if is_spin_tentative_match or is_parity_tentative_match:
        return 0.25
        
    return 0.0

if __name__ == "__main__":
    # Self-Test
    print("Test 1 (Firm Match):", calculate_physics_distance("3/2+", "3/2+"))
    print("Test 2 (Mismatch):", calculate_physics_distance("3/2+", "5/2+"))
    print("Test 3 (Tentative Spin):", calculate_physics_distance("(1,2)+", "1+"))
    print("Test 4 (Missing Info):", calculate_physics_distance("3/2+", ""))
    print("Test 5 (Tentative Parity):", calculate_physics_distance("(3/2)", "3/2+"))
    print("Test 6 (Firm List Veto):", calculate_physics_distance("0-,1-,2-", "3-"))
