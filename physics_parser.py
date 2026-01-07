import re
import numpy as np

# ==========================================
# FEATURE ENGINEERING & DATA TRANSFORMATION
# ==========================================

class PhysicsFeatureEngine:
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_z_score(e1, de1, e2, de2):
        if e1 is None or e2 is None: return 999.0
        sigma = np.sqrt(de1**2 + de2**2)
        if sigma == 0: sigma = 1.0 # Prevent div/0
        return abs(e1 - e2) / sigma

    @staticmethod
    def extract_features(level1, level2):
        """
        Transforms two level objects into a physical feature vector.
        Inputs: level1, level2 (dictionaries with keys energy_value, energy_uncertainty, Spin_Parity_List)
        Output: numpy array containing physics-informed features for XGBoost.
        """
        # 1. Energy Features (Z-Score)
        z_score = PhysicsFeatureEngine.calculate_z_score(
            level1.get('energy_value'), level1.get('energy_uncertainty', 10.0),
            level2.get('energy_value'), level2.get('energy_uncertainty', 10.0)
        )
        
        spin_parity_list_1 = level1.get('Spin_Parity_List', [])
        spin_parity_list_2 = level2.get('Spin_Parity_List', [])

        # 2. Comparison of Physics Properties
        spin_score = PhysicsFeatureEngine.evaluate_spin_match(spin_parity_list_1, spin_parity_list_2)
        parity_score = PhysicsFeatureEngine.evaluate_parity_match(spin_parity_list_1, spin_parity_list_2)
        
        # 3. Indepenedent Tentativeness
        # Value is 1.0 if any entry in the list is tentative, else 0.0.
        spin_is_tentative_1 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_1)
        spin_is_tentative_2 = any(state.get('isTentativeSpin', False) for state in spin_parity_list_2)
        spin_is_tentative = 1.0 if (spin_is_tentative_1 or spin_is_tentative_2) else 0.0

        parity_is_tentative_1 = any(state.get('isTentativeParity', False) for state in spin_parity_list_1)
        parity_is_tentative_2 = any(state.get('isTentativeParity', False) for state in spin_parity_list_2)
        parity_is_tentative = 1.0 if (parity_is_tentative_1 or parity_is_tentative_2) else 0.0

        # 4. Ambiguity Metric (Search Space Complexity)
        # More options in the ENSDF Spin/Parity field increase matching ambiguity.
        options_count_1 = len(spin_parity_list_1) if spin_parity_list_1 else 1
        options_count_2 = len(spin_parity_list_2) if spin_parity_list_2 else 1
        ambiguity = np.log10(max(1, options_count_1 * options_count_2)) * 0.5
        
        # Feature Vector ordering: [Z, Spin, Parity, Spin_Tent, Parity_Tent, Ambiguity]
        return np.array([z_score, spin_score, parity_score, spin_is_tentative, parity_is_tentative, ambiguity])

    @staticmethod
    def evaluate_spin_match(spin_parity_list_1, spin_parity_list_2):
        """
        Calculates a similarity score (0.0 to 1.0) based on nuclear spin heuristics.
        """
        if not spin_parity_list_1 or not spin_parity_list_2:
            return 0.5 # Neutral - cannot confirm or deny match

        # Extract spins (2*J) and tentative status
        spins_1 = [(state.get('twoTimesSpin'), state.get('isTentativeSpin', False)) 
                   for state in spin_parity_list_1 if state.get('twoTimesSpin') is not None]
        spins_2 = [(state.get('twoTimesSpin'), state.get('isTentativeSpin', False)) 
                   for state in spin_parity_list_2 if state.get('twoTimesSpin') is not None]

        if not spins_1 or not spins_2:
            return 0.5 # Neutral (Spins not defined)

        max_similarity_score = 0.0
        
        for two_times_j1, spin1_is_tentative in spins_1:
            for two_times_j2, spin2_is_tentative in spins_2:
                # Delta J (divide by 2 for physical units h-bar)
                spin_distance = abs(two_times_j1 - two_times_j2) / 2.0 
                
                pair_similarity = 0.0
                
                if spin_distance == 0.0:
                    # Match
                    if not spin1_is_tentative and not spin2_is_tentative:
                        pair_similarity = 1.0 # 2+ vs 2+ (Firm Match)
                    else:
                        pair_similarity = 0.9 # (2+) vs 2+ (Tentative Match)
                elif spin_distance == 1.0:
                    # Adjacent Spins (e.g., 2 vs 3)
                    if not spin1_is_tentative and not spin2_is_tentative:
                        pair_similarity = 0.0 # Veto (Strict Conflict)
                    else:
                        pair_similarity = 0.25 # (2) vs 3 (Weak Conflict)
                elif spin_distance == 2.0:
                    # Distant Spins (e.g., 2 vs 4)
                    if not spin1_is_tentative and not spin2_is_tentative:
                        pair_similarity = 0.0 # Veto
                    else:
                        pair_similarity = 0.1 # (2) vs 4 (Nearly Veto)
                else:
                    # Larger distances are considered hard vetos
                    pair_similarity = 0.0

                if pair_similarity > max_similarity_score:
                    max_similarity_score = pair_similarity
        
        return max_similarity_score

    @staticmethod
    def evaluate_parity_match(spin_parity_list_1, spin_parity_list_2):
        """
        Calculates a parity similarity score.
        """
        if not spin_parity_list_1 or not spin_parity_list_2:
            return 0.5 # Neutral (Parity information missing)

        # Extract parities (+1, -1) and tentative flags
        parities_1 = [(state.get('parity'), state.get('isTentativeParity', False)) 
                      for state in spin_parity_list_1 if state.get('parity') is not None]
        parities_2 = [(state.get('parity'), state.get('isTentativeParity', False)) 
                      for state in spin_parity_list_2 if state.get('parity') is not None]

        if not parities_1 or not parities_2:
            return 0.5 # Parity unknown

        max_parity_similarity_score = 0.0
        
        for parity_signal_1, is_tentative_1 in parities_1:
            for parity_signal_2, is_tentative_2 in parities_2:
                # Same Sign check (+1 == +1 or -1 == -1)
                is_match = (parity_signal_1 == parity_signal_2)
                
                pair_similarity = 0.0
                if is_match:
                    if not is_tentative_1 and not is_tentative_2:
                        pair_similarity = 1.0 # Firm match
                    else:
                        pair_similarity = 0.9 # Tentative match
                else:
                    # Mismatch (Conflict)
                    if not is_tentative_1 and not is_tentative_2:
                        pair_similarity = 0.0 # Veto (Strict Conflict)
                    else:
                        pair_similarity = 0.2 # Tentative Conflict
                
                if pair_similarity > max_parity_similarity_score:
                    max_parity_similarity_score = pair_similarity
                    
        return max_parity_similarity_score

    @staticmethod
    def get_training_data():
        """
        Returns (training_features, training_labels) for training the XGBoost Model.
        Feature Order: [Z_Score, Spin_Score, Parity_Score, Spin_Is_Tentative, Parity_Is_Tentative, Ambiguity]
        """
        training_records = []

        # 1. GOLD STANDARD MATCH
        # Perfect Z-score, Spin match, Parity match, Firm assignments, Low ambiguity
        training_records.append(([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 1.00))
        training_records.append(([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 0.95)) # Z degrades slightly

        # 2. SPIN VETO
        # Hard conflict in spin (firm 2 vs firm 4) means match is impossible regardless of energy.
        training_records.append(([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.00))

        # 3. PARITY VETO
        # Firm 2+ vs 2- is impossible matching.
        training_records.append(([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.00))

        # 4. TENTATIVE SENSITIVITY
        # (2+) vs 2+ is a good match (0.9 physics score) but slightly less certain than Gold.
        training_records.append(([0.0, 0.9, 1.0, 1.0, 0.0, 0.0], 0.90))

        # 5. ENERGY DEGRADATION (Far away levels)
        # Good physics, but Energy is 5 or 10 sigma away
        training_records.append(([5.0, 1.0, 1.0, 0.0, 0.0, 0.0], 0.10))
        training_records.append(([10.0, 1.0, 1.0, 0.0, 0.0, 0.0], 0.00))

        # 6. HIGH AMBIGUITY (Complex Spin strings)
        # Search space is large, lowering the coincidence probability.
        training_records.append(([0.0, 1.0, 1.0, 0.0, 0.0, 0.4], 0.80))
        
        # 7. INFORMATION VOID
        # No J or Pi information (0.5 scores). Match relies entirely on energy.
        training_records.append(([0.0, 0.5, 0.5, 0.0, 0.0, 0.0], 0.85))
        training_records.append(([2.0, 0.5, 0.5, 0.0, 0.0, 0.0], 0.40))

        training_features = np.array([record[0] for record in training_records])
        training_labels = np.array([record[1] for record in training_records])
        return training_features, training_labels


# ==========================================
# TEXT PARSING (Legacy helpers)
# ==========================================

class QuantumState:
    def __init__(self, spin_value, parity_value, spin_is_tentative=False, parity_is_firm=True):
        """
        Represents a nuclear quantum state with spin and parity property.
        """
        self.spin_value = spin_value        # Float or None
        self.parity_value = parity_value    # '+' or '-' or None
        self.spin_is_tentative = spin_is_tentative # Boolean flag
        self.parity_is_firm = parity_is_firm       # Boolean flag
        
    def __repr__(self):
        return f"(Spin={self.spin_value}, Parity={self.parity_value}, SpinTentative={self.spin_is_tentative}, ParityFirm={self.parity_is_firm})"
    
    def __eq__(self, other):
        return (self.spin_value == other.spin_value and 
                self.parity_value == other.parity_value and 
                self.spin_is_tentative == other.spin_is_tentative and 
                self.parity_is_firm == other.parity_is_firm)

    def __hash__(self):
        return hash((self.spin_value, self.parity_value, self.spin_is_tentative, self.parity_is_firm))

def parse_string_to_float(value_string):
    """Safely parses fractional (e.g., '3/2') or decimal strings to float."""
    try:
        value_string = value_string.strip()
        if '/' in value_string:
            numerator, denominator = value_string.split('/')
            return float(numerator) / float(denominator)
        return float(value_string)
    except (ValueError, ZeroDivisionError):
        return None

def extract_parity_character(text_string):
    """Separates the parity suffix (+ or -) from a spin-parity string."""
    if not text_string: return "", None
    if text_string.endswith('+'): return text_string[:-1], '+'
    if text_string.endswith('-'): return text_string[:-1], '-'
    return text_string, None

def expand_spin_parity_string(input_text, default_parity=None, is_parent_tentative=False):
    """
    Recursive parser for ENSDF Spin-Parity strings.
    Converts descriptive input (e.g., "(1/2, 3/2)+") into a list of QuantumState objects.
    """
    detected_states = []
    if not input_text: return detected_states
    
    # Normalize character formatting (e.g., replace unicode minus with standard hyphen)
    input_text = input_text.replace('−', '-').strip()
    
    if not input_text: return detected_states

    # Special Case: Isolated Parity markers (e.g. "+", "(-)")
    if input_text in ['+', '-', '(+)', '(-)']:
        parity_character = '+' if '+' in input_text else '-'
        # Entries in parentheses are treated as tentative parity
        parity_is_firm = not (input_text.startswith('(') and input_text.endswith(')'))
        detected_states.append(QuantumState(None, parity_character, spin_is_tentative=True, parity_is_firm=parity_is_firm))
        return detected_states

    # 1. Handle Mixed Tentative Parity Suffix (e.g. "7/2(+)", "5/2(-)")
    tentative_parity_match = re.match(r'^(.*)\((\+|\-)\)$', input_text)
    if tentative_parity_match:
        inner_content = tentative_parity_match.group(1).strip()
        parity_character = tentative_parity_match.group(2)
        
        # Recursively parse the spin content using this tentative parity
        recursive_states = expand_spin_parity_string(inner_content, default_parity=parity_character, is_parent_tentative=is_parent_tentative)
        for state in recursive_states:
            if state.parity_value is None: state.parity_value = parity_character
            state.parity_is_firm = False # Tentative because it was in parentheses
        return recursive_states

    # 2. Handle Grouped Parentheses (e.g. "(1/2,3/2)+", "(1,2)")
    grand_parentheses_match = re.match(r'^\s*\((.*)\)([\+\-]?)?\s*$', input_text)
    
    if grand_parentheses_match:
        inner_content = grand_parentheses_match.group(1)
        suffix_parity = grand_parentheses_match.group(2) # e.g., '+' or '-' or None
        
        if suffix_parity:
            # Explicit suffix outside parentheses implies firm parity for the entire group
            recursive_states = expand_spin_parity_string(inner_content, default_parity=None, is_parent_tentative=True)
            for state in recursive_states:
                state.parity_value = suffix_parity
                state.parity_is_firm = True
                state.spin_is_tentative = True # Parentheses make the spins tentative
            return recursive_states
        else:
            # No suffix -> Both spin and parity inherit the tentative nature of the parentheses
            recursive_states = expand_spin_parity_string(inner_content, default_parity=default_parity, is_parent_tentative=True)
            for state in recursive_states:
                state.parity_is_firm = False 
                state.spin_is_tentative = True
            return recursive_states

    # 3. Handle Comma-Separated Lists respecting nested parentheses
    if ',' in input_text:
        list_parts = []
        nesting_depth = 0
        current_part_buffer = []
        for character in input_text:
            if character == '(': nesting_depth += 1
            if character == ')': nesting_depth -= 1
            if character == ',' and nesting_depth == 0:
                list_parts.append("".join(current_part_buffer))
                current_part_buffer = []
            else:
                current_part_buffer.append(character)
        list_parts.append("".join(current_part_buffer))
        
        if len(list_parts) > 1:
            for part in list_parts:
                detected_states.extend(expand_spin_parity_string(part, default_parity, is_parent_tentative))
            return detected_states

    # 4. Handle Range Notation (e.g., "1/2:5/2")
    if ':' in input_text:
        range_parts = input_text.split(':')
        if len(range_parts) == 2:
            start_string, end_string = range_parts
            start_clean, start_parity = extract_parity_character(start_string.strip())
            end_clean, end_parity = extract_parity_character(end_string.strip())
            
            # Resolve common parity for the range
            range_parity = start_parity if start_parity else (end_parity if end_parity else default_parity)
            range_parity_is_firm = not is_parent_tentative if range_parity else False
            
            spin_start = parse_string_to_float(start_clean)
            spin_end = parse_string_to_float(end_clean)
            
            if spin_start is not None and spin_end is not None:
                current_spin = spin_start
                # Natural sorting safety
                if spin_end < current_spin: current_spin, spin_end = spin_end, current_spin 
                
                while current_spin <= spin_end + 0.001:
                    detected_states.append(QuantumState(current_spin, range_parity, spin_is_tentative=is_parent_tentative, parity_is_firm=range_parity_is_firm))
                    current_spin += 1.0
                return detected_states

    # 5. Handle Single Spin Item
    clean_text, explicit_parity = extract_parity_character(input_text)
    final_parity = explicit_parity if explicit_parity else default_parity
    
    numerical_spin = parse_string_to_float(clean_text)
    
    if numerical_spin is not None:
        parity_is_firm = not is_parent_tentative
        if final_parity is None: parity_is_firm = False
        
        detected_states.append(QuantumState(numerical_spin, final_parity, spin_is_tentative=is_parent_tentative, parity_is_firm=parity_is_firm))
    else:
        # Fallback for parity-only or malformed entries
        if final_parity:
             detected_states.append(QuantumState(None, final_parity, spin_is_tentative=True, parity_is_firm=not is_parent_tentative))
            
    return detected_states

def extract_json_compatibility_features(spin_parity_list_1, spin_parity_list_2):
    """
    Extracts a feature dictionary describing the compatibility of two Spin-Parity sets.
    """
    # 1. Neutral check for missing physics information
    if not spin_parity_list_1 or not spin_parity_list_2:
        return {
            'spin_conflict': 0.0,
            'parity_conflict': 0.0,
            'is_tentative': 0.0,
            'information_content': 0.0
        }
        
    match_found = False
    is_firm_match = False
    
    for state_1 in spin_parity_list_1:
        for state_2 in spin_parity_list_2:
            # check numerical spin compatibility
            spin_1_value = state_1.get('twoTimesSpin')
            spin_2_value = state_2.get('twoTimesSpin')
            spin_is_compatible = (spin_1_value is None or spin_2_value is None or spin_1_value == spin_2_value)
            
            # check parity signal compatibility
            parity_1_character = state_1.get('parity')
            parity_2_character = state_2.get('parity')
            parity_is_compatible = (parity_1_character is None or parity_2_character is None or parity_1_character == parity_2_character)
            
            if spin_is_compatible and parity_is_compatible:
                match_found = True
                
                # A match is firm only if neither attribute is marked as tentative
                state_1_is_tentative = state_1.get('isTentativeSpin') or state_1.get('isTentativeParity')
                state_2_is_tentative = state_2.get('isTentativeSpin') or state_2.get('isTentativeParity')
                
                if not state_1_is_tentative and not state_2_is_tentative:
                    is_firm_match = True
                    
    return {
        'spin_conflict': 0.0 if match_found else 1.0,
        'parity_conflict': 0.0, # detailed tracking can be added if required
        'is_tentative': 0.0 if is_firm_match else (1.0 if match_found else 0.0),
        'information_content': 1.0
    }

def calculate_json_compatibility_score(spin_parity_list_1, spin_parity_list_2):
    """
    Compatibility score wrapper for JSON data structures.
    """
    features = extract_json_compatibility_features(spin_parity_list_1, spin_parity_list_2)
    if features['spin_conflict'] > 0: return 1.0
    if features['is_tentative'] > 0: return 0.25
    if features['information_content'] == 0: return 0.5
    return 0.0

def compare_json_quantum_states(state_1, state_2):
    """
    Compares two individual J-Pi state dictionaries.
    """
    # 1. Spin comparison
    spin_1_value = state_1.get('twoTimesSpin')
    spin_2_value = state_2.get('twoTimesSpin')
    
    spin_mismatch_quality = 0.0
    
    if spin_1_value is None or spin_2_value is None:
        spin_mismatch_quality = 0.25 # Implicit uncertainty
    elif spin_1_value != spin_2_value:
        return 1.0 # Strict mismatch
    else:
        # Check if either value is tentative
        if state_1.get('isTentativeSpin') or state_2.get('isTentativeSpin'):
            spin_mismatch_quality = 0.25
            
    # 2. Parity comparison
    parity_1_character = state_1.get('parity')
    parity_2_character = state_2.get('parity')
    
    parity_mismatch_quality = 0.0
    
    if parity_1_character is None or parity_2_character is None:
        parity_mismatch_quality = 0.25
    elif parity_1_character != parity_2_character:
        return 1.0 # Strict mismatch
    else:
        if state_1.get('isTentativeParity') or state_2.get('isTentativeParity'):
            parity_mismatch_quality = 0.25
            
    # Aggregate results: Any tentative component makes the entire match tentative
    if spin_mismatch_quality > 0 or parity_mismatch_quality > 0:
        return 0.25
        
    return 0.0

if __name__ == "__main__":
    # Self-Test verification for the physics engine
    print("Test Logic 1 (Firm Match):", calculate_json_compatibility_score(
        [{'twoTimesSpin': 3, 'parity': '+', 'isTentativeSpin': False, 'isTentativeParity': False}],
        [{'twoTimesSpin': 3, 'parity': '+', 'isTentativeSpin': False, 'isTentativeParity': False}]
    ))
    print("Test Logic 2 (Tentative Match):", calculate_json_compatibility_score(
        [{'twoTimesSpin': 3, 'parity': '+', 'isTentativeSpin': True, 'isTentativeParity': False}],
        [{'twoTimesSpin': 3, 'parity': '+', 'isTentativeSpin': False, 'isTentativeParity': False}]
    ))
    print("Test Logic 3 (Strict Mismatch):", calculate_json_compatibility_score(
        [{'twoTimesSpin': 3, 'parity': '+', 'isTentativeSpin': False, 'isTentativeParity': False}],
        [{'twoTimesSpin': 5, 'parity': '+', 'isTentativeSpin': False, 'isTentativeParity': False}]
    ))
