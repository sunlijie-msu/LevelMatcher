import json
import re
import os
import argparse

def parse_jpi(jpi_str):
    """
    Parses a Jπ string into the project's structured JSON format.
    Handles ranges, lists, suffixes, and complex tentativeness logic (e.g. (1+, 2+), 3/2, 5/2(+)).
    """
    if not jpi_str:
        return []

    jpi_str = jpi_str.strip().rstrip('.')
    if not jpi_str or jpi_str.lower() == 'unknown':
        return []

    # 1. Check for Range e.g., 1:3
    if ':' in jpi_str and all(c.isdigit() or c.isspace() or c == ':' for c in jpi_str):
        try:
            parts = jpi_str.split(':')
            if len(parts) == 2:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return [{
                    "twoTimesSpin": s * 2,
                    "isTentativeSpin": False,
                    "parity": None,
                    "isTentativeParity": False
                } for s in range(start, end + 1)]
        except ValueError:
            pass 

    # 2. General Parsing Strategy
    is_wrapper_tentative = False
    content_str = jpi_str
    
    # Global Suffix Parity: (1, 2)+ or (1, 2)(+)
    suffix_pattern = r'^\((.*)\)(\(?[\+\-]\)?)$'
    suffix_match = re.match(suffix_pattern, jpi_str)
    
    global_suffix_parity_info = None 

    if suffix_match and ',' in suffix_match.group(1):
        content_str = suffix_match.group(1)
        is_wrapper_tentative = True 
        
        suffix_str = suffix_match.group(2)
        s_match = re.search(r'(\(?)([\+\-])(\)?)$', suffix_str)
        if s_match:
             global_suffix_parity_info = {
                 'symbol': s_match.group(2),
                 'is_tentative': (s_match.group(1) == '(' or s_match.group(3) == ')'),
                 'source': 'global_suffix'
             }
             
    # Global Parens without suffix: (1+, 2+)
    elif jpi_str.startswith('(') and jpi_str.endswith(')') and ',' in jpi_str:
        content_str = jpi_str[1:-1]
        is_wrapper_tentative = True

    # 3. Split Items
    raw_items = [x.strip() for x in content_str.split(',')]
    parsed_items = []
    
    for item in raw_items:
        current_spin_str = item
        current_parity_info = None 

        # Extract Parity from end of item
        p_match = re.search(r'(\(?)([\+\-])(\)?)$', item)
        
        if p_match:
            full_p_str = p_match.group(0)
            p_symbol = p_match.group(2)
            p_is_tentative = (p_match.group(1) == '(' or p_match.group(3) == ')')
            
            current_parity_info = {
                'symbol': p_symbol,
                'is_tentative': p_is_tentative,
                'source': 'local'
            }
            current_spin_str = item[:item.rfind(full_p_str)].strip()
            
        parsed_items.append({
            'spin_str': current_spin_str,
            'parity_info': current_parity_info,
        })
        
    # 4. Parity Distribution / Backfill
    if global_suffix_parity_info:
        for p_item in parsed_items:
            if p_item['parity_info'] is None:
                p_item['parity_info'] = global_suffix_parity_info
    # else:
        # REMOVED Backfill Logic per strict user rule:
        # "3/2,5/2,7/2(+) means 3, 5, 7(+) and 3,5 have no parities! only 7 has a tentative + parity!"
        # We only apply parity to multiple items if there is a global wrapper or suffix.

    # 5. Finalize Results
    results = []
    
    for p_item in parsed_items:
        s_str = p_item['spin_str']
        item_spin_tentative = is_wrapper_tentative
        
        # Check for local spin parens e.g. (1)
        if '(' in s_str or ')' in s_str:
            item_spin_tentative = True
            s_str = s_str.replace('(', '').replace(')', '')
            
        s_str = s_str.strip()
        if not s_str: 
            continue

        try:
            val = float(eval(s_str))
            two_spin = int(round(val * 2))
        except:
            continue 
           
        p_info = p_item['parity_info']
        out_parity = None
        out_par_tentative = False
        
        if p_info:
            out_parity = p_info['symbol']
            
            if p_info['source'] == 'global_suffix':
                 out_par_tentative = p_info['is_tentative']
                 
            elif p_info['source'] == 'local' or p_info['source'] == 'inherited':
                if is_wrapper_tentative:
                     out_par_tentative = True
                else:
                     out_par_tentative = p_info['is_tentative']
        
        results.append({
            "twoTimesSpin": two_spin,
            "isTentativeSpin": item_spin_tentative,
            "parity": out_parity,
            "isTentativeParity": out_par_tentative
        })
        
    return results

def parse_log_line(line):
    # Expected format: E_level = 1000(3) keV; Jπ: unknown. [Gammas: 1500(3) keV (BR: 100), 2000(1) keV (BR: 50).]
    # Regex: E_level = (value)((unc)) keV; Jπ: (string) [optional gamma section]
    
    # First extract energy and Jπ
    pattern = r"E_level\s*=\s*([\d\.]+)\(?([\d\.]*)\)?\s*keV;\s*Jπ:\s*([^\.]+)"
    match = re.search(pattern, line)
    
    if not match:
        return None
        
    energy_val = float(match.group(1))
    unc_str = match.group(2)
    unc_val = float(unc_str) if unc_str else 0.0
    jpi_raw = match.group(3).strip()
    
    spin_parity_data = parse_jpi(jpi_raw)
    
    level_obj = {
        "energy": {
            "value": energy_val,
            "uncertainty": { "value": unc_val },
            "evaluatorInput": f"{energy_val:.0f} {unc_val:.0f}" if unc_val >= 1 else f"{energy_val} {unc_val}"
        },
        "spinParity": {
            "values": spin_parity_data,
            "evaluatorInput": jpi_raw
        }
    }
    
    # Parse gamma decays if present
    # Format: Gammas: 1500(3) keV (BR: 100), 2000(1) keV (BR: 50).
    gamma_pattern = r"Gammas:\s*(.+?)(?:\.|$)"
    gamma_match = re.search(gamma_pattern, line)
    
    if gamma_match:
        gamma_str = gamma_match.group(1).strip()
        gamma_decays = []
        
        # Parse individual gamma entries: "1500(3) keV (BR: 100)"
        gamma_entries = re.findall(r'([\d\.]+)\(([\d\.]+)\)\s*keV\s*\(BR:\s*([\d\.]+)\)', gamma_str)
        
        for gamma_energy_str, gamma_unc_str, branching_ratio_str in gamma_entries:
            gamma_decays.append({
                "energy": float(gamma_energy_str),
                "branching_ratio": float(branching_ratio_str)
            })
        
        # Normalize branching ratios relative to strongest branch (ENSDF convention: strongest = 100)
        if gamma_decays:
            maximum_branching_ratio = max(g["branching_ratio"] for g in gamma_decays)
            if maximum_branching_ratio > 0:
                normalization_factor = 100.0 / maximum_branching_ratio
                for gamma in gamma_decays:
                    gamma["branching_ratio"] *= normalization_factor
            
            level_obj["gamma_decays"] = gamma_decays
    
    return level_obj

def convert_log_to_datasets(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_dataset = None
    datasets = {}  # Key: 'A', 'B', etc. Value: list of levels with temp gamma data

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for dataset header
        header_match = re.search(r"#\s*Dataset\s*([A-Z0-9]+)\s*:", line, re.IGNORECASE)
        if header_match:
            current_dataset = header_match.group(1)
            datasets[current_dataset] = []
            print(f"Found Dataset: {current_dataset}")
            continue
        
        # Parse Level Line
        if line.startswith("E_level"):
            level_data = parse_log_line(line)
            if level_data and current_dataset:
                datasets[current_dataset].append(level_data)
    
    # Write JSON files with proper ENSDF structure
    for code, levels in datasets.items():
        filename = f"test_dataset_{code}.json"
        
        # Build gammasTable from level gamma_decays
        gammas_table = []
        gamma_index = 0
        
        for level_index, level in enumerate(levels):
            if "gamma_decays" in level:
                level_gamma_indices = []
                initial_level_energy = level["energy"]["value"]
                
                for gamma_data in level["gamma_decays"]:
                    gamma_energy = gamma_data["energy"]
                    
                    # Calculate final level: E_final = E_initial - E_gamma
                    final_level_energy = initial_level_energy - gamma_energy
                    
                    # Find the level index that matches this energy (closest match within tolerance)
                    final_level_index = 0  # Default to ground state
                    tolerance = 50.0  # Increased tolerance for datasets with large uncertainties
                    min_diff = float('inf')
                    best_match_idx = 0
                    
                    for idx, lvl in enumerate(levels):
                        lvl_energy = lvl["energy"]["value"]
                        diff = abs(lvl_energy - final_level_energy)
                        if diff < min_diff:
                            min_diff = diff
                            best_match_idx = idx
                            
                    if min_diff < tolerance:
                        final_level_index = best_match_idx
                    
                    # Create gamma entry in ENSDF format
                    gamma_entry = {
                        "energy": {
                            "value": gamma_energy,
                            "unit": "keV"
                        },
                        "gammaIntensity": {
                            "value": gamma_data["branching_ratio"]
                        },
                        "initialLevel": level_index,
                        "finalLevel": final_level_index
                    }
                    gammas_table.append(gamma_entry)
                    level_gamma_indices.append(gamma_index)
                    gamma_index += 1
                
                # Replace gamma_decays with gamma indices array
                level["gammas"] = level_gamma_indices
                del level["gamma_decays"]
        
        # Build final structure
        structure = {
            "levelsTable": {
                "levels": levels
            }
        }
        
        # Add gammasTable only if there are gammas
        if gammas_table:
            structure["gammasTable"] = {
                "gammas": gammas_table
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=4)
        print(f"Generated {filename} with {len(levels)} levels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ENSDF-style log files to structured JSON datasets.")
    parser.add_argument("input_file", nargs='?', default="evaluatorInput.log", help="Path to the input log file")
    
    args = parser.parse_args()
    convert_log_to_datasets(args.input_file)
