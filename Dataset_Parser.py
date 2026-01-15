import json
import re
import os
import argparse

def parse_jpi(jpi_str):
    """
    Parses Jπ string into structured format.
    Handles ranges "1:3", tentative "(1)-", and lists "1+,2+".
    """
    if not jpi_str or jpi_str.lower() in ['unknown', 'none']: 
        return []

    clean_str = jpi_str.strip().rstrip('.')

    # 1. Range support (e.g. 1:3)
    if ':' in clean_str and all(c.isdigit() or c.isspace() or c == ':' for c in clean_str):
        try:
            start_s, end_s = map(int, clean_str.split(':'))
            return [{
                "twoTimesSpin": s * 2,
                "isTentativeSpin": False,
                "parity": None,
                "isTentativeParity": False
            } for s in range(start_s, end_s + 1)]
        except ValueError: pass

    # 2. List parsing
    # Handle global parentheses like (1,2)+
    is_global_tentative = clean_str.startswith('(') and clean_str.endswith(')') and ',' in clean_str
    if is_global_tentative: 
        clean_str = clean_str[1:-1]

    parts = [p.strip() for p in clean_str.split(',')]
    results = []

    for part in parts:
        if not part: continue
        
        # Heuristic parsing for "3/2+" or "(5/2-)" or "1(+)"
        # Extract parity
        parity = '+' if '+' in part else '-' if '-' in part else None
        
        # Check tentativeness (local parens)
        is_local_tentative = '(' in part or ')' in part
        is_tentative = is_global_tentative or is_local_tentative

        # Extract numeric spin: remove parens, parity signs
        spin_raw = part.replace('+', '').replace('-', '').replace('(', '').replace(')', '').strip()
        
        try:
            val = float(eval(spin_raw))
            two_spin = int(round(val * 2))
            
            results.append({
                "twoTimesSpin": two_spin,
                "isTentativeSpin": is_tentative,
                "parity": parity,
                "isTentativeParity": is_tentative if parity else False
            })
        except: continue
        
    return results

def calculate_absolute_uncertainty(val_str, unc_str):
    """
    Calculates absolute uncertainty based on ENSDF convention.
    123(12) -> 12
    123.4(12) -> 1.2
    0.123(4) -> 0.0004
    """
    if not unc_str:
        return 0.0
    
    val_str = val_str.strip()
    if '.' in val_str:
        decimal_part = val_str.split('.')[1]
        decimals = len(decimal_part)
    else:
        decimals = 0
        
    return float(unc_str) * (10 ** -decimals)

def format_evaluator_input(val_str, unc_str):
    if not unc_str:
        return val_str
    return f"{val_str} {unc_str}"

def parse_log_line(line):
    # Regex for Level: E_level = 1000(3) [keV]; Jπ: ... [Gammas: ...]
    pattern = r"E_level\s*=\s*([\d\.]+)(?:\(([\d\.]+)\))?(?:\s*keV)?;\s*Jπ:\s*([^;\.]+)(?:(?:\.|;)\s*Gammas:\s*(.*))?"
    match = re.search(pattern, line)
    
    if not match: return None
        
    # Capture raw strings for formatting
    energy_str = match.group(1)
    unc_str = match.group(2)
    jpi_raw = match.group(3).strip()
    
    # Calculate values
    energy_val = float(energy_str)
    unc_val = calculate_absolute_uncertainty(energy_str, unc_str)
    
    level_obj = {
        "energy": {
            "value": energy_val,
            "uncertainty": { "value": unc_val, "type": "symmetric" } if unc_val > 0 else {"value": 0.0},
            "evaluatorInput": format_evaluator_input(energy_str, unc_str)
        },
        "spinParity": {
            "values": parse_jpi(jpi_raw),
            "evaluatorInput": jpi_raw
        }
    }
    
    # Parse Gammas
    gamma_str = match.group(4)
    if gamma_str:
        gamma_decays = []
        # Regex: 1400(3) [keV] (BR: 100(5))
        entries = re.findall(r'([\d\.]+)(?:\(([\d\.]+)\))?(?:\s*keV)?\s*\(BR:\s*([\d\.]+)(?:\(([\d\.]+)\))?\)', gamma_str)
        
        for g_en_str, g_unc_str, g_br_str, g_br_unc_str in entries:
            g_unc_val = calculate_absolute_uncertainty(g_en_str, g_unc_str)
            g_br_unc_val = calculate_absolute_uncertainty(g_br_str, g_br_unc_str)
            
            gamma_decays.append({
                "energy": float(g_en_str),
                "energy_unc": g_unc_val,
                "energy_input": format_evaluator_input(g_en_str, g_unc_str),
                "branching_ratio": float(g_br_str),
                "bra_unc": g_br_unc_val,
                "br_input": format_evaluator_input(g_br_str, g_br_unc_str)
            })
                            
            level_obj["gamma_decays"] = gamma_decays
            
    return level_obj

def convert_log_to_datasets(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    datasets = {}
    current_ds = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith("# Dataset"):
            parts = line.split("Dataset")
            if len(parts) > 1:
                current_ds = parts[1].split(":")[0].strip()
                datasets[current_ds] = []
                print(f"Processing Dataset {current_ds}...")
            continue
            
        if line.startswith("E_level") and current_ds:
            lvl = parse_log_line(line)
            if lvl: datasets[current_ds].append(lvl)

    # Output JSONs
    for code, levels in datasets.items():
        gammas_table = []
        gamma_counter = 0
        
        for lvl_idx, level in enumerate(levels):
            if "gamma_decays" in level:
                level_gammas = []
                initial_E = level["energy"]["value"]
                
                for g_data in level["gamma_decays"]:
                    g_E = g_data["energy"]
                    final_E_target = initial_E - g_E
                    
                    # Match Final Level
                    best_match = 0
                    min_diff = 1e9
                    for c_idx, cand in enumerate(levels):
                        diff = abs(cand["energy"]["value"] - final_E_target)
                        if diff < min_diff:
                            min_diff = diff
                            best_match = c_idx
                            
                    final_idx = best_match if min_diff <= 50.0 else 0
                    
                    # Structuring Gamma Entry
                    g_entry = {
                        "energy": {
                            "value": g_E,
                            "uncertainty": { "value": g_data["energy_unc"], "type": "symmetric" } if g_data["energy_unc"] > 0 else {"value": 0.0},
                            "evaluatorInput": g_data["energy_input"]
                        },
                        "gammaIntensity": {
                            "value": g_data["branching_ratio"],
                            "uncertainty": { "value": g_data["bra_unc"], "type": "symmetric" } if g_data["bra_unc"] > 0 else {"value": 0.0},
                            "evaluatorInput": g_data["br_input"]
                        },
                        "initialLevel": lvl_idx,
                        "finalLevel": final_idx
                    }
                    gammas_table.append(g_entry)
                    level_gammas.append(gamma_counter)
                    gamma_counter += 1
                
                level["gammas"] = level_gammas
                del level["gamma_decays"]
        
        output = { "levelsTable": { "levels": levels } }
        if gammas_table:
            output["gammasTable"] = { "gammas": gammas_table }
            
        with open(f"test_dataset_{code}.json", 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"Dataset {code} saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default="evaluatorInput.log")
    args = parser.parse_args()
    convert_log_to_datasets(args.input_file)
