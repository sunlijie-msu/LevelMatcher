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
            spin_value = float(eval(spin_raw))
            two_spin = int(round(spin_value * 2))
            
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

def parse_ensdf_line(line):
    """
    Parses a single ENSDF line (L or G record) using fixed-width slicing.
    """
    if len(line) < 8:
        return None, None
        
    record_type = line[7]
    if record_type not in ['L', 'G']:
        return None, None
        
    energy_str = line[9:19].strip()
    unc_str = line[19:21].strip()
    
    if not energy_str:
        return None, None
        
    energy_val = float(energy_str)
    unc_val = calculate_absolute_uncertainty(energy_str, unc_str)
    
    if record_type == 'L':
        jpi_raw = line[22:39].strip()
        data = {
            "energy": {
                "unit": "keV",
                "value": energy_val,
                "uncertainty": { "value": unc_val, "type": "symmetric" } if unc_val > 0 else {"type": "unreported"}
            },
            "isStable": False,
            "gamma_decays": []
        }
        if energy_str:
            data["energy"]["evaluatorInput"] = format_evaluator_input(energy_str, unc_str)
            
        jpi_values = parse_jpi(jpi_raw)
        if jpi_values or jpi_raw:
            data["spinParity"] = {}
            if jpi_values:
                data["spinParity"]["values"] = jpi_values
            if jpi_raw:
                data["spinParity"]["evaluatorInput"] = jpi_raw
                
        return "L", data
    
    elif record_type == 'G':
        ri_str = line[22:29].strip()
        dri_str = line[29:31].strip()
        
        ri_val = float(ri_str) if ri_str else 0.0
        dri_val = calculate_absolute_uncertainty(ri_str, dri_str) if dri_str else 0.0
        
        data = {
            "energy_val": energy_val,
            "energy_unc": unc_val,
            "energy_input": format_evaluator_input(energy_str, unc_str) if energy_str else None,
            "branching_ratio": ri_val,
            "bra_unc": dri_val,
            "br_input": format_evaluator_input(ri_str, dri_str) if ri_str else None
        }
        return "G", data

    return None, None

def convert_log_to_datasets(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    datasets = {}
    current_ds = None
    last_level = None
    
    for line in lines:
        raw_line = line # Keep for slicing
        line = line.strip()
        if not line: continue
        
        if line.startswith("# Dataset"):
            parts = line.split("Dataset")
            if len(parts) > 1:
                current_ds = parts[1].split(":")[0].strip()
                datasets[current_ds] = []
                last_level = None
                print(f"Processing Dataset {current_ds}...")
            continue
            
        rec_type, rec_data = parse_ensdf_line(raw_line)
        
        if rec_type == "L" and current_ds is not None:
            datasets[current_ds].append(rec_data)
            last_level = rec_data
        elif rec_type == "G" and last_level is not None:
            last_level["gamma_decays"].append(rec_data)

    # Output JSONs
    for code, levels in datasets.items():
        gammas_table = []
        gamma_counter = 0
        
        for lvl_idx, level in enumerate(levels):
            if "gamma_decays" in level:
                has_gammas = len(level["gamma_decays"]) > 0
                level_gammas = []
                initial_E = level["energy"]["value"]
                
                for g_data in level["gamma_decays"]:
                    g_E = g_data["energy_val"]
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
                            "unit": "keV",
                            "value": g_E,
                            "uncertainty": { "value": g_data["energy_unc"], "type": "symmetric" } if g_data["energy_unc"] > 0 else {"type": "unreported"}
                        },
                        "gammaIntensity": {
                            "value": g_data["branching_ratio"],
                            "uncertainty": { "value": g_data["bra_unc"], "type": "symmetric" } if g_data["bra_unc"] > 0 else {"type": "unreported"}
                        },
                        "initialLevel": lvl_idx,
                        "finalLevel": final_idx
                    }
                    if g_data.get("energy_input"):
                        g_entry["energy"]["evaluatorInput"] = g_data["energy_input"]
                    if g_data.get("br_input"):
                        g_entry["gammaIntensity"]["evaluatorInput"] = g_data["br_input"]
                        
                    gammas_table.append(g_entry)
                    level_gammas.append(gamma_counter)
                    gamma_counter += 1
                
                if level_gammas:
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
