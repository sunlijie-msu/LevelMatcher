"""
Dataset Parser for Nuclear Level Matcher
=========================================

# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Start] -> [Raw Evaluated Nuclear Structure Data File Strings] -> [Parser Engine] -> [Structured Output]
                                     |
                                     v
                             [Inference Rules]
                            /        |        \
      [Uncertainties (Precision)] [Spin-Parity Lists]  [Gamma Decays]
                            \        |        /
                             v       v       v
                     [Standardized JSON Schema (Levels/Gammas)]

Technical Steps:
1. Parse raw strings from Evaluated Nuclear Structure Data File (ENSDF) for energies, uncertainties, spins, and parities.
2. Infer uncertainties from precision (significant figures) where explicit values are missing.
3. Standardize data into a consistent JSON schema for the pipeline.
4. Output structured files to `data/raw/` for ingestion by Feature Engineer.

Architecture:
- `infer_uncertainty_from_precision`: Heuristic engine for uncertainty estimation.
- `standardize_spin_parity`: Normalizes Spin-Parity strings (e.g., "(3/2,5/2)+" -> distinct hypotheses).
- `generate_datasets`: Main driver creating input files from embedded raw data (or file inputs).
"""

import json
import re
import os
import argparse

def infer_uncertainty_from_precision(value_string):
    """
    Infers uncertainty from reported precision in Evaluated Nuclear Structure Data File (ENSDF) evaluatorInput string.
    
    Nuclear physics convention: Uncertainty = 0.5 × least_significant_digit_place_value
    
    Examples:
    - "2000" → ±5 keV (integer, precision to 10s place)
    - "2.0E3" → ±500 keV (scientific notation, 1 decimal in mantissa)
    - "2.00E3" → ±50 keV (scientific notation, 2 decimals in mantissa)
    - "1234.5" → ±0.5 keV (1 decimal place)
    - "567.89" → ±0.05 keV (2 decimal places)
    
    Returns: Inferred uncertainty as float, or 5.0 (conservative default) if cannot parse
    """
    if not value_string:
        return 5.0
    
    value_string = value_string.strip()
    
    # Handle scientific notation (e.g., "2.0E3" or "1.5e+02")
    if 'E' in value_string.upper():
        parts = value_string.upper().split('E')
        if len(parts) != 2:
            return 5.0
        
        mantissa = parts[0]
        try:
            exponent = int(parts[1])
        except ValueError:
            return 5.0
        
        # Count decimal places in mantissa (DO NOT strip trailing zeros)
        # Example: "2.0E3" has 1 decimal → ±500 keV
        # Example: "2.00E3" has 2 decimals → ±50 keV
        if '.' in mantissa:
            decimal_part = mantissa.split('.')[1]
            decimal_places = len(decimal_part)
            
            # Uncertainty: 5 × 10^(-decimal_places) × 10^exponent
            mantissa_uncertainty = 5.0 * (10 ** (-decimal_places))
            return mantissa_uncertainty * (10 ** exponent)
        else:
            # Integer mantissa: "2E3" → ±5×10^3 = ±5000
            return 5.0 * (10 ** exponent)
    
    # Handle regular decimal notation
    if '.' in value_string:
        # Count decimal places WITHOUT stripping trailing zeros
        # Example: "2000.0" has 1 decimal place → precision to 0.1 keV → ±0.5 keV
        # Example: "1234.56" has 2 decimal places → precision to 0.01 keV → ±0.05 keV
        decimal_part = value_string.split('.')[1]
        decimal_places = len(decimal_part)
        return 5.0 * (10 ** (-decimal_places))
    else:
        # Integer: precision to nearest 10
        return 5.0

def parse_spin_parity(spin_parity_string):
    """
    Parses Spin-Parity string into structured format.
    Handles ranges "1:3", tentative "(1)-", and lists "1+,2+".
    """
    if not spin_parity_string or spin_parity_string.lower() in ['unknown', 'none']: 
        return []

    clean_string = spin_parity_string.strip().rstrip('.')

    # 1. Range support (e.g. 1:3)
    if ':' in clean_string and all(c.isdigit() or c.isspace() or c == ':' for c in clean_string):
        try:
            start_spin, end_spin = map(int, clean_string.split(':'))
            return [{
                "twoTimesSpin": spin_index * 2,
                "isTentativeSpin": False,
                "parity": None,
                "isTentativeParity": False
            } for spin_index in range(start_spin, end_spin + 1)]
        except ValueError: pass

    # 2. List parsing
    # Handle global parentheses like (1,2)+
    is_global_tentative = clean_string.startswith('(') and clean_string.endswith(')') and ',' in clean_string
    if is_global_tentative: 
        clean_string = clean_string[1:-1]

    parts = [part.strip() for part in clean_string.split(',')]
    results = []

    for part in parts:
        if not part: continue
        
        # Heuristic parsing for "3/2+" or "(5/2-)" or "1(+)"
        # Extract parity
        parity_value = '+' if '+' in part else '-' if '-' in part else None
        
        # Check tentativeness (local parentheses)
        is_local_tentative = '(' in part or ')' in part
        is_tentative = is_global_tentative or is_local_tentative

        # Extract numeric spin: remove parentheses, parity signs
        spin_raw = part.replace('+', '').replace('-', '').replace('(', '').replace(')', '').strip()
        
        try:
            spin_value_parsed = float(eval(spin_raw))
            two_times_spin_value = int(round(spin_value_parsed * 2))
            
            results.append({
                "twoTimesSpin": two_times_spin_value,
                "isTentativeSpin": is_tentative,
                "parity": parity_value,
                "isTentativeParity": is_tentative if parity_value else False
            })
        except: continue
        
    return results

def calculate_absolute_uncertainty(value_string, uncertainty_string):
    """
    Calculates absolute uncertainty based on Evaluated Nuclear Structure Data File convention.
    123(12) -> 12
    123.4(12) -> 1.2
    0.123(4) -> 0.0004
    """
    if not uncertainty_string:
        return 0.0
    
    value_string = value_string.strip()
    if '.' in value_string:
        decimal_part = value_string.split('.')[1]
        decimals = len(decimal_part)
    else:
        decimals = 0
        
    return float(uncertainty_string) * (10 ** -decimals)

def format_evaluator_input(value_string, uncertainty_string):
    if not uncertainty_string:
        return value_string
    return f"{value_string} {uncertainty_string}"

def parse_ensdf_line(line):
    """
    Parses a single ENSDF line (L or G record) using fixed-width slicing.
    Applies precision-based uncertainty inference when explicit uncertainty is missing.
    """
    if len(line) < 8:
        return None, None
        
    record_type = line[7]
    if record_type not in ['L', 'G']:
        return None, None
        
    energy_string = line[9:19].strip()
    uncertainty_string = line[19:21].strip()
    
    if not energy_string:
        return None, None
        
    energy_value = float(energy_string)
    
    # Calculate uncertainty: explicit if provided, otherwise infer from precision
    if uncertainty_string:
        uncertainty_value = calculate_absolute_uncertainty(energy_string, uncertainty_string)
    elif energy_value == 0.0:
        # Ground state (0.0 keV) is the absolute reference => 0 uncertainty
        uncertainty_value = 0.0
    else:
        uncertainty_value = infer_uncertainty_from_precision(energy_string)
    
    if record_type == 'L':
        spin_parity_raw = line[22:39].strip()
        data = {
            "energy": {
                "unit": "keV",
                "value": energy_value,
                "uncertainty": { 
                    "value": uncertainty_value, 
                    "type": "symmetric" if uncertainty_string else "inferred"
                }
            },
            "isStable": False,
            "gamma_decays": []
        }
        if energy_string:
            data["energy"]["evaluatorInput"] = format_evaluator_input(energy_string, uncertainty_string)
            
        spin_parity_values = parse_spin_parity(spin_parity_raw)
        if spin_parity_values or spin_parity_raw:
            data["spinParity"] = {}
            if spin_parity_values:
                data["spinParity"]["values"] = spin_parity_values
            if spin_parity_raw:
                data["spinParity"]["evaluatorInput"] = spin_parity_raw
                
        return "L", data
    
    elif record_type == 'G':
        relative_intensity_string = line[22:29].strip()
        delta_relative_intensity_string = line[29:31].strip()
        
        relative_intensity_value = float(relative_intensity_string) if relative_intensity_string else 0.0
        
        # Infer intensity uncertainty if not explicitly provided
        if delta_relative_intensity_string:
            intensity_uncertainty_value = calculate_absolute_uncertainty(relative_intensity_string, delta_relative_intensity_string)
        elif relative_intensity_value > 0:
            intensity_uncertainty_value = infer_uncertainty_from_precision(relative_intensity_string)
        else:
            intensity_uncertainty_value = 0.0
        
        data = {
            "energy_value": energy_value,
            "energy_uncertainty": uncertainty_value,
            "energy_input_string": format_evaluator_input(energy_string, uncertainty_string) if energy_string else None,
            "relative_intensity_value": relative_intensity_value,
            "intensity_uncertainty_value": intensity_uncertainty_value,
            "intensity_input_string": format_evaluator_input(relative_intensity_string, delta_relative_intensity_string) if relative_intensity_string else None
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
    current_dataset_id = None
    last_level_data = None
    
    for line in lines:
        raw_line = line # Keep for slicing
        line = line.strip()
        if not line: continue
        
        if line.startswith("# Dataset"):
            parts = line.split("Dataset")
            if len(parts) > 1:
                current_dataset_id = parts[1].split(":")[0].strip()
                datasets[current_dataset_id] = []
                last_level_data = None
                print(f"Processing Dataset {current_dataset_id}...")
            continue
            
        record_type, record_data = parse_ensdf_line(raw_line)
        
        if record_type == "L" and current_dataset_id is not None:
            datasets[current_dataset_id].append(record_data)
            last_level_data = record_data
        elif record_type == "G" and last_level_data is not None:
            last_level_data["gamma_decays"].append(record_data)

    # Output JSON Files
    for dataset_code, level_list in datasets.items():
        gammas_table_list = []
        gamma_counter = 0
        
        for level_index, level_item in enumerate(level_list):
            if "gamma_decays" in level_item:
                level_gamma_indices = []
                initial_level_energy = level_item["energy"]["value"]
                
                for gamma_data in level_item["gamma_decays"]:
                    gamma_energy_value = gamma_data["energy_value"]
                    final_energy_target = initial_level_energy - gamma_energy_value
                    
                    # Match Final Level
                    best_match_index = 0
                    minimum_difference = 1e9
                    for candidate_index, candidate_level in enumerate(level_list):
                        difference = abs(candidate_level["energy"]["value"] - final_energy_target)
                        if difference < minimum_difference:
                            minimum_difference = difference
                            best_match_index = candidate_index
                            
                    final_level_index = best_match_index if minimum_difference <= 50.0 else 0
                    
                    # Structuring Gamma Entry
                    gamma_entry = {
                        "energy": {
                            "unit": "keV",
                            "value": gamma_energy_value,
                            "uncertainty": { "value": gamma_data["energy_uncertainty"], "type": "symmetric" } if gamma_data["energy_uncertainty"] > 0 else {"type": "unreported"}
                        },
                        "gammaIntensity": {
                            "value": gamma_data["relative_intensity_value"],
                            "uncertainty": { "value": gamma_data["intensity_uncertainty_value"], "type": "symmetric" } if gamma_data["intensity_uncertainty_value"] > 0 else {"type": "unreported"}
                        },
                        "initialLevel": level_index,
                        "finalLevel": final_level_index
                    }
                    if gamma_data.get("energy_input_string"):
                        gamma_entry["energy"]["evaluatorInput"] = gamma_data["energy_input_string"]
                    if gamma_data.get("intensity_input_string"):
                        gamma_entry["gammaIntensity"]["evaluatorInput"] = gamma_data["intensity_input_string"]
                        
                    gammas_table_list.append(gamma_entry)
                    level_gamma_indices.append(gamma_counter)
                    gamma_counter += 1
                
                if level_gamma_indices:
                    level_item["gammas"] = level_gamma_indices
                del level_item["gamma_decays"]
        
        output_data_structure = { "levelsTable": { "levels": level_list } }
        if gammas_table_list:
            output_data_structure["gammasTable"] = { "gammas": gammas_table_list }
            
        # Robustly determine output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, 'data', 'raw', f"test_dataset_{dataset_code}.json")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data_structure, f, indent=4)
        print(f"Dataset {dataset_code} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default="evaluatorInput.log")
    args = parser.parse_args()
    convert_log_to_datasets(args.input_file)
