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
    # Expected format: E_level = 1000(3) keV; Jπ: unknown.
    # Regex: E_level = (value)((unc)) keV; Jπ: (string)
    pattern = r"E_level\s*=\s*([\d\.]+)\(([\d\.]+)\)\s*keV;\s*Jπ:\s*(.*)"
    match = re.search(pattern, line)
    if match:
        energy_val = float(match.group(1))
        unc_val = float(match.group(2))
        jpi_raw = match.group(3).strip().rstrip('.')
        
        # Reconstruct evaluatorInput string for Energy (e.g., "1000 3")
        # Assuming unc is absolute from log format context in example
        
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
        return level_obj
    return None

def convert_log_to_datasets(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_dataset = None
    datasets = {}  # Key: 'A', 'B', etc. Value: list of levels

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for dataset header
        # Relaxed regex: case insensitive, allows space before colon
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
    
    # Write JSON files
    for code, levels in datasets.items():
        filename = f"test_dataset_{code}.json"
        
        # If file exists, try to preserve structure (like wrapping in "levelsTable")
        # The prompt implies strictly "update ... according to evaluator input"
        # We will generate the standard structure used in this project
        
        structure = {
            "levelsTable": {
                "levels": levels
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=4)
        print(f"Generated {filename} with {len(levels)} levels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ENSDF-style log files to structured JSON datasets.")
    parser.add_argument("input_file", nargs='?', default="evaluatorInput.log", help="Path to the input log file")
    
    args = parser.parse_args()
    convert_log_to_datasets(args.input_file)
