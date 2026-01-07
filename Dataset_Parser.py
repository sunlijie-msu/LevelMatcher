import json
import re
import os

def parse_jpi(jpi_str):
    """
    Parses a Jπ string into the project's structured JSON format.
    Handles:
    - Single values: "2+", "1-"
    - Tentative values: "(1)-", "2(+)", "(2+)"
    - Lists: "1-, 2-"
    - Ranges: "1:3"
    - Global tentativeness: "(1, 2)+" or "(1+, 2+)"
    """
    jpi_str = jpi_str.strip().rstrip('.')
    if not jpi_str or jpi_str.lower() == 'unknown':
        return []
    
    # Handle range 1:3 -> 1, 2, 3
    # Logic: if contains ':', try to parse as integer range
    if ':' in jpi_str:
        # Example: 1:3
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
            pass # Fallback to standard parsing if not simple ints

    # Handle lists and global parentheses
    # Example: (1+, 2+, 3+) implies all tentative
    is_global_tentative = False
    content = jpi_str
    if jpi_str.startswith('(') and jpi_str.endswith(')') and ',' in jpi_str:
        is_global_tentative = True
        content = jpi_str[1:-1]
    
    items = [x.strip() for x in content.split(',')]
    results = []
    
    for item in items:
        # Parse Spin and Parity from strings like "2+", "(1)-", "2(+)", "5/2-"
        
        parity = None
        is_tentative_parity = False
        
        # Regex to find parity part: (+), (-), +, -
        # We look for a + or - at the end, possibly in parens
        p_match = re.search(r'(\(?)([\+\-])(\)?)$', item)
        spin_part = item
        
        if p_match:
            full_p_str = p_match.group(0)
            p_symbol = p_match.group(2)
            has_parens = p_match.group(1) == '(' or p_match.group(3) == ')'
            
            parity = p_symbol
            is_tentative_parity = has_parens
            
            # Remove the parity part from the item to isolate spin
            # We use rsplit to ensure we only remove the last occurrence found by regex
            spin_part = item[:item.rfind(full_p_str)]
        
        spin_part = spin_part.strip()
        
        # Check for spin tentativeness: (1) or global parens
        is_tentative_spin = is_global_tentative
        if spin_part.startswith('(') and spin_part.endswith(')'):
            is_tentative_spin = True
            spin_part = spin_part[1:-1].strip()
        
        # Parse numeric spin value
        try:
            two_times_spin = 0
            if '/' in spin_part:
                n, d = spin_part.split('/')
                # Assuming denominator is 2 for half-integer spins
                two_times_spin = int(n)
            elif spin_part.isdigit():
                two_times_spin = int(spin_part) * 2
            else:
                # If we cannot parse the spin (e.g. empty or non-numeric), skip this item
                continue
                
            results.append({
                "twoTimesSpin": two_times_spin,
                "isTentativeSpin": is_tentative_spin,
                "parity": parity,
                "isTentativeParity": is_tentative_parity
            })
        except ValueError:
            continue

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
    convert_log_to_datasets("evaluatorInput.log")
