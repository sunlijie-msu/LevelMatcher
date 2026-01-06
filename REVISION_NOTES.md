# LevelMatcher Revision Notes (Gemini Refactor)

## Summary of Changes
This revision transitions the project from a hardcoded, column-based logic (Spin, Parity columns) to a flexible, ENSDF-compliant architecture (Spin_Parity string).

## New Architecture

### 1. Physics Parser (`physics_parser.py`)
- **Centralized Logic:** All string parsing for Spin/Parity (e.g., `"3/2+,5/2+"`, `"(1,2)-"`) is now handled here.
- **QuantumState Class:** Represents a parsed state with:
  - `J` (Spin): List of possible floats (e.g., `[1.5, 2.5]`).
  - `Pi` (Parity): Int `+1` or `-1`.
  - `Is_Tentative`: Boolean flag derived from parentheses.
- **Physics Distance:** Replced the binary "Veto" with a continuous distance metric:
  - `0.00`: **Strict Match** (Identical J/Pi).
  - `0.25`: **Tentative Match** (One side is tentative, or overlap exists).
  - `0.50`: **Neutral/Unknown** (Missing data on one side).
  - `1.00`: **Mismatch** (Conflict in firm assignments).

### 2. Data Schema (`.json` files)
- Removed `Spin` and `Parity` columns.
- Added `Spin_Parity` field (string).
- Example: `"Spin_Parity": "3/2+"` or `"Spin_Parity": null`.

### 3. Main Matcher (`Level_Matcher_Gemini.py`)
- **Simplified Loop:** No longer parses regex inline. Calls `calculate_physics_distance(sp1, sp2)`.
- **ML Model Update:** XGBoost now trains on 2 features: `[Z_Score, Physics_Distance]`.
- **Logic:**
  - `Physics_Distance` allows the model to learn "Soft Vetos" (e.g., Tentative mismatches punish probability less than Hard mismatches).

## Usage
1. Ensure `physics_parser.py` is in the directory.
2. Run `python Level_Matcher_Gemini.py`.
3. Adjust `training_data` in the main script to tune how strictly the model penalizes the Physics Distance.
