# Feature Correlation Implementation Task Plan

**Created:** January 15, 2026  
**Objective:** Implement feature correlation where perfect spin+parity can "rescue" mediocre energy similarity  
**Status:** ✅ COMPLETE - All phases validated

---

## Implementation Status

### ✅ ALL COMPLETE

1. **Feature_Engineer.py:**
   - Added `Feature_Correlation` config block to `Scoring_Config`
   - Updated `generate_synthetic_training_data()` with Physics Rescue logic
   - Updated header docstring with Feature Correlation explanation

2. **Level_Matcher.py:**
   - Threshold correctly set: `pairwise_output_threshold = 0.001` (0.1%)
   - Header docstring updated with Feature Correlation

3. **README.md:**
   - Workflow section updated (line 86): Feature Correlation documented
   - Configuration section complete with all parameters

4. **Runtime Validation:**
   - Exit Code: 0 ✅
   - Output: 127 level pairs (>0%) generated
   - Key metric: `A_3005 <-> B_3000` probability = **81.6%** (vs baseline ~56.5%)
   - Feature correlation successfully boosted pairs with perfect spin+parity

5. **Visual Validation:**
   - Exit Code: 0 ✅
   - Both PNGs generated: `Input_Level_Scheme.png`, `Output_Cluster_Scheme.png`

---

## Remaining Tasks

### Phase 1: Verify Current Implementation
**Purpose:** Ensure Feature_Engineer.py code is correct and complete

**Actions:**
1. Read `Feature_Engineer.py` lines 570-650 → Verify `generate_synthetic_training_data()` Physics Rescue logic
2. Read `Feature_Engineer.py` lines 1-30 → Verify header docstring mentions Feature Correlation
3. Read `Level_Matcher.py` lines 1-35 → Verify threshold is 0.001 and docstring updated

**Success Criteria:**
- Physics Rescue code present and syntactically correct
- Both file headers document Feature Correlation
- Threshold = 0.001 confirmed

---

### Phase 2: Update README.md
**Purpose:** Document Feature Correlation for end users

**Location:** `README.md` Section "Workflow" → Step 2 "Model Training"

**Current Text (Line ~55):**
```markdown
2.  **Model Training (Supervised Learning):**
    *   Generates 580+ synthetic training samples encoding physics rules across six scenarios...
```

**Add After Existing Bullets:**
```markdown
    *   Implements **Feature Correlation**: Perfect spin+parity (≥0.95) triggers "Physics Rescue" where energy similarity is boosted via sqrt transformation
```

**Success Criteria:**
- README.md updated
- No other changes to README.md needed (Configuration section already complete)

---

### Phase 3: Runtime Validation (MANDATORY per copilot-instructions.md)
**Purpose:** Verify code executes without errors and produces expected output

**Actions:**
1. Run `python Level_Matcher.py` in terminal
2. Check Exit Code → MUST be 0
3. Verify console output shows: `[INFO] Pairwise Inference Complete: XXX level pairs (>0%)`
4. Spot-check `Output_Level_Pairwise_Inference.txt`:
   - Find pair `A_3005 <-> B_3000`
   - Verify probability >70% (feature correlation boost expected)
   - Verify features show: Energy_Sim ~0.6, Spin_Sim=1.0, Parity_Sim=1.0

**Success Criteria:**
- Exit Code 0
- Output file generated
- A_3005 <-> B_3000 probability increased from baseline (~56.5% → ~80%+)

---

### Phase 4: Visual Verification
**Purpose:** Ensure visualizer still works after all changes

**Actions:**
1. Run `python Combined_Visualizer.py` in terminal
2. Check Exit Code → MUST be 0
3. Verify both PNG files generated:
   - `Input_Level_Scheme.png`
   - `Output_Cluster_Scheme.png`

**Success Criteria:**
- Exit Code 0
- Both PNGs present with recent timestamps

---

## Recovery Instructions (If Crashes Occur)

**If VS Code crashes during Phase 2 (README.md update):**
- Resume from Phase 2
- Open `README.md`
- Navigate to line ~55 (Workflow section, step 2)
- Add Feature Correlation bullet as specified above
- Proceed to Phase 3

**If VS Code crashes during Phase 3 (Testing):**
- Resume from Phase 3
- Run `python Level_Matcher.py` only
- Skip visual verification if terminal tests pass

**If implementation is complete but untested:**
- Skip directly to Phase 3 (Runtime Validation)
- Testing is MANDATORY per `.github/copilot-instructions.md`

---

## Expected Final State

**Files Modified:**
1. `Feature_Engineer.py` — Code + header docstring ✅
2. `Level_Matcher.py` — Threshold + header docstring ✅
3. `README.md` — Workflow section updated ⏳
4. All outputs tested ⏳

**Key Metrics After Completion:**
- Training data size: 580+ samples (up from ~200 baseline)
- Pair `A_3005 <-> B_3000`: Probability >70% (vs ~56.5% baseline)
- Feature Correlation: Active in `Scoring_Config['Feature_Correlation']['Enabled'] = True`

---

## Technical Details

### Physics Rationale
Two levels with **identical quantum numbers** (Jπ) but **different energies** may still be the same level if:
1. They are both **isolated** (no nearby levels with same Jπ)
2. Energy disagreement caused by **calibration error** or **systematic uncertainty**
3. Spin+parity agreement is **firm** (not tentative, similarity ≥0.95)

### Mathematical Implementation
```python
# In generate_synthetic_training_data()
if spin_sim >= 0.95 and parity_sim >= 0.95:
    effective_energy = energy_sim ** Rescue_Exponent  # Default: 0.5 (sqrt)
```

**Transformation Examples:**
- energy_sim=0.4 → effective=0.63 (+58% boost)
- energy_sim=0.6 → effective=0.77 (+28% boost)
- energy_sim=0.8 → effective=0.89 (+11% boost)

### XGBoost Learning
Model learns interaction via training examples:
- `[e=0.4, s=1.0, p=1.0] → label=high` (with rescue)
- `[e=0.4, s=0.8, p=0.8] → label=low` (no rescue)

Tree splits on spin/parity BEFORE energy when both features high, effectively learning:
```
IF (spin ≥ 0.95 AND parity ≥ 0.95):
    energy_penalty = gentle
ELSE:
    energy_penalty = strict
```

---

## Compliance Checklist

- ✅ Read `.github/copilot-instructions.md` before starting
- ✅ Use `replace_string_in_file` for all edits (preserve VS Code diff)
- ⏳ Test immediately after each code change (Phase 3 pending)
- ✅ Update documentation to reflect code changes
- ✅ Preserve all existing comments and code structure
- ⏳ Provide compliance checklist to user

---

**End of Task Plan**
