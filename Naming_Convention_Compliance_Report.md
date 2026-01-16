# Naming Convention Compliance Report (CORRECTED)

**Date:** January 15, 2026  
**Rules Checked:**
1. **No Acronyms:** Variable names must be spelled out completely (e.g., `error` not `err`, `index` not `idx`)
2. **No ALL CAPS:** Avoid ALL CAPS where EVERY letter is uppercase. Use `Title_Case` (e.g., `Scoring_Config`) or `snake_case` (e.g., `scoring_config`) instead.

**Rule Clarification:** ALL CAPS means EVERY character is uppercase (e.g., `FONT_CONFIG`, `MAX_VALUE`). Mixed case like `Scoring_Config` or `Font_Config` is ALLOWED.

---

## ❌ VIOLATIONS FOUND

### 1. ALL CAPS Variable Names

#### **Combined_Visualizer.py (Line 34)**
```python
FONT_CONFIG = {  # ❌ ALL CAPS VIOLATION
```
**Issue:** Dictionary name uses ALL CAPS (every letter uppercase)  
**Rule Violation:** "Avoid using ALL CAPS for variable names, constants, or comments"  
**Fix Required:** Rename to `Font_Config` (Title_Case) or `font_config` (snake_case)  
**Impact:** 15 references throughout Combined_Visualizer.py

---

### 2. ALL CAPS in Comments

#### **Feature_Engineer.py (Line 628)**
```python
        # FEATURE CORRELATION: Physics Rescue  # ❌ VIOLATION
```
**Issue:** Comment header uses ALL CAPS phrase "FEATURE CORRELATION"  
**Rule Violation:** "Avoid using ALL CAPS for comments"  
**Fix Required:** Change to `# Feature Correlation: Physics Rescue` or `# Feature correlation: Physics rescue`  
**Impact:** 1 occurrence

---

### 3. Acronym Violations

#### **Combined_Visualizer.py (Line 243)**
```python
for idx, gamma_data in enumerate(all_gamma_data):  # ❌ VIOLATION
```
**Issue:** Variable `idx` is acronym for "index"  
**Rule Violation:** "Do not use acronyms for variable names. Spell out long variable names completely"  
**Fix Required:** Rename `idx` → `index`  
**Impact:** 2 references (lines 243, 252)

---

#### **Dataset_Parser.py (Lines 52-53)**
```python
val = float(eval(spin_raw))  # ❌ VIOLATION
two_spin = int(round(val * 2))
```
**Issue:** Variable `val` is acronym for "value"  
**Rule Violation:** Same as above  
**Fix Required:** Rename `val` → `spin_value`  
**Impact:** 2 references

---

## Legacy Folder (Low Priority - Deprecated Code)

### **Legacy/Level_Matcher_Legacy.py**

#### ALL CAPS Variables (Lines 35-48)
```python
MATCH_EXCELLENT = [...]  # ❌ VIOLATION
MATCH_TRANSITION = [...]  # ❌ VIOLATION
MATCH_WEAK = [...]       # ❌ VIOLATION
MATCH_VETOED = [...]     # ❌ VIOLATION
```
**Note:** Legacy code, not actively maintained. **Ignore unless reactivated.**

#### Acronym Violations (Lines 280-283)
```python
err = row['DE_level'] if pd.notna(row['DE_level']) else 10.0  # ❌ VIOLATION
vals.append((row['E_level'], err))
weights = [1/err**2 for _, err in vals]
```
**Issue:** Variable `err` is acronym for "error"  
**Note:** Legacy code. **Ignore unless reactivated.**

---

## ✅ NOT VIOLATIONS (Previously Misidentified)

### **Feature_Engineer.py (Line 41)**
```python
Scoring_Config = {  # ✅ ALLOWED (Title_Case)
```
**Status:** COMPLIANT - Uses Title_Case (not ALL CAPS)  
**Explanation:** `Scoring_Config` has mixed case - `S`, `c`, `o`, `r`, `i`, `n`, `g` are lowercase. Only `S` and `C` are uppercase. This is **Title_Case**, which is explicitly allowed per instructions: "Use `Title_Case` (for example, `Scoring_Config`)".

### **Feature_Engineer.py - Dictionary Keys**
```python
'Sigma_Scale': 0.1,              # ✅ ALLOWED (Title_Case)
'Default_Uncertainty': 10.0      # ✅ ALLOWED (Title_Case)  
'Match_Firm': 1.0,               # ✅ ALLOWED (Title_Case)
'Feature_Correlation': {...}     # ✅ ALLOWED (Title_Case)
'Rescue_Exponent': 0.5           # ✅ ALLOWED (Title_Case)
```
**Status:** ALL COMPLIANT - These use Title_Case with underscores  
**Explanation:** None of these are ALL CAPS. They all have lowercase letters mixed in. Per instructions: "Use `Title_Case` (for example, `Scoring_Config`) or `snake_case`" - both are acceptable.

---

## Violation Count Summary

| File | ALL CAPS Variables | ALL CAPS Comments | Acronyms | Total |
|------|-------------------|-------------------|----------|-------|
| Combined_Visualizer.py | 1 (`FONT_CONFIG`) | 0 | 1 (`idx`) | **2** |
| Feature_Engineer.py | 0 | 1 (`FEATURE CORRELATION`) | 0 | **1** |
| Dataset_Parser.py | 0 | 0 | 1 (`val`) | **1** |
| **Active Total** | **1** | **1** | **2** | **4** |
| Legacy (ignored) | 4 | 0 | 1 (`err`) | 5 |

---

## Compliance Checklist

- ✅ Understood rule correctly: ALL CAPS means EVERY letter uppercase (e.g., `FONT_CONFIG`)
- ✅ Verified `Scoring_Config` is Title_Case (COMPLIANT, not a violation)
- ✅ Verified all `Scoring_Config` dictionary keys are Title_Case (COMPLIANT)
- ✅ Searched for ALL CAPS comments (found `FEATURE CORRELATION`)
- ✅ Searched for common acronyms (`idx`, `err`, `val`)
- ✅ Excluded test files and documentation
- ✅ Prioritized active codebase over Legacy folder

---

## Recommendation Priority

### 🟡 Medium Priority
1. **Combined_Visualizer.py**: Rename `FONT_CONFIG` → `Font_Config` or `font_config` (15 references)
2. **Feature_Engineer.py line 628**: Change `# FEATURE CORRELATION:` → `# Feature Correlation:` (1 reference)

### 🟢 Low Priority
3. **Combined_Visualizer.py**: Rename `idx` → `index` (2 references)
4. **Dataset_Parser.py**: Rename `val` → `spin_value` (2 references)

---

## Compliance Status

**Overall:** ❌ **NON-COMPLIANT**  
**Active Violations:** 4 total
- 1 ALL CAPS variable (`FONT_CONFIG`)
- 1 ALL CAPS comment (`FEATURE CORRELATION`)
- 2 acronyms (`idx`, `val`)

**Legacy Violations:** 5 (ignored unless code reactivated)

---

## Apology for Previous Error

**Previous Report Error:** Incorrectly identified `Scoring_Config` and its dictionary keys as ALL CAPS violations. These are actually Title_Case (mixed uppercase/lowercase), which is explicitly allowed per instructions: "Use `Title_Case` (for example, `Scoring_Config`)".

**Root Cause:** Misunderstood definition of "ALL CAPS" - thought it included Title_Case with underscores like `Sigma_Scale`. Correct definition: ALL CAPS means EVERY single character is uppercase (e.g., `FONT_CONFIG`, `SCORING_CONFIG`).

**Corrective Action:** Updated `.github/copilot-instructions.md` with explicit examples to prevent future confusion.
