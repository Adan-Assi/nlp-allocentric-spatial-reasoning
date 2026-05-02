# 📄 Incident Report: 01-Pipeline Sparsity Bias

**Status:** Resolved (Architecture Change)  
**Date:** 2026-04-19  
**Issue:** Philadelphia (Orange) missing results in `mask_landmark` and `mask_both` variants.  
**Root Cause:** "Landmark-Gating" logic in `underspecify_instructions.py`.

---

## 🔍 Discovery & Root Cause Analysis

During Stage 4 (Cross-City Generalization) visualization, it was observed that **Philadelphia** (Orange) failed to render any bars for landmark-masked variants. 

### The "Landmark-Gating" Failure
The original `underspecify_instructions.py` relied on the raw RVS `landmarks` metadata dictionary. 
* **Manhattan (Dense):** High frequency of proper-noun POIs in metadata. Masking triggered successfully.

* **Philadelphia (Sparse/Residential):** As noted in the RVS paper, instructors were often forced to use **generic descriptions** (e.g., "the pharmacy") or street names.
 
* **The Crash:** Because these generic terms did not always appear in the `landmarks` metadata dictionary, the generation script skipped the landmark-masking logic entirely, assuming no landmarks were present to mask.

---

## 🛠️ The Permanent Fix: "Solver-Driven Masking"

To ensure the pipeline is robust across different urban morphologies, we are shifting the **Source of Truth** from the raw JSON metadata to the output of our own `SymbolicSolver`.

### 1. Logic Transition
| Feature | Original Logic (Broken) | Fixed Logic (Robust) |
| :--- | :--- | :--- |
| **Input Source** | Raw `data.json` | `silver_standard.parquet` |
| **Target Key** | `sample['landmarks']` dictionary | `row['extracted_noun']` |
| **Philly Success** | **Failed** (No proper names found) | **Passed** (Masks generic categories) |

### 2. Implementation Strategy
The `underspecify_instructions.py` script now consumes the output of `batch_labeling.py`. By using the `extracted_noun` column, we guarantee that the text being masked is exactly what the Symbolic Solver identified as the navigational goal, regardless of whether it is a proper name (Manhattan) or a generic category (Philly).

---

## 📈 Impact on Dataset Integrity

This fix recovers **1,035 Answerable samples** for Philadelphia that were previously excluded from the Stress-Test stage. 

**Revised Experiment Flow:**
1. **Labeling:** `batch_labeling.py` extracts nouns/categories from raw text.
2. **Underspecification:** `underspecify_instructions.py` reads these extracted nouns to create consistent masks.
3. **Inference:** `evaluate_llm_masked.py` benchmarks the LLM against the recovered variants.
