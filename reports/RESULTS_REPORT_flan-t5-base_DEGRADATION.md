# 📊 Results Report: LLM Evaluation under Systematic Underspecification

**Status:** Complete

**Date:** 2026-05-01

**Pipeline Step:** Step 5 → Step 6 (evaluate_llm_masked.py → analysis notebooks)

**Model:** google/flan-t5-base (250M parameters)

**Dataset:** LLM_DEGRADATION_INPUT.parquet — 21,049 masked variants across 3 cities

---

## 1. Oracle 2 Label Distribution (Masked Variants)

Oracle 2 (`mode="label"`) labels each masked variant by counting reachable candidates
after constraint resolution — preserving Ambiguous as a research signal, unlike Oracle 1.

| City | Mask Type | N | Answerable | Ambiguous | Contradictory |
|:---|:---|---:|---:|---:|---:|
| Manhattan | mask_landmark | 5,468 | 32.1% | 18.8% | 49.1% |
| Manhattan | mask_directions | 5,299 | 2.2% | 87.1% | 10.7% |
| Manhattan | mask_both | 5,299 | 31.5% | 11.3% | 57.2% |
| Pittsburgh | mask_landmark | 705 | 2.8% | 14.6% | 82.6% |
| Pittsburgh | mask_directions | 668 | 2.8% | 91.8% | 5.4% |
| Pittsburgh | mask_both | 668 | 1.6% | 13.5% | 84.9% |
| Philadelphia | mask_landmark | 990 | 13.8% | 21.2% | 64.9% |
| Philadelphia | mask_directions | 976 | 2.9% | 84.4% | 12.7% |
| Philadelphia | mask_both | 976 | 9.5% | 9.0% | 81.5% |

**Key finding:** Directional constraints are the primary disambiguation mechanism.
`mask_directions` produces 85–92% Ambiguous across all cities — removing direction
collapses unique resolution. `mask_landmark` drives Contradictory at rates inversely
correlated with graph density (Manhattan 49.1% vs Pittsburgh 82.6%).

---

## 2. LLM Evaluation — Overall Results

**Model:** FLAN-T5-base | **Evaluation threshold:** 250m (strict: 100m) per RVS paper

| Metric | Value |
|:---|---:|
| Total samples | 21,049 |
| Symbolic resolution rate | 34.3% (7,218 / 21,049) |
| Unresolved outputs | 65.7% (13,831) |
| Overall Acc@250m (resolved only) | 39.2% |
| Overall Acc@100m (resolved only) | 16.2% |

**Note:** Resolution rate corrected from an initial 37.8% after a pipeline bug was
identified and fixed (see INCIDENT_REPORT_07). The corrected rate excludes 729 mask-token
outputs that were previously resolving to fallback coordinates via substring match on
`[MASK]` → `mask`.

### 2.1 Resolution Rate by Oracle Label

| Oracle Label | Resolution Rate | N |
|:---|---:|---:|
| Ambiguous | 40.7% | 8,169 |
| Answerable | 30.6% | 3,850 |
| Contradictory | 30.1% | 9,030 |

### 2.2 Accuracy by Variant Type (resolved only)

| Variant Type | Acc@250m | Acc@100m |
|:---|---:|---:|
| mask_directions | 38.0% | 15.6% |
| mask_landmark | 42.9% | 17.8% |
| mask_both | 39.0% | 15.9% |

### 2.3 Accuracy by Oracle Label (resolved only)

| Oracle Label | Acc@250m | Acc@100m |
|:---|---:|---:|
| Answerable | 37.7% | 15.4% |
| Ambiguous | 39.0% | 16.0% |
| Contradictory | 40.1% | 16.3% |

**Critical finding:** Accuracy is statistically flat across oracle labels. The model
achieves equivalent or higher accuracy on Contradictory variants (40.1%) compared to
Answerable (37.7%), confirming complete insensitivity to constraint satisfiability.

---

## 3. Level 1 Failure Mode Analysis (Unresolved Outputs, N=13,831)

Unresolved outputs are LLM predictions that `oracle.resolve_landmark()` could not map
to any graph node. These represent 65.7% of all outputs.

| Failure Mode | Count | % |
|:---|---:|---:|
| Mask passthrough | 6,220 | 45.0% |
| Unresolved name | 4,077 | 29.5% |
| Street echo | 3,336 | 24.1% |
| Instruction fragment | 196 | 1.4% |
| Generic category | 1 | 0.0% |
| Empty output | 1 | 0.0% |
| **Total** | **13,831** | **100.0%** |

### Breakdown by Variant Type (%)

| Variant Type | Mask Passthrough | Street Echo | Unresolved Name | Instruction Fragment |
|:---|---:|---:|---:|---:|
| mask_landmark | 69.1% | 8.7% | 19.7% | 2.4% |
| mask_directions | 9.0% | 42.2% | 47.6% | 1.1% |
| mask_both | 42.4% | 29.9% | 27.4% | 0.3% |

**Key finding:** Failure modes are variant-dependent, not general.
- `mask_landmark` → model echoes `[MASK]` token (69.1% passthrough)
- `mask_directions` → model defaults to street name (42.2% street echo) or
  confabulates a plausible but graph-absent landmark (47.6% unresolved name)

---

## 4. Level 2 Failure Mode Analysis (Resolved Outputs, N=7,218)

### 4.1 Summary

| Mode | Description | N | % Resolved | Acc@250m |
|:---|:---|---:|---:|---:|
| 2a | Overconfident on Ambiguous | 3,322 | 46.0% | 39.0% |
| 2b | Overconfident on Contradictory | 2,717 | 37.6% | 40.1% |
| 2c | Answerable, wrong location | 734 | 10.2% | N/A |
| 2d | Salient anchor outputs | 1,025 | 14.2% | see §4.4 |

**83.6% of all resolved outputs are overconfident** (2a + 2b): the model produces
specific landmark predictions on Ambiguous and Contradictory variants at accuracies
indistinguishable from Answerable cases.

### 4.2 Overconfidence Signal (2a, 2b)

Resolution rate and accuracy are decoupled across oracle labels. Contradictory variants
resolve at 30.1% but achieve 40.1% accuracy among resolved samples — higher than
Answerable (30.6% resolution, 37.7% accuracy). The resolution gate is the only signal
of difficulty; the model has no internal representation of spatial constraint satisfiability.

### 4.3 Wrong-location Characterization (2c)

Of 734 verifiable wrong-location Answerable predictions:

- **99.2% (728/734)** resolve to a categorically unrelated POI
- **0.8% (6/734)** resolve to correct category, wrong instance

Mean error distance: **968m** (wrong) vs **118m** (correct). The bimodal gap confirms
the model either approximately identifies the right area or is entirely off — no
intermediate spatial reasoning is observed.

The 0.8% same-category rate indicates near-complete absence of category-level spatial
grounding. In same-category cases, the model anchors to a high-salience brand
(Starbucks, Urban Outfitters) rather than the correct instance.

### 4.4 Salient Landmark Anchoring (2d)

High-frequency anchors are dominated by street names with near-zero accuracy:

| City | Anchor | Freq | Acc@250m |
|:---|:---|---:|---:|
| Manhattan | Broadway | 3.2% | 6.5% |
| Manhattan | Bank of America | 2.0% | 0.0% |
| Manhattan | Starbucks | 1.6% | 2.2% |
| Pittsburgh | Penn Avenue | 9.6% | 15.6% |
| Pittsburgh | Carson Street | 4.8% | 12.5% |
| Philadelphia | Market Street | 9.7% | 12.0% |
| Philadelphia | Walnut Street | 8.2% | 9.4% |

High-accuracy anchors (Smithfield Street 100%, Goodwill 100%, Rittenhouse Square 81.8%)
all verified as coincidence: gold targets are located within 300m of the anchor
street/POI, not evidence of genuine spatial inference.

---

## 5. Success Case Analysis (N=445, 37.7% of Answerable resolved)

### By Category

Top-performing categories: LIBRARY (71.4%), GARDEN (70.0%), HOSPITAL (69.2%),
PARK (54.5%), SCHOOL (52.9%). These share low within-city instance density and
high name salience — the model retrieves them from pretraining without spatial reasoning.

Bottom-performing: CHURCH (11.1%), PHARMACY (16.7%), CLOTHES (20.0%) — high-density
categories with many per-city instances where spatial constraints are required.

### By City

| City | Correct | Total | Rate |
|:---|---:|---:|---:|
| Pittsburgh | 10 | 18 | 55.6% |
| Manhattan | 413 | 1,094 | 37.8% |
| Philadelphia | 22 | 67 | 32.8% |

Pittsburgh's higher rate is a graph sparsity artifact, not evidence of better spatial
reasoning — fewer POIs means proximity-based correctness is more likely.

### By Variant Type

| Variant Type | Correct | Total | Rate |
|:---|---:|---:|---:|
| mask_directions | 42 | 86 | 48.8% |
| mask_landmark | 138 | 354 | 39.0% |
| mask_both | 265 | 739 | 35.9% |

`mask_directions` achieves the highest correct rate (48.8%), confirming landmark name
is the primary resolution signal — directional constraints contribute minimally.

**Qualitative note:** All three sampled correct predictions are street echoes or brand
anchors, not genuine spatial inferences. Even successful cases are surface-pattern driven.

---

## 6. Key Findings Summary

1. **Directional constraints are the primary disambiguation mechanism.** Their removal
   produces 85–92% Ambiguous across all cities and variant types.

2. **The model is completely insensitive to constraint satisfiability.** Acc@250m is
   flat across Answerable (37.7%), Ambiguous (39.0%), and Contradictory (40.1%) oracle
   labels — the model cannot distinguish well-formed from ill-formed instructions.

3. **83.6% of resolved outputs are overconfident** — the model produces specific
   landmark predictions on Ambiguous and Contradictory variants without evidential basis.

4. **Wrong-location errors are categorically wrong in 99.2% of cases.** The model has
   near-zero category-level spatial grounding.

5. **Success concentrates in high-salience, low-density categories** (LIBRARY, GARDEN,
   HOSPITAL) where name retrieval from pretraining is sufficient without spatial reasoning.

6. **All failure modes are surface-pattern driven:** mask passthrough (45.0%), street
   echo (24.1%), salient anchoring (14.2% of resolved), and brand anchoring in
   correct predictions.

---

## 7. Evaluation Design Note

Per Paz-Argaman et al. (EACL 2024), success is defined as predicted coordinates
falling within 250m (strict: 100m) of the human-validated gold target. This threshold
introduces a tolerance zone: a model predicting a nearby street name may count as
correct even if it did not identify the exact gold POI. This is an inherited design
choice, not a pipeline flaw, and is noted explicitly to contextualize the 37.7%
Answerable correct rate reported above.
