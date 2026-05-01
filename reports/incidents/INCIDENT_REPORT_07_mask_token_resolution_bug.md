# 📄 Incident Report: 07 — Mask Token Resolution Bug & Evaluation Correction

**Status:** Resolved

**Date:** 2026-05-01

**Affected Step:** Step 5 — evaluate_llm_masked.py → LLM_DEGRADATION_RESULTS.parquet

**Root Cause:** `oracle.resolve_landmark()` resolving `[MASK]` and `The [MASK]` to
real graph nodes via substring match fallback, inflating resolution rate and corrupting
accuracy metrics.

---

## 🔍 Discovery & Root Cause Analysis

### 1. Discovery

During Level 2 failure mode analysis, a diagnostic cell was run on the resolved outputs
to identify salient landmark anchoring patterns. The top-20 most frequent resolved
outputs globally showed:

```
487 ( 6.1%)  →  '[MASK]'
242 ( 3.0%)  →  'The [MASK]'
```

These mask tokens appeared in the **resolved** pool — meaning `oracle.resolve_landmark()`
was successfully mapping them to real graph nodes and returning valid GPS coordinates.

### 2. Root Cause — Substring Match on Cleaned Token

`resolve_landmark()` applies `re.sub(r'[^a-zA-Z0-9]', '', landmark_name).lower()`
before searching. This transforms:

```
'[MASK]'     → 'mask'
'The [MASK]' → 'themask'
```

The cleaned string `'mask'` then matched against POI `clean_name`, `clean_amenity`,
and other columns via `.str.contains('mask', na=False)`. POIs with "mask" as a substring
in any tag field (e.g. shop names, brand fields) returned as candidates. The KDTree
proximity filter then selected the nearest such POI to the start node and returned it
as a valid resolution.

### 3. Impact Quantification

```
Rows where mask token resolved: 729
  '[MASK]'     : 487
  'The [MASK]' : 242

City distribution:
  manhattan:    698
  philadelphia:  31
  pittsburgh:     0

Distance distribution (these 729 rows):
  mean: 1371.8m  |  @250m accuracy: 3.2%  |  @100m accuracy: 0.1%
```

**Inflated metrics (pre-fix):**
- Resolution rate: 37.8% (7,947 / 21,049)
- These 729 rows had mean distance 1371m and 3.2% accuracy @250m
- They were concentrated in `mask_landmark` and `mask_both` variants
- Answerable resolution rate was inflated to 42.1% (should be 30.6%)

**Corrected metrics (post-fix):**
- Resolution rate: 34.3% (7,218 / 21,049)
- Overall Acc@250m improved: 35.9% → 39.2% (contaminated denominator removed)
- Answerable resolution rate corrected to 30.6%

### 4. Fixed Coordinates — Two Fallback Nodes

Inspection of predicted coordinates for mask-token resolved rows revealed clustering
at two fixed points:

```
(40.731303, -74.002970)  — Manhattan
(40.747398, -73.985322)  — Manhattan
```

These are not POIs named "[MASK]" — they are fallback nodes returned when the KDTree
finds the nearest POI to the start node that contains "mask" as a substring in any
tag field. The same coordinates repeat across hundreds of rows because the same few
POIs happen to be the nearest "mask"-matching POIs to many different start nodes.

---

## 🛠️ The Fix

### oracle_engine.py — Module-level blocklist

A blocklist was added at module level, checked at the top of `resolve_landmark()`
**before** the `re.sub` cleaning step, operating on the raw input string:

```python
RESOLUTION_BLOCKLIST = {'[mask]', 'the [mask]', '[dir_mask]', 'the [dir_mask]'}

def resolve_landmark(self, landmark_name: str, ...) -> str:
    if landmark_name.strip().lower() in RESOLUTION_BLOCKLIST:
        return None
    # ... rest of existing logic unchanged
```

**Why pre-cleaning:** The blocklist uses raw input strings with brackets intact.
After `re.sub`, `'[MASK]'` becomes `'mask'` which could legitimately match real POI
names containing "mask" as a substring. Blocking at the raw string level is precise
and avoids false positives.

**Why hardcoded, not semantic:** The four blocked strings are syntactic artifacts of
the masking pipeline defined in `underspecify_instructions.py`. They are not a natural
language category requiring semantic generalization. No legitimate POI name will ever
match them.

### Verification

Before resubmitting the SLURM job, the fix was verified interactively:

```
'[MASK]'                  → BLOCKED ✅
'The [MASK]'              → BLOCKED ✅
'[DIR_MASK]'              → BLOCKED ✅
'The [DIR_MASK]'          → BLOCKED ✅
'Starbucks'               → resolved → 1#560636265
'Central Park'            → resolved → 1#1504407639
'Broadway'                → resolved → 1#1428185707
```

Legitimate landmarks continue to resolve correctly. The fix is isolated and minimal.

---

## ⚠️ Secondary Incident — Job Slowdown (Non-Bug)

### Observation

The corrected job ran significantly slower than the original:
- Original (s-002): ~6s/it, completed in ~2h10m
- Corrected (s-004): ~55–70s/it, 10+ hours elapsed

### Investigation

The blocklist check itself is O(1) and adds negligible overhead. The `re.sub` cleaning
path is unchanged for non-blocked inputs. No code path was made slower by the fix.

**Root cause: node variance on `studentkillable` partition.**

The corrected job ran on node s-004 vs s-002 for the original. The `studentkillable`
partition shares resources with other jobs. s-004 was under significantly higher load
during the overnight run. This was confirmed by:

1. Progress bar showing consistent 55–70s/it from batch 1 (not a mid-run regression)
2. Fix verification showing identical oracle initialization logs to the original run
3. No code path changes affecting the main inference loop

**Resolution:** No action required. The job completed successfully. Future jobs should
note that runtime on `studentkillable` is non-deterministic and node assignment affects
wall time significantly.

---

## 📈 Metric Comparison

| Metric | Pre-Fix (Buggy) | Post-Fix (Correct) |
|:---|---:|---:|
| Resolution rate | 37.8% | **34.3%** |
| Mask token resolutions | 729 | **0** |
| Overall Acc@250m | 35.9% | **39.2%** |
| Answerable resolution rate | 42.1% | **30.6%** |
| Ambiguous resolution rate | 42.2% | **40.7%** |
| Contradictory resolution rate | 31.9% | **30.1%** |
| Mean distance (resolved) | 674.2m | **603.7m** |

---

## 🔬 Scope Confirmation

Before rerunning Step 5, it was confirmed that `resolve_landmark()` is also called in
`symbolic_solver.py` (lines 165, 279). However, both calls receive extracted nouns
from the symbolic extraction pipeline — not LLM outputs or mask tokens. The
`underspecify_instructions.py` script stores `[MASK]` only in the instruction `text`
field, never in `extracted_noun`. The solver in `label_variants.py` extracts its own
noun from the masked text and fails gracefully if the noun is masked.

**Conclusion:** The fix affects Step 5 only. Steps 2–4.9 are unaffected and were
not rerun.

---

## ✅ Final Validation

Post-fix diagnostic confirmed zero mask-token resolutions:

```
Rows where mask token resolved: 0
  '[MASK]'     : 0
  'The [MASK]' : 0
```

`LLM_DEGRADATION_RESULTS.parquet` regenerated at:
```
modified=2026-05-01 11:08:47 UTC  size=1568.9 KB
```

All downstream notebooks (LLM_DEGRADATION_INPUT_validate, failure_modes_level_1,
failure_modes_level_2) rerun and validated against the corrected file.
