# Incident Report: Philadelphia Labeling Stagnation

**Date:** April 22, 2026
**Status:** Pending (Logic Identified)
**Incident Lead:** adanassi

---

## 1. Executive Summary
Despite multiple code deployments and bytecode purges, the Philadelphia dataset labeling distribution remained frozen at **490 Contradictory / 430 Answerable / 358 Ambiguous**. Investigations revealed a "Silent Failure" loop where 70% of Philadelphia samples were being discarded by a logic gate in the Symbolic Solver before reachability or fallbacks could be attempted.

## 2. Root Cause Analysis (RCA)
The failure was caused by a two-stage "bottleneck" in the labeling pipeline:

1.  **The Extraction Gap:** The `extract_rvs_target` utility returned `UNKNOWN` for **892/1278** samples (69.8%). This occurred because Philadelphia's vocabulary (e.g., "ATM," "parking lot," "car sharing") was not mapped to specific categories in `config.TEXT_TO_GROUP_MAP`.
2.  **The Logic Gatekeeper:** In `symbolic_solver.py`, a strict conditional check (`if category != "UNKNOWN"`) prevented the **Philly Fallback** (Step 2) from executing for any sample the extractor couldn't categorize. 
3.  **The Fallback Crash:** A missing scalar `haversine` function in `src/utils.py` caused the script to throw `AttributeError` during Step 3 (Fuzzy Fallback). The `try-except` block in `batch_labeling.py` caught these errors but suppressed the traceback, leading to "Sample None" logs and skipped records.

## 3. Timeline of Events
* **01:29 AM:** First run; IDs show as `None`. Logic fails to trigger.
* **01:40 AM:** Bytecode purged. `DEBUG_CRITICAL` confirms IDs are correct (9126, 9127), but counts do not move.
* **01:59 AM:** Pipeline crashes with `AttributeError: haversine`. Discovered the solver was calling a non-existent utility.
* **02:13 AM:** `haversine` utility added. Pipeline runs to completion, but distribution remains identical (**490 Contradictory**).
* **02:30 AM:** Diagnostics run on Philly dataset; confirmed **892 UNKNOWNs** are causing the solver to abort searches prematurely.

## 4. Corrective Actions Taken
| Component | Action | Result |
| :--- | :--- | :--- |
| **`src/utils.py`** | Implemented scalar `haversine` formula. | Restored Step 3 Fallback functionality. |
| **`config.py`** | Expanded `TEXT_TO_GROUP_MAP` with Philly-specific keywords. | Reduced `UNKNOWN` count by providing category bridges. |
| **`src/symbolic_solver.py`** | Removed `category != "UNKNOWN"` requirement for Step 2. | Enabled "Fuzzy-Noun" searches for unmapped landmarks. |
| **`batch_labeling.py`** | Unified `current_id` variable for logs and appends. | Resolved "Sample None" log synchronization issue. |

## 5. Prevention & Monitoring
* **Aggressive Fallback:** The solver will now attempt a spatial search based on `raw_noun` and `target_dir` regardless of whether a category is found.
* **Log Transparency:** Future `try-except` blocks in `batch_labeling.py` should log the specific Error type to prevent silent "Continue" loops.
* **Vocabulary Audit:** Periodic runs of `debug_philly_extraction.py` should be used to ensure high "Mapped Category" percentages in new cities.