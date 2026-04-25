# INCIDENT_REPORT_03_PHILLY_GATEKEEPER_LOGIC_BYPASS.md

**Date:** April 22, 2026  
**Status:** ✅ RESOLVED  
**Incident Lead:** adanassi  

---

## 1. Executive Summary
The Philadelphia labeling pipeline reached a state of **distribution stagnation**, remaining frozen at **490 Contradictory** samples. Investigations identified a "Logic Gatekeeper" in the `SymbolicSolver` as the primary bottleneck. By requiring a mapped `category` before proceeding with spatial searches, the solver was discarding a majority of the dataset as `UNKNOWN`. Combined with an extraction "leak" where articles were captured as nouns, the solver was effectively paralyzed.

---

## 2. Evidence: Extraction & Mapping Evolution
Diagnostics performed via `debug_philly_extraction.py` tracked the transition from high-noise `UNKNOWN` states to high-signal `Mapped` states.

### Extraction Table (Post-Fix Sample)
| ID | CATEGORY | NOUN | INSTRUCTION (Snippet) |
| :--- | :--- | :--- | :--- |
| **9134** | <span style="color:green">BENCH</span> | waste basket | Meet me at the waste basket southwest... |
| **9141** | <span style="color:green">PARKING</span> | car sharing | Meet me at the car sharing on West Jefferson... |
| **9145** | <span style="color:orange">UNKNOWN</span> | **None** | Meet me at the recycling place... (Plugged Leak) |
| **9158** | <span style="color:green">POST</span> | post box | We can meet at the post box on West Girard... |
| **9167** | <span style="color:green">RESTAURANT</span> | Wendy's fast food | Meet me west of your location at Wendy's... |
| **9170** | <span style="color:orange">UNKNOWN</span> | **None** | We can meet at the parking entrance... (Plugged Leak) |

**Statistical Progress:**
* **Initial Unknowns:** 892 (69.80%)
* **Current Unknowns:** 686 (**53.68%**)
* **Net Gain:** +206 samples successfully bridged to OSM Categories.

---

## 3. Root Cause Analysis (RCA)
1.  **The Gatekeeper Constraint:** `symbolic_solver.py` contained the condition `if category != "UNKNOWN"`. This caused 892 samples to skip "Step 2: Spatial Fallback," leading to false-positive Contradictory labels.
2.  **Lexical Mapping Gap:** Philly-specific nouns ("car sharing", "post box", "convenience shop") were present in OSM data but absent from the `TEXT_TO_GROUP_MAP` trigger list.
3.  **The "The" Leak:** The extraction regex was capturing definite articles ("the") as landmarks. This created junk metadata and caused the `underspecify.py` script to erroneously mask articles instead of landmarks.

---

## 4. Corrective Measures
* **Logic Bypass:** Removed the `UNKNOWN` category requirement in `SymbolicSolver`. The system now attempts a noun-based spatial search regardless of category mapping.
* **Surgical Leak Plug:** Updated `extract_rvs_target` to return `None` if the extracted noun is a stop-word (e.g., "the", "and"). This prevents noise from entering the solver and variant generator.
* **Bridge Expansion:** Added 15+ new keyword triggers to `config.py` to link Philly urban dialect to canonical `LANDMARK_GROUPS`.

---

## 5. Conclusion
The "490 Wall" was a result of overly-restrictive logic and vocabulary mismatch. By granting the solver **"permission to search"** even when categorization fails and cleaning the extraction noise, we expect the Philadelphia distribution to finally reflect the true reachability of the dataset.