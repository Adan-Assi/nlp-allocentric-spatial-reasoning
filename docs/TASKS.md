# 🛰️ TASKS.md: Oracle Implementation & Labeling Pipeline

## ✅ Phase 1: Synchronization & Configuration (COMPLETED)
*Goal: Align on constants and eliminate hard-coded values.*

- [x] **Task 1.1: Finalize Review Items**
    - Done: Implemented standard shortest-path logic and geodesic bearings.
- [x] **Task 1.2: Centralize Constants (`config.py`)**
    - Done: Created `config.py` with `POI_NODE_PREFIX` and data paths.

---

## 🏗️ Phase 2: The Logic Overhaul
*Goal: Implement the mathematical rules defined in our Oracle Protocol.*

- [x] **Task 2.1: Implement Node Bridge & Graph Loading**
    - Done: Resolved NetworkX 3.0+ `gpickle` issues; successfully loading 74k nodes via `pickle.load`.
- [x] **Task 2.4: Implement OSM Landmark POI Buffer**
    - Done: Built `OracleEngine` with regex-based fuzzy matching and `osmid` resolution from the 198MB POI pickle.
- [ ] **Task 2.2: Implement Clamped Radius (PENDING)**
    - *To do:* Add `get_search_radius(distance)` using the max(D * 1.1, D + 80) logic to the Solver.
- [ ] **Task 2.3: Implement Vector "Away/Toward" Logic (PENDING)**
    - *To do:* Write the dot-product filter for relative movement (e.g., "moving away from the park").
- [ ] **Task 2.5: Data-Driven Landmark Mapping**
    - Run frequency analysis on RVS instruction nouns to identify top landmarks.
    - Map top-frequency spatial nouns to OSM tags in `config.py`(e.g., "deli", "church").
    - Ensure 90% of landmark-based instructions have a corresponding mapping.

---

## 🧪 Phase 3: Validation & The Labeling Pipeline
*Goal: Generate the Silver Standard dataset for ML training.*

- [x] **Task 3.1: The "Sanity Check" Suite**
    - Done: Created `tests/test_symbolic_solver.py` with 5 integration tests (Graph stats, Reachability, Shortest Path, Bearing, and Landmark Navigation). All PASSING.
- [ ] **Task 3.2: The Diagnostic Object**
    - *To do:* Update `SymbolicSolver` to return dicts like `{"state": "Ambiguous", "count": 3}` instead of raw paths.
- [ ] **Task 3.3: Batch Labeling Script**
    - *To do:* Create the script to iterate through the Manhattan dataset and save to Parquet.

---

## 📉 Phase 4: Degradation & Validation
*Goal: Prove the impact of underspecification.*

- [ ] **Task 4.1: The Masking Engine (`mask_instructions.py`)**
- [ ] **Task 4.2: Automated Degradation Analysis**
    - Generate plots showing `Answerable` → `Ambiguous` flips.

---

## ⚠️ Technical Debt (Added Mar 2026)
- [ ] **Refactor Legacy Tests:** Update `sanity_check_all_graphs.py` and `sanity_check.py` to use `pickle.load` instead of `nx.read_gpickle`.

---

## 👥 Suggested Team Assignments

| Role | Tasks | Primary Goal |
| :--- | :--- | :--- |
| **Backend/Geometry Lead** | 2.2, 2.3 | Geometry logic & filtering |
| **Data/ML Lead** | 3.2, 3.3 | Labeling pipeline & Parquet generation |
| **NLP/Robustness Lead** | 4.1, 4.2 | Masking engine & degradation plots |