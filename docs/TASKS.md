# 🛰️ TASKS.md: Oracle Implementation & Labeling Pipeline

## ✅ Phase 1: Synchronization & Configuration (COMPLETED)
- [x] **Task 1.1: Finalize Review Items**
    - Implemented standard shortest-path logic and geodesic bearings.
- [x] **Task 1.2: Centralize Constants (`config.py`)**
    - Created `config.py` with `POI_NODE_PREFIX` and data paths.

## ✅ Phase 2: Logic & Extraction Overhaul (COMPLETED)
- [x] **Task 2.1: Implement Node Bridge & Graph Loading**
    - Resolved NetworkX 3.0+ issues; loading 74k nodes via `pickle.load`.
- [x] **Task 2.2: Implement Clamped Radius**
    - Added `get_search_radius(distance)` logic (max(D*1.1, D+80)).
- [x] **Task 2.3: Implement Vector "Away/Toward" Logic**
    - Wrote dot-product filter for relative movement.
- [x] **Task 2.4: Implement OSM Landmark POI Buffer**
    - Built `OracleEngine` with regex-based fuzzy matching.
- [x] **Task 2.5: Categorical Grounding & Brand Resolution**
    - Developed `CategoricalMatcher` to bridge brands (Starbucks) to OSM tags (CAFE).
    - Implemented Dependency Parsing in `extraction_utils.py` to identify goal objects.
    - Result: Reduced Manhattan ambiguous samples from 307 to 2.

## ✅ Phase 3: Validation & The Labeling Pipeline (COMPLETED)
- [x] **Task 3.1: The "Sanity Check" Suite**
    - Created `tests/test_symbolic_solver.py`; 5 integration tests PASSING.
- [x] **Task 3.2: The Diagnostic Object**
    - Updated `SymbolicSolver` to return structured dicts (`Answerable`, `Ambiguous`, `Contradictory`).
- [x] **Task 3.3: Batch Labeling Script**
    - Created `scripts/batch_labeling.py` with KD-Tree optimization (~120 it/s).
    - Generated `manhattan_silver_standard.parquet` (6,998 answerable samples).

## 🌍 Phase 4: Multi-City Expansion (IN PROGRESS)
*Goal: Generalize the Oracle to the full RVS dataset (Pittsburgh & Philadelphia).*

- [x] **Task 4.1: Multi-City Asset Ingestion & Validation**
    - Import `pittsburgh_graph.gpickle` and `philadelphia_graph.gpickle` from RVS Drive.
    - Run compatibility checks to ensure `pickle.load` works across NetworkX versions.
- [ ] **Task 4.2: City-Agnostic Solver Refactor**
    - Update `batch_labeling.py` to dynamically load the correct graph/POI index and apply city-specific success radii (80m for MHT, 100m for PIT/PHL).
- [ ] **Task 4.3: Generate Full Silver Standard**
    - [ ] **Task 4.3.1: Slurm Job Configuration (`scripts/submit_labeling.sh`)**
        - *Note: Use Slurm here to run MHT, PIT, and PHL in parallel to save time.*
    - [ ] **Run labeling pipeline for all RVS splits to create a complete training/testing corpus.**

## 📉 Phase 5: Evaluation & Degradation
*Goal: Prove the impact of underspecification and benchmark SOTA models.*

- [ ] **Task 5.1: LLM Benchmarking (`scripts/evaluate_llm.py`)**
    - ⚠️ **SLURM RECOMMENDED:** Use cluster GPUs for model inference/evaluation (T5-base / Pythia / GPT-4o).
    - Compare LLM spatial predictions against the Symbolic Oracle's "Ground Truth."
- [ ] **Task 5.2: The Masking Engine (`mask_instructions.py`)**
    - Implement RVS-style masking (removing landmarks) to test model degradation.
- [ ] **Task 5.3: Automated Degradation Analysis**
    - Generate plots showing `Answerable` → `Ambiguous` flips as information is removed.

---

### 🚀 Slurm Quick-Reference
| Task | Why use the Cluster? |
| :--- | :--- |
| **4.3 (Labeling)** | Parallelize processing of ~10k examples across 3 cities. |
| **5.1 (LLM Eval)** | Access high-end GPUs for local model inference (~1B params). |

---

## ⚠️ Technical Debt
- [x] **Refactor Legacy Tests:** (Done) Updated all scripts to use `pickle.load` for graph loading.
- [ ] **Error Handling:** Add robust catch for the remaining 2 malformed Manhattan instructions.

## 👥 Suggested Team Assignments

| Role | Tasks | Primary Goal |
| :--- | :--- | :--- |
| **Backend/Geometry Lead** | 4.1, 4.2, 4.3.1 | Multi-city infrastructure & Slurm setup |
| **Data/ML Lead** | 4.3, 5.1 | Full dataset generation & LLM Eval |
| **NLP/Robustness Lead** | 5.2, 5.3 | Masking engine & degradation analysis |