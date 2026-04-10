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
    - Result: Reduced Manhattan ambiguous samples significantly via density-aware logic.

## ✅ Phase 3: Validation & The Labeling Pipeline (COMPLETED)
- [x] **Task 3.1: The "Sanity Check" Suite**
    - Created `tests/test_symbolic_solver.py`; 5 integration tests PASSING.
- [x] **Task 3.2: The Diagnostic Object**
    - Updated `SymbolicSolver` to return structured dicts (`Answerable`, `Ambiguous`, `Contradictory`).
- [x] **Task 3.3: Batch Labeling Script**
    - Created `scripts/batch_labeling.py` with KD-Tree optimization (~120 it/s).
    - Generated Manhattan samples with 77.8% yield.

## ✅ Phase 4: Multi-City Expansion (COMPLETED)
- [x] **Task 4.1: Multi-City Asset Ingestion & Validation**
    - Imported Pittsburgh and Philadelphia assets; verified NetworkX compatibility.
- [x] **Task 4.2: City-Agnostic Solver Refactor**
    - Implemented dynamic parameter tuning: **0.5 Salience Ratio** (PHL/PIT) and **0.7 Salience Ratio** (MHT).
- [x] **Task 4.3: Generate Full Silver Standard**
    - [x] **Task 4.3.1: Forensic Audit**
        - Conducted range-check and existence-check (e.g., "Pretzel Factory" case study).
        - Verified 1500m search horizon aligns with Paz-Argaman et al. (2020) local-context constraints.
    - [x] **Task 4.3.2: Create Master Dataset**
        - Merged 3 cities into `RVS_MASTER_SILVER_STANDARD.parquet`.
        - Result: **7,263 Answerable** training samples (78.1% Yield).

## 📉 Phase 5: Evaluation & Degradation (IN PROGRESS)
*Goal: Prove the impact of underspecification and benchmark SOTA models.*

- [ ] **Task 5.1: LLM Benchmarking (`scripts/evaluate_llm.py`)**
    - ⚠️ **SLURM RECOMMENDED:** Use cluster GPUs for model inference/evaluation (T5-base / Pythia / GPT-4o).
    - Compare LLM spatial predictions against the Symbolic Oracle's "Ground Truth."
- [ ] **Task 5.2: The Masking Engine (`mask_instructions.py`)**
    - Implement RVS-style masking (removing landmarks) to test model degradation.
- [ ] **Task 5.3: Automated Degradation Analysis**
    - Generate plots showing `Answerable` → `Ambiguous` flips as information is removed.

## 🧠 Phase 6: Modeling & Training (PLANNED)
- [ ] **Task 6.1: Train/Val/Test Splitting**
    - Create city-balanced splits from the 7,263 answerable rows.
- [ ] **Task 6.2: Neural Baseline Construction**
    - Set up a sequence-to-sequence or regression head model to map instructions to coordinates.

---

### 🚀 Slurm Quick-Reference
| Task | Why use the Cluster? |
| :--- | :--- |
| **5.1 (LLM Eval)** | Access high-end GPUs for local model inference (~1B params). |
| **6.2 (Training)** | Accelerate backprop for multi-epoch spatial reasoning training. |

---

## ⚠️ Technical Debt
- [x] **Type Safety:** (Done) Standardized `sample_id` as String to prevent PyArrow conversion errors during merge.
- [ ] **Error Handling:** Add robust catch for the remaining malformed instructions if encountered during inference.