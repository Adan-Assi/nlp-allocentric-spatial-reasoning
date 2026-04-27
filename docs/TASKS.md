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

## ✅ Phase 5: Evaluation & Degradation (COMPLETED)
*Goal: Quantify the impact of underspecification and benchmark baseline model resolution.*

- [x] **Task 5.1: LLM Benchmarking (`scripts/evaluate_llm.py`)**
    - Leveraged SLURM-based cluster GPUs for T5-base model inference.
    - Benchmarked LLM spatial predictions against the Symbolic Oracle's "Ground Truth."
    - Discovered the **"Resolution Limit"**: 90%+ Topological Grounding vs. <2% Semantic Specificity.
- [x] **Task 5.2: The Masking Engine (`scripts/underspecify_instructions.py`)**
    - Implemented RVS-style masking logic to programmatically remove landmarks and cardinal directions.
    - Created a "Hard Mode" (Mask Both) to test extreme information decay.
- [x] **Task 5.3: Automated Degradation Analysis**
    - Generated "Information Decay" and "Spatial Drift" (KDE) visualizations.
    - Confirmed the **Hierarchical Spatial Resilience** of LLMs under semantic underspecification.

## 🧠 Phase 6: Modeling & Training (PLANNED)
- [ ] **Task 6.1: Spatially-Aware Data Stratification**
    - Create city-balanced splits (Train/Val/Test) from the 7,263 answerable rows.
    - **Refinement:** Ensure "Near-Miss" samples (from Phase 5.3) are represented in the Test set to measure resolution improvement.

- [ ] **Task 6.2: Neural Architecture - The "Precision Head"**
    - Implement a Dual-Head Architecture:
        1. **Classification Head:** Predicts the Topological Node (Street/Intersection).
        2. **Regression Head:** Predicts exact Latitude/Longitude coordinates to bridge the "Last Mile" gap.
    - Baseline models: T5-base (seq2seq) or a Lightweight MLP over LLM embeddings.

- [ ] **Task 6.3: Custom Spatial Loss Function**
    - Implement **Haversine Distance Loss** or **Geodesic Loss** instead of standard MSE.
    - This penalizes the model based on actual physical distance (meters) rather than abstract numeric error.

---

### 🚀 Slurm Quick-Reference
| Task | Why use the Cluster? |
| :--- | :--- |
| **5.1 (LLM Eval)** | Access high-end GPUs for local model inference (~1B params). |
| **6.2 (Training)** | Accelerate backprop for multi-epoch spatial reasoning training. |

---

## ⚠️ Technical Debt & System Stability
*Ongoing maintenance tasks to ensure pipeline integrity.*

- [x] **Type Safety:** Standardized `sample_id` as String to prevent PyArrow conversion errors during master dataset merges.
- [ ] **Data Sanitization:** Implement robust filtering for malformed instructions (e.g., empty strings or null POIs) to prevent Batch Inference crashes in Phase 6.
- [ ] **Path Normalization:** Ensure `config.py` handles cross-platform pathing (Windows/Linux) for seamless SLURM cluster deployment.