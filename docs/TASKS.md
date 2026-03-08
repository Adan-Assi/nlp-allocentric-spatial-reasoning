# 📋 TASKS.md: Oracle Implementation & Labeling Pipeline

This document outlines the step-by-step workplan to transition our system from a navigation solver to an **Answerability Oracle**.

---

## 🟢 Phase 1: Synchronization & Configuration
*Goal: Align on constants and eliminate hard-coded values.*

- [ ] **Task 1.1: Finalize Review Items**
    - Reach team consensus on:
        - Distance Multiplier ($1.1\times$ vs $1.2\times$).
        - Landmark Proximity (Is 20m enough for high-density areas?).
- [ ] **Task 1.2: Centralize Constants (`config.py`)**
    - Create a centralized configuration file.
    - **No hard-coding** the 45° wedge, 80m buffer, or 20m $S_0$ radius inside functions.
    - *Purpose:* Allows us to tune the "Physics" of our world globally.

---

## 🟡 Phase 2: The Logic Overhaul (Parallel Work)
*Goal: Implement the mathematical rules defined in our Oracle Protocol.*

- [ ] **Task 2.1: Implement $S_0$ (The Multi-Start)**
    - Update solver to find all nodes within 20m of a starting geocode.
    - Treat this set as the origin for all path/candidate searches.
    - *Success Metric:* Solver handles complex intersections with multiple nodes without crashing.
    

- [ ] **Task 2.2: Implement Clamped Radius**
    - Create `get_search_radius(distance)` using the $\max(D \times 1.1, D + 80)$ logic.

- [ ] **Task 2.3: Implement Vector "Away/Toward" Logic**
    - Write a dot-product filter for relative movement.
    - Calculate the vector between the `start_node` and `landmark_center`.
    

- [ ] **Task 2.4: Implement OSM Landmark POI Buffer**
    - Update the landmark filter to check for matching POIs within 20m of a candidate node, rather than just checking the node's own tags.

---

## 🟠 Phase 3: Validation & The Labeling Pipeline
*Goal: Generate the Silver Standard dataset for ML training.*

- [ ] **Task 3.1: The "Sanity Check" Suite (CRITICAL)**
    - Before starting the 50,000-row batch, the team must agree on a "Sanity Check" suite of 10 hand-picked instructions (like the ones in the scenarios).
    - Run the new logic against these 10 cases to ensure the output matches our manual expectations.

- [ ] **Task 3.2: The Diagnostic Object**
    - Modify the solver return statement.
    - Instead of a path, return: `{"state": "Ambiguous", "candidate_count": 3, "candidates": [id1, id2, id3]}`.
    

- [ ] **Task 3.3: Batch Labeling Script**
    - Create a script to iterate through the full dataset and save results as `gold_standard_train.parquet`.

---

## 🔵 Phase 4: Degradation & Validation
*Goal: Prove the impact of underspecification.*

- [ ] **Task 4.1: The Masking Engine (`mask_instructions.py`)**
    - Create logic to strip landmarks (replace with `[MASK]`) or directions to simulate underspecification.

- [ ] **Task 4.2: Automated Degradation Analysis**
    - Run the solver on "Full" vs. "Masked" instructions.
    - Generate a table/plot showing the rate at which `Answerable` instructions flip to `Ambiguous`.
    

---

## 📊 Suggested Team Assignments (The "Who Does What")

| Role | Tasks | Primary Goal |
| :--- | :--- | :--- |
| **Backend/Geometry Lead** | 2.1, 2.3 | Math integrity (Vectors & $S_0$). |
| **Data Engineer** | 2.2, 2.4, 3.3 | Scaling (BallTree & POI lookups). |
| **ML Lead** | 4.1, 4.2 | Preparing data for DistilBERT/T5. |
| **Project Lead** | 1.1, 1.2, 3.1, 3.2 | Systems integration & Decision-making. |