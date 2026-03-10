# Quality Assurance & Sanity Checks

This folder contains the "Gatekeeper" scripts for our project. These tests verify that our spatial logic, graph structures, and NLP-to-Symbolic bridges are mathematically sound and data-consistent.

## Files

* **`test_symbolic_solver.py` (New)**: The primary integration test. Verifies the end-to-end pipeline: loading the Manhattan graph/POI pickle, resolving landmarks via the `OracleEngine`, and calculating paths/bearings via the `SymbolicSolver`.
* **`sanity_check_all_graphs.py`**: Validates that `.gpickle` files for all regions are loadable, connected, and support shortest-path calculations.
* **`sanity_check.py`**: A lightweight version specifically for Manhattan; used for quick debugging of coordinate-to-bearing logic.
* **`test_variants.py`**: Verifies the NLP logic in `src/instruction_degrader.py`, ensuring degraded instructions (missing cardinal directions or proximity) are formatted correctly.

## When to Run
1.  **Data Ingestion**: After adding new `.gpickle` or `.pkl` files to `data/`, run the sanity checks.
2.  **Logic Updates**: If you modify `src/symbolic_solver.py`, `src/oracle_engine.py`, or `src/utils.py`, run the integration test.
3.  **Before Deployment**: All tests must pass before merging into the main branch.

## Execution

Always execute from the **project root** using the module flag to ensure correct path resolution:

```bash
# Run Integration Tests
python -m tests.test_symbolic_solver

# Run Graph Sanity Checks
python -m tests.sanity_check_all_graphs

# Run NLP Logic Tests
python -m tests.test_variants.py
```
---

## ⚠️ Technical Debt & Refactoring
The following legacy scripts likely require refactoring to align with the **NetworkX 3.0+** and **Pickle-based** data loading implemented in Phase 2:

* **`sanity_check_all_graphs.py`**
* **`sanity_check.py`**

**Issue:** These scripts still utilize `nx.read_gpickle()`, which is deprecated/removed in modern NetworkX versions. They should be updated to use `pickle.load(open(path, 'rb'))` as seen in `src/oracle_engine.py`.