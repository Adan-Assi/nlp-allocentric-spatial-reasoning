# Quality Assurance & Sanity Checks

This folder contains the "Gatekeeper" scripts for our project. Before moving to Phase 3 (Model Training), all scripts in this folder must pass. They verify that our spatial logic and graph structures are mathematically sound.

## Files

* **`sanity_check_all_graphs.py`**: Validates that the `.gpickle` files for all regions are loadable, connected, and support shortest-path calculations.
* **`sanity_check.py`**: A lightweight version specifically for Manhattan; used for quick debugging of coordinate-to-bearing logic.
* **`test_variants.py`**: Verifies the NLP logic in `src/instruction_degrader.py`. It ensures that when we drop "North" or "Proximity" landmarks, the resulting text is formatted correctly.

## When to Run
1.  **After Building Graphs**: Run `sanity_check_all_graphs.py` to ensure OSMnx didn't produce any isolated islands in the graph.
2.  **After Logic Updates**: If you modify `SymbolicSolver.py` or `utils.py`, run these tests to ensure no regressions were introduced.

## Execution

Always execute from the **project root**:

```bash
python tests/sanity_check_all_graphs.py
python tests/test_variants.py
```