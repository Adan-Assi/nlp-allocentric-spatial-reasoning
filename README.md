# Robust Allocentric Spatial Reasoning under Underspecified Instructions

Course project (NLP): evaluating the robustness of Large Language Models (LLMs) in **allocentric spatial reasoning**.

This project investigates how models handle navigation instructions when spatial information is **systematically underspecified**, using a fixed, geodetically grounded urban map graph as the source of truth.

It uses the **Rendezvous (RVS)** dataset:
- human-written allocentric navigation instructions
- grounded in real-world urban map graphs
- clear, graph-backed ground truth (reachability, paths, directions)

---

## Research question

> How robust are large language models at allocentric spatial reasoning when language is systematically underspecified, but a consistent underlying map exists?

The symbolic solver labels each (possibly underspecified) instruction as one of three classes, per the project proposal:

| Label | Meaning |
| :--- | :--- |
| **Answerable** | Exactly one reachable POI satisfies the remaining constraints. |
| **Ambiguous** | Multiple reachable POIs satisfy the remaining constraints. |
| **Contradictory** | No reachable POI satisfies the remaining constraints. |

This three-class output is the dependent variable of the experiment.

---

## What changed in this version

A short summary of the corrections; full rationale lives in [`CHANGES.md`](CHANGES.md).

1. **Ambiguity is preserved.** The previous solver collapsed multi-candidate cases by picking the most salient POI and labeling them *Answerable*. The current solver returns *Ambiguous* in that case, matching the proposal's definition. A `mode="resolve"` flag is available for RVS-style original-goal recovery, but it should not be used for underspecification labels.
2. **Direction reasoning is 8-way.** Instructions like "head northeast" are no longer collapsed to "north". The classifier returns one of `N, NE, E, SE, S, SW, W, NW`. Cardinal targets (N) match adjacent intercardinals (NW, N, NE); intercardinal targets (NE) require an exact match.
3. **Reachability is filtered first.** The candidate count used for the 3-way label only includes POIs that are reachable from the start node. The old code counted before the reachability filter, which inflated *Answerable*.
4. **No more silent-junk extraction.** Bare cardinal words inside named entities (e.g. "East 49th Street", "North Face") no longer trigger a direction. Direction is only extracted in valid grammatical frames (motion verb, "<dir> of", "on/to my <dir>", "<n> blocks <dir>").
5. **Reproducible salience tiebreaker.** When two candidates tie on salience tier and distance, the lexicographically-smallest `node_id` wins. Without this, KDTree iteration order leaked into labels and reproducibility broke across `numpy`/`scipy` versions.
6. **Missing dependencies declared.** `thefuzz` and `python-Levenshtein` are now in `requirements.txt`.

---

## Repository structure

```text
nlp-allocentric-spatial-reasoning/
├── config.py                                # Centralized multi-city settings
├── requirements.txt
├── CHANGES.md                               # File-by-file change log
├── README.md                                # This file
├── src/
│   ├── extraction_utils.py                  # Rule-based NLP (8-way directions, context-aware)
│   ├── oracle_engine.py                     # POI resolution & graph queries (Where)
│   ├── symbolic_solver.py                   # Master controller, 3-class labeling (How)
│   └── utils.py                             # Math primitives + 8-way helpers
├── scripts/
│   ├── batch_labeling.py                    # Per-city silver-standard labeling
│   ├── label_variants_with_oracle.py        # Constraint-based variant labeling
│   ├── attach_target_node_all_regions.py    # Snap (target_lat, target_lon) → node_id
│   ├── stress_test_oracle.py                # Variant-level oracle validation
│   └── ...
├── tests/
│   └── sanity_check_logic.py                # Data-free smoke test (run anytime)
├── data/
│   └── <city>/<city>_graph.gpickle, ..._poi.pkl, ...
├── docs/                                    # Strategy & specs
└── reports/                                 # Audits, plots, methodology
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the data-free sanity check (no graph data needed)

```bash
python tests/sanity_check_logic.py
```

You should see **`✅ ALL CHECKS PASSED`**. This validates the direction extractor, the 8-way classifier, and the cardinal-vs-intercardinal compatibility rule.

### 3. Label a city

```bash
python scripts/batch_labeling.py --city manhattan
```

Output goes to `data/manhattan/manhattan_silver_standard.parquet`. The label distribution should now include a non-empty **Ambiguous** bucket.

### 4. Use the solver from your own code

```python
import config
from src.oracle_engine import OracleEngine
from src.symbolic_solver import SymbolicSolver

config.CURRENT_CITY = "manhattan"
oracle = OracleEngine(
    config.get_graph_path(),
    config.get_poi_path(),
    config.get_node_prefix(),
    "manhattan",
)
solver = SymbolicSolver(oracle, search_radius=config.get_success_radius())

# Default mode is "label" — preserves ambiguity per the project proposal.
result = solver.solve("Head northeast to meet me at the cafe", start_node="1#7977067481")

# state ∈ {"Answerable", "Ambiguous", "Contradictory"}
print(result["state"])
print(result["candidate_count"], "candidates,",
      result["reachable_candidate_count"], "reachable")

# In Answerable cases, target_node is set:
print(result.get("target_node"))

# In Ambiguous cases (label mode), candidate_nodes lists up to 50 IDs:
print(result.get("candidate_nodes"))
```

For RVS-style original-goal recovery (single answer always returned), use the resolver mode:

```python
result = solver.solve(text, start_node, mode="resolve")
# state will be Answerable or Contradictory in this mode.
# result["selection_strategy"] tells you whether a salience pick was used.
```

---

## High-level pipeline

1. **Instruction input** — original RVS text or an underspecified variant.
2. **NLP extraction** — `extract_rvs_target` returns `(category, noun, direction)`.
3. **Oracle (Where)** — resolves POI candidates via OSM tag matching, name fuzzy-search, and direction filtering.
4. **Solver (How)** — applies reachability, then assigns one of the 3 labels.
5. **LLM inference** — small open-weight models (e.g. T5, Pythia) attempt the same task.
6. **Comparison** — error taxonomy on the 3-class output.

---

## Documentation hub

- [**FAQs**](docs/FAQs.md) — read first; the "why" behind Symbolic Masking, 1.5km Reasoning Horizon, Logic Loop pipeline.
- [**Strategy & Calibration**](docs/STRATEGY_CALIBRATION.md) — high-level vision and stages.
- [**Oracle Specification**](docs/ORACLE_SPEC.md) — math behind the Symbolic Solver (vectors / bearings).
- [**Pipeline Data Flow**](docs/PIPELINE_FLOW.md) — trace from raw text to answerability labels.
- [**Data & Graph Guide**](docs/DATA_GUIDE.md) — graph + POI structure reference.
- [**CHANGES.md**](CHANGES.md) — file-by-file change log for this iteration.

---

## Team

- Adan Assi
- Shaimaa Hoji
- Noor Mhajne

(Tel Aviv University — NLP Course Project 2025–2026)
