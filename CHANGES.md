# CHANGES — Improved Version

A consolidated, single-pass review of every code change against the version in `/mnt/project/`, the working files you uploaded, and the `nlp_fixed_logic_patch.zip`. Below is **what changed**, **why**, and **what stays the same** so downstream callers (e.g. `batch_labeling.py`) keep working.

---

## TL;DR

The original repo had three behaviors that contradicted the project proposal's own research question. This version:

1. **Preserves ambiguity** instead of collapsing it (the project's own definition of *Ambiguous* was being silently overridden by salience-picking).
2. **Preserves 8-way directions** (NE/NW/SE/SW) instead of squashing them to N/S/E/W.
3. **Filters reachability before counting**, so candidate counts reflect genuinely reachable POIs.
4. **Stops false-positive direction extraction** ("East 49th Street", "North Face").
5. Adds the missing `thefuzz` / `python-Levenshtein` dependencies that `extraction_utils.py` already imports.

These match the patch's intent. Where the patch and the working uploaded files diverged, this version takes the better of the two and keeps a deterministic salience tiebreaker so labels are reproducible across `numpy`/`scipy` versions.

---

## File-by-file

### `src/symbolic_solver.py` — biggest behavioral change

| Aspect | Before | After |
|---|---|---|
| Signature | `solve(text, start_node)` | `solve(text, start_node, mode="label")` |
| When `>1` candidates exist | Always picked the most salient → returned `Answerable` | `mode="label"`: returns `Ambiguous`. `mode="resolve"`: same as before. |
| Reachability | Checked after salience-collapse | Filtered FIRST, then counted |
| Tiebreaking | Two equal-salience candidates → KDTree iteration order leaked in | Lexicographically-smallest `node_id` always wins |
| Output keys | `state, candidate_count, target_node, extracted_*` | adds `mode, reachable_candidate_count, resolution_stage`; `target_node` only set when state=`Answerable`; `candidate_nodes` (≤50) added when state=`Ambiguous` (label mode); `selection_strategy` added in resolve mode |

**Why** — the project proposal explicitly defines:

> "labels each variant as **Answerable (unique solution), Ambiguous (multiple solutions), or Contradictory (no solution)**."

The previous code silently turned every Ambiguous case into Answerable via the salience pick, which would inflate Answerable accuracy in the eval and erase the central research signal.

**`batch_labeling.py` compatibility** — verified. It reads `state, candidate_count, extracted_*, target_node` via `.get()`. All still present; `target_node` is `None` for Ambiguous cases (which is correct — there's no single goal).

### `src/extraction_utils.py` — direction extraction hardened twice

Two fixes layered together:

1. **Context-aware direction regex** (already in your uploaded version). A bare cardinal word in a name (e.g. `East 49th Street`, `North Face`) no longer triggers a direction. Direction is only matched in:
   - motion verb + dir: `walk north`, `head northeast`
   - dir + of: `north of the park`
   - on/to (the/my/your) + dir: `on my south`
   - `<n> blocks <dir>`: `2 blocks east`
2. **8-way preserved** (this version's new fix). Instead of `.upper()[0]` which collapsed `northeast→N`, we map to one of `N, NE, E, SE, S, SW, W, NW`. This is essential because the project example *"Head northeast to meet me at the café"* literally hinges on whether NE is distinguishable from N.

### `src/utils.py` — adds 8-way helpers

Added two new functions while keeping every existing helper untouched:

- `get_direction_8way(lat1, lon1, lat2, lon2) → str` — 45° sector classifier returning N/NE/E/.../NW.
- `direction_matches(actual, target) → bool` — comparison with the natural-language compatibility rule: a cardinal target (N) accepts adjacent intercardinals (NW, N, NE), but an intercardinal target (NE) requires an exact match. This mirrors how a human says "head north" and means "anywhere from NW to NE", but "head northeast" means specifically NE.

The legacy `get_dominant_direction()` is kept for the proximity sanity tests and any other 4-way callers.

### `src/oracle_engine.py` — direction filter cleaned up

`filter_candidates_by_direction()` was riddled with debug prints and was collapsing `target_direction[0]` to a single letter. After:

- Uses `utils.get_direction_8way()` for candidate bearings.
- Uses `utils.direction_matches()` for the compatibility comparison.
- Debug prints removed.
- All other Oracle methods unchanged.

### `config.py` — minor cleanup only

Removed one duplicated `LANDMARK_GROUPS` entry: `"STORE": {"shop": "yes"}` (line 167 in the old file) was being shadowed two lines later by a more-specific entry. Same effective behavior, less confusing.

The multi-city support (`CITY_SETTINGS`, `get_graph_path()`, etc.) is preserved verbatim.

### `requirements.txt`

Added the two missing entries that the existing `extraction_utils.py` already imports:

```
thefuzz
python-Levenshtein
```

Without these, a clean install would crash on import.

---

## Pipeline scripts that were also broken in the original repo (and are now fixed)

These three scripts were broken in the original repo against the current solver/oracle API. They are fixed here:

### `scripts/label_variants_with_oracle.py`

Three concrete bugs:
1. Constructor was being called as `SymbolicSolver(str(gpath))` — the actual constructor takes an `OracleEngine`, not a path string.
2. Called `solver.nodes_within_radius(start_lat, start_lon, radius_m)` and `solver.filter_nodes_by_direction(...)` — neither method exists on `SymbolicSolver`.
3. The directions returned by `underspec_constraints.extract_constraints()` are spelled-out words ("north", "southwest"); the Oracle's direction filter expects abbreviations ("N", "SW"). No conversion was happening.

The fix builds an `OracleEngine` first, then passes it to `SymbolicSolver`, snaps `(start_lat, start_lon)` via `oracle.find_nearest_node()`, calls the existing `oracle.get_candidates_within_radius()` and `oracle.filter_candidates_by_direction()`, and maps direction words to abbreviations.

### `scripts/attach_target_node_all_regions.py`

One bug: called `solver.find_nearest_node(lat, lon)`. That method lives on `OracleEngine`, not `SymbolicSolver`. Switched to using the Oracle directly (`SymbolicSolver` was only being instantiated to access this method anyway).

### `scripts/stress_test_oracle.py`

Two structural bugs and one design improvement:
1. Referenced `config.GRAPH_PATH`, `config.POI_PATH`, `config.VARIANTS_JSON`, `config.RVS_DATA_JSON` — these module-level constants were removed when config went multi-city. Replaced with the per-city resolvers and explicit per-city paths.
2. Called `OracleEngine(graph, poi)` with two args — the constructor now requires `(graph_path, poi_path, node_prefix, city_name)`.
3. Re-implemented candidate filtering inline. Now delegates to `solver.solve(text, start_node, mode="label")` so the stress test agrees with the rest of the pipeline.

A `--city` CLI argument was added so the stress test can run against any of the three cities.

## Tests

A new data-free smoke test was added at `tests/sanity_check_logic.py`. It exercises:

1. 8-way direction extraction including the false-positive guards ("East 49th Street", "North Face").
2. The 8-way compass-sector classifier (`get_direction_8way`) at all 8 angles.
3. The `direction_matches` compatibility rule (cardinals coarse, intercardinals exact).
4. The contract that `extract_rvs_target` always returns a 3-tuple, even on empty / nonsensical input.

Run from the repo root:

```bash
python tests/sanity_check_logic.py
```

Expected output ends with `✅ ALL CHECKS PASSED` (32 checks).

## README

The repo's `README.md` is updated to reflect the corrected behavior — specifically the 3-class label semantics, the 8-way direction system, and the new `mode="label"` / `mode="resolve"` API. The original RVS dataset wording and team credits are preserved.

## What this version *deliberately* does NOT change

- **No changes to `batch_labeling.py`.** It already works against the new `solve()` signature because the new `mode` parameter has a default value (`"label"`) that gives the project's wanted behavior. The output keys (`state`, `candidate_count`, `extracted_*`, `target_node`) are all present in the new return; `target_node` is `None` for Ambiguous rows because there's no single goal — which is correct.
- **No changes to `audit_failures.py`.** It already works because it does dynamic method resolution (`getattr(solver, m)` for `m in ('solve', ...)`) and falls through to whichever solve method exists.

---

## How to call `solve()` after the upgrade

```python
# In batch_labeling.py — no change needed, default mode is "label":
label_info = solver.solve(instruction, start_node)

# Inspect the result:
label_info["state"]                       # "Answerable" | "Ambiguous" | "Contradictory"
label_info["candidate_count"]             # POIs found before reachability filter
label_info["reachable_candidate_count"]   # POIs found AND reachable
label_info.get("target_node")             # set when state == "Answerable"
label_info.get("candidate_nodes")         # list (≤50) when state == "Ambiguous"
```

For RVS-style original-goal recovery (e.g. when comparing your oracle to the RVS gold goal), use the resolver mode:

```python
label_info = solver.solve(instruction, start_node, mode="resolve")
# state will always be Answerable or Contradictory in this mode.
# label_info["selection_strategy"] tells you whether a salience pick was used.
```

---

## Verification

Run this in a Python shell after installing requirements (no graph data needed):

```python
from src.extraction_utils import extract_rvs_target
from src.utils import direction_matches, get_direction_8way

# 1. Direction extraction is 8-way and context-aware
assert extract_rvs_target("Head northeast to the cafe")[2] == "NE"
assert extract_rvs_target("Head northwest to the bank")[2] == "NW"
assert extract_rvs_target("Meet at the cafe on East 49th Street")[2] is None
assert extract_rvs_target("Head to North Face on Broadway")[2] is None

# 2. direction_matches: cardinals are coarse, intercardinals are exact
assert direction_matches("NE", "N") is True
assert direction_matches("SE", "N") is False
assert direction_matches("NE", "NE") is True
assert direction_matches("N", "NE") is False

# 3. 8-way classifier
assert get_direction_8way(40.0, -74.0, 40.01, -74.01) == "NW"  # +lat -lon
print("All checks passed.")
```

---

## Drop-in instructions

```bash
# 1. Replace these four files in your repo:
cp config.py                    /path/to/repo/config.py
cp src/extraction_utils.py      /path/to/repo/src/extraction_utils.py
cp src/utils.py                 /path/to/repo/src/utils.py
cp src/oracle_engine.py         /path/to/repo/src/oracle_engine.py
cp src/symbolic_solver.py       /path/to/repo/src/symbolic_solver.py

# 2. Update dependencies
cp requirements.txt             /path/to/repo/requirements.txt
pip install -r requirements.txt

# 3. Re-run the labeling pipeline; expect a meaningful Ambiguous bucket.
python scripts/batch_labeling.py --city manhattan
```

You should now see a non-empty `Ambiguous` row in the label distribution. That's the behavior the project proposal asks for.

---

## Suggested addendum for the project README

> The symbolic solver labels each underspecified variant as **Answerable** (one reachable POI satisfies the constraints), **Ambiguous** (multiple reachable POIs satisfy the constraints), or **Contradictory** (no reachable POI satisfies the constraints). This three-class output is the dependent variable of the experiment. Direction reasoning is 8-way (N, NE, E, SE, S, SW, W, NW); cardinal directions in instructions are treated as coarse (accepting adjacent intercardinals), while intercardinals require exact matches.
