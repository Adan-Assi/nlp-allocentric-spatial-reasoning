# 📄 Incident Report: 05-Pittsburgh Silver Standard Regression

## Summary
A series of cascading bugs caused Pittsburgh's silver standard Answerable rate to drop from a baseline of **76.0%** to **24.6%**, then were incrementally diagnosed and resolved back to **~63%+** through systematic debugging across node ID normalization, candidate resolution, directional filtering, NLP extraction, and oracle labeling logic.

---

## Timeline & Root Causes

### BUG 1 — Silent KeyError in `filter_candidates_by_direction`
**Symptom:** Warnings of the form `⚠️ Warning: Failed to process sample 289. Error: '8146635481'` — raw node ID strings as exception messages.  
**Root Cause:** `self.G.nodes[node_id]` raised a `KeyError` because candidate node IDs from the POI dataframe were strings while graph nodes were a different type/format.  
**Fix:** Wrapped lookup in `get_graph_node()` with graceful `continue` on failure.

---

### BUG 2 — Node ID Format Mismatch (Philadelphia)
**Symptom:** Philadelphia crash: `KeyError: '3010327442'` in `filter_candidates_by_direction` at `self.G.nodes[node_id]`.  
**Root Cause:** Philadelphia's graph nodes use string prefixes (`1#...`, `#...`), but candidate IDs from the POI dataframe were plain integer strings with no prefix.  
**Fix:** Added `get_graph_node()` call inside `filter_candidates_by_direction` to normalize all candidate IDs before graph lookup.

---

### BUG 3 — `graph_node_id` Column Not Being Used
**Symptom:** Despite `_prepare_poi_data()` correctly building `graph_node_id`, direction filter still dropped 80-90% of candidates.  
**Root Cause:** Three separate locations in `oracle_engine.py` were doing ad-hoc prefix reconstruction (`f"1#{osmid}"`) instead of using the pre-normalized `graph_node_id` column:
- `resolve_all_candidates` (line 444)
- `resolve_by_tags` (line 367)
- `resolve_landmark` (line 188)

**Fix:** Replaced all three with `row.get('graph_node_id')` and added `if node_id not in self.G.nodes: continue` guard.

---

### BUG 4 — OSM Type Prefix Not Stripped in `normalize_node_id`
**Symptom:** 502 POI rows showing as not in graph despite valid node IDs. Missing IDs had format `node/281362013`.  
**Root Cause:** `normalize_node_id` stripped `#` but not OSM type prefixes (`node/`, `way/`, `relation/`), producing lookups like `1#node/281362013`.  
**Fix:** Added prefix stripping loop before variant construction:
```python
for osm_prefix in ('node/', 'way/', 'relation/'):
    if raw_id.startswith(osm_prefix):
        raw_id = raw_id[len(osm_prefix):]
        break
```

---

### BUG 5 — Candidates Valid in Graph but Still Failing Direction Filter
**Symptom:** Candidates like `1#357379432` confirmed to exist in graph via direct test, yet still dropped by direction filter.  
**Root Cause:** `get_graph_node()` was never being called — diagnosed via missing debug prints. Root issue was `.pyc` bytecode cache running stale version of `oracle_engine.py`.  
**Fix:** 
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {}+
```

---

### BUG 6 — High Contradictory Rate from UNKNOWN Category (504/535)
**Symptom:** 49% of samples getting `UNKNOWN` category, vs ~5% in old pipeline.  
**Root Cause:** `TEXT_TO_GROUP_MAP` in `config.py` was missing hundreds of common landmark terms ("bar", "boutique shop", "fitness center", "pharmacy variants", "garden", "fountain", etc.).  
**Fix:** Expanded `TEXT_TO_GROUP_MAP` with ~80 new entries grouped by category using `dict.fromkeys()` pattern.

---

### BUG 7 — Mid-Word Truncation in Noun Extraction
**Symptom:** "gas station" → `gas stati`, "tennis court" → `tenn`, "shelter pavilion" → `shelter pavili`.  
**Root Cause:** `_AT_THE_RE` lookahead `(?=on\b|...)` matched `on` inside words like "stati**on**", "pavili**on**". Similarly `_STOPS` matched single letters inside words.  
**Fix:** Added `\s+` before boundary tokens in `_AT_THE_RE` lookahead:
```python
r"(?=\s+on\b|\s+is\b|\s+at\b|\s+near\b|\s+just\b|\s+in\b|,|\.|\s+and\b|$)"
```
Removed over-broad tokens (`a`, `an`, `the`, `on`) from `_STOPS`.

---

### BUG 8 — False ROAD Classification
**Symptom:** "tennis court" → `ROAD`, "fountain in the square" → `ROAD`.  
**Root Cause:** `_ROAD_SUFFIXES` included `court`, `ct`, `way`, `square` — matching legitimate POI nouns.  
**Fix:** Removed `court|ct|way|square` from `_ROAD_SUFFIXES`.

---

### BUG 9 — `_STOPS` Matching Inside Valid Noun Phrases
**Symptom:** "corner store" → `None` (span trimmed to "the"). "shelter pavilion in the park" not trimming at "in".  
**Root Cause:** `corner` in `_STOPS` matched inside "the **corner** store". `in` not in `_AT_THE_RE` lookahead so full phrase captured before `_STOPS` applied.  
**Fix:** 
- Removed `corner`, `end` from `_STOPS`
- Added `\s+in\b` to `_AT_THE_RE` lookahead
- Added `on` back to `_STOPS` (safe now that `_AT_THE_RE` uses `\s+on\b`)

---

### BUG 10 — Wrong Ambiguous Logic (Core Label Regression)
**Symptom:** Answerable at 30% with high Ambiguous rate (32%), despite candidates being found.  
**Root Cause:** `solve()` returned `STATE_AMBIGUOUS` when multiple candidates existed and nearest was >250m. This had no basis in the RVS paper — the 250m threshold is an *evaluation metric*, not a labeling criterion.  
**Root Cause (deeper):** Per RVS paper, ground truth is always a unique pre-selected node chosen by salience. `STATE_AMBIGUOUS` belongs only in `underspecify_instructions.py` (Step 4), not in silver standard labeling.  
**Fix:** Replaced distance-based Ambiguous logic with salience-based single candidate selection per RVS LANDMARK baseline hierarchy:
1. Wikipedia
2. Wikidata
3. Brand
4. Tourism
5. Amenity
6. Shop

`STATE_AMBIGUOUS` removed from `solve()` entirely. Silver standard now only produces `Answerable` / `Contradictory`.

---

## Final State

| Metric | Before | After |
|--------|--------|-------|
| Answerable | 24.6% | ~63%+ |
| Contradictory | 56.5% | ~37% |
| Ambiguous | 18.9% | 0% |
| UNKNOWN Contradictory | 504 | ~343 |

---

## Files Modified
- `src/oracle_engine.py` — `filter_candidates_by_direction`, `resolve_all_candidates`, `resolve_by_tags`, `resolve_landmark`, `_prepare_poi_data`, `normalize_node_id`
- `src/symbolic_solver.py` — `solve()`, added `_pick_by_salience()`, added `SALIENCE_COLS`
- `src/extraction_utils.py` — `_AT_THE_RE`, `_STOPS`, `_ROAD_SUFFIXES`, `_JUNK_PATTERNS`
- `config.py` — `TEXT_TO_GROUP_MAP` expanded and restructured
