# SOTA Phases: NLP Project Roadmap

## Roadmap: From Rule-Based to SOTA

### 1. Semantic Matching (Lectures 1-2) (Complete ✅)

**Current:** `TEXT_TO_GROUP_MAP` does exact string matching — "doctor's office" fails if not in the dict.

**SOTA approach:** Replace the dictionary lookup in `CategoricalMatcher.get_category()` with embedding-based similarity:

```python
# Instead of:
if text_lower in self.text_lookup:
    return self.text_lookup[text_lower]

# Use sentence embeddings:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-encode all category descriptions once
category_descriptions = {
    "HOSPITAL": "hospital clinic medical center emergency room doctors",
    "OFFICE": "office corporate headquarters workplace business",
    "GYM": "fitness center gym sports workout exercise",
    # ...
}
category_embeddings = {k: model.encode(v) for k, v in category_descriptions.items()}

# At query time:
query_emb = model.encode(noun)
best_category = max(category_embeddings,
                    key=lambda k: cosine_similarity(query_emb, category_embeddings[k]))
```

This handles "doctor's office" vs "corporate office" through vector space geometry — "doctor's office" will be closer to HOSPITAL than OFFICE in embedding space.

**Why it matters for Our project:** `extracted_noun` is the exact thing being sent to `get_category()`. Replacing step 3 (partial match) and step 4 (fuzzy) in `CategoricalMatcher` with embedding similarity would eliminate most of the remaining UNKNOWN cases without hand-coding synonyms.

---
### ~~2. LLM as Orchestrator (Lecture 7)~~
<details>

**Current:** Rule-based pipeline: regex → string match → OSM filter → KDTree.

**SOTA approach:** The LLM routes intent to our symbolic tools rather than regex doing it:

```
User instruction
      ↓
LLM (intent router) ── tool_call: search_poi(category="HOSPITAL", direction="N", radius=500)
      ↓
OracleEngine.resolve_nearby_candidates()
      ↓
Result back to LLM for final grounding decision
```

Concretely, this means defining our oracle methods as **tools** in an OpenAI-style function calling schema:

```python
tools = [
    {
        "name": "search_landmarks",
        "description": "Search for POIs matching a category near a location",
        "parameters": {
            "category": {"type": "string", "enum": list(LANDMARK_GROUPS.keys())},
            "direction": {"type": "string", "enum": ["N","NE","E","SE","S","SW","W","NW"]},
            "landmark_name": {"type": "string"}
        }
    }
]
```

The LLM then calls `search_landmarks(category="HOSPITAL", direction="NE", landmark_name="UPMC")` instead of our regex extracting those fields. This is exactly the **tool-use** paradigm from Lecture 7.

**Why it matters:** Our current `extract_rvs_target()` is doing what an LLM with tool-calling does natively and more accurately. The LLM understands "the place where I got my stitches" → HOSPITAL; regex doesn't.
</details>

### **Decision: REJECT**

**Reasoning:**
- Our pipeline already has a working deterministic extraction system (`extract_rvs_target`) + semantic matcher (Phase 1) + dense retrieval (Phase 3). LLM orchestration would replace stable, auditable components with a black box that produces unpredictable outputs
- The course explicitly warns against over-reliance on LLMs for core system logic — orchestration IS core system logic
- Tool-calling reliability at 1B-7B scale is not guaranteed. Models at this scale frequently hallucinate tool arguments, call wrong tools, or produce malformed JSON
- Your oracle is symbolic and deterministic by design, introducing a probabilistic router upstream corrupts the research validity. If the LLM misroutes "pharmacy" → SHOP, you can't tell whether downstream failures are oracle failures or routing failures
- The research question is about LLM spatial reasoning under underspecification. The LLM is the **subject of study**, not the infrastructure. Using it as an orchestrator conflates the evaluator with the evaluated

**Risks:**
- Unstable tool-call outputs break the labeling pipeline mid-run on 9301 samples
- Non-reproducible results — same instruction routed differently on different runs
- No guarantee small models support structured tool-calling schemas
- Debugging becomes extremely difficult when failures occur inside LLM reasoning
- Violates instructor warning against LLM-as-judge / LLM-as-core-logic explicitly

**Suggested alternative approach:**
- Our current `extract_rvs_target()` + `CategoricalMatcher` already performs the routing job reliably and deterministically

- Document the decision in the paper: "We considered LLM-based intent routing but chose deterministic extraction to ensure oracle reliability and research reproducibility"

---

### 3. Information Retrieval (Lecture 11)

**Current:** KDTree spatial lookup → score by name match + category match → sort by score.

**SOTA approach:** Treat landmark retrieval as a **dense retrieval** problem:

```
Query: "the big pharmacy near the church"
      ↓
Bi-encoder: encode query → query vector
      ↓
FAISS index of all POI descriptions → top-k nearest POIs
      ↓
Cross-encoder reranker: score(query, each_POI) → final ranked list
```

Each POI in the dataset gets encoded as a text description:

```python
def poi_to_text(row):
    return f"{row['name']} {row['amenity']} {row['shop']} near {row['addr:street']}"
```

This is the **bi-encoder + cross-encoder** pipeline from Lecture 11 (Dense Passage Retrieval style). Currently the `resolve_all_candidates()` scoring (`name_mask * 2.5 + cat_mask * 1.5`) is a sparse BM25-style scorer — upgrading to dense retrieval would handle paraphrases and descriptions naturally.

**Why it matters:** "The place where you pick up prescriptions" would retrieve pharmacies even though the word "pharmacy" never appears, because the embedding space captures the semantic relationship.

---

### 4. Evaluation (Lecture 9)

**Current:** We measure `oracle_label` distribution (Answerable %) and geographic distance (250m accuracy).

**SOTA approach — three evaluation layers:**

**Layer 1 — Intrinsic (category extraction accuracy):**
Create a small gold-labeled test set of 100 instructions with manually verified categories, then measure:

```python
# Precision/Recall per category
from sklearn.metrics import classification_report
print(classification_report(gold_categories, predicted_categories))
```

**Layer 2 — Extrinsic (end-to-end grounding):**
Our current 250m metric. But also report:
- **Mean Reciprocal Rank (MRR)** — where does the correct POI rank among candidates?
- **Recall@k** — is the correct POI in the top-k candidates before filtering?

**Layer 3 — Alignment (Lecture 9 specific):**
Measure whether the oracle's Answerable/Ambiguous labels align with human judgment on a sample:

```python
# Human study: show 50 masked variants to annotators
# Ask: "Is this instruction uniquely answerable?"
# Compare to oracle label → Cohen's Kappa
```

This directly addresses the alignment question from Lecture 9 — are our automatic labels aligned with what humans would say?

---

## Summary Roadmap

| Component | Current | SOTA Upgrade | Lecture |
|-----------|---------|--------------|---------|
| `get_category()` | Dict + fuzzy string match | Sentence embedding similarity | 1-2 |
| `extract_rvs_target()` | Regex | LLM tool-call intent extraction | 6-7 |
| `resolve_all_candidates()` | Sparse score (name+cat mask) | Dense retrieval (FAISS + bi-encoder) | 11 |
| `filter_candidates_by_direction()` | Rule-based bearing | Keep — symbolic is correct here | — |
| Evaluation | Answerable % + 250m | + MRR, Recall@k, Cohen's Kappa | 9-10 |

> **What to actually implement for the course:** The embedding-based `get_category()` (Lectures 1-2) is the highest-impact, lowest-effort upgrade. It directly replaces the `TEXT_TO_GROUP_MAP` which is the most brittle part of the pipeline. Everything else can be discussed as future work.

---

## Phase 1 vs Phase 3: Clear Distinction

Think of the pipeline as two completely separate questions:

> **Phase 1 answers:** "What TYPE of place is the user looking for?"
> **Phase 3 answers:** "Which SPECIFIC place in the city matches?"

---

### Concrete Example

**Instruction:** "Take me to the nearest pharmacy near Central Park"

#### Phase 1 — Category Extraction (already done)

```
"pharmacy" → CategoricalMatcher → "PHARMACY"
tags = {"amenity": "pharmacy", "brand": "yes"}
```

Phase 1 is finished here. It produced a category label and OSM tags. **It never touches the map, the graph, or any actual POI data.**

#### Phase 3 — Candidate Retrieval (what we're improving)

```
tags = {"amenity": "pharmacy"}
start_lat, start_lon = (40.7829, -73.9654)  # near Central Park

# Current sparse scoring:
CVS on 72nd St          → name contains "pharmacy"? No → score 0.0 ❌ missed
Duane Reade on 66th     → amenity="pharmacy"? Yes → score 1.5 ✓
Rite Aid on Broadway    → amenity="pharmacy"? Yes → score 1.5 ✓
"The Pharmacy Bar"      → name contains "pharmacy"? Yes → score 2.5 ✗ wrong!
```

The current scoring gives "The Pharmacy Bar" a **higher score** than actual pharmacies because it matches by name string. It has no understanding that "The Pharmacy Bar" is a bar, not a pharmacy.

---

### Why Phase 1 Can't Solve This

Phase 1 operates on the **instruction text** — it extracts intent from language.
Phase 3 operates on the **POI database** — it retrieves actual places from OSM data.

They operate on completely different inputs:

| | Phase 1 | Phase 3 |
|--|---------|---------|
| Input | User's instruction text | POI dataframe rows |
| Output | Category + tags | Ranked list of candidate nodes |
| Knows about | Language, synonyms, intent | Map data, coordinates, OSM attributes |
| Example | "pharmacy" → PHARMACY | PHARMACY → [CVS, Duane Reade, Rite Aid] |

---

### What Phase 3 Specifically Improves

The current `resolve_all_candidates` scoring is:

```python
name_mask  → +2.5  (does POI name contain the query string?)
cat_mask   → +1.5  (does POI have the right OSM tag?)
```

This is **sparse keyword matching** — the same paradigm as BM25 from Lecture 11. Problems:

**1. Name matching is unreliable:**
- "The Pharmacy Bar" scores 2.5 — wrong type, high score
- "CVS" scores 0.0 for query "pharmacy" — right type, zero score

**2. No semantic understanding of POI descriptions:**
- A POI with `name="Duane Reade"` and `amenity="pharmacy"` scores 1.5
- A POI with `name="pharmacy"` and `amenity="bar"` scores 2.5
- The bar wins despite being wrong

**3. No ranking by relevance to the full instruction context:**
- "nearest pharmacy near Central Park" — current code ignores "near Central Park" entirely after spatial pruning

---

### What Dense Retrieval Fixes (Lecture 11)

Instead of keyword overlap, encode each POI as a text description and rank by vector similarity to the query:

```python
# POI description (built from OSM columns):
"CVS pharmacy on West 72nd Street, amenity: pharmacy, brand: CVS"

# Query (from the instruction):
"pharmacy near Central Park"

# Cosine similarity between their embeddings → meaningful relevance score
# "The Pharmacy Bar" description includes "bar, pub, drinks" → low similarity to "pharmacy"
# "CVS pharmacy" description → high similarity to "pharmacy"
```

This is the **bi-encoder** from Lecture 11 — the same model that powers modern search engines.

---

### How They Work Together

```
Instruction: "Take me to the nearest pharmacy near Central Park"
        │
        ▼
PHASE 1 (extraction_utils.py)
"pharmacy" → PHARMACY
tags = {"amenity": "pharmacy"}
direction = None
        │
        ▼
PHASE 3 (oracle_engine.py — resolve_all_candidates)
Spatial filter: POIs within 1500m of start node
Dense retrieval: rank by embedding similarity to "pharmacy near Central Park"
Returns: [CVS (0.89), Duane Reade (0.87), Rite Aid (0.82), Pharmacy Bar (0.21)]
        │
        ▼
solve() → _pick_by_salience() → single target node
```

Phase 1 tells the system **what to look for**.
Phase 3 tells the system **where it actually is**.

Without Phase 3 improvement, even perfect Phase 1 extraction gets undermined by a retrieval system that ranks "The Pharmacy Bar" above real pharmacies.