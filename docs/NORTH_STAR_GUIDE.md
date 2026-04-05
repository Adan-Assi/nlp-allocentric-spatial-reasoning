# 🧭 Project Roadmap: Allocentric Spatial Reasoning & Robustness

## 📝 The Core Research Question
> **How does an LLM's spatial reasoning degrade when we systematically "break" a clear instruction?**
> Even if the map stays the same, does the model realize when an instruction is no longer "Answerable"?

---

## 🛠️ The Logic Flow: From Raw Data to Evaluation

### 1. Data Ingestion (The "Base World")
We start with the **Rendezvous (RVS)** dataset. 
- **Input:** A starting point (S0), a human instruction, and a gold goal (G).
- **Environment:** Real-world map graphs (Manhattan, Pittsburgh, Philadelphia).

### 2. Identity Resolution (The Oracle Engine)
Before we can do math, we must find the landmarks in the real world.
- **File:** `src/oracle_engine.py`
- **Input:** Instruction string (e.g., "Meet me at the Starbucks near Bryant Park").
- **Logic:**
    - **Normalization:** Cleans text (lowercase, removes punctuation).
    - **Lookup:** Queries `LANDMARK_GROUPS` in `config.py` to identify the "Type" (e.g., Coffee Shop).
    - **Spatial Matching:** Searches city-specific `.pkl` files for the specific name or type.
- **Output:** A set of candidate **Graph Node IDs** (OSMIDs) representing the mentioned landmarks.

### 3. Geometric Reasoning (The Symbolic Solver)
Now that we have the physical nodes, we test the spatial constraints.
- **File:** `src/symbolic_solver.py`
- **Logic:** It applies filters like "North of," "Within 100m," or "Away from" using vector math and shortest-path algorithms.
- **The Labels:**
    - **Answerable:** Only **1** node in the entire city fits all constraints.
    - **Ambiguous:** **Multiple** nodes fit (The instruction is now vague).
    - **Contradictory:** **0** nodes fit (The instruction is now impossible).



---

## 🔍 Clarification: Oracle Truth vs. Human Goal

A common point of confusion is: *If we have the original Human Goal from the RVS dataset, why do we need the Oracle to generate new labels?*

**The Answer:** Because we are testing **Reasoning**, not just **Memory.**

1. **The "Lucky Guess" Problem:** If we mask "Starbucks" to "[MASK]", there might be 20 different coffee shops that fit the description. Even if the "Human Goal" was one specific Starbucks, a masked instruction that fits 20 locations is scientifically **Ambiguous**. 

2. **Measuring Hallucination:** If the Oracle says "Ambiguous (20 matches)" but the LLM confidently picks the original Human Goal, the LLM is **hallucinating certainty**. It isn't reasoning; it's guessing based on surface patterns.

3. **The Robustness Metric:** We evaluate the LLM on whether its behavior matches the **Oracle's mathematical certainty**. If the Oracle says it's impossible (Contradictory), a robust LLM should say "I can't find that," rather than picking the original goal node.

---

## 🚀 Execution Strategy (Phase 4 & 5)

### Step A: Multi-City Scaling
- Ensure `config.py` toggles correctly between cities and applies city-specific success radii (80m for MHT, 100m for PIT/PHL).

### Step B: The Slurm Power-Up
- Processing 10k instructions × 3 cities × 5 variants = **~150k simulations**.
- Use Slurm to parallelize these across the university cluster.

### Step C: Degradation Analysis
- Create plots where the X-axis is "Information Removed" and the Y-axis is "LLM Accuracy relative to the Oracle."

---

## 📂 Key Files to Remember

| File | Purpose |
| :--- | :--- |
| `config.py` | The "Remote Control" (Switch cities, change radii). |
| `src/oracle_engine.py` | The "Eyes" (Finds landmarks/identities in the data). |
| `src/symbolic_solver.py` | The "Brain" (Performs geometry and graph math). |
| `scripts/batch_labeling.py` | The "Factory" (Processes thousands of rows). |
| `scripts/submit_labeling.sh` | The "Slurm Key" (Unlocks cluster power). |