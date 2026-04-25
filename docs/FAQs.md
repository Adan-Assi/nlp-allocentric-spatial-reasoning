# Justifications & FAQs

### Q1: Why filter the dataset into a "Silver Standard" (7,263 rows)?
1. **Noise Filtering:** Human-generated datasets contain "impossible" instructions (e.g., paths through buildings). Verification ensures failure is due to **LLM logic**, not **bad data**.
2. **The "Clean Room" Requirement:** To scientifically test a model, the ground truth must be 100% reachable. The Silver Standard provides a "Certified Answer Key."
3. **Trajectory Granularity:** It provides coordinates for every intermediate landmark, enabling partial credit analysis rather than just "pass/fail."

### Q2: Why use Oracle labels if we already have the original Human Goal?
**The Answer:** Because we are testing **Reasoning**, not just **Memory.**
1. **The "Lucky Guess" Problem:** If we mask a landmark, there might be 20 matches that fit the description. Even if the original goal was one specific shop, a masked instruction that fits 20 locations is scientifically **Ambiguous**.
2. **The Hallucination Metric:** If the Oracle identifies 20 matches (Ambiguity) but the LLM confidently picks the original Human Goal, the LLM is **hallucinating certainty**. We measure if the LLM's confidence matches the Oracle's mathematical probability.
3. **The Robustness Metric:** If the Oracle says a path is impossible (Contradictory), a robust LLM should state it cannot find the destination, rather than "guessing" a coordinate.

### Q3: Why mask the instructions (e.g., [MASK])?
1. **Anti-Leakage:** Large models have memorized city maps. Masking landmarks prevents them from "cheating" via training memory.
2. **Dead Reckoning:** Forces the model to rely on **geometric vectors** (turn angles and distances) rather than specific names.
3. **Generalization:** It proves the model understands the **physics of space**; a skill that should work in any city, even those not in its training data.

---

## 🎨 The Logic Loop
```text
[ RAW RVS DATA ] 
      |
[ ORACLE ENGINE ] ----> Normalizes text & resolves Landmark IDs.
      |
[ SYMBOLIC SOLVER ] --> Validates geometry & filters contradictions.
      |
[ SILVER STANDARD ] --> RESULTS: 7,263 verified rows (The "Answer Key").
      |
      +---- [ MASKED PROMPT ] --> Sent to LLM (Pythia/T5) on Cluster GPUs.
      |                                |
      +---- [ EVALUATION ] <-----------+ Compare Predicted vs. Oracle Coordinates.
```