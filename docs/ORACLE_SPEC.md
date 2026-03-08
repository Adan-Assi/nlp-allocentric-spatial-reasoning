# 📜 ORACLE_SPEC.md: Answerability Logic & Project Decisions

This document serves as the "Source of Truth" for our symbolic solver and data labeling pipeline. It defines the mathematical and physical constraints of our map-world to ensure consistent training for our Answerability Classifier.

---

## 🎯 The "Oracle" Mission Statement
We are shifting from a **Navigation Task** (where the AI guesses a path) to an **Information Theory Task** (where the AI judges if an instruction is sufficient).

* **Goal:** Use a symbolic "Oracle" to label our dataset as *Answerable*, *Ambiguous*, or *Contradictory*.
* **Purpose:** This creates the "Silver Standard" labels needed to train our Answerability Classifier (DistilBERT/T5).

### 🔍 Why are we locking down the "Oracle" now?
Before we begin training our models, we must have a mathematical "Ground Truth." In Machine Learning, if labels are inconsistent, the model will never converge. This protocol serves three purposes:

1.  **Building the "Labeling Factory":** Automatically generate labels for 50,000+ rows ($0$=Answerable, $1$=Ambiguous, $2$=Contradictory).
2.  **Ensuring Model Convergence:** A precise Oracle ensures the classifier learns the linguistic patterns of underspecification without "vague" label noise.
3.  **Scientific Control:** Provides a mathematical baseline for thesis validation—proving the AI isn't just guessing.

**The Goal:** Move from "Garbage In, Garbage Out" to a "Silver Standard" dataset.

---

## 📚 Glossary of Sets
* **$S_0$**: The set of possible starting nodes (accounts for intersection ambiguity).
* **$C$**: The initial pool of candidates (The BallTree search radius).
* **$S$**: The final filtered set of valid destinations.

---

## ⚖️ The "Oracle" Laws (Standardized Constants)

| Parameter | Rule/Decision | Justification |
| :--- | :--- | :--- |
| **Starting Set ($S_0$)** | Nodes within **20m** of geocode | OSM can have multiple nodes per intersection; $S_0$ prevents "Starting Point" failure. |
| **Search Radius ($C$)** | $\max(D \times 1.1, D + 80\text{m})$ | Prevents "Search Explosion" at long distances while allowing human vagueness at short ones. |
| **Directional Wedge** | $\text{Target} \pm 45^\circ$ | Accounts for "Fuzzy" human directions (e.g., "North-ish"). |
| **Landmark Matching** | OSM Tag + **20m** Buffer | A node matches if it has the tag OR is within 20m of a POI with that tag. |
| **Default Radius** | **500m** | Used when no distance is specified in the instruction. |



---

## 🛠️ Implementation: The Diagnostic Object
The `symbolic_solver.py` no longer returns a path. It returns a Classification based on the Candidate Set ($S$):

$$
Score = \begin{cases} \text{Answerable} & \text{if } |S| = 1 \\ \text{Ambiguous} & \text{if } |S| > 1 \\ \text{Contradictory} & \text{if } |S| = 0 \end{cases}
$$

### 🧭 Vector-Based "Away/Toward" Logic
For constraints relative to a landmark (e.g., "Walk away from the park"), we use vector projection:

* **"Away"** (Angle $> 90^\circ$): $\text{dot}(\vec{V}_{candidate}, \vec{V}_{landmark}) < 0$
* **"Toward"** (Angle $< 90^\circ$): $\text{dot}(\vec{V}_{candidate}, \vec{V}_{landmark}) > 0$

*Why: Mathematically cleaner than radius hacks; handles complex map geometry.*



---

## 🟢 Scenario Walkthroughs

### 1. Answerable (Unique Target)
*"Go north from 5th & 42nd for 200m to the library."*
1.  **Step 1:** $S_0$ identifies all 4 corner nodes of the intersection.
2.  **Step 2:** BallTree identifies candidates $(C)$ within $280\text{m}$ of $S_0$ ($\max(220, 200+80)$).
3.  **Step 3:** Filter for nodes at Bearing $0^\circ \pm 45^\circ$ AND `amenity: library`.
4.  **Result:** All paths from $S_0$ converge on Node #101. $|S| = 1$ (Answerable).

### 2. Ambiguous (Multiple Candidates)
*"Walk away from the park for two blocks."*
1.  **Step 1:** Radius set to $300\text{m}$ (estimated 2 blocks).
2.  **Step 2:** **Vector Filter:** Remove nodes where dot product with park vector is $> 0$.
3.  **Step 3:** No landmark filter provided.
4.  **Result:** $|S| = 3$. **Output: Ambiguous.**



### 3. Contradictory (Zero Candidates)
*"Go south from the southern tip of Manhattan for 1km."*
1.  **Step 1:** BallTree looks 1km South.
2.  **Step 2:** Graph check finds only water/empty coordinates.
3.  **Result:** $|S| = 0$. **Output: Contradictory.**

---

## 📈 Research Application: Degradation Analysis
We evaluate model robustness by systematically "breaking" instructions:
* **Full Constraints:** Should be `Answerable`.
* **Landmark Masked:** Does $|S|$ increase? (Measures landmark importance).
* **Distance Masked:** Does the default 500m radius create too much noise/ambiguity?

---

## 📏 Finalized Spatial Constants (Updated Mar 2026)

The team has converged on the following constants for the Manhattan environment:

* **Search Radius ($R$):** Uses the "Human Error Buffer" formula: $\max(D \times 1.1, D + 80\text{m})$.
* **Landmark Proximity ($S_0$):** $20\text{m}$ radius around the starting geocode to handle complex intersections.
* **Node Prefix:** `1#` for projected street nodes (as defined in `config.py`).