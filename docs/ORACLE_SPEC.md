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
* **$R_{search}$**: The dynamic, clamped distance used to define "proximity" to a landmark.
* **$C$**: The initial pool of candidates (The BallTree search radius).
* **$S$**: The final filtered set of valid destinations.

---

## ⚖️ The "Oracle" Laws (Standardized Constants)

| Parameter | Rule/Decision | Justification |
| :--- | :--- | :--- |
| **Starting Set ($S_0$)** | Nodes within **20m** of geocode | OSM can have multiple nodes per intersection; $S_0$ prevents "Starting Point" failure. |
| **Distance Error** | $\max(D \times 1.1, D + 80\text{m})$ | Accounts for human over/under-estimation of **path length** ($1.1$ = 10% buffer). |
| **Landmark Scale** | Area-based + **1.2x** Scale | Defines the **influence zone** (sidewalks/entries) around a POI using Clamped Radius. |
| **Directional Wedge** | $\text{Target} \pm 45^\circ$ | Accounts for "Fuzzy" human directions (e.g., "North-ish"). |
| **Landmark Matching** | OSM Tag + $R_{search}$ | A node matches if it has the required tag AND falls within the landmark's Clamped Radius. |
| **Default Radius** | **500m** | Fallback search distance used when no distance is specified in the instruction. |
---

## 📐 Spatial Search: The Clamped Radius Logic

In allocentric navigation, "at the landmark" is a subjective distance that scales with the landmark's physical size. The Oracle uses a **Clamped Radius** to model this human perception.

### 1. Why "Clamped"?
A simple fixed radius (e.g., always 50m) fails to capture human intent at the extremes:
* **Small Landmarks (e.g., Mailbox/Pole):** A 50m radius is too loose. The agent could be across the street, and the Oracle would incorrectly mark it as "at the mailbox." These require a tight radius ($r \approx 15\text{m}$).
* **Large Landmarks (e.g., Bryant Park/Hospital):** A 50m radius measured from a center point might not even reach the sidewalk. Humans consider themselves "at" the park the moment they hit the perimeter. These require a wide radius ($r > 100\text{m}$).



### 2. The Formula
To balance precision (not grabbing too many nodes) and recall (not missing the landmark), the search radius ($R_{search}$) is calculated as:

$$R_{search} = \min\left(\max\left(R_{min}, \sqrt{\frac{\text{Area}}{\pi}} \cdot \text{Scale}\right), R_{max}\right)$$

* **Scaling:** We multiply the physical footprint by a `RADIUS_SCALE_FACTOR` (default 1.2x) to account for the "influence zone" (the sidewalk/street immediately adjacent).
* **Lower Bound ($R_{min}$):** Prevents the search area from disappearing for point-nodes (e.g., "at the lamp post").
* **Upper Bound ($R_{max}$):** Prevents massive landmarks from "swallowing" the graph and creating false positives in our candidate set $S$.

### 3. Usage
This logic is used during **Landmark Matching** to determine if a graph node qualifies as being "at" a specific POI. If the distance from Node $N$ to POI $P$ is $\le R_{search}$, the node is added to the candidate pool.

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