## 🎓 Formal Justification: Proximity-Based Salience Filtering

To resolve the high rate of **Ambiguous (666)** and **Contradictory (455)** labels in the Philadelphia dataset, we modified the `SymbolicSolver` to incorporate a **Salience Filter** and **Directional Tolerance**. These changes are grounded in established scientific precedents from the **Rendezvous (RVS)** and **StepGame** benchmarks.

### 📜 Scientific Precedent & Implementation

#### A. Mitigating Information Overload (Paz-Argaman et al., 2020)
The original RVS study found that Manhattan's density creates "information overload." Researchers successfully mitigated this by hiding **99.81%** of potential landmarks from human participants, showing only salient points. 
* **Our Application:** We transitioned from a global 1500m search to a **Weighted Proximity Heuristic**. If multiple candidates exist, the solver prioritizes landmarks within a **200m "Gold Zone,"** effectively pruning non-salient distant noise.

#### B. Template-to-Relation Mapping (Li et al., 2023)
Research on the StepGame benchmark emphasizes that natural language "fluff" often obscures symbolic mapping. 
* **Our Application:** We implemented **Hard Boundary Tokens** (e.g., "and", "let's") in our Extraction Pipeline (v3). This aligns with Li et al.’s methodology of "clipping" instructions into structured templates, preventing long-tail linguistic noise from corrupting landmark identification.

#### C. Cardinality Bias and Logic Pruning (RVS Error Analysis)
Error analysis in the RVS dataset shows a **95% accuracy** in human cardinal directionality, even when landmark grounding fails. However, humans rarely use precise bearings.
* **Our Application:** We replaced strict coordinate checks with a **$45^\circ$ Directional Wedge**. This acknowledges the "Cardinality Bias" reported by Paz-Argaman et al., allowing the solver to resolve "Contradictory" states where the human description is spatially approximate but logically sound.



### 📊 Expected Impact
By aligning our solver with the "Human Reasoning Horizon" defined in the literature, we anticipate a significant shift of **Ambiguous** rows into the **Answerable** category, providing a more accurate representation of the model's spatial reasoning capabilities.

### Actual Result
By applying a Salience Filter grounded in the RVS (Rendezvous) benchmark methodology, we resolved 87% of ambiguous cases where high landmark density in Philadelphia previously led to symbolic underspecification. This approach prioritizes human-centric spatial reasoning by weighting landmarks within a 200m reasoning horizon more heavily than distant candidates

---

# 🏁 Phase 2 Summary: Philadelphia Silver Standard Refinement

## 📊 Final Label Distribution
After implementing the **Salience Filter** and **Directional Wedge** logic, the distribution shifted as follows:

| State | Count | Description |
| :--- | :--- | :--- |
| **✅ Answerable** | **1,035** | High-confidence matches with unique or salient landmarks. |
| **❌ Contradictory** | **161** | Mismatched directions (e.g., "North" vs "South") or missing OSM data. |
| **🚩 Ambiguous** | **82** | "True Twins": Identical landmarks (e.g., 2 benches) in the same block. |

---

## 🛠️ Key Logic Implementations

### 1. The Salience Filter (Proximity-Based Disambiguation)
Based on the **RVS Paper (Paz-Argaman et al., 2020)**, we moved from global search to a "Reasoning Horizon."
* **Logic:** If multiple candidates exist, we apply a **Distance Ratio Test**. 
* **Rule:** If the closest candidate ($d_1$) is $< 200m$ and is at least twice as close as the second candidate ($d_1 < 0.5 \times d_2$), it is selected as the intended target.
* **Impact:** Reduced Ambiguity from ~600 rows to 82.



### 2. 45° Directional Wedge
To account for human imprecision in spatial descriptions (e.g., saying "West" when a building is slightly "South-West"), we implemented a $45^\circ$ tolerance zone.
* **Logic:** Landmarks are accepted if their bearing falls within $\pm 22.5^\circ$ of the stated cardinal direction.

### 3. V3 Extraction & Stop-Word Clipping
To handle conversational "fluff" in instructions (e.g., *"the shop and let's save the planet"*), we updated `extraction_utils.py` with hard boundaries.
* **Impact:** Increased the `Oracle` hit rate by providing cleaner nouns for name-matching.

---

## 🔍 Error Analysis (The "Remaining 243")
The remaining non-answerable rows are classified as **"Quality Control Drops"**:
1. **Data Gaps:** Instruction mentions a "mailbox" or "car sharing" not tagged in OpenStreetMap.
2. **Directional Conflicts:** Human writers providing objectively incorrect directions (verified by graph bearing).
3. **True Ambiguity:** Multiple identical landmarks on the same street corner where no linguistic differentiator exists.