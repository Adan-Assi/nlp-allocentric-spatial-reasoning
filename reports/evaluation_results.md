# 📊 Research Summary: Spatial Grounding & Robustness Audit
*Derived from analysis in: [final_paper_evaluation.ipynb](../notebooks/final_paper_evaluation.ipynb)*

## 1. Executive Summary: The "Trifecta of Failure"
The following table summarizes the performance of the LLM across 22,173 instructions.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Grounding Rate** | **~8%** | Only 1 out of 10 instructions prompted a confident coordinate. |
| **Precision Score** | **0.57%** | Of those confident answers, less than 1% actually hit the target. |
| **Logic Alignment** | **8.69% Error** | Rate at which the LLM "hallucinates certainty" in broken/ambiguous worlds. |

---

## 2. Statistical Evidence & Visual Trends

### A. The "1.5km Reasoning Horizon" (Geometric Limit)
LLM error curves across all cities (Manhattan, Pittsburgh, Philadelphia) converge sharply at **1,500 meters**. This represents a physical boundary in the model's weights: it can place an instruction in a general neighborhood but lacks the topological resolution to navigate to a specific coordinate.

> **[INSERT PLOT: CDF Curve - image_80fb40.png]**

> **[INSERT PLOT: Error Distribution Histogram - image_80fb09.png]**

### B. Fragile Robustness (Information Limit)
Analysis shows that removing **Directional cues** is significantly more damaging to success rates than removing **Landmark names**. This suggests the model attempts to follow a "linguistic path" rather than a geographic one.

> **[INSERT PLOT: Robustness Decay (Zoomed) - image_80fb25.png]**
> **[INSERT PLOT: Robustness Decay (0-100% Scale) - image_8106df.png]**

### C. Complexity vs. Accuracy
The **Semantic Drift Plot** confirms that increasing instruction complexity (word count) does not resolve localization errors. The model fails to utilize additional "spatial breadcrumbs" to improve its geodetic precision.

> **[INSERT PLOT: Semantic Drift (Complexity vs Error) - image_80ff61.png]**

---

## 3. Data Integrity & Safety Check
To ensure the **0.57% Precision Score** was a genuine model failure rather than a pipeline error, we verified the following:

1. **Global Consistency (100.00%)**: Every LLM prediction landed within the correct city, confirming correct data alignment.
2. **Local Clustering (~1.4km)**: Errors are clustered in the correct urban area, proving the model identifies the right street but fails the relative logic.
3. **The "Perfect Match" Test**: Valid successful matches (0.0m error) confirm the evaluation logic is **capable of detecting hits**.

---

## 4. Qualitative Audit: Anatomy of a "Local Failure"
A manual review of a decisive "Happy Path" sample reveals how the model prioritizes keyword matching over directional constraints.

* **Instruction**: "Let's get together and meet at the restaurant. It's on **Canal Street**. It's **east of the Verizon Wireless**... A couple of blocks **east of the restaurant is a church**."
* **City**: Manhattan
* **Oracle (Ground Truth)**: `40.7140, -73.9929` (Canal & Division St)
* **LLM Prediction**: `40.7171, -73.9985` (Canal & Broadway)
* **Resulting Error**: **585.02 meters**

### Diagnostic Conclusion: Directional Negligence
**The LLM ignored the directional cues completely.** While the instruction required the target to be **East** of the Verizon store, the LLM "snapped" to an intersection nearly **600 meters to the West**. 

The **Oracle** successfully resolved the nested spatial ordering: **[Verizon (West) < Restaurant (Target) < Church (East)]**. In contrast, the **LLM** exhibited **Keyword Anchoring**; identifying the correct street name but failing to resolve the geometric logic of the sentence.

---

## 5. Logic Alignment & Hallucination
The model fails to distinguish between perfectly valid paths and logically impossible ones.

> **[INSERT PLOT: Logic Alignment Heatmap - image_80fb46.png / image_801a09.png]**

The model provides a "Decisive" coordinate for **Contradictory** instructions at nearly the same rate as it does for **Answerable** ones, proving it relies on pattern matching rather than symbolic reasoning.

---

## 🏛️ Final Conclusion for Research Paper
The LLM acts as a **Semantic Compass**, not a **Geometric Navigator**. While it possesses high "City-Level" awareness, it fails to translate allocentric instructions into precise coordinates. 

Our results demonstrate that while LLMs can act as "Decisive" agents, their grounding is unanchored from symbolic logic. The consistent collapse of precision at the 1.5km mark suggests a fundamental limit in how neural models process spatial instructions without a dedicated symbolic reasoning layer.