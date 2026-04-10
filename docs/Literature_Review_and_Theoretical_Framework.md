# 📚 Literature Review & Theoretical Framework
**Project:** Robust Allocentric Spatial Reasoning in LLMs under Underspecified Map Instructions  
**Goal:** This document synthesizes the core findings of RVS and StepGame to justify our Symbolic Solver and Geodesic Pruning strategy.

---

## 1. Core Research Foundations

### A. The Ground-Truth Environment (Paz-Argaman et al., 2024)
* **Key Contribution:** Established the **Rendezvous (RVS)** dataset, providing human-written allocentric instructions grounded in OpenStreetMap (OSM) graphs.
* **The "2km Problem":** Authors identified that standard T5 models fail to reason beyond a **2000m (2km)** range, often defaulting to locations near the starting point.
* **Project Alignment:** Our **1500m Geodesic Gatekeeper** is theoretically supported by this "Limited Range" observation. We optimize for the high-confidence human reasoning zone while using the 2km limit as a "Recall Ceiling."

### B. Multi-Hop Robustness (Shi et al., 2022)
* **Key Contribution:** Introduced **StepGame**, testing if models maintain accuracy as "reasoning hops" increase.
* **Project Alignment:** While Shi et al. focused on adding "Noise," our project focuses on **Underspecification** (removing signal). Their work provides the precedent for measuring *Robustness* rather than just *Accuracy*.

### C. LLM Limitations & Template Accuracy (Li et al., 2024)
* **Key Contribution:** An evaluation showing LLMs can map text to relations (e.g., "North" = "Up") but fail at the **Logic-Based Reasoning** required to chain them.
* **Project Alignment:** This justifies our **Symbolic Graph-Based Solver**. Since LLMs cannot reliably "chain" spatial nodes, we use the graph solver as the absolute "Ground Truth."

---

## 2. Empirical Error Taxonomy (From Literature)

### **Part A: RVS Baseline Errors (Paz-Argaman et al., 2024)**
*Analysis of T5+Graph model in the Manhattan (Seen-City) split.*

| Error Type | % in Split | Scientific Description |
| :--- | :--- | :--- |
| **Cardinal Direction Bias** | **95%** | Model learns global layout (e.g., "North") but fails on specific node resolution. |
| **Entity Type Matching** | **50%** | Model grounds the category (e.g., 'bank') but picks the wrong instance. |
| **Street-Level Alignment** | **45%** | Predicted goal is on the correct street, usually when the street is named in text. |
| **S2-Cell Granularity** | **25%** | Failure due to map discretization; model finds the neighborhood but not the node. |

### **Part B: StepGame Template Errors (Li et al., 2024)**
*Failures in original benchmarks that our Symbolic Solver is designed to resolve.*

| Error Category | Impact on Reasoning | Example Failure |
| :--- | :--- | :--- |
| **Irreparable Failures** | Zero information provided. | "Object A is above $o1$" (omitting the target $o2$). |
| **Self-Reference** | Circular logic. | "$o1$ is diagonally left and above $o1$." |
| **Mapping Inconsistency** | Incorrect spatial anchors. | Using "10 o'clock" to label a "Lower-Left" relation. |
| **Ambiguous Multi-Mapping**| Label noise. | One text string mapped to 3 different ground-truth labels. |

---

## 3. Theoretical Justification for Implementation

### **I. Geodesic Pruning (The 1500m Threshold)**
* **Scientific Basis:** RVS Appendix C shows "Signal-to-Noise" drops sharply as distance approaches 2km.
* **Implementation:** We use R=1500m as a **Cognitive Filter** that mimics the "Observable Horizon" of the human participants who wrote the RVS instructions.

### **II. Answerability Classification**
Based on the Solver, we categorize every instruction as:
1.  **Answerable:** 1 unique node satisfies all constraints.
2.  **Ambiguous:** $>1$ nodes satisfy constraints (common in our masking experiments).
3.  **Contradictory:** 0 nodes satisfy constraints.

### **III. Hypotheses for Underspecification**
1.  **The "Hallucination" Check:** When masking a landmark name, will the model exhibit the **Cardinal Direction Bias (95%)**? We predict the model will pick a node in the correct direction despite having zero textual evidence for it.
2.  **Ambiguity Sensitivity:** If the solver says an instruction is **Ambiguous**, and the LLM confidently picks a single node, we have documented a **Robustness Failure**.
3.  **Multi-Hop Degradation:** Following **Li et al.**, we expect error rates to explode as masking increases the number of "logic hops" the model must fill in (7.6% at 1-hop $\rightarrow$ 54.3% at 10-hops).

---
**References:**
- Paz-Argaman, T., et al. (2024). *Rendezvous.* EACL.
- Shi, Z., et al. (2022). *StepGame.* AAAI.
- Li, F., et al. (2024). *Advancing Spatial Reasoning.* AAAI.