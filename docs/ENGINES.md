This document outlines the sequential processing engines used to evaluate LLM robustness in allocentric spatial reasoning under systematic underspecification.

---

## 1. The Perturbation Engine (`underspecify_instructions.py`)
* **Input:** Raw RVS Human-written instructions (Ground Truth).
* **Action:** Applies rule-based logic to identify spatial-relation phrases, landmarks, and cardinal directions. It replaces these elements with `[MASK]` or `[DIR_MASK]` tokens.
* **Output:** `underspecified_variants.json` containing multiple versions of each instruction (e.g., `mask_near`, `mask_directions`, `mask_both`).
* **Role:** Creates the "experimental conditions" by removing information while the underlying map graph remains fixed.

---

## 2. The Extraction Engine (`extract_rvs_target`)
* **Input:** Masked text strings from the Perturbation Engine.
* **Action:** The first stage of the symbolic solve process. It parses the string to identify the broad **Category** (e.g., `SHOP`) and the **Specific Noun** (e.g., `coffee shop`).
* **Role:** Acts as the NLP-to-Symbolic translator, converting human language into structured queries for the map.

---

## 3. The Oracle Engine (`OracleEngine`)
* **Input:** Extracted Categories/Nouns + City Map Graph ($G$).
* **Action:** Performs "Semantic Grounding" by querying the spatial database/POI (Point of Interest) layer. It identifies all physical nodes on the map that match the extracted description within a defined search radius.
* **Output:** A list of candidate nodes and their spatial coordinates.

---

## 4. The Symbolic Solver (`SymbolicSolver`)
* **Input:** Candidate list from the Oracle + Reachability logic.
* **Action:** The **Master Controller**. It evaluates the "Answerability" of the instruction based on the number of valid candidates found:
    * **Answerable:** Exactly 1 unique node satisfies all constraints.
    * **Ambiguous:** $>1$ nodes satisfy the constraints (due to masking).
    * **Contradictory:** 0 nodes satisfy the constraints.
* **Output:** The "Silver Standard" ground truth label for the perturbed instruction.

---

## 5. The Evaluation Engine (1B-Parameter LLMs)
* **Input:** The same masked text used in Engine #2.
* **Action:** The target LLM (e.g., T5-base, Pythia) attempts to predict the goal coordinate or classify the answerability.
* **Role:** The research subject. Its performance is compared against the **Symbolic Solver's** output to characterize error patterns, such as over-confidence on ambiguous inputs.

---

### Summary of Data Flow
`Raw Data` $\rightarrow$ **Perturbation** $\rightarrow$ **Extraction** $\rightarrow$ **Oracle** $\rightarrow$ **Solver (Truth)** $\rightarrow$ **Comparison with LLM**