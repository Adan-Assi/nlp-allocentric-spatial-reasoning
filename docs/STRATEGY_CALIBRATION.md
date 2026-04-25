# 🧠 Project Strategy: Building & Calibrating the Spatial Brain

We are building a **Symbolic Brain** that understands the "Physics of Navigation." Unlike traditional ML, we are not teaching a model to guess; we are building the logic centers that define how space is reasoned over.

## 🔬 Stage 1: The Manhattan Lab (Calibration)
Manhattan is our "Physics Laboratory." Since the RVS dataset has human-verified success paths here, it serves as our **Ground Truth**.

* **Goal**: Build the hardware (Oracle/Solver) and calibrate the software constants in `config.py`.
* **Optimization**: We tune our distance buffers and landmark matching until our Brain's logic matches the human's "Success" labels. 
* **Result**: A "Calibrated Brain" that perceives the world like a human navigator.



## 🏗️ Stage 2: The Silver Standard (Labeling)
Once calibrated, the Brain processes the entire RVS dataset to generate a "Diagnostic Answer Key."
* **Output**: A `.parquet` file where every instruction is labeled: 
    * ✅ **Answerable**: 1 clear destination.
    * ⚠️ **Ambiguous**: Multiple possible destinations.
    * ❌ **Invalid**: No destination fits the logic.


## 🧪 Stage 3: The "Final Exam" (LLM Robustness)
We use our "Perfect Brain" to judge **LLM Robustness**. 
1.  **Masking**: We strip landmarks or directions from instructions.
2.  **Comparison**: Does the LLM get confused at the same point our Symbolic Brain does? If the Brain says "Ambiguous" but the LLM "guesses" right, we know the LLM is hallucinating patterns rather than reasoning.

## 🌍 Stage 4: Generalization (Cross-City Testing)
Because our logic is universal (physics-based), we can swap the Manhattan map for **Pittsburgh or London**. The Brain doesn't need "training" for new cities; it already knows how to think.