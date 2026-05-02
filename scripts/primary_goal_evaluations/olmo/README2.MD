# OLMo Primary Goal Evaluations: In-Context Learning (ICL) Setup

This README documents the evaluation setup for the OLMo model used in:

`scripts/primary_goal_evaluations/`

Model used:

```python
model_name: str = "allenai/OLMo-2-0425-1B-Instruct"
````

---

## Evaluation Paradigm: In-Context Learning (ICL)

What we are doing is **in-context learning (ICL)**: we provide labeled examples directly inside the prompt, and the model generalizes from them at inference time without any parameter updates.

No fine-tuning, no gradient updates, and no prompt tuning are involved.

This matches the lecture definition:

> “Showing the model how to solve a task through example, requiring only inference.”

This distinction is important because our evaluation is strictly inference-only.

---

## Current OLMo Evaluation Variants

We evaluate OLMo under several ICL configurations:

### 4-shot ICL (Current Setup)

Four labeled examples are included inside the prompt before the test instance.

This is our main evaluation setting.

Purpose:

* provide demonstrations of the task format
* allow the model to infer the decision rule from examples
* simulate practical few-example prompting without training

---

### Zero-shot (Proposed Ablation)

No examples are included.

Only:

* task definition
* label definitions
* prompt instructions

Purpose:

* test how much performance depends on demonstrations
* measure the contribution of examples themselves

---

### Shuffled-Order (Proposed Abllation)

Same four examples as 4-shot ICL, but their order is changed.

Purpose:

* test sensitivity to example ordering
* verify whether prompt order affects model behavior

This is still ICL because the demonstrations remain inside the prompt.

---

## Why This Matters for the Paper

This gives us cleaner and more academically correct terminology.

We should:

### Instead of:

* “few-shot prompting”

write:

* **4-shot in-context learning**

---

### Instead of:

* “we prompted the model”

write:

* **we evaluated models under in-context learning with k=0 and k=4 demonstrations**

---

### Instead of:

* “prompt ablation”

write:

* **ablation of demonstration count and order under ICL**

This framing is stronger and aligns directly with lecture material.

---

## Methodology Section Wording

A clean version for the paper:

> We evaluate all models under in-context learning (ICL), requiring only inference access. No model parameters are updated. We use k=4 demonstrations drawn from oracle-verified examples, with label definitions provided in the prompt preamble.

This should be stated explicitly in the methodology section and connected to the distinction between:

* ICL
* fine-tuning
* prompt tuning

as discussed in the final slides of Lecture 6 (LLMs).

---

## Note for Final Report

The paper should explicitly state that:

**Prompt ablations are ICL ablations**, not training procedures.

This improves:

* lecture alignment
* terminology precision
* methodological clarity
* academic correctness
* reviewer confidence
