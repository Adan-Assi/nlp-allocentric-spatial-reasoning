What we are doing is **in-context learning (ICL)**: we provide labeled examples directly in the prompt and the model generalizes from them at inference time without any parameter updates.

Our olmo evaluations use:
- **4-shot ICL** — four labeled examples in the prompt (current setup)
- **Zero-shot** — no examples, only task definition (proposed ablation)
- **Shuffled-order** — same examples, different order (proposed ablation)

All three are ICL variants. None involve gradient updates. This maps directly to the lecture's definition: "showing the model how to solve a task through example, requires only inference."

---

## Why this matters for the paper

This gives us precise academic terminology to use throughout. We:

- Do not write "few-shot prompting" loosely. Instead we write: "4-shot in-context learning"
- Do not write "we prompted the model". Instead we write: "we evaluated models under in-context learning with k=0 and k=4 demonstrations"
- The prompt ablation is properly framed as "ablation of demonstration count and order under ICL"

Note for self: the methodology section should explicitly cite ICL as the evaluation paradigm and distinguish it from fine-tuning and prompt tuning. The last slides in lecture 6 (LLMs) make exactly this distinction.

---

## For the paper's methodology section

One clean framing:

> "We evaluate all models under in-context learning (ICL), requiring only inference access. No model parameters are updated. We use k=4 demonstrations drawn from oracle-verified examples, with label definitions provided in the prompt preamble."