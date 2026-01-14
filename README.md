# Robust Allocentric Spatial Reasoning under Underspecified Instructions

Course project (NLP): evaluating how robust large language models (LLMs) are at **allocentric spatial reasoning** when navigation instructions are **systematically underspecified**, while the underlying world (urban map graph) remains fixed and well-defined.

---

## 🔹 Guiding Principle

**De-risk early.**  
Prove feasibility quickly, freeze infrastructure, then focus on **analysis**, not engineering.

---

## 🧠 Project Overview

LLMs often perform well on clean, fully specified spatial instructions.  
However, real-world spatial language is frequently incomplete and relies on implicit shared knowledge.

This project studies whether LLMs **truly reason over spatial structure** or rely on surface-level cues by:

- grounding navigation instructions in **real urban map graphs** (RVS dataset),
- systematically **removing or weakening spatial information** in the instructions,
- comparing LLM outputs against **graph-derived ground truth**.

---

## 🗺️ Dataset

We use the **Rendezvous (RVS)** dataset:

- Human-written allocentric navigation instructions
- Grounded in real-world urban map graphs
- Clear, graph-backed ground truth (reachability, paths, directions)

The dataset allows us to modify instructions while keeping the environment fixed and answerable.

---

## 🔧 High-Level Pipeline

1. **Instruction** (original or underspecified)
2. **LLM inference** (small, open-weight models)
3. **LLM answer**
4. **Graph-based solver**
   - reachability
   - shortest path
   - coarse direction checks
5. **Comparison & error analysis**

The symbolic solver is intentionally minimal and does *not* perform language understanding.

---

## 🧩 Symbolic Solver Scope

The solver **does**:
- compute reachability
- compute shortest paths
- check coarse relative direction (N / S / E / W)

The solver **does not**:
- parse natural language
- perform logical inference
- resolve ambiguity
- go beyond graph queries

This scope is fixed early to avoid engineering creep.

---

## 🧠 Instruction Underspecification

We define a small number of systematic underspecification strategies, such as:

- removing spatial relations,
- replacing exact directions with vague ones,
- dropping intermediate landmarks.

These are applied to a controlled subset of instructions while preserving answerability via the graph.

---

## 📊 Evaluation Strategy

- LLM answers are compared against **graph-derived ground truth**.
- Outputs are labeled as:
  - correct
  - incorrect
  - inconsistent / hallucinated

We focus on **qualitative error patterns** and robustness trends rather than leaderboard performance.

---

## 🗓️ Project Plan (High-Level)

**Week 1 — Feasibility & Presentation Readiness**
- Load RVS graph and run basic queries
- Freeze solver scope
- Define underspecification strategies
- Prepare preliminary presentation

**Week 2 — Minimal End-to-End Pipeline**
- Run LLM inference (small models)
- Compare LLM outputs to graph answers
- Document failure modes

**Week 3 — Analysis & Scaling Plan**
- Define error taxonomy
- Plan larger-scale experiments and comparisons

---

## 👥 Team

- Adan Assi  
- Shaimaa Hoji
- Noor Mhajne

(Tel Aviv University – NLP course project)

---

## 📌 Notes

- The project prioritizes **clarity and feasibility** over model scale.
- We start with sub-1B parameter models and scale only if justified.
- All experiments are designed to be reproducible and graph-grounded.

---
