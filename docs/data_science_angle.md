# 🧪 RVS Pipeline: Data Science Perspective

This document maps the project's spatial reasoning workflow to standard Data Science (DS) and Machine Learning (ML) terminology.

| Workflow Stage | Description & Process | Data Science Equivalent |
|:---|:---|:---|
| **Data Ingestion & Cleaning** | Normalizing raw RVS text, fixing coordinate formats, and canonicalizing city names. | **Data Preprocessing & Normalization** |
| **City Scaling** | Applying different success radii (80m for MHT, 100m for PIT/PHL) based on city density. | **Context-Aware Thresholding / Hyperparameter Tuning** |
| **Instruction Parsing** | Breaking instructions into landmarks, cardinal directions, and distances. | **Feature Extraction / Entity Recognition (NER)** |
| **Node Snapping** | Using a KDTree to map raw coordinates to the nearest graph nodes. | **Spatial Indexing & Grounding** |
| **Symbolic Solving** | Using a geometric oracle (wedges/circles) to find the "mathematically certain" goal. | **Heuristic-Based Programmatic Labeling** |
| **Ambiguity Detection** | Categorizing instructions as "Answerable," "Ambiguous," or "Contradictory" based on solver results. | **Classification / Categorical Labeling** |
| **Information Degradation** | Systematically removing landmarks or directions from the input (Masking). | **Adversarial Perturbation / Ablation Setup** |
| **Experimental Variants** | Creating 22k+ variants with different levels of missing info. | **Synthetic Data Augmentation** |
| **LLM Inference** | Testing the LLM's performance on full vs. masked data without fine-tuning. | **Zero-Shot Probing & Robustness Testing** |
| **Drift Measurement** | Calculating the distance between predicted and symbolic goal coordinates. | **Error Variance / Residual Analysis** |
| **Density Visualization** | Using KDE plots to show where the model "guesses" vs. "knows." | **Kernel Density Estimation (KDE) Analysis** |
---

## 🔬 Core Methodologies Used

### 1. Programmatic Labeling (The Symbolic Oracle)
Instead of manual human annotation, we utilize a **Symbolic Solver** as a ground-truth oracle. This follows the principles of **Snorkel-style weak supervision**, where rules generate labels that are then used to evaluate complex models.


### 2. Ablation Studies via Masking
By masking specific columns (e.g., `masked_landmark`), we perform a **Feature Importance Study**. This identifies which "signals" (Spatial Relations vs. Points of Interest) are the primary drivers of the LLM’s spatial reasoning capabilities.

### 3. Spatial Drift Analysis
The "success" of a model isn't just binary (Correct/Incorrect). We treat the output as a **Continuous Error Variable**, measuring the spatial drift from the ground truth to quantify the model's **Resolution Limit**.


### 4. High-Dimensional Spatial Lookups
The use of **KDTree** indexing allows the pipeline to scale. We perform $O(\log n)$ nearest-neighbor searches across thousands of graph nodes, ensuring the feature grounding phase is computationally efficient.

---

## Diagram

<pre>
+-------------------------------------------------------+
|                 RVS Data Science Pipeline             |
+-------------------------------------------------------+
                                       
+-------------------------------------------------------+
|                 RVS Data Science Pipeline             |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  1. Preprocessing: Fuzzy Matching & Normalization     |
|     [ Levenshtein T=80 | Coordinate Canonicalization ] |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  2. Spatial Grounding: WGS-84 & Haversine Pruning     |
|     [ 1500m Observable Horizon | KDTree Snapping ]    |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  3. Topological Logic: SCC Reachability Check         |
|     [ Ensuring O(1) Path Verification ]               |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  4. Symbolic Oracle: Heuristic Labeling               |
|     [ Generating Silver/Gold Standard Goals ]         |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  5. Adversarial Ablation: Semantic Masking            |
|     [ Creating 22k Underspecified Variants ]          |
+-------------------------------------------------------+
                             |
                             V
+-------------------------------------------------------+
|  6. Evaluation: KDE & Spatial Drift Analysis          |
|     [ Measuring Resolution Limits & Decay ]           |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                 Final Evaluation Reports              |
+-------------------------------------------------------+
</pre>