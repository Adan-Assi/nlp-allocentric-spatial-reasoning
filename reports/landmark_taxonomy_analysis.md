# 🏷️ Technical Report: Landmark Taxonomy & Semantic Alignment

## 1. Executive Summary
During the batch labeling of the Manhattan dataset (7,000 samples), we identified a "Semantic Gap" between human instructions and OpenStreetMap (OSM) data structures. This report documents the "Symbolic Normalization Layer" developed to bridge this gap, resulting in a **92.53% production accuracy** and the recovery of 310 samples previously lost to noise.

---

## 2. Identified Failure Patterns & Taxonomy Optimization

### 2.1 Multimodal "Parking" Collision
* **Pattern:** Users refer to "bike parking," but engines prioritize "Car Parking" (the OSM `amenity=parking` default).
* **Resolution:** Expanded `LANDMARK_GROUPS["PARKING"]` to include `bicycle_parking` and `motorcycle_parking`.

### 2.2 Fuzzy Symbolic Normalization (The $T=80$ Rule)
We implemented a normalization layer using Levenshtein Distance to handle human orthographic noise without requiring LLM re-training.
* **Correction Example:** `musuem` (Score: 83) → `MUSEUM` (✅ Corrected).
* **Safety Buffer:** `starbucks` (Score: 60) → `REJECTED`. This ensures brand-specific entities are passed through for high-precision string matching rather than being forced into a general category.

---

## 3. Optimized Spatial Grounding & Scalability (V4/V5)

The transition to a **Proximity-Aware Oracle** addressed "Global Ambiguity" by anchoring searches to the agent's local horizon.

### 3.1 The Geodesic Gatekeeper (R=1500m)
Based on Landmark Recall Sensitivity Analysis, we identified a 1500m "Human Observable Horizon." To maintain throughput, we implemented a **Dual-Precision Pruning Pipeline**:

1.  **Vectorized Coarse Filter:** Uses a curvature-aware bounding box and the **Haversine formula** to prune city-wide POIs ($50\times$ speedup).
2.  **WGS-84 Precision Verification:** Uses the **WGS-84 Ellipsoid model** (via `geopy.geodesic`) for final 1500m verification.
3.  **Curvature Correction:** Longitude is adjusted dynamically per city:
    $$\Delta_{lat} = \frac{1500}{111000}, \quad \Delta_{lon} = \frac{1500}{111000 \cdot \cos(\text{lat})}$$

### 3.2 O(1) Reachability via Strongly Connected Components (SCC)
To resolve the "Manhattan Freeze" (where graph traversal caused exponential latency), we introduced a pre-computed topological lookup. Reachability is now a simple parity check: 
$$\text{is\_reachable} = (\text{SCC\_ID}_{\text{start}} == \text{SCC\_ID}_{\text{target}})$$

---

## 4. Empirical Scalability Results
The optimizations allowed the system to handle the massive Philadelphia graph with zero timeouts.

| City | Graph Nodes | Baseline Latency (V1) | Optimized Latency (V5) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Pittsburgh** | ~3,500 | 120ms / sample | 25ms / sample | 4.8x |
| **Manhattan** | ~7,000 | 450ms / sample | 38ms / sample | 11.8x |
| **Philadelphia** | **~14,000** | **1,200ms (Timeout)** | **52ms / sample** | **23.0x** |
