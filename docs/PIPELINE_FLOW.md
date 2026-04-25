# 🌊 System Data Flow: From Text to Label

This document describes how data moves through our system to determine if a navigation instruction is "Answerable."

## 1. Input Stage (Raw Data)
* **Source**: RVS Dataset (Manhattan).
* **Data**: Raw instruction text + Target Latitude/Longitude.
* **Action**: The system extracts the user's intent and the intended destination coordinates.

## 2. Identity Resolution (The Oracle Engine)
* **Input**: Instruction string (e.g., "Meet me at the Starbucks near Bryant Park").
* **Logic**:
    1.  **Normalization**: Cleans text (lowercase, removes punctuation).
    2.  **Lookup**: Queries `LANDMARK_GROUPS` in `config.py` to identify the "Type" (e.g., Coffee Shop).
    3.  **Spatial Matching**: Searches `manhattan_poi.pkl` for the specific name or type.
* **Output**: A set of candidate **Graph Node IDs** (OSMIDs) representing the mentioned landmarks.

## 3. Geometric Reasoning (The Symbolic Solver)
* **Input**: Landmark Node IDs + Agent's current location.
* **Logic**:
    1.  **Pathfinding**: Runs Dijkstra/A* on `manhattan_graph.gpickle` to find the shortest path.
    2.  **Distance Filtering**: Applies the **Clamped Radius** ($\max(D \times 1.1, D + 80)$) to see if the target node is within a "reasonable" walking distance.
    3.  **Vector Analysis**: Checks if the movement direction (N/S/E/W) matches the instruction's cardinal descriptors.
* **Output**: A list of valid candidate nodes that satisfy all spatial constraints.

## 4. Final Labeling (The Decision Layer)
* **Input**: The list of valid candidate nodes.
* **Decision Logic**:
    * **Count == 1**: ✅ **Answerable** (The instruction points to exactly one location).
    * **Count > 1**: ⚠️ **Ambiguous** (Multiple locations fit the description).
    * **Count == 0**: ❌ **Unreachable/Invalid** (The instruction is physically impossible on this graph).
* **Result**: The instruction is saved with its status to `gold_standard_train.parquet`.

## Summary Visualization
`Instruction` ➔ `Oracle (Identity)` ➔ `Solver (Geometry)` ➔ `Diagnostic Label`