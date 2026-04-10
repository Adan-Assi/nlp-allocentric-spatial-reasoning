# Source Code (Core Logic)

This directory contains the core engines for spatial reasoning and landmark resolution.

## 🏗️ Core Components

### 1. `oracle_engine.py` (The Knowledge Base)
The `OracleEngine` acts as the bridge between Natural Language strings and Graph Theory IDs.
- **Landmark Resolution**: Uses regex normalization and fuzzy matching to map strings (e.g., "Hell's Kitchen") to OpenStreetMap IDs (OSMIDs).
- **Data Handling**: Efficiently loads large-scale spatial data (~200MB Pickles) and Manhattan street networks.

### 2. `symbolic_solver.py` (The Navigator)
The `SymbolicSolver` performs the heavy mathematical lifting on the graph.
- **Pathfinding**: Implements Dijkstra’s algorithm via NetworkX to find the shortest path between any two nodes.
- **Spatial Bearings**: Calculates cardinal directions (N, S, E, W) between nodes using geodesic coordinates.
- **Landmark Navigation**: Combines with the Oracle to compute paths to named locations.

### 3. `utils.py`
Contains shared helper functions for coordinate transformation, distance calculation (Haversine/Geodesic), and string cleaning.

## ⚙️ Requirements
- **NetworkX 3.0+**: Essential for modern graph processing.
- **Pandas**: Used for high-speed POI (Point of Interest) lookups.
- **Geopy**: Used for accurate Earth-surface distance and bearing calculations.

## 🛠️ Design Principle: "The Separation of Concerns"
The **Oracle** knows *where* things are (Identity), while the **Solver** knows *how to get there* (Geometry). This allows us to swap out the Manhattan map for another city without changing the Solver logic.


## 🧪 Case Study: The "Final Logic Flow" Verification
To verify the integration between the **Solver** and the **Oracle**, consider the command: *"Go to the Church."*

### 1. NLP Extraction (Solver)
- **Input:** `"Go to the Church"`
- **Action:** The Solver identifies the keyword **"CHURCH"** and matches it against `config.LANDMARK_GROUPS`.
- **Result:** Maps to `{ 'amenity': 'place_of_worship' }`.

### 2. Spatial Grounding (Oracle)
- **Action:** The Solver calls `oracle.resolve_by_tags(current_pos, tags)`.
- **Logic:** 
    1. The Oracle filters the 1,033 columns in `manhattan_poi.pkl` for `amenity == 'place_of_worship'`.
    2. It calculates the Euclidean distance from the agent's current `(x, y)` to all matching POI `geometry` points.
    3. It identifies the nearest match (e.g., `osmid: #666`).
- **Transformation:** The Oracle cleans the ID and applies the project prefix to return: **`1#666`**.

### 3. Path Computation (Solver)
- **Action:** The Solver verifies `1#666` exists in the Manhattan Graph (`self.G`).
- **Result:** It executes `nx.shortest_path(self.G, current_node, "1#666", weight='length')`.

### 4. Output
- **Final Result:** A list of Node IDs representing the exact street-level path to the landmark.