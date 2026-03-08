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