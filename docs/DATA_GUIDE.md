# 🗺️ Manhattan Navigation: Data & Graph Guide

This document serves as the single source of truth for the Manhattan RVS dataset, the underlying street graph, and Point-of-Interest (POI) metadata.

---

## 1. Core Physical Files
These files represent the "Digital Twin" of the environment.

| File | Type | Role | Key Metric |
| :--- | :--- | :--- | :--- |
| **`manhattan_graph.gpickle`** | NetworkX MultiDiGraph | The "Physics Engine" | Weights = Distance in **meters**. |
| **`manhattan_poi.pkl`** | GeoPandas DataFrame | The "World Brain" | 20,979 unique landmarks. |
| **`manhattan_streets.pkl`** | GeoPandas DataFrame | The "Roadmap" | Geometry and street name data. |

---

## 2. Linking Landmarks to the Graph
To navigate to a landmark, the Oracle must link the **POI Metadata** to a **Graph Node**.

* **Node ID Format:** Nodes are stored as strings using the pattern: `[Prefix]#[OSMID]`.
* **The Prefix:**
    * `#123456`: Represents the raw OSM coordinate.
    * `1#123456`: Represents a "Projected Node" where the landmark meets the actual street.
* **Navigation Tip:** Always search for the `1#` (or similar prefix) nodes first, as these are the ones connected to the navigable street edges.



---

## 3. Querying Landmarks (The Oracle's Cheat Sheet)
When the Oracle parses a noun from an instruction (e.g., "Library"), it should query `manhattan_poi.pkl` using these primary keys:

| Landmark Type | Primary OSM Column | Common Values |
| :--- | :--- | :--- |
| **Public Services** | `amenity` | `library`, `post_office`, `university`, `school`, `hospital` |
| **Green Space** | `leisure` / `fountain` | `park`, `playground`, `fountain=yes` |
| **Culture/Sightseeing** | `tourism` | `museum`, `gallery`, `attraction`, `hotel`, `viewpoint` |
| **Commercial** | `shop` / `brand` | `supermarket`, `clothes`, `bakery`, `Starbucks`, `Dunkin` |
| **Transport** | `railway` / `subway` | `station`, `subway_entrance`, `stop` |

* **Fallback Logic:** If a specific tag is missing, the Oracle performs a case-insensitive string search on the `name` column.

---

## 4. The Instruction Dataset
* **File:** `underspecified_instructions.csv`
* **Structure:**
    * `original_instruction`: High-precision human text (e.g., "2 blocks North of Central Park").
    * `underspecified_instruction`: Corrupted text for robustness testing (e.g., "near Central Park").
    * `target_node_id`: The "Ground Truth" node in the `.gpickle` where the instruction ends.
* **Strategy:** The Oracle's success is measured by how close the agent's final node is to this `target_node_id`.

---

## 5. Quick Integration (Python Snippet)
```python
import pickle
import pandas as pd

# Load the "World"
with open('data/manhattan/manhattan_graph.gpickle', 'rb') as f:
    G = pickle.load(f)
poi_df = pd.read_pickle('data/manhattan/manhattan_poi.pkl')

# Example: Find all Museums
museums = poi_df[poi_df['tourism'] == 'museum']
print(f"Found {len(museums)} museums in Manhattan.")