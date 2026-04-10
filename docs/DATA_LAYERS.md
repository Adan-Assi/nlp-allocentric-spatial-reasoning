# 🗺️ Manhattan Spatial Intelligence Layers

This document outlines the four distinct data layers provided by the `manhattan_poi.pkl` dataset (Reminder: POI = `Points of Interest`). By utilizing these 40+ high-density columns, the Oracle can move beyond simple "Point A to B" navigation and perform "Intent-Based" reasoning.

---

## 1. The Commercial & Retail Layer (Intent)
*This layer identifies **what** a place does and **who** owns it. Use this for specific brand or service requests.*

| Column | Density (Count) | Logic Application |
| :--- | :--- | :--- |
| `name` | 14,076 | Primary identifier for specific landmarks (e.g., "Joe's Pizza"). |
| `shop` | 3,539 | Categorizes retail stores (e.g., `bakery`, `clothes`, `supermarket`). |
| `brand` | 2,333 | Enables corporate-specific searches (e.g., `Starbucks`, `CVS`, `Chase`). |
| `cuisine` | 2,673 | Maps food cravings to specific POIs (e.g., `sushi`, `dim_sum`, `pizza`). |
| `takeaway` | 1,098 | Distinguishes sit-down dining from quick-service spots. |

---

## 2. The Infrastructure & Urban Context Layer
*This layer describes the **physical characteristics** of the environment. Use this for accessibility or visual descriptions.*

| Column | Density (Count) | Logic Application |
| :--- | :--- | :--- |
| `building` | 2,950 | Identifies structural types (e.g., `apartments`, `train_station`). |
| `height` | 2,231 | Used for "Look for the tall building" style navigation instructions. |
| `wheelchair`| 1,406 | Enables accessibility-aware routing (checking for `yes`/`no`). |
| `cityracks.*`| ~1,915 | A specialized NYC dataset for finding bicycle parking and rack sizes. |
| `leisure` | 1,411 | Identifies parks, playgrounds, and recreational spaces. |

---

## 3. The Digital & Connectivity Layer (Metadata)
*This layer provides **out-of-graph** information. Use this for data enrichment or verifying status.*

| Column | Density (Count) | Logic Application |
| :--- | :--- | :--- |
| `website` | 5,242 | Links to external menus, hours, or official information. |
| `phone` | 4,325 | Provides contact info for "ground-truthing" a landmark's existence. |
| `opening_hours`| 4,072 | Allows the Oracle to check if a landmark is "active" at the current time. |
| `wikidata` | 1,796 | Connects the POI to global knowledge graphs for historical context. |
| `wikipedia` | 1,015 | Direct link to long-form descriptions of famous Manhattan landmarks. |

---

## 4. The Geocoding & Address Layer (Grounding)
*This layer provides the **mathematical anchor** to the street network. Use this when the Name is unknown.*

| Column | Density (Count) | Logic Application |
| :--- | :--- | :--- |
| `addr:street` | 9,095 | Allows filtering by street name (e.g., "Find a cafe on Broadway"). |
| `addr:housenumber`| 8,999 | Provides exact coordinate grounding for house-level precision. |
| `addr:postcode`| 7,093 | Useful for broad-area filtering (Neighborhood/Zip Code level). |
| `geometry` | 20,979 | The raw Point/Polygon shape used for all distance calculations. |
| `centroid` | 20,979 | The mathematical center of the geometry (essential for Polygon math). |

---

## ⚙️ Recommended Usage in OracleEngine
To maximize accuracy, the `resolve_by_tags` method should prioritize these layers in the following order:
1. **Name Match** (`name`)
2. **Brand Match** (`brand`)
3. **Specific Category** (`cuisine`, `shop`, `amenity`)
4. **General Tags** (`building`, `leisure`, `office`)