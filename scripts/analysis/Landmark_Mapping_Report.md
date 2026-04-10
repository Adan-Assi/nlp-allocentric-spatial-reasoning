# 🏺 Rosetta Stone: Landmark Mapping Report (Comprehensive)
**Date:** March 2026

**Project Phase:** Task 2.5 (Vocabulary) → Task 3.1 (Parsing)

**Verification Status:** 100.00% Instruction Coverage Confirmed

---

## 1. Executive Summary: The "90% Goal" Challenge
In this project, we were tasked with ensuring that our Symbolic Solver can understand the landmarks mentioned in 7,000+ Manhattan navigation instructions. 

Initially, we faced a **Discovery Gap**:
* **The Problem:** Manhattan has thousands of unique landmark names (e.g., "The Red Bench," "60 Hudson Street").
* **The Failure:** If we only look for exact name matches, we only cover **24.83%** of the data. This would cause the agent to get "lost" in 75% of instructions.
* **The Solution:** We moved to **Root Keyword Mapping**. By identifying the core "Root" of a landmark (e.g., "Church" instead of "St. Paul's Episcopal Church"), we reached **100% Coverage**. Every single instruction now has at least one anchor point the Oracle can find on the map.

---

## 2. Coverage Analysis Results
We performed two distinct passes over the data to determine the most efficient way to build the `config.py` vocabulary.

### Pass 1: Exact String Analysis (The "Instance" Level)
This pass counted exact matches for specific landmark strings.
| Rank | Landmark Category | Count | Cumulative % |
| :--- | :--- | :--- | :--- |
| 1 | CHURCH | 3228 | 1.65% |
| 12 | POST OFFICE | 1477 | 12.75% |
| 19 | BENCH | 1047 | 17.35% |
| 26 | 7-ELEVEN | 733 | 20.30% |
| **Total** | **Top 40 Strings** | **--** | **24.83%** |

### Pass 2: Root Category Analysis (The "Logic" Level)
This pass used a script to extract the "Root Noun" from every landmark name.
| Rank | Root Category | Count | Cumulative % |
| :--- | :--- | :--- | :--- |
| 1 | BUILDING | 9235 | 4.71% |
| 2 | CHURCH | 8903 | 9.25% |
| 8 | BENCH | 3470 | 23.58% |
| 13 | STREET | 2904 | 31.71% |
| **Total** | **Top 40 Roots** | **--** | **56.48% (Raw Instances)** |

**Why is 56% better than 24%?**
While only 56% of *individual objects* are covered by these 40 roots, **100% of instructions** contain at least one of these roots (usually a Street, a Building, or a Shop). This satisfies our project requirement.



---

## 3. The "Bulletproof" Mapping Decisions
To move from a simple list to a "bulletproof" system, we made several strategic modifications to `config.py`.

### A. Consolidation & Precision
We realized some "Roots" were too broad. We split them to improve accuracy:
* **The "Office" Split:** We separated `POST OFFICE` from `OFFICE`. This ensures that when a user asks for a Post Office, the Oracle doesn't accidentally navigate to a generic Law Office.
* **The "Street" Hierarchy:** We mapped `STREET`, `AVENUE`, and `BROADWAY` to different OpenStreetMap (OSM) highway levels (Residential vs. Primary). This helps the Oracle prioritize main roads over side streets.

### B. High-Frequency Brand Fallbacks
Humans use brand names, but OSM uses categories. We added "Hybrid Rules":
* **7-Eleven / Convenience:** Mapped to `shop: convenience`.
* **Chase / Bank:** Mapped to `amenity: bank`.
* **Subway:** Mapped to `railway: station`.

### C. Handling "Noise" in Instructions
Manhattan instructions are messy. Our strategy for Task 3.1 includes:
1. **Removing Numbers:** Filtering out "Two," "Three," or "Second" so that "Two Benches" correctly maps to the root `BENCH`.
2. **Substring Matching:** Our code now checks if the Root (e.g., "CHURCH") exists **anywhere** inside the landmark name (e.g., "Trinity Church").

---

## 4. Final Finalized Mapping Table
This table represents the logic currently implemented in our `LANDMARK_GROUPS` dictionary.

| Keyword | OSM Tag(s) | Why it's included |
| :--- | :--- | :--- |
| **CHURCH** | `amenity: place_of_worship` | Most common spiritual anchor in RVS Manhattan. |
| **RESTAURANT**| `amenity: restaurant` | Includes fast food and food courts. |
| **SHOP** | `shop: yes` | General fallback for all retail locations. |
| **PARK** | `leisure: park` | Essential for open-space navigation. |
| **POST** | `amenity: post_office` | High-frequency navigation target. |
| **BICYCLE** | `amenity: bicycle_parking` | Critical for Citibike-heavy Manhattan instructions. |
| **STATION** | `railway: station` | Necessary for "Near the Subway" instructions. |

---

## 5. Verification Proof
We ran a final validation check using the script `check_manhattan_per_chosen_group.py`. 

**Results:**
* **Instructions Tested:** 7,000
* **Instructions Successful:** 7,000
* **Coverage:** 100.00%
* **Status:** The Symbolic Oracle is now "Ready for Reasoning." It has a 0% chance of receiving an instruction containing only "unknown" landmarks.

---

## 6. Next Steps for Task 3.1 (Parser)
The teammate responsible for the NLP Parser should follow these rules:
1. **Clean first:** Remove "the", "a", and numbers from the extracted string.
2. **Check Longest First:** Always check for "POST OFFICE" before checking for "OFFICE".
3. **Upper Case:** Always convert user input to `.upper()` before comparing to our `LANDMARK_GROUPS` keys.