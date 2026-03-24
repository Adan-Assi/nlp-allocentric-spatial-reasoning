"""
config.py
Centralized configuration for the Symbolic Solver Oracle.
Ref: docs/DATA_GUIDE.md | docs/Literature_Review_and_Theoretical_Framework.md
"""

import os

# --- Phase 0: Path Management (Fixes Import/Pylance Errors) ---
# This ensures that whether we run from /scripts or the root, paths stay valid
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths (Updated for our 1B stress test)
GRAPH_PATH = os.path.join(BASE_DIR, "data", "manhattan", "manhattan_graph.gpickle")
POI_PATH = os.path.join(BASE_DIR, "data", "manhattan", "manhattan_poi.pkl")
VARIANTS_JSON = os.path.join(BASE_DIR, "data", "manhattan", "underspecified_variants.json")
RVS_DATA_JSON = os.path.join(BASE_DIR, "data", "manhattan", "manhattan.json")

# Output Reports
AMBIGUITY_REPORT_CSV = os.path.join(BASE_DIR, "data", "manhattan", "ambiguity_report.csv")

# --- Phase 0.5: Search & Radius Constants (The "Scientific" Layer) ---
# JUSTIFICATION: Based on RVS (Paz-Argaman et al., 2024) Appendix C.
# This represents the 'Human Observable Horizon.'
GLOBAL_SEARCH_HORIZON_METERS = 1500

# --- Phase 1: Search & Radius Constants ---
DISTANCE_SCALE_FACTOR = 1.1 
DISTANCE_FIXED_BUFFER = 80 
DEFAULT_SEARCH_RADIUS = 500 

# --- Phase 2: Starting Location (S0) & Reachability ---
S0_BUFFER_METERS = 20
# Toggle for SCC Optimization in reachability checks (Phase 2)
USE_SCC_OPTIMIZATION = True

# --- Phase 3: Directional & Vector Logic ---
DIRECTIONAL_WEDGE_DEGREES = 45  

# --- Phase 4: Landmark Grounding (Clamped Radius) ---
# The influence zone scaling: 1.2 means 20% larger than the physical footprint
RADIUS_SCALE_FACTOR = 1.2 
# Bounds to prevent search areas from being too small (points) or too large (parks)
RADIUS_MIN = 15.0  
RADIUS_MAX = 150.0 
# Fallback radius if area data is missing for a landmark in manhattan_poi.pkl
DEFAULT_LANDMARK_BUFFER = 50.0

# The prefix found in manhattan_graph.gpickle for projected POI nodes
POI_NODE_PREFIX = "1#" 
#POI_NODE_PREFIX = "999"  # Updated from RVS Readme logic

# --- Oracle Classification Labels ---
STATE_ANSWERABLE = "Answerable"
STATE_AMBIGUOUS = "Ambiguous"
STATE_CONTRADICTORY = "Contradictory"


# --- Multi-City Settings ---
CITY_SETTINGS = {
    "manhattan": {
        "success_radius": 80,
        "raw_json": "manhattan.json", 
        "graph_file": "manhattan_graph.gpickle",
        "poi_file": "manhattan_poi.pkl"
    },
    "pittsburgh": {
        "success_radius": 100,
        "raw_json": "pittsburgh.json",
        "graph_file": "pittsburgh_graph.gpickle",
        "poi_file": "pittsburgh_poi.pkl"
    },
    "philadelphia": {
        "success_radius": 100,
        "raw_json": "philadelphia.json",
        "graph_file": "philadelphia_graph.gpickle",
        "poi_file": "philadelphia_poi.pkl"
    }
}

# --- OSM Metadata Mapping ---
# Maps keywords from instructions to specific columns and values in manhattan_poi.pkl
# Task 2.5 Final Mapping - based on coverarge results
LANDMARK_GROUPS = {
    "CHURCH": {"amenity": ["place_of_worship", "monastery"]},
    "RESTAURANT": {"amenity": ["restaurant", "fast_food", "food_court"]},
    "SHOP": {"shop": ["yes", "supermarket", "convenience", "clothes", "mall"]},
    "PARK": {"leisure": ["park", "recreation_ground"], "boundary": "park"},
    "GARDEN": {"leisure": "garden"},
    "THEATRE": {"amenity": ["theatre", "arts_centre"]},
    "STREET": {"highway": ["residential", "tertiary", "secondary", "unclassified"]},
    "AVENUE": {"highway": ["primary", "secondary"]}, 
    "BICYCLE": {"amenity": ["bicycle_parking", "bicycle_rental"]},
    "PHARMACY": {"amenity": "pharmacy"},
    "BANK": {"amenity": "bank"},
    "CAFE": {"amenity": "cafe"},
    "PARKING": {"amenity": "parking"},
    "MUSEUM": {"tourism": "museum", "historic": "museum"},
    "WATER": {"natural": "water", "waterway": "river"},
    "BENCH": {"amenity": "bench"},

    # --- NEW: Mapping the "Hit List" Misses ---
    "FOOD": {"amenity": ["restaurant", "food_court", "cafe"]}, # Freq: 540
    "BIKE": {"amenity": ["bicycle_parking", "bicycle_rental"]}, # Freq: 467
    "RENTAL": {"amenity": ["bicycle_rental", "car_rental"]}, # Freq: 398
    "BUILDING": {"building": "yes"}, # Freq: 377
    "BROADWAY": {"highway": "primary", "name": "Broadway"}, # Freq: 373
    "POST": {"amenity": "post_office"}, # Freq: 346
    "BAR": {"amenity": ["bar", "pub"]}, # Freq: 338
    "SCHOOL": {"amenity": ["school", "university", "college"]}, # Freq: 331
    "LIBRARY": {"amenity": "library"}, # Freq: 326
    "STATION": {"railway": ["station", "subway_entrance"], "amenity": "bus_station"}, # Freq: 304
    "STORE": {"shop": "yes"}, # Freq: 297
    "OFFICE": {"office": "yes"}, # Freq: 290
    "HOTEL": {"tourism": ["hotel", "hostel", "motel", "guest_house"]}, # Frequency fix: 660
}

# Use these words to trigger the OSM tag searches in LANDMARK_GROUPS
TEXT_TO_GROUP_MAP = {
    "supermarket": "SHOP", "grocery": "SHOP", "deli": "SHOP", "pharmacy": "PHARMACY",
    "chemists": "PHARMACY", "drugstore": "PHARMACY", "wine": "SHOP", "vitamins": "SHOP",
    "synagogue": "CHURCH", "temple": "CHURCH", "mosque": "CHURCH",
    "pub": "BAR", "nightclub": "BAR", "club": "BAR", "biergarten": "BAR",
    "theatre": "THEATRE", "cinema": "THEATRE", "movie": "THEATRE",
    "college": "SCHOOL", "university": "SCHOOL", "campus": "SCHOOL",
    "playground": "PARK", "recreation": "PARK",
    "bench": "BENCH", "seat": "BENCH",
    "pier": "STATION", "dock": "STATION", "terminal": "STATION"
}

# Standard tags for broad fallback searches
POI_SEARCH_COLUMNS = ['amenity', 'tourism', 'leisure', 'shop', 'historic', 'name']

# --- Phase 5: Scientific Evaluation Constants ---
# Use these in our report to show "Methodological Rigor"
METRIC_AMBIGUITY_RATE = "ambiguity_rate"
METRIC_REACHABILITY_FAILURE = "unreachable_rate"

# Thresholds for the resolve_all_candidates method
SEMANTIC_THRESHOLD = 0.5
