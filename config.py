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
#GRAPH_PATH = os.path.join(BASE_DIR, "data", "manhattan", "manhattan_graph.gpickle")
#POI_PATH = os.path.join(BASE_DIR, "data", "manhattan", "manhattan_poi.pkl")
#VARIANTS_JSON = os.path.join(BASE_DIR, "data", "manhattan", "underspecified_variants.json")
#RVS_DATA_JSON = os.path.join(BASE_DIR, "data", "manhattan", "manhattan.json")

# --- Dynamic Path Resolution ---
# Replace the previously hardcoded GRAPH_PATH/POI_PATH with dynamic getters:

CURRENT_CITY = "manhattan" # Default

# --- Multi-City Settings ---
CITY_SETTINGS = {
    "manhattan": {
        "success_radius": 80,
        "salience_ratio": 0.7, #attempt
        "raw_json": "manhattan.json", 
        "graph_file": "manhattan_graph.gpickle",
        "poi_file": "manhattan_poi.pkl",
        "node_prefix": "1#",
        "geo_col": "centroid"
    },
    "pittsburgh": {
        "success_radius": 100,
        "salience_ratio": 0.5, #attempt
        "raw_json": "pittsburgh.json",
        "graph_file": "pittsburgh_graph.gpickle",
        "poi_file": "pittsburgh_poi.pkl",
        "node_prefix": "1#",
        "geo_col": "centroid"
    },
    "philadelphia": {
        "success_radius": 250,  # Matches the RVS "Coarse" baseline for Philly
        "salience_ratio": 0.5, #attempt
        "raw_json": "philadelphia.json",
        "graph_file": "philadelphia_graph.gpickle",
        "poi_file": "philadelphia_poi.pkl",
        "node_prefix": "1#",
        "geo_col": "centroid"
    }
}

def get_graph_path():
    """Dynamically builds the path to the graph file based on current city."""
    city_data = CITY_SETTINGS.get(CURRENT_CITY)
    return os.path.join(BASE_DIR, "data", CURRENT_CITY, city_data["graph_file"])

def get_poi_path():
    """Dynamically builds the path to the POI file based on current city."""
    city_data = CITY_SETTINGS.get(CURRENT_CITY)
    return os.path.join(BASE_DIR, "data", CURRENT_CITY, city_data["poi_file"])

def get_success_radius():
    """Returns the city-specific bibliographic success radius."""
    return CITY_SETTINGS.get(CURRENT_CITY)["success_radius"]

def get_node_prefix():
    """Returns the city-specific node prefix for POI nodes."""
    return CITY_SETTINGS.get(CURRENT_CITY)["node_prefix"]

def get_salience_ratio():
    """Returns the city-specific salience ratio for ambiguity resolution."""
    return CITY_SETTINGS.get(CURRENT_CITY)["salience_ratio"]

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

# --- Oracle Classification Labels ---
STATE_ANSWERABLE = "Answerable"
STATE_AMBIGUOUS = "Ambiguous"
STATE_CONTRADICTORY = "Contradictory"

# --- Geographic & Spatial Pruning Constants ---
# Roughly 111km per degree of latitude. 
# Used for fast bounding-box pre-filtering (S2 approximation).
METERS_PER_DEGREE_LATITUDE = 111000.0

# --- OSM Metadata Mapping ---
# Maps keywords from instructions to specific columns and values in manhattan_poi.pkl
# Task 2.5 Final Mapping - based on coverarge results
LANDMARK_GROUPS = {
    "CHURCH": {"amenity": ["place_of_worship", "monastery"]},
    "RESTAURANT": {"amenity": ["restaurant", "fast_food", "food_court"], "brand": "yes"},
    "SHOP": {"shop": ["yes", "supermarket", "convenience", "clothes", "mall"], "brand": "yes"},
    "PARK": {"leisure": ["park", "recreation_ground"], "boundary": "park"},
    "GARDEN": {"leisure": "garden"},
    "THEATRE": {"amenity": ["theatre", "arts_centre"]},
    "STREET": {"highway": ["residential", "tertiary", "secondary", "unclassified"]},
    "AVENUE": {"highway": ["primary", "secondary"]}, 
    "BICYCLE": {"amenity": ["bicycle_parking", "bicycle_rental"]},
    "PHARMACY": {"amenity": "pharmacy", "brand": "yes"},
    "BANK": {"amenity": "bank", "brand": "yes"},
    "CAFE": {"amenity": "cafe", "brand": "yes"},
    "PARKING": {
        "amenity": ["parking", "bicycle_parking", "motorcycle_parking"],
        "parking": ["surface", "multi-storey", "underground"] # Add this!
    },
    "MUSEUM": {"tourism": "museum", "historic": ["museum", "yes"]}, # Expanded historic tag for better coverage
    "WATER": {"natural": "water", "waterway": "river"},
    "BENCH": {"amenity": "bench"},

    # --- NEW: Mapping the "Hit List" & RVS Metadata ---
    "FOOD": {"amenity": ["restaurant", "food_court", "cafe"], "brand": "yes"}, # Freq: 540
    "BIKE": {
        "amenity": ["bicycle_parking", "bicycle_rental"], # Matches Citi Bike
        "shop": ["bicycle", "yes"],                       # Matches Franks / Retail
        "brand": "yes"
    },# Freq: 467
    "RENTAL": {"amenity": ["bicycle_rental", "car_rental"]}, # Freq: 398

    # Updated BUILDING group with RVS material/roof metadata (Freq: 377)
    "BUILDING": {
        "building": "yes", 
        "building:material": "yes", 
        "roof:shape": "yes", 
        "roof:material": "yes",
        "colour": "yes",            # <--- Added
        "building:colour": "yes",   # <--- Added
        "roof:colour": "yes"        # <--- Added
    },

    "BROADWAY": {"highway": "primary", "name": "Broadway"}, # Freq: 373
    "POST": {"amenity": ["post_office", "post_box"]}, # Freq: 346
    "BAR": {"amenity": ["bar", "pub"]}, # Freq: 338
    "SCHOOL": {"amenity": ["school", "university", "college"]}, # Freq: 331
    "LIBRARY": {"amenity": "library"}, # Freq: 326
    "STATION": {"railway": ["station", "subway_entrance"], "amenity": "bus_station"}, # Freq: 304
    "OFFICE": {"office": "yes"}, # Freq: 290
    "HOTEL": {"tourism": ["hotel", "hostel", "motel", "guest_house"]}, # Frequency fix: 660

    # Common typos found in our audit
    "ENTRANCE": {"amenity": ["subway_entrance", "entrance"], "railway": "subway_entrance"},

    # Consolidate ambiguous terms
    "CLOTHES": {"shop": ["clothes", "fashion", "boutique"], "brand": "yes"},
    "STORE": {"shop": ["yes", "convenience", "supermarket", "clothes"], "brand": "yes"},

    "MONUMENT": {"historic": ["monument", "memorial"], "tourism": "artwork"},
    "MARKET": {"amenity": "marketplace", "shop": "market"}
}

# Maps common instruction keywords to canonical categories for the CategoricalMatcher
# Maps common instruction keywords to canonical categories for the CategoricalMatcher
TEXT_TO_GROUP_MAP = {
    # SHOP
    **dict.fromkeys(["convenience shop", "alcohol shop", "atm", "supermarket", "grocery",
        "wine", "vitamins", "boutique", "boutique shop", "antique", "antiques",
        "antique shop", "antiques shop", "vacant shop", "books shop", "bookshop",
        "bookstore", "beauty shop", "florist shop", "florist", "massage shop",
        "hairdresser shop", "hairdresser", "optician shop", "optician", "bike repair",
        "car repair", "gas station", "petrol station", "fuel station", "pep boys",
        "gift shop", "salon", "barbershop", "laundry", "laundromat", "dry cleaner",
        "outfitters", "7-eleven", "wawa", "ben & jerry's", "ice cream", "creamery",
        "aldi", "shop"], "SHOP"),

    # PHARMACY
    **dict.fromkeys(["pharmacy", "chemists", "drugstore", "drug store",
        "medicine shoppe"], "PHARMACY"),

    # BAR
    **dict.fromkeys(["bar", "pub", "nightclub", "club", "biergarten",
        "tavern", "brewery", "taproom"], "BAR"),

    # RESTAURANT / FOOD
    **dict.fromkeys(["restaurant", "diner", "wendy's", "mcdonald's",
        "burger king"], "RESTAURANT"),
    **dict.fromkeys(["pizza", "burger", "papa john's", "papa johns"], "FOOD"),

    # CAFE
    **dict.fromkeys(["coffee", "coffee shop", "bakery", "cafe", "starbucks",
        "dunkin", "peet's"], "CAFE"),

    # CLOTHES
    **dict.fromkeys(["clothing", "clothes shop", "american eagle"], "CLOTHES"),

    # STORE
    **dict.fromkeys(["store", "deli", "target", "walmart"], "STORE"),

    # CHURCH
    **dict.fromkeys(["synagogue", "temple", "mosque", "church",
        "cathedral", "chapel"], "CHURCH"),

    # SCHOOL
    **dict.fromkeys(["college", "university", "campus", "school", "library",
        "hospital", "u of pittsburgh", "university of pittsburgh"], "SCHOOL"),

    # OFFICE
    **dict.fromkeys(["doctor", "dentist", "aspen dental", "community centre",
        "social facility", "studio", "courthouse", "warehouse", "vfw",
        "state farm", "gateway center", "headquarters", "company headquarters"], "OFFICE"),

    # PARK
    **dict.fromkeys(["playground", "recreation", "garden", "park", "pitch",
        "bing pitch", "tennis", "tennis court", "court", "pavilion",
        "shelter pavilion", "picnic shelter", "picnic", "leisure garden",
        "little garden", "recreation center"], "PARK"),

    # PARKING
    **dict.fromkeys(["parking lot", "parking entrance", "garage", "car sharing",
        "parking space", "parking spaces", "3 parking"], "PARKING"),

    # MONUMENT
    **dict.fromkeys(["monument", "memorial", "fountain", "gateway", "ruins",
        "historic ruins", "historical building", "fort pitt",
        "love sculpture", "love", "sculpture",
        "allegheny river", "river", "river entrance",
        "water feature", "drinking water feature"], "MONUMENT"),

    # MUSEUM
    **dict.fromkeys(["gallery", "museum"], "MUSEUM"),

    # THEATRE
    **dict.fromkeys(["theatre", "cinema", "movie", "casino"], "THEATRE"),

    # MARKET
    **dict.fromkeys(["marketplace", "market"], "MARKET"),

    # HOTEL
    **dict.fromkeys(["hotel", "motel", "inn"], "HOTEL"),

    # BANK
    **dict.fromkeys(["bank", "chase bank"], "BANK"),

    # GYM
    **dict.fromkeys(["fitness center", "fitness", "gym", "recreation center"], "GYM"),

    # BENCH
    **dict.fromkeys(["bench", "seat", "benches", "waste basket",
        "drinking water"], "BENCH"),

    # BICYCLE
    **dict.fromkeys(["bicycle parking", "bike parking"], "BICYCLE"),

    # STATION
    **dict.fromkeys(["pier", "dock", "terminal"], "STATION"),

    # POST
    **dict.fromkeys(["post box", "post office", "post-office", "mailbox"], "POST"),
}


# Standard tags for broad fallback searches
# Updated to include all critical RVS-identified OSM keys
POI_SEARCH_COLUMNS = [
    'amenity', 'tourism', 'leisure', 'shop', 'historic', 
    'name', 'brand', 'building', 'office', 'craft', 'healthcare',
    'building:material', 'roof:shape', 'roof:material', 
    'colour', 'building:colour', 'roof:colour', 'parking'
]

# --- Phase 5: Scientific Evaluation Constants ---
# Use these in our report to show "Methodological Rigor"
METRIC_AMBIGUITY_RATE = "ambiguity_rate"
METRIC_REACHABILITY_FAILURE = "unreachable_rate"

# Thresholds for the resolve_all_candidates method
SEMANTIC_THRESHOLD = 0.5
