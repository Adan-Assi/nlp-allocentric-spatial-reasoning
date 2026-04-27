"""
config.py
Centralized configuration for the Symbolic Solver Oracle.
Ref: docs/DATA_GUIDE.md | docs/Literature_Review_and_Theoretical_Framework.md
"""

import os

# --- Phase 0: Path Management (Fixes Import/Pylance Errors) ---
# This ensures that whether we run from /scripts or the root, paths stay valid
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
#DIRECTIONAL_WEDGE_DEGREES = 45
COMPASS_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_SECTOR_COUNT = 8
COMPASS_SECTOR_ANGLE = 360 / COMPASS_SECTOR_COUNT  # 45.0
COMPASS_CENTERING_OFFSET = COMPASS_SECTOR_ANGLE / 2  # 22.5

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
# --- MAPPING LOGIC REMINDER ---
# 1. TEXT_TO_GROUP_MAP: Human -> Concept (Translates slang/synonyms to a Category)
#    Example: "Starbucks" or "coffee" -> "CAFE"
#
# 2. LANDMARK_GROUPS: Concept -> Data (Translates a Category to OSM Tags)
#    Example: "CAFE" -> {"amenity": "cafe", "brand": "yes"}
# ------------------------------

# Organized by alphabetic order (based on coverage results)
LANDMARK_GROUPS = {
    "AVENUE": {"highway": ["primary", "secondary"]}, 

    "BANK": {"amenity": "bank", "brand": "yes"},
    "BAR": {"amenity": ["bar", "pub"]}, # Freq: 338
    "BENCH": {"amenity": "bench"},
    "BIKE": {
        "amenity": ["bicycle_parking", "bicycle_rental"], # Matches Citi Bike
        "shop": ["bicycle", "yes"],                       # Matches Franks / Retail
        "network": ["Citi Bike", "citibike"], 
        "operator": ["NYC Bike Share", "Motivate"],
        "brand": "yes"
    },# Freq: 467
    "BROADWAY": {"highway": "primary", "name": "Broadway"}, # Freq: 373
    "BUILDING": {"building": ["yes", "commercial", "residential", 
                            "office", "retail", "apartments"]},


    "CAFE": {"amenity": "cafe", "brand": "yes"},
    "CHURCH": {"amenity": ["place_of_worship", "monastery"]},
    "CLOTHES": {"shop": ["clothes", "fashion", "boutique"], "brand": "yes"},

    # Common typos found in our audit
    "ENTRANCE": {"amenity": ["subway_entrance", "entrance"],
                "railway": "subway_entrance"},
    
    "FOOD": {"amenity": ["restaurant", "food_court", "cafe"], "brand": "yes"}, # Freq: 540

    "GARDEN": {"leisure": ["garden", "nature_reserve"],
            "landuse": "grass"},
    "GYM": {"leisure": ["fitness_centre", "sports_centre"], 
        "amenity": ["gym", "sports_centre"],
        "shop": "sports"}, 

    "HOSPITAL": {"amenity": ["hospital", "clinic", "doctors"]},
    "HOTEL": {"tourism": ["hotel", "hostel", "motel", "guest_house"]}, # Frequency fix: 660

    "LIBRARY": {"amenity": "library"}, # Freq: 326

    "MARKET": {"amenity": "marketplace", "shop": "market"},
    "MONUMENT": {"historic": ["monument", "memorial"], "tourism": ["artwork", "viewpoint"]},
    "MUSEUM": {"tourism": "museum", "historic": ["museum", "yes"]}, # Expanded historic tag for better coverage

    "OFFICE": {
        "office": "yes",
        "amenity": ["police", "townhall", "courthouse", "embassy"]
    },

    "PARK": {"leisure": ["park", "recreation_ground"], "boundary": "park"},
    "PARKING": {
        "amenity": ["parking", "bicycle_parking", "motorcycle_parking"],
        "parking": ["surface", "multi-storey", "underground"]
    },
    "PHARMACY": {"amenity": "pharmacy", "brand": "yes"},
    "POST": {"amenity": ["post_office", "post_box"]}, # Freq: 346

    "RENTAL": {"amenity": ["bicycle_rental", "car_rental"]}, # Freq: 398
    "RESTAURANT": {"amenity": ["restaurant", "fast_food", "food_court"], "brand": "yes"},
    
    "SCHOOL": {"amenity": ["school", "university", "college"]}, # Freq: 331
    "SHOP": {"shop": ["yes", "supermarket", "convenience", "clothes", "mall"], "brand": "yes"},
    "STATION": {"railway": ["station", "subway_entrance"], "amenity": "bus_station"}, # Freq: 304
    "STORE": {"shop": ["yes", "convenience", "supermarket", "clothes"], "brand": "yes"}, # Freq: 297
    "STREET": {"highway": ["residential", "tertiary", "secondary", "unclassified"]},

    "THEATRE": {"amenity": ["theatre", "arts_centre"]},

    "WATER": {"natural": "water", "waterway": "river"}
}


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
        "aldi", "shop", "Gap"], "SHOP"),

    # PHARMACY
    **dict.fromkeys(["pharmacy", "chemists", "drugstore", "drug store",
        "medicine shoppe", "cvs"], "PHARMACY"),

    # BAR
    **dict.fromkeys(["bar", "pub", "nightclub", "club", "biergarten",
        "tavern", "brewery", "taproom"], "BAR"),

    # RESTAURANT / FOOD
    **dict.fromkeys(["restaurant", "diner", "wendy's", "mcdonald's",
        "burger king", "potbelly"], "RESTAURANT"),
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

    # SCHOOL — academic institutions only
    **dict.fromkeys(["college", "university", "campus", "school",
        "u of pittsburgh", "university of pittsburgh"], "SCHOOL"),

    # LIBRARY — separate from SCHOOL, for correct OSM tag matching
    **dict.fromkeys(["library", "public library", "branch library"], "LIBRARY"),

    # HOSPITAL — medical facilities
    **dict.fromkeys(["hospital", "medical center", "clinic", "urgent care",
        "emergency room", "health center", "medical centre"], "HOSPITAL"),

    # OFFICE
    **dict.fromkeys(["doctor", "dentist", "aspen dental", "community centre",
        "social facility", "studio", "courthouse", "warehouse", "vfw",
        "state farm", "gateway center", "headquarters",
        "company headquarters"], "OFFICE"),

    # PARK
    **dict.fromkeys(["playground", "recreation", "park", "pitch",
        "bing pitch", "tennis", "tennis court", "court", "pavilion",
        "shelter pavilion", "picnic shelter", "picnic", "leisure garden",
        "recreation center"], "PARK"),

    # GARDEN
    **dict.fromkeys(["garden", "botanical garden", "community garden",
    "nature reserve", "green space"], "GARDEN"),

    # PARKING
    **dict.fromkeys(["parking lot", "parking entrance", "garage", "car sharing",
        "parking space", "parking spaces", "3 parking"], "PARKING"),

    # MONUMENT
    **dict.fromkeys(["monument", "memorial", "fountain", "gateway", "ruins",
        "historic ruins", "historical building", "historic building", "fort pitt",
        "love sculpture", "love", "sculpture",
        "river entrance", "water feature", "drinking water feature"], "MONUMENT"),

    # MUSEUM
    **dict.fromkeys(["gallery", "museum"], "MUSEUM"),

    # THEATRE
    **dict.fromkeys(["theatre", "cinema", "movie", "casino"], "THEATRE"),

    # MARKET
    **dict.fromkeys(["marketplace", "market"], "MARKET"),

    # HOTEL
    **dict.fromkeys(["hotel", "motel", "inn", "best western plus", "best western"], "HOTEL"), # Reclaiming Philly specific hotel terms

    # BANK
    **dict.fromkeys(["bank", "chase bank"], "BANK"),

    # GYM
    **dict.fromkeys(["fitness center", "fitness", "gym", "recreation center",
        "sports center", "sports centre", "fitness centre",
        "crossfit", "yoga studio", "pilates"], "GYM"),

    # BENCH
    **dict.fromkeys(["bench", "seat", "benches", "waste basket",
        "drinking water"], "BENCH"),

    # BIKE — cycling infrastructure
    **dict.fromkeys([
        "bicycle parking", "bike parking", "bicycle rack",
        "citi bike", "citibike", "citi bike rental",
        "citi bike bicycle rental", "bike station", "bicycle station",
    ], "BIKE"),

    # RENTAL — motorized or general rentals
    **dict.fromkeys([
        "car rental", "rental shop", "enterprise", "hertz", "avis",
        "scooter rental", "rental car",
    ], "RENTAL"),

    # STATION
    **dict.fromkeys(["pier", "dock", "terminal"], "STATION"),

    # POST
    **dict.fromkeys(["post box", "post office", "post-office", "mailbox"], "POST"),

    # WATER
    **dict.fromkeys(["river", "lake", "pond", "waterfront", "water",
        "hudson river", "east river", "allegheny river",
        "monongahela", "schuylkill"], "WATER"),

    # BUILDING
    **dict.fromkeys(["building", "office building", "apartment building",
        "residential building", "commercial building"], "BUILDING"),
    
    # ENTRANCE — subway and building entrances
    **dict.fromkeys(["entrance", "subway entrance", "building entrance",
        "metro entrance", "station entrance"], "ENTRANCE"),
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
