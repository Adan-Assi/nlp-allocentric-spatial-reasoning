"""
config.py
Centralized configuration for the Symbolic Solver Oracle.
Ref: docs/DATA_GUIDE.md
"""

# --- Phase 1: Search & Radius Constants ---
DISTANCE_SCALE_FACTOR = 1.1 
DISTANCE_FIXED_BUFFER = 80 
DEFAULT_SEARCH_RADIUS = 500 

# --- Phase 2: Starting Location (S0) ---
S0_BUFFER_METERS = 20  

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

# --- Oracle Classification Labels ---
STATE_ANSWERABLE = "Answerable"
STATE_AMBIGUOUS = "Ambiguous"
STATE_CONTRADICTORY = "Contradictory"

# --- OSM Metadata Mapping ---
# Maps keywords from instructions to specific columns and values in manhattan_poi.pkl
LANDMARK_GROUPS = {
    'cafe':       {'column': 'amenity', 'value': 'cafe'},
    'coffee':     {'column': 'amenity', 'value': 'cafe'},
    'library':    {'column': 'amenity', 'value': 'library'},
    'museum':     {'column': 'tourism', 'value': 'museum'},
    'park':       {'column': 'leisure', 'value': 'park'},
    'fountain':   {'column': 'fountain', 'value': 'yes'}, # Based on our 'fountain' col
    'subway':     {'column': 'railway', 'value': 'station'},
    'station':    {'column': 'railway', 'value': 'station'},
    'hotel':      {'column': 'tourism', 'value': 'hotel'},
    'restaurant': {'column': 'amenity', 'value': 'restaurant'},
    'theatre':    {'column': 'amenity', 'value': 'theatre'},
    'pharmacy':   {'column': 'amenity', 'value': 'pharmacy'}
}

# Standard tags for broad fallback searches
POI_SEARCH_COLUMNS = ['amenity', 'tourism', 'leisure', 'shop', 'historic', 'name']