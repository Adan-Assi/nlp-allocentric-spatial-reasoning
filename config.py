"""
config.py
Centralized configuration for the Symbolic Solver Oracle.
Ref: docs/DATA_GUIDE.md
"""

# --- Phase 1: Search & Radius Constants ---
DISTANCE_MULTIPLIER = 1.1 
FIXED_DISTANCE_BUFFER = 80 
DEFAULT_SEARCH_RADIUS = 500 

# --- Phase 2: Starting Location (S0) ---
S0_BUFFER_METERS = 20  

# --- Phase 3: Directional & Vector Logic ---
DIRECTIONAL_WEDGE_DEGREES = 45  

# --- Phase 4: Landmark Grounding & Graph Linking ---
# Distance from a street node to a POI to consider it a "match"
LANDMARK_PROXIMITY_BUFFER = 20  
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

# --- OSM Metadata Mapping (Updated for Task 2.5) ---
# Maps keywords from instructions to specific columns and values in manhattan_poi.pkl
LANDMARK_GROUPS = {
    # Existing mappings
    'cafe':       {'column': 'amenity', 'value': 'cafe'},
    'coffee':     {'column': 'amenity', 'value': 'cafe'},
    'library':    {'column': 'amenity', 'value': 'library'},
    'museum':     {'column': 'tourism', 'value': 'museum'},
    'park':       {'column': 'leisure', 'value': 'park'},
    'fountain':   {'column': 'fountain', 'value': 'yes'},
    'subway':     {'column': 'railway', 'value': 'station'},
    'station':    {'column': 'railway', 'value': 'station'},
    'hotel':      {'column': 'tourism', 'value': 'hotel'},
    'restaurant': {'column': 'amenity', 'value': 'restaurant'},
    'theatre':    {'column': 'amenity', 'value': 'theatre'},
    'pharmacy':   {'column': 'amenity', 'value': 'pharmacy'},
    
    # --- NEW: Discovered Brand/Specific Mappings ---
    'starbucks':       {'column': 'name', 'value': 'Starbucks'},
    'chase bank':      {'column': 'name', 'value': 'Chase'},
    'duane reade':     {'column': 'name', 'value': 'Duane Reade'},
    'cvs':             {'column': 'name', 'value': 'CVS'},
    'citi bike':       {'column': 'amenity', 'value': 'bicycle_rental'}
}

# Standard tags for broad fallback searches
POI_SEARCH_COLUMNS = ['amenity', 'tourism', 'leisure', 'shop', 'historic', 'name']