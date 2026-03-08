import math
import networkx as nx
from geopy.distance import geodesic
import config

# --- NODE COORDINATE HELPERS ---

def get_node_coords(G, node_id):
    """Extracts (lat, lon) from the graph node metadata."""
    node_data = G.nodes[node_id]
    # OSMnx and RVS graphs use 'y' for Latitude and 'x' for Longitude
    return (node_data['y'], node_data['x'])

# --- DISTANCE & PROXIMITY UTILITIES ---

def get_geodesic_dist_raw(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates high-precision geodesic distance between two GPS points in meters.
    Uses WGS-84 ellipsoid (more accurate than great_circle for city grids).
    """
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def get_euclidean_dist(G, node_a, node_b):
    """Calculates straight-line distance in meters between two graph nodes."""
    coords_a = get_node_coords(G, node_a)
    coords_b = get_node_coords(G, node_b)
    return get_geodesic_dist_raw(coords_a[0], coords_a[1], coords_b[0], coords_b[1])

def get_walking_dist(G, start_node, end_node):
    """
    Calculates the actual street-path distance in meters.
    Returns infinity if no path exists.
    """
    try:
        # 'length' is the standard weight attribute in Manhattan .gpickle files
        return nx.shortest_path_length(G, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        return float('inf')

def is_within_buffer(G, agent_node, target_node, buffer=config.LANDMARK_PROXIMITY_BUFFER):
    """Checks if agent is within the 'At/Near' threshold of a landmark node."""
    return get_euclidean_dist(G, agent_node, target_node) <= buffer

# --- DIRECTIONAL & BEARING UTILITIES ---

def get_dominant_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Calculates N, S, E, or W based on which axis has the larger change.
    Used for global strategy and filtering landmarks (e.g., 'North of the park').
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    if abs(dlat) >= abs(dlon):
        return 'N' if dlat > 0 else 'S'
    else:
        return 'E' if dlon > 0 else 'W'

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two GPS points (0-360 degrees)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def get_node_bearing(G, start_node, end_node):
    """Calculates bearing specifically between two Graph Nodes."""
    lat1, lon1 = get_node_coords(G, start_node)
    lat2, lon2 = get_node_coords(G, end_node)
    return calculate_bearing(lat1, lon1, lat2, lon2)

def get_coarse_direction(bearing):
    """Maps a degree bearing to N, S, E, W. Best for local turn instructions."""
    # N: 315-45, E: 45-135, S: 135-225, W: 225-315
    if 45 <= bearing < 135: return "E"
    if 135 <= bearing < 225: return "S"
    if 225 <= bearing < 315: return "W"
    return "N"