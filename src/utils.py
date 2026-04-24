import math
import numpy as np
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


def get_clamped_radius(area_m2):
    """
    Calculates search radius based on landmark size.
    Formula: R = clip(sqrt(Area/π) * scale, min_r, max_r)
    """
    if area_m2 <= 0:
        return config.DEFAULT_LANDMARK_BUFFER
        
    # Radius from area: Area = πr²
    base_radius = math.sqrt(area_m2 / math.pi)
    scaled_radius = base_radius * config.RADIUS_SCALE_FACTOR
    
    # Clamp between config limits
    return max(config.RADIUS_MIN, min(scaled_radius, config.RADIUS_MAX))


def is_within_buffer(G, agent_node, landmark_coords, radius=None):
    """
    Checks if agent node is within a specific radius of landmark coordinates.
    """
    if radius is None:
        # Dynamically fetch the radius based on config.CURRENT_CITY
        radius = config.get_success_radius() 

    agent_coords = get_node_coords(G, agent_node)
    dist = get_geodesic_dist_raw(agent_coords[0], agent_coords[1], 
                                 landmark_coords[0], landmark_coords[1])
    return dist <= radius

# --- DIRECTIONAL & BEARING UTILITIES ---

def get_dominant_direction(lat1, lon1, lat2, lon2):
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    half_wedge = config.DIRECTIONAL_WEDGE_DEGREES / 2
    
    # Check against cardinal axes with the configurable wedge
    if (360 - half_wedge) <= bearing or bearing < half_wedge: return 'N'
    if (90 - half_wedge) <= bearing < (90 + half_wedge): return 'E'
    if (180 - half_wedge) <= bearing < (180 + half_wedge): return 'S'
    if (270 - half_wedge) <= bearing < (270 + half_wedge): return 'W'
    
    # Fallback to the old "largest change" logic if it's in a dead-zone
    dlat, dlon = lat2 - lat1, lon2 - lon1
    if abs(dlat) >= abs(dlon): return 'N' if dlat > 0 else 'S'
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

# --- VECTOR RELATIONAL LOGIC ---

def is_past_landmark(start_coords, landmark_coords, candidate_coords):
    """
    Task 2.3: Uses Scalar Projection to check if a candidate node is 'past' a landmark.
    Logic: The projection of S0->C onto S0->L must be greater than the length of S0->L.
    """
    v_l = np.array([landmark_coords[0] - start_coords[0], landmark_coords[1] - start_coords[1]])
    v_c = np.array([candidate_coords[0] - start_coords[0], candidate_coords[1] - start_coords[1]])
    
    mag_l_sq = np.dot(v_l, v_l)
    if mag_l_sq == 0: return False
    
    # This value 'p' tells us how many 'landmark-lengths' we've traveled along the vector
    # p > 1.0 means we have physically overshot the landmark's position on that axis
    p = np.dot(v_c, v_l) / mag_l_sq
    
    return p > 1.0

# --- GEODESIC GATEKEEPER ---

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Uses high-performance NumPy vectorization for large-scale spatial queries.
    """
    # Earth radius in meters
    R = 6371000
    
    # Use NumPy's native radian conversion (fast for arrays)
    # This replaces the slow map(np.radians, ...) call we previously had
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # Using arctan2 is more robust than arcsin for floating point errors
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
    
    return R * c


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points 
    on the earth in meters (scalar version).
    """
    R = 6371000  # Earth radius in meters
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def apply_geodesic_gatekeeper(agent_coords, candidates_df, radius=1500):
    """
    Shared helper to filter POIs based on the 'Human Reasoning Horizon'.
    
    Args:
        agent_coords (tuple): (lat, lon)
        candidates_df (pd.DataFrame): POIs matching the NLP category
        radius (int): Default 1500m (Paz-Argaman et al. threshold)
    """
    if candidates_df.empty:
        return candidates_df

    # Extract coordinates for vectorized calculation
    cand_lats = candidates_df.geometry.y.values
    cand_lons = candidates_df.geometry.x.values
    
    # Calculate all distances at once (Vectorized is 50x faster than .apply)
    distances = haversine_vectorized(agent_coords[0], agent_coords[1], cand_lats, cand_lons)
    
    return candidates_df[distances <= radius].copy()

# --- CONNECTIVITY & GRAPH OPTIMIZATION ---

def get_connectivity_map(G):
    """
    Generates a mapping of Node ID -> Component ID.
    Uses Undirected components to represent physical urban connectivity.
    """
    # 'Strongly' implies directionality (Car logic)
    # 'Connected' implies physical access (Human logic)
    undirected_G = G.to_undirected()

    # Convert to undirected to match the origianl RVS paper's "Physical Connectivity"
    components = list(nx.connected_components(undirected_G))
    
    conn_map = {}
    for i, component in enumerate(components):
        for node in component:
            conn_map[node] = i
    return conn_map, len(components)


def is_reachable_fast(scc_map, start_node, end_node):
    """
    Instant reachability check using SCC mapping.
    Replaces expensive nx.has_path() calls.
    """
    # If either node is missing from map (not in graph), they aren't reachable
    if start_node not in scc_map or end_node not in scc_map:
        return False
        
    # In a Strongly Connected Component, every node can reach every other node
    return scc_map[start_node] == scc_map[end_node]