"""
utils.py
Shared math primitives: distance, bearings, directions, reachability.

Improvements over the original repo version:
  - Adds get_direction_8way() for 8-way compass classification (N/NE/E/.../NW)
  - Adds direction_matches() for cardinal-vs-intercardinal matching with a
    small "cardinals are coarse, intercardinals are exact" compatibility rule.
  - Keeps the legacy get_dominant_direction() for any caller that still wants
    coarse 4-way output (the proximity buffer & sanity tests use it).
"""

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


def meters_to_degree_radius(meters, latitude):
    """
    Convert a meter distance to a degree radius safe for KDTree ball queries
    over (lat, lon) coordinate pairs.

    The KDTree treats (lat, lon) as Euclidean coordinates, so `query_ball_point`
    draws a circle in degree-space. But one degree of latitude spans ~111 km
    everywhere, while one degree of longitude spans only ~111 km × cos(lat).
    At NYC (~40°N) a longitude-degree is ~23% shorter than a latitude-degree.

    Using `meters / METERS_PER_DEGREE_LATITUDE` alone would UNDER-fetch in the
    east-west direction — real POIs within the target meter-radius but offset
    east/west would fall outside the ball. This helper instead uses the
    longitude-degree width, so the degree-circle fully CONTAINS the meter-
    circle. Downstream haversine / score filters trim the over-fetch.

    Note:
        Breaks down near the poles (cos(lat) → 0). Fine for all RVS cities
        (Manhattan, Pittsburgh, Philadelphia are all ~40°N).
    """
    return meters / (config.METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude)))


def is_within_buffer(G, agent_node, target_node, radius=None):
    """
    Checks if `agent_node` is within `radius` meters of `target_node`.

    Both arguments are graph node IDs; coordinates are looked up from G.
    Signature mirrors `get_euclidean_dist(G, node_a, node_b)` for consistency.

    Args:
        G: NetworkX graph whose nodes carry 'x' (lon) and 'y' (lat) attributes.
        agent_node: Node ID of the agent's current position.
        target_node: Node ID of the target landmark.
        radius: Success radius in meters. Defaults to `config.get_success_radius()`
                (city-aware via `config.CURRENT_CITY`).

    Returns:
        True if the geodesic distance between the two nodes is <= radius.

    Raises:
        KeyError: if either node ID is not present in G.
    """
    if radius is None:
        # Dynamically fetch the radius based on config.CURRENT_CITY
        radius = config.get_success_radius()

    # Reuse get_euclidean_dist so both helpers share identical distance math.
    dist = get_euclidean_dist(G, agent_node, target_node)
    return dist <= radius


# --- DIRECTIONAL & BEARING UTILITIES ---

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


def get_dominant_direction(lat1, lon1, lat2, lon2):
    """
    Legacy 4-way classifier. Kept for backward compatibility with proximity
    sanity checks, but new code should prefer get_direction_8way().
    """
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    half_wedge = config.DIRECTIONAL_WEDGE_DEGREES / 2

    # Check against cardinal axes with the configurable wedge
    if (360 - half_wedge) <= bearing or bearing < half_wedge:
        return 'N'
    if (90 - half_wedge) <= bearing < (90 + half_wedge):
        return 'E'
    if (180 - half_wedge) <= bearing < (180 + half_wedge):
        return 'S'
    if (270 - half_wedge) <= bearing < (270 + half_wedge):
        return 'W'

    # Fallback to "largest change" logic if bearing falls in a dead-zone
    dlat, dlon = lat2 - lat1, lon2 - lon1
    if abs(dlat) >= abs(dlon):
        return 'N' if dlat > 0 else 'S'
    return 'E' if dlon > 0 else 'W'


def get_coarse_direction(bearing):
    """Maps a degree bearing to N, S, E, W. Best for local turn instructions."""
    # N: 315-45, E: 45-135, S: 135-225, W: 225-315
    if 45 <= bearing < 135:
        return "E"
    if 135 <= bearing < 225:
        return "S"
    if 225 <= bearing < 315:
        return "W"
    return "N"


def get_direction_8way(lat1, lon1, lat2, lon2):
    """
    Maps the bearing between two GPS points to an 8-way compass direction.

    Returns one of: N, NE, E, SE, S, SW, W, NW

    Used by the oracle's directional filter so instructions like "northeast"
    and "northwest" are no longer collapsed into the same label.
    """
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)

    # 8 equal compass sectors, each 45 degrees wide.
    # Adding 22.5 shifts the sector boundaries so N is centered on 0/360.
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((bearing + 22.5) % 360) // 45)
    return directions[idx]


def direction_matches(actual_direction: str, target_direction: str) -> bool:
    """
    Direction comparison with a small cardinal-vs-intercardinal compatibility layer.

    Rules:
      - If either argument is empty/None, treat as compatible (no constraint).
      - If target_direction is intercardinal (NE/NW/SE/SW), it must match exactly.
      - If target_direction is cardinal (N/E/S/W), it ALSO matches the two
        adjacent intercardinal sectors. Natural-language "north" is coarse and
        often includes "northeast" or "northwest" too, so we accept either.
    """
    if not actual_direction or not target_direction:
        return True

    actual = actual_direction.upper().strip()
    target = target_direction.upper().strip()

    if target in {"NE", "NW", "SE", "SW"}:
        return actual == target

    compatible = {
        "N": {"NW", "N", "NE"},
        "E": {"NE", "E", "SE"},
        "S": {"SE", "S", "SW"},
        "W": {"SW", "W", "NW"},
    }
    return actual in compatible.get(target, {target})


# --- VECTOR RELATIONAL LOGIC ---

def is_past_landmark(start_coords, landmark_coords, candidate_coords):
    """
    Uses Scalar Projection to check if a candidate node is 'past' a landmark.
    Logic: The projection of S0->C onto S0->L must be greater than the length of S0->L.
    """
    v_l = np.array([landmark_coords[0] - start_coords[0], landmark_coords[1] - start_coords[1]])
    v_c = np.array([candidate_coords[0] - start_coords[0], candidate_coords[1] - start_coords[1]])

    mag_l_sq = np.dot(v_l, v_l)
    if mag_l_sq == 0:
        return False

    # 'p' is how many 'landmark-lengths' we've traveled along the vector.
    # p > 1.0 means we have physically overshot the landmark on that axis.
    p = np.dot(v_c, v_l) / mag_l_sq

    return p > 1.0


# --- GEODESIC GATEKEEPER ---

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).
    Uses high-performance NumPy vectorization for large-scale spatial queries.
    """
    R = 6371000  # Earth radius in meters

    # Use NumPy's native radian conversion (fast for arrays)
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    )

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # c = 2 ⋅ atan2( √a, √(1−a) )
    # arctan2 is more robust than arcsin for floating-point edge cases.
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

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

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
    Builds a Node ID -> Component ID map for fast O(1) reachability checks.

    Uses WEAKLY connected components (i.e. treats G as undirected) rather
    than strongly connected components. This matches the RVS paper's
    pedestrian model, where one-way street directions are not barriers:
    a walker on the opposite side of a one-way street can still reach
    the goal. If you ever need a vehicle-routing variant, swap this to
    `nx.strongly_connected_components(G)` on the directed graph.

    Returns:
        (conn_map, num_components): dict{node_id -> int}, int
    """
    # Convert to undirected to model physical pedestrian connectivity.
    undirected_G = G.to_undirected()
    components = list(nx.connected_components(undirected_G))

    conn_map = {}
    for i, component in enumerate(components):
        for node in component:
            conn_map[node] = i
    return conn_map, len(components)


def is_reachable_fast(conn_map, start_node, end_node):
    """
    O(1) reachability check using a pre-computed connectivity map.

    Two nodes are reachable iff they belong to the same weakly-connected
    component (see `get_connectivity_map` for the rationale).

    Args:
        conn_map: dict{node_id -> component_id} built by get_connectivity_map.
        start_node: source node ID.
        end_node: destination node ID.

    Returns:
        True iff both nodes are in the map and share the same component.
    """
    # If either node isn't in the map (e.g. not in the graph), they aren't reachable.
    if start_node not in conn_map or end_node not in conn_map:
        return False
    return conn_map[start_node] == conn_map[end_node]
