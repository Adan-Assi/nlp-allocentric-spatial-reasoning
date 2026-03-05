import math

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two points.
    Returns degrees (0-360).
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def get_coarse_direction(bearing):
    """Maps a degree bearing to N, S, E, W."""
    if 45 <= bearing < 135: return "E"
    if 135 <= bearing < 225: return "S"
    if 225 <= bearing < 315: return "W"
    return "N"