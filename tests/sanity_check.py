import pickle
from pathlib import Path
import networkx as nx

# --- Load graph ---
graph_path = Path(__file__).parent.parent / "data" / "manhattan" / "manhattan_graph.gpickle"

with open(graph_path, "rb") as f:
    G = pickle.load(f)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# --- Coordinate helpers ---
def get_coords(node):
    data = G.nodes[node]
    return data["y"], data["x"]   # (lat, lon)

def coarse_direction(a, b):
    lat1, lon1 = get_coords(a)
    lat2, lon2 = get_coords(b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    if abs(dlat) > abs(dlon):
        return "N" if dlat > 0 else "S"
    else:
        return "E" if dlon > 0 else "W"

# --- One navigation query (no LLM) ---
nodes = list(G.nodes)
start = nodes[0]
goal = nodes[500]

path = nx.shortest_path(G, start, goal, weight="length")
first_step = path[1]

direction = coarse_direction(start, first_step)

print(f"Navigation decision: first step goes {direction}")
