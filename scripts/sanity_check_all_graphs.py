import pickle
from pathlib import Path
import networkx as nx


# Where your graphs live (adjust if your folder differs)
GRAPHS = {
    "manhattan": Path("data/graphs/manhattan.gpickle"),
    "philadelphia": Path("data/graphs/philadelphia.gpickle"),
    "pittsburgh": Path("data/graphs/pittsburgh.gpickle"),
}


def get_coords(G, node):
    """
    OSMnx graphs store coordinates as:
      x = longitude
      y = latitude
    """
    data = G.nodes[node]
    return data["y"], data["x"]  # (lat, lon)


def coarse_direction(G, a, b):
    """
    Dominant-axis coarse direction from node a to node b.
    Returns one of: N/E/S/W
    """
    lat1, lon1 = get_coords(G, a)
    lat2, lon2 = get_coords(G, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    if abs(dlat) >= abs(dlon):
        return "N" if dlat > 0 else "S"
    else:
        return "E" if dlon > 0 else "W"


def pick_two_nodes(nodes, idx_a=0, idx_b=500):
    """
    Picks two nodes by index, safely.
    """
    if len(nodes) == 0:
        return None, None
    a = nodes[min(idx_a, len(nodes) - 1)]
    b = nodes[min(idx_b, len(nodes) - 1)]
    return a, b


def run_sanity_check(region_name: str, graph_path: Path):
    print("\n" + "=" * 80)
    print(f"SANITY CHECK — {region_name.upper()}")
    print("=" * 80)

    if not graph_path.exists():
        print(f"⚠️ Graph file not found: {graph_path}")
        return

    # Load graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print("Graph path:", graph_path)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # Pick start/goal safely
    nodes = list(G.nodes)
    start, goal = pick_two_nodes(nodes, 0, 500)

    if start is None or goal is None:
        print("⚠️ Could not pick nodes for test (graph empty).")
        return

    # Shortest path check (uses edge "length" if available)
    # If "length" doesn't exist, fallback to unweighted shortest path.
    try:
        path = nx.shortest_path(G, start, goal, weight="length")
        weight_used = "length"
    except Exception:
        path = nx.shortest_path(G, start, goal)
        weight_used = "unweighted"

    print(f"Shortest path computed ({weight_used}). Path nodes:", len(path))

    # Coarse direction of first move
    if len(path) >= 2:
        first_step = path[1]
        direction = coarse_direction(G, start, first_step)
        print(f"Navigation decision: first step goes {direction}")
    else:
        print("⚠️ Path length < 2 (start==goal or weird graph).")


def main():
    for region, path in GRAPHS.items():
        run_sanity_check(region, path)


if __name__ == "__main__":
    main()
