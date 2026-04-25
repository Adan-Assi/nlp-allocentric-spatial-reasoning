from pathlib import Path
import osmnx as ox

OUT_DIR = Path("data/graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regions we want graphs for
REGIONS = {
    "manhattan": "Manhattan, New York City, New York, USA",
    "philadelphia": "Philadelphia, Pennsylvania, USA",
    "pittsburgh": "Pittsburgh, Pennsylvania, USA",
}

# Choose street network type.
# "walk" is often best for navigation instructions (pedestrian-friendly).
NETWORK_TYPE = "walk"

def main():
    for region_key, query in REGIONS.items():
        out_path = OUT_DIR / f"{region_key}.gpickle"
        print(f"\n=== Building graph for: {region_key} ===")
        print(f"Query: {query}")
        print(f"Saving to: {out_path}")

        # Download and build graph from OpenStreetMap
        G = ox.graph_from_place(query, network_type=NETWORK_TYPE, simplify=True)

        # Optional: keep largest connected component (usually cleaner)
        G = ox.truncate.largest_component(G, strongly=False)

        # Save as gpickle
        ox.save_graphml(G, OUT_DIR / f"{region_key}.graphml")  # optional debug format
        ox.save_graph_geopackage(G, OUT_DIR / f"{region_key}.gpkg")  # optional
        ox.save_graphml(G, OUT_DIR / f"{region_key}.graphml")  # safe re-save
        # gpickle via networkx (osmnx sometimes removed direct save in some versions)
        import pickle
        with open(out_path, "wb") as f:
            pickle.dump(G, f)

        print(f"✅ Saved: {out_path}")
        print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

if __name__ == "__main__":
    main()
