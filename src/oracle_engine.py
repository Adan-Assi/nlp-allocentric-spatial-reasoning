import pickle
import pandas as pd
import networkx as nx
import config
import src.utils as utils
import re
from typing import Tuple

class OracleEngine:
    def __init__(self, graph_path_or_obj, poi_path_or_df):
        # Handle Graph (Path or Object)
        if isinstance(graph_path_or_obj, str):
            print(f"Loading graph via pickle from {graph_path_or_obj}...")
            with open(graph_path_or_obj, 'rb') as f:
                self.G = pickle.load(f)
        else:
            self.G = graph_path_or_obj

        # Handle POI (Path or DataFrame)
        if isinstance(poi_path_or_df, str):
            if poi_path_or_df.endswith('.pkl'):
                print(f"Loading POIs via pickle from {poi_path_or_df}...")
                self.poi_df = pd.read_pickle(poi_path_or_df)
            else:
                self.poi_df = pd.read_csv(poi_path_or_df)
        else:
            self.poi_df = poi_path_or_df

        self.prefix = config.POI_NODE_PREFIX

        # Now this will work because self.poi_df is definitely a DataFrame
        self.poi_df['clean_name'] = (
            self.poi_df['name']
            .str.replace(r'[^a-zA-Z0-9]', '', regex=True)
            .str.lower()
        )

    
    def resolve_landmark(self, landmark_name):
        """
        Translates a human-readable string into a valid Graph Node ID.
        
        Args:
            landmark_name (str): The name of the landmark (e.g., "Hell's Kitchen").
            
        Returns:
            str or None: The projected Node ID (e.g., "1#666") if found in G, else None.
        """
        # 1. Normalize the INPUT string to match our pre-calculated format
        # This turns "Hell's Kitchen" into "hellskitchen"
        clean_input = re.sub(r'[^a-zA-Z0-9]', '', landmark_name).lower()
        
        if not clean_input:
            return None

        # 2. Use the pre-calculated search column from __init__
        search_col = self.poi_df['clean_name']

        # --- STEP 2: Exact Search ---
        matches = self.poi_df[search_col == clean_input]

        # --- STEP 3: Partial Search (Fallback) ---
        if matches.empty:
            matches = self.poi_df[search_col.str.contains(clean_input, na=False)]

        # --- STEP 4: OSMID Extraction & Bridge Construction ---
        if not matches.empty:
            # We take the first match. In your data, matches.iloc[0] is the safest bet.
            osmid = str(matches.iloc[0]['osmid']).replace('#', '')
            target_node = f"{self.prefix}{osmid}"

            # --- STEP 5: Graph Verification ---
            if target_node in self.G.nodes:
                return target_node

        return None

    def verify_proximity(self, agent_node, landmark_name):
        """
        Checks if the agent is 'at' or 'near' a specific landmark.
        """
        target_node = self.resolve_landmark(landmark_name)

        if not target_node:
            return False, 0.0, None
        
        is_near = utils.is_within_buffer(self.G, agent_node, target_node)
        distance = utils.get_euclidean_dist(self.G, agent_node, target_node)
        
        return is_near, distance, target_node
    

    def get_candidate_nodes(self, landmark_name: str, radius_m: float = 500.0) -> list:
        """
        Finds all graph nodes within a radius of a named landmark.
        
        Args:
            landmark_name: e.g., "Hell's Kitchen"
            radius_m: Search radius in meters (default 500m)
            
        Returns:
            List of Node IDs (e.g., ['1#101', '1#102'])
        """
        # 1. Resolve the name to a center point node
        center_node = self.resolve_landmark(landmark_name)
        if not center_node:
            print(f"⚠️ Oracle: Cannot generate candidates for unknown landmark: {landmark_name}")
            return []

        # 2. Iterate through graph nodes and filter by distance
        candidates = []
        for node_id in self.G.nodes:
            # We use the high-precision geodesic math from utils
            dist = utils.get_euclidean_dist(self.G, center_node, node_id)
            if dist <= radius_m:
                candidates.append(node_id)
        
        return candidates
    

    def filter_candidates_by_direction(self, origin_node: str, candidate_ids: list, target_direction: str) -> list:
        # 1. Standardize
        target_direction = target_direction.strip().upper()[0] # Gets 'N', 'S', etc.
        
        # 2. Get Origin Coords
        origin_data = self.G.nodes[origin_node]
        lat1, lon1 = origin_data['y'], origin_data['x']

        kept = []
        for node_id in candidate_ids:
            node_data = self.G.nodes[node_id]
            # 3. Call utils directly
            actual_dir = utils.get_dominant_direction(
                lat1, lon1, node_data['y'], node_data['x']
            )
            
            if actual_dir == target_direction:
                kept.append(node_id)
        return kept
    

    def find_nearest_node(self, lat: float, lon: float) -> Tuple[str, float]:
        """
        Finds the closest graph node to a given (lat, lon).
        Moved from Solver to Oracle as it is a spatial grounding task.
        """
        best_node = None
        min_dist = float('inf')

        for node_id, data in self.G.nodes(data=True):
            # Use our unified geodesic math from utils
            d = utils.get_geodesic_dist_raw(lat, lon, data['y'], data['x'])
            if d < min_dist:
                min_dist = d
                best_node = node_id
        
        return best_node, min_dist