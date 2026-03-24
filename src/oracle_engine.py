import pickle
import pandas as pd
import networkx as nx
import config
import src.utils as utils
import re
import sys # Added for path mapping
from typing import Tuple
from geopy.distance import geodesic

# --- PANDAS 2.0 COMPATIBILITY PATCH ---
# This fixes the 'ModuleNotFoundError' and 'AttributeError' for older .pkl files
import pandas.core.indexes.base
if not hasattr(pandas.core.indexes, 'numeric'):
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pd.Index
# --------------------------------------

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
    

    def get_candidates_within_radius(self, origin: str, radius_m: float = 500.0) -> list:
            """
            Task 2.2 Integration: Finds all street nodes within a buffered radius.
            Accepts: Node ID (e.g., '101') or Landmark Name (e.g., 'Cafe').
            """
            # 1. Resolve Origin to a Node ID
            if origin in self.G:
                center_node = origin
            else:
                center_node = self.resolve_landmark(origin)
                
            if not center_node:
                return []

            # 2. Extract center coordinates once
            lat1, lon1 = utils.get_node_coords(self.G, center_node)

            candidates = []
            for node_id, node_data in self.G.nodes(data=True):
                # 3. Filter: Skip POI nodes, we want walkable street nodes
                if str(node_id).startswith(self.prefix):
                    continue
                    
                # 4. Use high-precision geodesic distance
                dist = utils.get_geodesic_dist_raw(
                    lat1, lon1, node_data['y'], node_data['x']
                )
                
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
    
    
    def resolve_by_tags_old(self, current_node_id: str, tags: dict, landmark_name: str = "") -> str:
        """
        Task 3.1 Optimized: Semantic-aware resolution.
        Balances tag categories with name-based keyword verification.
        """
        curr_node_data = self.G.nodes[current_node_id]
        curr_x, curr_y = curr_node_data['x'], curr_node_data['y']

        # 1. Broad Filter by Tags (The Category)
        filtered_df = self.poi_df.copy()
        for key, value in tags.items():
            if key in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]

        if filtered_df.empty:
            return None

        # 2. Distance Calculation (Spatial Score)
        # We normalize distance so closer = higher score (0 to 1 range)
        dist = ((filtered_df.geometry.x - curr_x)**2 + (filtered_df.geometry.y - curr_y)**2)**0.5
        spatial_score = 1 / (1 + dist * 1000) # Simple inverse decay

        # 3. Enhanced Semantic Score (Deep Search)
        semantic_score = 0
        if landmark_name:
            query = landmark_name.lower()
            
            # Layer A: Direct Name Match (+2.0 points)
            name_match = filtered_df['name'].str.lower().str.contains(query, na=False)
            
            # Layer B: Tag-Value Match (+1.5 points)
            # Check if our query exists inside the values of the columns (like 'cuisine' or 'office')
            # This catches "dim_sum" in the cuisine column even if not in the name.
            tag_match = filtered_df.apply(
                lambda row: any(query in str(val).lower() for val in row.values), axis=1
            )
            
            # Combine them: Name matches are strongest, Tag matches are second strongest
            semantic_score = (name_match.astype(float) * 2.0) + (tag_match.astype(float) * 1.5)

        # 4. Final Ranking
        # Combined score ensures a slightly further 'Church' beats a closer 'Synagogue'
        filtered_df['final_score'] = spatial_score + semantic_score
        
        nearest_idx = filtered_df['final_score'].idxmax()
        
        # 5. ID Construction
        raw_osmid = str(filtered_df.loc[nearest_idx, 'osmid']).replace('#', '')
        return f"{self.prefix}{raw_osmid}"
    
    
    def resolve_by_tags(self, current_node_id: str, tags: dict, landmark_name: str = "") -> str:
        curr_node_data = self.G.nodes[current_node_id]
        curr_x, curr_y = curr_node_data['x'], curr_node_data['y']

        # 1. Initial Category Filter (e.g., 'amenity': 'restaurant')
        filtered_df = self.poi_df.copy()
        for key, value in tags.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value] if not isinstance(value, list) else filtered_df[filtered_df[key].isin(value)]

        if filtered_df.empty: return None

        # 2. Distance Score
        poi_centroids = filtered_df.geometry.centroid
        dist = ((poi_centroids.x - curr_x)**2 + (poi_centroids.y - curr_y)**2)**0.5
        spatial_score = 1 / (1 + dist * 1000)

        # 3. Deep Semantic Intent Search
        semantic_score = 0
        if landmark_name:
            query = landmark_name.lower().replace(" ", "_") # Handle "dim sum" -> "dim_sum"
            
            # Check the 'name' first (High priority: 2.0)
            name_hit = filtered_df['name'].str.lower().str.contains(query.replace("_", " "), na=False)
            
            # Check ALL other columns for the intent (Medium priority: 1.5)
            # This catches things in 'cuisine', 'shop', or 'description'
            tag_hit = filtered_df.apply(lambda row: any(query in str(val).lower() for val in row.values), axis=1)
            
            semantic_score = (name_hit.astype(float) * 2.0) + (tag_hit.astype(float) * 1.5)

        # 4. Final Rank
        filtered_df['final_score'] = spatial_score + semantic_score
        nearest_idx = filtered_df['final_score'].idxmax()
        
        raw_osmid = str(filtered_df.loc[nearest_idx, 'osmid']).replace('#', '')
        return f"{self.prefix}{raw_osmid}"

    
    def resolve_all_candidates(self, tags: dict, landmark_name: str = "", score_threshold: float = 0.5) -> list:
        """
        Diagnostic Method: Finds ALL possible landmarks matching the constraints.
        Essential for labeling 'Ambiguous' instructions in the 1B Stress Test.
        """
        # 1. Category Filter (Same as resolve_by_tags)
        filtered_df = self.poi_df.copy()
        for key, value in tags.items():
            if key in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]

        if filtered_df.empty:
            return []

        # 2. Score calculation (No current_node_id needed here to keep it broad)
        semantic_scores = []
        if landmark_name:
            query = landmark_name.lower().replace(" ", "_")
            
            # Name match (2.0) and Tag match (1.5)
            name_hits = filtered_df['name'].str.lower().str.contains(query.replace("_", " "), na=False)
            tag_hits = filtered_df.apply(lambda row: any(query in str(val).lower() for val in row.values), axis=1)
            
            filtered_df['semantic_score'] = (name_hits.astype(float) * 2.0) + (tag_hits.astype(float) * 1.5)
        else:
            filtered_df['semantic_score'] = 1.0 # Default if no name provided

        # 3. Filter by Threshold instead of picking Max
        # This allows us to find multiple "Church" candidates on the same street
        candidates_df = filtered_df[filtered_df['semantic_score'] >= score_threshold]
        
        results = []
        for idx, row in candidates_df.iterrows():
            raw_osmid = str(row['osmid']).replace('#', '')
            node_id = f"{self.prefix}{raw_osmid}"
            if node_id in self.G:
                results.append({
                    "node_id": node_id,
                    "name": row['name'],
                    "score": row['semantic_score'],
                    # This is the 'Robust Grounding' logic required for scientific accuracy
                    # Centroid returns the center for Polygons (Type 2) 
                    # and the point itself for Points (Type 1).
                    "coords": (row.geometry.centroid.y, row.geometry.centroid.x)
                })
        
        return results
    
    from geopy.distance import geodesic

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculates geodesic distance (meters) to match RVS path constraints."""
        return geodesic((lat1, lon1), (lat2, lon2)).meters

    def resolve_nearby_candidates(self, tags, center_lat, center_lon, radius_m=1500):
        """
        Finds all landmarks matching tags within the RVS-standard distance.
        Uses .centroid to handle 'Type 2' large POIs mentioned in README.
        """
        candidates = self.resolve_all_candidates(tags)
        nearby = []
        for c in candidates:
            # README link: Using centroid for polygons/areas
            dist = self.calculate_distance(center_lat, center_lon, c['coords'][0], c['coords'][1])
            if dist <= radius_m:
                nearby.append(c)
        return nearby