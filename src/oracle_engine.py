import pickle
import pandas as pd
import networkx as nx
import config
import src.utils as utils
import re
import sys
from typing import Tuple
from geopy.distance import geodesic
import math

# --- PANDAS 2.0 COMPATIBILITY PATCH ---
import pandas.core.indexes.base
if not hasattr(pandas.core.indexes, 'numeric'):
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pd.Index

class OracleEngine:
    def __init__(self, graph_path_or_obj, poi_path_or_df):
        # 1. Load Graph
        if isinstance(graph_path_or_obj, str):
            with open(graph_path_or_obj, 'rb') as f:
                self.G = pickle.load(f)
        else:
            self.G = graph_path_or_obj

        # 2. Load POI
        if isinstance(poi_path_or_df, str):
            if poi_path_or_df.endswith('.pkl'):
                self.poi_df = pd.read_pickle(poi_path_or_df)
            else:
                self.poi_df = pd.read_csv(poi_path_or_df)
        else:
            self.poi_df = poi_path_or_df

        self.prefix = config.get_node_prefix()
        self._prepare_poi_data()

    def _prepare_poi_data(self):
        """
        V4 Optimization: Pre-extracts coordinates and cleans all high-density columns
        once during initialization to make the search loop fly.
        """
        geo_col = None
        if 'centroid' in self.poi_df.columns:
            geo_col = 'centroid'
        elif 'geometry' in self.poi_df.columns:
            geo_col = 'geometry'

        if geo_col:
            # DELETE PRINT LATER
            print(f"📍 Extracting coordinates from {config.CURRENT_CITY} '{geo_col}' column...")
            # Some objects are Points (have .x), some are Polygons (need .centroid.x)
            self.poi_df['x'] = self.poi_df[geo_col].apply(lambda p: p.x if hasattr(p, 'x') else p.centroid.x)
            self.poi_df['y'] = self.poi_df[geo_col].apply(lambda p: p.y if hasattr(p, 'y') else p.centroid.y)
        else:
            print(f"⚠️ Warning: No spatial column found for {config.CURRENT_CITY}!")

        self.prefix = config.get_node_prefix()

        # Clean all relevant search columns (The 'Vocabulary' Expansion)
        search_cols = ['name', 'amenity', 'shop', 'tourism', 'leisure', 'historic', 'man_made']
        for col in search_cols:
            if col in self.poi_df.columns:
                self.poi_df[f'clean_{col}'] = (
                    self.poi_df[col]
                    .astype(str)
                    .str.replace(r'[^a-zA-Z0-9]', '', regex=True)
                    .str.lower()
                )
            else:
                self.poi_df[f'clean_{col}'] = ""

    def resolve_landmark(self, landmark_name: str, context_node: str = None, radius_m: float = 1500.0) -> str:
        """
        The "Proximity-Aware" Oracle. Matches human-readable names to Node IDs 
        within a specific radius of a context node (usually gold_goal_node).
        """
        # 1. Clean input
        target = re.sub(r'[^a-zA-Z0-9]', '', landmark_name).lower()
        if not target:
            return None

        # 2. Define Search Area (Bounding Box)
        deg_buffer = (radius_m / config.METERS_PER_DEGREE_LATITUDE) 
        
        if context_node and context_node in self.G.nodes:
            goal_lat = self.G.nodes[context_node]['y']
            goal_lon = self.G.nodes[context_node]['x']
            
            nearby_df = self.poi_df[
                (self.poi_df['y'] >= goal_lat - deg_buffer) & (self.poi_df['y'] <= goal_lat + deg_buffer) &
                (self.poi_df['x'] >= goal_lon - deg_buffer) & (self.poi_df['x'] <= goal_lon + deg_buffer)
            ].copy()
        else:
            nearby_df = self.poi_df

        if nearby_df.empty:
            return None

        # 3. Deep Column Search (This was the missing piece!)
        match_mask = (
            nearby_df['clean_name'].str.contains(target, na=False) |
            nearby_df['clean_amenity'].str.contains(target, na=False) |
            nearby_df['clean_shop'].str.contains(target, na=False) |
            nearby_df['clean_tourism'].str.contains(target, na=False) |
            nearby_df['clean_leisure'].str.contains(target, na=False) |
            nearby_df['clean_historic'].str.contains(target, na=False) |
            nearby_df['clean_man_made'].str.contains(target, na=False)
        )
        
        candidates = nearby_df[match_mask].copy()

        # 4. Process Results with Fuzzy Prefix Fix
        if not candidates.empty:
            if context_node:
                candidates['dist'] = ((candidates['y'] - goal_lat)**2 + (candidates['x'] - goal_lon)**2)**0.5
                best_match = candidates.sort_values('dist').iloc[0]
            else:
                best_match = candidates.iloc[0]
            
            # --- START OF FUZZY PREFIX FIX ---
            raw_id = str(best_match['osmid']).replace('#', '').strip()
            
            # These are the 3 variations we found in your Philly Graph diagnostic
            possible_node_ids = [f"1#{raw_id}", f"#{raw_id}", raw_id]
            
            # Find the first one that actually exists in the Graph
            target_node = next((node for node in possible_node_ids if node in self.G.nodes), None)
            
            if target_node:
                return target_node
            # --- END OF FUZZY PREFIX FIX ---

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
    
    def resolve_by_tags(self, current_node_id: str, tags: dict, landmark_name: str = "") -> str:
        curr_node_data = self.G.nodes[current_node_id]
        curr_x, curr_y = curr_node_data['x'], curr_node_data['y']

        # 1. CATEGORY FILTER (No .copy(), using boolean indexing)
        # This narrows 7000 rows down to ~100-300 rows instantly
        mask = pd.Series(True, index=self.poi_df.index)
        for key, value in tags.items():
            if key in self.poi_df.columns:
                mask &= (self.poi_df[key].isin(value) if isinstance(value, list) else self.poi_df[key] == value)
        
        filtered_df = self.poi_df[mask]
        if filtered_df.empty: return None

        # 2. VECTORIZED DISTANCE (Using pre-calculated floats)
        dist = ((filtered_df['x'] - curr_x)**2 + (filtered_df['y'] - curr_y)**2)**0.5
        spatial_score = 1 / (1 + dist * 1000)

        # 3. SNIPER SEMANTIC SEARCH
        semantic_score = 0
        if landmark_name:
            query = landmark_name.lower().replace(" ", "_")
            
            # 3a. Primary Check: Name (High Weight)
            name_hit = filtered_df['name'].str.lower().str.contains(query.replace("_", " "), na=False)
            
            # 3b. Secondary Check: Only columns that our notebook proved are useful (updated based on V2 Discovery Analysis)
            search_cols = [
                'amenity', 'shop', 'description', 'cuisine', 'office', 
                'brand', 'tourism', 'leisure', 'building', 'artwork_type']
            tag_hit = pd.Series(False, index=filtered_df.index)
            
            for col in search_cols:
                if col in filtered_df.columns:
                    tag_hit |= filtered_df[col].str.lower().str.contains(query, na=False)
            
            semantic_score = (name_hit.astype(float) * 2.5) + (tag_hit.astype(float) * 1.5)

        # 4. Final Rank
        # Combining scores without adding columns to the DF (saves memory)
        final_scores = spatial_score + semantic_score
        nearest_idx = final_scores.idxmax()
            
        raw_osmid = str(filtered_df.loc[nearest_idx, 'osmid']).replace('#', '').strip()
        possible_node_ids = [f"1#{raw_osmid}", f"#{raw_osmid}", raw_osmid]
        
        return next((node for node in possible_node_ids if node in self.G.nodes), None)
    
    # --------- DIAGNOSTIC METHODS ---------

    def resolve_all_candidates(self, tags: dict, landmark_name: str = "", score_threshold: float = 0.5, bounds: tuple = None) -> list:
        """
        Diagnostic Method: Finds ALL possible landmarks matching constraints.
        Optimization: Uses pre-calculated 'x' and 'y' for fast spatial pruning.
        """
        df = self.poi_df
        
        # --- 1. SMART SPATIAL PRUNING ---
        # We check if the bounds provided match the city the data is currently in.
        filtered_df = df
        if bounds:
            lat_min, lat_max, lon_min, lon_max = bounds
            data_lat_avg = df['y'].mean()
            
            # City detection logic:
            # Philly: ~39.9 | Pittsburgh: ~40.4 | Manhattan: ~40.7            
            is_pgh_or_philly_data = data_lat_avg < 40.6
            is_pgh_or_philly_search = lat_min < 40.6

            # Only prune if the search city matches the data city
            if is_pgh_or_philly_data == is_pgh_or_philly_search:
                filtered_df = df[
                    (df['y'] >= lat_min) & (df['y'] <= lat_max) &
                    (df['x'] >= lon_min) & (df['x'] <= lon_max)
                ].copy()
            else:
                # If they don't match, we skip pruning so we don't get 0 hits.
                # This allows Manhattan tests to run on Manhattan data 
                # and PGH tests to run on PGH data without interference.
                filtered_df = df.copy()

        if filtered_df.empty: return []

        # --- 2. WEIGHTED SCORING (The "Love, Pittsburgh" Fix) ---
        query = str(landmark_name).lower()
        safe_query = re.escape(query)
        name_mask = filtered_df['name'].fillna("").str.lower().str.contains(safe_query, na=False, regex=True)


        cat_mask = pd.Series(False, index=filtered_df.index)
        if tags:
            for k, v in tags.items():
                if k in filtered_df.columns:
                    matches = filtered_df[k].isin(v) if isinstance(v, list) else (filtered_df[k] == v)
                    cat_mask |= matches.fillna(False)

        # Apply scores to a copy to keep original data clean
        res_df = filtered_df.copy()
        res_df['score'] = 0.0
        res_df.loc[name_mask, 'score'] += 2.5
        res_df.loc[cat_mask, 'score'] += 1.5
        
        # --- 3. FILTER & FORMAT ---
        hits = res_df[res_df['score'] >= score_threshold].copy()
        
        results = []
        for _, row in hits.iterrows():
            results.append({
                "node_id": str(row.get('osmid', 'unknown')),
                "name": row.get('name', 'Unknown'),
                "score": row['score'],
                "coords": (row['y'], row['x'])
            })

        # Sort by best score
        return sorted(results, key=lambda x: x['score'], reverse=True)
        

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculates geodesic distance (meters) to match RVS path constraints."""
        return geodesic((lat1, lon1), (lat2, lon2)).meters

    
    def resolve_nearby_candidates(self, tags, center_lat, center_lon, radius_m=1500, landmark_name=""):
        # 1. Standard Latitude Buffer: 1 degree lat is ~111,000 meters
        lat_buffer = (radius_m / config.METERS_PER_DEGREE_LATITUDE)
        
        # 2. Adjusted Longitude Buffer (Accounting for Earth's curvature)
        # Longitude shrinks as you move toward the poles
        # cos(radians(lat)) scales the degree width correctly
        lon_buffer = radius_m / (config.METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(center_lat)))        
        
        bounds = (
            center_lat - lat_buffer, 
            center_lat + lat_buffer, 
            center_lon - lon_buffer, 
            center_lon + lon_buffer
        )

        # Pass the bounds into the optimized resolver
        candidates = self.resolve_all_candidates(
            tags=tags, 
            landmark_name=landmark_name,
            bounds=bounds
        )

        nearby = []
        for c in candidates:
            # High precision check only for those inside the "Rough Box"
            dist = self.calculate_distance(center_lat, center_lon, c['coords'][0], c['coords'][1])
            if dist <= radius_m:
                nearby.append(c)
        return nearby