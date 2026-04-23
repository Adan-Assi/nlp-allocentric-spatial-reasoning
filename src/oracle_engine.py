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
import numpy as np
from scipy.spatial import KDTree

# --- PANDAS 2.0 COMPATIBILITY PATCH ---
import pandas.core.indexes.base
if not hasattr(pandas.core.indexes, 'numeric'):
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pd.Index

class OracleEngine:
    def __init__(self, graph_path_or_obj, poi_path_or_df, node_prefix, city_name):
        
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

        self.prefix = node_prefix
        self.city_name = city_name
        
        self._prepare_poi_data()

        # --- ADDED FOR WARP SPEED: GRAPH SPATIAL INDEX ---
        # 3. Build a KDTree for all nodes in the Graph
        # This prevents the slow 'for node in G.nodes' loops (specficially encountered with Msnhattan)
        self.node_ids = list(self.G.nodes())
        # Create a matrix of [Lat, Lon] for every node
        coords = np.array([[self.G.nodes[n]['y'], self.G.nodes[n]['x']] for n in self.node_ids])
        self.node_tree = KDTree(coords)
        
        # 4. Build a KDTree for POIs
        # This makes resolve_landmark much faster
        self.poi_coords = self.poi_df[['y', 'x']].values
        self.poi_tree = KDTree(self.poi_coords)
        print(f"DEBUG: POI Tree Bounds - Lat: {self.poi_coords[:,0].min()} to {self.poi_coords[:,0].max()}")
        print(f"DEBUG: POI Tree Bounds - Lon: {self.poi_coords[:,1].min()} to {self.poi_coords[:,1].max()}")

        # --- ADDED FOR O(1) REACHABILITY: SCC LOOKUP ---
        # Manhattan's graph is massive; nx.has_path() will take ~1s per call otherwise.
        # Centralized in utils to ensure undirected logic is applied
        
        print(f"DEBUG: Calculating Connectivity Map for {config.CURRENT_CITY}...", flush=True)
        self.scc_lookup, num_comp = utils.get_connectivity_map(self.G)
        
        print(f"DEBUG: Oracle initialized with {len(self.node_ids)} nodes and {num_comp} components.", flush=True)
        print(f"DEBUG: [CITY: {self.city_name}] Using Salience Ratio: {config.get_salience_ratio()}")


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
    

    def get_graph_node(self, rvs_id: str) -> str:
        """
        Attempts to resolve an RVS Node ID to a valid Graph Node ID.
        Handles the prefix inconsistencies (#, 1#) found in Philly and Pittsburgh.
        """
        if not rvs_id:
            return None
            
        # Clean the input ID
        raw_id = str(rvs_id).replace('#', '').strip()
        
        # Check the 3 known variations in order of likelihood
        possible_ids = [f"1#{raw_id}", f"#{raw_id}", raw_id]
        
        for variant in possible_ids:
            if variant in self.G.nodes:
                return variant
                
        return None


    def resolve_landmark(self, landmark_name: str, context_node: str = None, radius_m: float = 1500.0) -> str:
        """
        The "Proximity-Aware" Oracle. Matches human-readable names to Node IDs 
        within a specific radius of a context node.
        """
        # 1. Clean input
        target = re.sub(r'[^a-zA-Z0-9]', '', landmark_name).lower()
        if not target:
            return None

        # 2. Define Search Area using KDTree
        # Replaces the previous Pandas Bounding Box filter, which was too slow on large datasets like Manhattan
        if context_node and context_node in self.G.nodes:
            goal_lat = self.G.nodes[context_node]['y']
            goal_lon = self.G.nodes[context_node]['x']
            
            # 1 degree lat is ~111,000 meters. 
            # We query the ball to get INDICES of nearby rows instantly.
            deg_buffer = (radius_m / config.METERS_PER_DEGREE_LATITUDE) 
            indices = self.poi_tree.query_ball_point([goal_lat, goal_lon], r=deg_buffer)
            
            # Select only the relevant rows without scanning the whole DF
            nearby_df = self.poi_df.iloc[indices].copy()
        else:
            nearby_df = self.poi_df

        if nearby_df.empty:
            return None

        # 3. Deep Column Search
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
            
            # These are the 3 variations we found in the Philly Graph diagnostic
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

        # 3. --- FAST SPATIAL QUERY ---
        # Convert meters to approximate degrees (rough estimate for the tree)
        # 111,000 meters is approx 1 degree
        degree_radius = radius_m / config.METERS_PER_DEGREE_LATITUDE
        
        # This query returns indices of nodes within the 'rough' box instantly
        indices = self.node_tree.query_ball_point([lat1, lon1], r=degree_radius)

        candidates = []
        for idx in indices:
            node_id = self.node_ids[idx]
            
            # 4. Filter: Skip POI nodes (This is the 'Scientific' Layer that matches human perception of 'streets' vs 'landmarks')
            if str(node_id).startswith(self.prefix):
                continue
                
            # 5. Precise Verification (Only on the handful of nearby nodes)
            node_data = self.G.nodes[node_id]
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
        """
        # Query the KDTree for the nearest node (returns distance in degrees and index)
        dist_deg, idx = self.node_tree.query([lat, lon])
        node_id = self.node_ids[idx]
        
        # Convert to meters using our high-precision utility
        # This keeps the 'grounding' logic compatible with the rest of the pipeline
        actual_dist_m = utils.get_geodesic_dist_raw(
            lat, lon, 
            self.G.nodes[node_id]['y'], 
            self.G.nodes[node_id]['x']
        )
        
        return node_id, actual_dist_m

    
    def resolve_by_tags(self, current_node_id: str, tags: dict, landmark_name: str = "") -> str:
        """
        Advanced Landmark Resolver: Combines spatial pruning, category filtering, and semantic search.
        This is the "Love, Pittsburgh" Fix that handles ambiguous instructions by using all available signals
        """
        curr_node_data = self.G.nodes[current_node_id]
        curr_x, curr_y = curr_node_data['x'], curr_node_data['y']

        # --- OPTIMIZATION: SPATIAL PRE-FILTER ---
        # 1. Using the newly added POI tree to only consider things within a reasonable distance (e.g. 1500m)
        # This reduces Manhattan (50k rows) down to a few hundred before doing ANY logic.
        deg_radius = config.GLOBAL_SEARCH_HORIZON_METERS / config.METERS_PER_DEGREE_LATITUDE
        nearby_indices = self.poi_tree.query_ball_point([curr_y, curr_x], r=deg_radius)
        
        if not nearby_indices:
            return None
            
        # Create the small subset we actually care about
        subset_df = self.poi_df.iloc[nearby_indices].copy()

        # 2. CATEGORY FILTER (On the subset)
        mask = pd.Series(True, index=subset_df.index)
        for key, value in tags.items():
            if key in subset_df.columns:
                mask &= (subset_df[key].isin(value) if isinstance(value, list) else subset_df[key] == value)
        
        filtered_df = subset_df[mask]
        if filtered_df.empty: 
            return None

        # 3. VECTORIZED DISTANCE (On the filtered subset)
        dist = ((filtered_df['x'] - curr_x)**2 + (filtered_df['y'] - curr_y)**2)**0.5
        spatial_score = 1 / (1 + dist * 1000)

        # 4. SNIPER SEMANTIC SEARCH (Only on filtered subset)
        semantic_score = 0
        if landmark_name:
            query = landmark_name.lower().replace(" ", "_")
            name_hit = filtered_df['name'].str.lower().str.contains(query.replace("_", " "), na=False)
            
            search_cols = [
                'amenity', 'shop', 'description', 'cuisine', 'office', 
                'brand', 'tourism', 'leisure', 'building', 'artwork_type']
            tag_hit = pd.Series(False, index=filtered_df.index)
            
            for col in search_cols:
                if col in filtered_df.columns:
                    tag_hit |= filtered_df[col].str.lower().str.contains(query, na=False)
            
            semantic_score = (name_hit.astype(float) * 2.5) + (tag_hit.astype(float) * 1.5)

        # 5. Final Rank
        final_scores = spatial_score + semantic_score
        nearest_idx = final_scores.idxmax()
            
        raw_osmid = str(filtered_df.loc[nearest_idx, 'osmid']).replace('#', '').strip()
        possible_node_ids = [f"1#{raw_osmid}", f"#{raw_osmid}", raw_osmid]
        
        return next((node for node in possible_node_ids if node in self.G.nodes), None)


    def resolve_all_candidates(self, tags: dict, landmark_name: str = "", score_threshold: float = 0.5, bounds: tuple = None) -> list:
        """
        City-Agnostic Candidate Resolution.
        Uses Spatial Indexing (KDTree) for pruning and config-driven logic.
        """
        df = self.poi_df
        
        # --- 1. SPATIAL PRUNING (City-Agnostic) ---
        if bounds:
            lat_min, lat_max, lon_min, lon_max = bounds
            
            # Instead of checking lat < 40.6, we check if the bounds 
            # actually overlap with the data currently loaded in this Oracle.
            data_lat_min, data_lat_max = df['y'].min(), df['y'].max()
            data_lon_min, data_lon_max = df['x'].min(), df['x'].max()

            # Only prune if the requested bounds overlap with our city's data
            # This handles the "Philly search on Manhattan data" edge case dynamically
            if not (lat_max < data_lat_min or lat_min > data_lat_max or 
                    lon_max < data_lon_min or lon_min > data_lon_max):
                
                # Use KDTree for instant subsetting
                center_y = (lat_min + lat_max) / 2
                center_x = (lon_min + lon_max) / 2
                # Calculate search radius for the tree (half-diagonal of the box)
                r_deg = (((lat_max - lat_min)**2 + (lon_max - lon_min)**2)**0.5) / 2
                
                indices = self.poi_tree.query_ball_point([center_y, center_x], r=r_deg)
                filtered_df = df.iloc[indices].copy()
                
                # Precise crop to the exact bounding box
                filtered_df = filtered_df[
                    (filtered_df['y'] >= lat_min) & (filtered_df['y'] <= lat_max) &
                    (filtered_df['x'] >= lon_min) & (filtered_df['x'] <= lon_max)
                ]
            else:
                # Outside city data range
                # Usually, if bounds don't match data, we shouldn't scan the whole city
                return []
        else:
            filtered_df = df.copy()

        if filtered_df.empty: return []

        # --- 2. WEIGHTED SCORING (Scalable Column Logic) ---
        query = str(landmark_name).lower()
        safe_query = re.escape(query)
        
        # Use columns defined in config.POI_SEARCH_COLUMNS for portability
        search_cols = getattr(config, 'POI_SEARCH_COLUMNS', ['name'])
        
        # Calculate scores without hardcoding column names
        name_mask = filtered_df['name'].fillna("").str.lower().str.contains(safe_query, na=False)
        
        cat_mask = pd.Series(False, index=filtered_df.index)
        if tags:
            for k, v in tags.items():
                if k in filtered_df.columns:
                    matches = filtered_df[k].isin(v) if isinstance(v, list) else (filtered_df[k] == v)
                    cat_mask |= matches.fillna(False)

        res_df = filtered_df.copy()
        res_df['score'] = 0.0
        res_df.loc[name_mask, 'score'] += 2.5
        res_df.loc[cat_mask, 'score'] += 1.5
        
        # --- 3. FILTER & FORMAT ---
        hits = res_df[res_df['score'] >= score_threshold]
        
        results = []
        for _, row in hits.iterrows():
            osmid = str(row.get('osmid', 'unknown')).replace('#', '').strip()
            node_id = f"1#{osmid}" if f"1#{osmid}" in self.G.nodes else osmid
            
            results.append({
                "node_id": node_id, 
                "name": row.get('name', 'Unknown'),
                "score": row['score'],
                "coords": (row['y'], row['x'])
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)


    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculates geodesic distance (meters) to match RVS path constraints."""
        return geodesic((lat1, lon1), (lat2, lon2)).meters

    
    def resolve_nearby_candidates(self, tags, center_lat, center_lon, radius_m=1500, landmark_name=""):
        # Standard Latitude Buffer: 1 degree lat is ~111,000 meters
        lat_buffer = (radius_m / config.METERS_PER_DEGREE_LATITUDE)
        
        # Adjusted Longitude Buffer (Accounting for Earth's curvature)
        # Longitude shrinks as we move toward the poles
        # cos(radians(lat)) scales the degree width correctly
        lon_buffer = radius_m / (config.METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(center_lat)))        
        
        bounds = (
            center_lat - lat_buffer, center_lat + lat_buffer, 
            center_lon - lon_buffer, center_lon + lon_buffer
        )

        # 1. Get candidates in the "Rough Box" (O(log N) via KDTree)
        candidates = self.resolve_all_candidates(
            tags=tags, landmark_name=landmark_name, bounds=bounds
        )
        if not candidates:
            return []

        # VECTORIZED DISTANCE FILTER (Hopefully the 1.93s/it Fix)
        # Extract all lats and lons into numpy arrays
        cand_lats = np.array([c['coords'][0] for c in candidates])
        cand_lons = np.array([c['coords'][1] for c in candidates])
        
        # Calculate all distances in a single clock cycle
        distances = utils.haversine_vectorized(center_lat, center_lon, cand_lats, cand_lons)
        
        # Filter using the results
        nearby = []
        for i, c in enumerate(candidates):
            if distances[i] <= radius_m:
                c['dist'] = distances[i] # Useful for sorting later
                nearby.append(c)
                
        return nearby