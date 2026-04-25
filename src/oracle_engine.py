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

        # --- GRAPH SPATIAL INDEX (built before POI prep so the snap-fallback
        # in _prepare_poi_data can resolve ways/relations to nearest graph node) ---
        self.node_ids = list(self.G.nodes())
        # Set form for O(1) membership in vectorized filters (resolve_all_candidates).
        self.graph_node_set = set(self.node_ids)
        # Lat/lon matrix for KDTree queries.
        coords = np.array([[self.G.nodes[n]['y'], self.G.nodes[n]['x']] for n in self.node_ids])
        self.node_tree = KDTree(coords)
        self._node_id_array = np.array(self.node_ids, dtype=object)

        self._prepare_poi_data()
        
        # 4. Build a KDTree for POIs
        # This makes resolve_landmark much faster
        self.poi_coords = self.poi_df[['y', 'x']].values
        self.poi_tree = KDTree(self.poi_coords)
        print(f"DEBUG: POI Tree Bounds - Lat: {self.poi_coords[:,0].min()} to {self.poi_coords[:,0].max()}")
        print(f"DEBUG: POI Tree Bounds - Lon: {self.poi_coords[:,1].min()} to {self.poi_coords[:,1].max()}")

        # --- ADDED FOR O(1) REACHABILITY: CONNECTIVITY LOOKUP ---
        # Manhattan's graph is massive; nx.has_path() will take ~1s per call otherwise.
        # We use WEAKLY connected components (pedestrian model — see utils.get_connectivity_map).
        
        print(f"DEBUG: Calculating Connectivity Map for {config.CURRENT_CITY}...", flush=True)
        self.conn_lookup, num_comp = utils.get_connectivity_map(self.G)
        
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

        # Pick the column carrying real OSM ids. RVS POI files store the OSM
        # node id under 'id' (large ints like 42437096); 'osmid' in this dataset
        # holds row-internal indices that don't match the graph. Prefer 'id'
        # and fall back to 'osmid' for legacy files.
        id_col = 'id' if 'id' in self.poi_df.columns else (
            'osmid' if 'osmid' in self.poi_df.columns else None
        )

        if id_col:
            # Build a reverse index raw_osm_id -> graph_node_id. The graph uses
            # several prefixes ('#', '1#', '2#', '4#', ...), so we can't just
            # try a fixed list of variants per POI — we'd miss every '2#'/'4#'
            # node. Walking G.nodes once gives the full mapping.
            #
            # Tiebreak when the same raw id appears under multiple prefixes
            # (e.g. both '1#666' and '#666' exist): prefer '1#' > '#' > others,
            # matching the priority of the previous variant-list approach.
            def _split(node_id):
                s = str(node_id)
                if '#' not in s:
                    return '', s
                p, r = s.split('#', 1)
                return p, r

            def _priority(prefix):
                if prefix == '1':
                    return 0
                if prefix == '':
                    return 1
                return 2

            reverse_index = {}
            for n in self.G.nodes():
                prefix, raw = _split(n)
                cur = reverse_index.get(raw)
                if cur is None or _priority(prefix) < _priority(_split(cur)[0]):
                    reverse_index[raw] = n

            def normalize_node_id(raw_id):
                if pd.isna(raw_id):
                    return None
                raw = str(raw_id).replace('#', '').strip()
                # Strip OSM type prefixes like 'node/', 'way/', 'relation/'
                for osm_prefix in ('node/', 'way/', 'relation/'):
                    if raw.startswith(osm_prefix):
                        raw = raw[len(osm_prefix):]
                        break
                # Returns None when the POI's OSM id is not present anywhere in
                # the graph — callers skip those rows.
                return reverse_index.get(raw)

            self.poi_df['graph_node_id'] = self.poi_df[id_col].apply(normalize_node_id)
            mapped_direct = self.poi_df['graph_node_id'].notna().sum()
            print(
                f"DEBUG: POI->graph direct id match via column '{id_col}': "
                f"{mapped_direct:,}/{len(self.poi_df):,} "
                f"({100*mapped_direct/len(self.poi_df):.1f}%)"
            )

        # --- Snap-fallback: ways/relations have OSM ids that don't appear in
        # the street graph (which only contains intersections). For those POIs
        # we snap their centroid to the nearest graph node within SNAP_MAX_M.
        # POIs whose centroid is farther than that are usually outside the
        # street network entirely (parks-as-polygons covering water, etc.) and
        # stay unmapped on purpose — snapping them would invent a fake target.
        SNAP_MAX_M = 100.0
        if (
            'graph_node_id' in self.poi_df.columns
            and 'x' in self.poi_df.columns
            and 'y' in self.poi_df.columns
        ):
            unmapped_mask = (
                self.poi_df['graph_node_id'].isna()
                & self.poi_df['x'].notna()
                & self.poi_df['y'].notna()
            )
            n_unmapped = int(unmapped_mask.sum())
            if n_unmapped > 0:
                coords = self.poi_df.loc[unmapped_mask, ['y', 'x']].to_numpy()
                # query_ball_point would be wasteful here; one nearest neighbor
                # per row is exactly what `query(..., k=1)` gives us, vectorized.
                _, idxs = self.node_tree.query(coords, k=1)
                snapped_node_ids = self._node_id_array[idxs]

                # Recompute distances in METERS so the threshold is honest
                # (KDTree distance is in degree-space, which over/under-fetches
                # depending on latitude — see utils.meters_to_degree_radius).
                snapped_lats = np.array(
                    [self.G.nodes[n]['y'] for n in snapped_node_ids]
                )
                snapped_lons = np.array(
                    [self.G.nodes[n]['x'] for n in snapped_node_ids]
                )
                dists_m = utils.haversine_vectorized(
                    coords[:, 0], coords[:, 1], snapped_lats, snapped_lons
                )

                accepted = dists_m <= SNAP_MAX_M
                final_ids = np.where(accepted, snapped_node_ids, None)
                self.poi_df.loc[unmapped_mask, 'graph_node_id'] = final_ids

                n_snapped = int(accepted.sum())
                total_mapped = int(self.poi_df['graph_node_id'].notna().sum())
                print(
                    f"DEBUG: POI->graph snap fallback: "
                    f"{n_snapped:,}/{n_unmapped:,} unmapped POIs snapped "
                    f"(within {SNAP_MAX_M:.0f}m). "
                    f"Total mapped: {total_mapped:,}/{len(self.poi_df):,} "
                    f"({100*total_mapped/len(self.poi_df):.1f}%)"
                )

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
        if not rvs_id:
            return None
        raw_id = str(rvs_id).replace('#', '').strip()
        for osm_prefix in ('node/', 'way/', 'relation/'):
            if raw_id.startswith(osm_prefix):
                raw_id = raw_id[len(osm_prefix):]
                break
        possible_ids = [f"1#{raw_id}", f"#{raw_id}", raw_id]
        for variant in possible_ids:
            if variant in self.G.nodes:
                return variant
        # Always print on miss - no counter limit
        print(f"MISS: input={rvs_id!r} tried={possible_ids}", flush=True)
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
            
            # Convert meters → degree radius using longitude-degree width at this
            # latitude so the KDTree ball fully contains the meter-circle (avoids
            # under-fetching in east-west direction at mid-latitudes).
            deg_buffer = utils.meters_to_degree_radius(radius_m, goal_lat)
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

        # Drop POIs whose OSM id could not be mapped to a graph node —
        # they're unusable downstream (no path can be computed to them).
        if not candidates.empty and 'graph_node_id' in candidates.columns:
            candidates = candidates[candidates['graph_node_id'].notna()]

        # 4. Process Results with Fuzzy Prefix Fix
        if not candidates.empty:
            if context_node:
                candidates['dist'] = ((candidates['y'] - goal_lat)**2 + (candidates['x'] - goal_lon)**2)**0.5
                best_match = candidates.sort_values('dist').iloc[0]
            else:
                best_match = candidates.iloc[0]

            target_node = best_match['graph_node_id']

            # Membership guard: only return nodes that actually exist in the
            # graph. notna() above handles the None case; this is belt-and-
            # suspenders in case graph_node_id was populated by another path.
            if target_node and target_node in self.G.nodes:
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

        # 3. --- FAST SPATIAL QUERY ---
        # Convert meters to a KDTree-safe degree radius at this latitude
        # (see utils.meters_to_degree_radius — avoids the mid-latitude
        # longitude under-fetch that plain lat-degree math would introduce).
        degree_radius = utils.meters_to_degree_radius(radius_m, lat1)
        
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
    

    def filter_candidates_by_direction(self, origin_node, candidate_ids, target_direction):
        """
        Filters candidate graph nodes by allocentric direction from origin_node.

        Important fix:
        - Preserves 8-way directions (NE/NW/SE/SW) instead of collapsing them
          to N/S/E/W.
        - Uses utils.get_direction_8way() for candidate bearings.
        - Uses utils.direction_matches() so cardinal directions remain coarse
          while intercardinal directions must match exactly.
        """
        if not target_direction or not candidate_ids:
            return candidate_ids

        target_direction = target_direction.strip().upper()
        origin_data = self.G.nodes[origin_node]
        lat1, lon1 = origin_data['y'], origin_data['x']

        kept = []
        failed_count = 0

        for node_id in candidate_ids:
            if node_id not in self.G.nodes:
                failed_count += 1
                continue

            node_data = self.G.nodes[node_id]
            actual_dir = utils.get_direction_8way(
                lat1, lon1, node_data['y'], node_data['x']
            )

            if utils.direction_matches(actual_dir, target_direction):
                kept.append(node_id)

        if failed_count:
            print(
                f"⚠️ Direction filter skipped {failed_count} candidates "
                f"because their node IDs were not present in the graph."
            )

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
        # Safe degree radius: uses longitude-degree width at curr_y so the
        # KDTree ball fully contains the meter-horizon circle.
        deg_radius = utils.meters_to_degree_radius(config.GLOBAL_SEARCH_HORIZON_METERS, curr_y)
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

        # Drop POIs with no usable graph node — scoring unresolvable rows
        # risks returning the best name-match but no reachable path.
        if 'graph_node_id' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['graph_node_id'].notna()]

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

        target_node = filtered_df.loc[nearest_idx, 'graph_node_id']
        # Belt-and-suspenders: the notna() filter above should guarantee this,
        # but guard anyway so every return path is a valid graph node or None.
        if target_node and target_node in self.G.nodes:
            return target_node
        return None


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
        # Vectorized: prune unresolvable POIs and POIs whose ids aren't in the
        # graph using DataFrame ops, then materialize result dicts from column
        # arrays. Replaces a per-row iterrows() loop that dominated runtime on
        # categories with thousands of tag matches (BUILDING / STREET / BENCH).
        hits = res_df[res_df['score'] >= score_threshold]
        if hits.empty or 'graph_node_id' not in hits.columns:
            return []

        hits = hits.dropna(subset=['graph_node_id'])
        if hits.empty:
            return []

        hits = hits[hits['graph_node_id'].isin(self.graph_node_set)]
        if hits.empty:
            return []

        hits = hits.sort_values('score', ascending=False)

        nids = hits['graph_node_id'].to_numpy()
        scores = hits['score'].to_numpy()
        ys = hits['y'].to_numpy()
        xs = hits['x'].to_numpy()
        names = (
            hits['name'].to_numpy()
            if 'name' in hits.columns
            else np.array(['Unknown'] * len(hits), dtype=object)
        )

        return [
            {"node_id": n, "name": nm, "score": float(s),
             "coords": (float(y), float(x))}
            for n, nm, s, y, x in zip(nids, names, scores, ys, xs)
        ]


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