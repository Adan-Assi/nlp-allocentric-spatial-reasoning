import math
import pickle
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import heapq
from matplotlib import category
import networkx as nx

# Internal Project Imports
import config
from src.oracle_engine import OracleEngine
import src.utils as utils

from src.extraction_utils import extract_rvs_target


class SymbolicSolver:
    """
    Symbolic graph operations for RVS map queries.
    
    This class handles the 'How to get there' logic, using the OracleEngine
    as the 'Where - Sense Organ' to resolve landmarks and locations.
    """
    
    def __init__(self, oracle: 'OracleEngine', search_radius: Optional[int] = None):
        """
        Initialize the solver with a pre-loaded Oracle.
        
        Args:
            oracle: An instance of OracleEngine already containing the graph and POIs.
        """
        self.oracle = oracle
        self.G = oracle.G  # Access the NetworkX graph directly from the Oracle
        
        # --- City-Specific Success Radius ---
        # This replaces the hardcoded reliance on config.DISTANCE_FIXED_BUFFER
        self.search_radius = search_radius if search_radius is not None else config.DISTANCE_FIXED_BUFFER

        # --- Shared Connectivity Map (for Instant Reachability) ---
        # We reference the map already built by the Oracle to ensure 
        # consistency and save memory.
        self.scc_lookup = oracle.scc_lookup
        
        # Verify connectivity status
        num_comp = len(set(self.scc_lookup.values())) if self.scc_lookup else 0
        print(f"✅ Solver Initialized: Using Oracle's map with {num_comp} components.")
        
        # Salvaged properties for the Dijkstra logic
        self.nodes = self.G.nodes(data=True)
        self.edges = self.G.adj

    # ========== CAPABILITY 1: REACHABILITY ==========
    
    def check_reachability(self, start_node: str, end_node: str) -> bool:
            """
            Check if path exists between nodes using the Oracle's graph.
            (Uses pre-computed SCC lookup for instant O(1) reachability.)
            
            Args:
                start_node: Starting node ID (e.g., '1#666')
                end_node: Target node ID
                
            Returns:
                True if reachable, False otherwise
            """    
            # 1. Validation: Ensure both nodes actually exist in the graph
            if start_node not in self.scc_lookup or end_node not in self.scc_lookup:
                return False
                
            # 2. SCC Check: If they are on the same 'island', return True
            return self.scc_lookup[start_node] == self.scc_lookup[end_node]
    
    # ========== CAPABILITY 2: SHORTEST PATH ==========
    
    def compute_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        """
        Find shortest path using actual geographic distance as the edge weight.
        """
        # 1. Validation
        if start_node not in self.G or end_node not in self.G:
            return []
        
        # 2. Check reachability first (to avoid expensive calculations if impossible)
        if not self.check_reachability(start_node, end_node):
            return []

        # 3. Use NetworkX Dijkstra with a dynamic weight function
        # Instead of a pre-calculated 'weight' attribute, we use our utils math
        try:
            path = nx.shortest_path(
                self.G, 
                source=start_node, 
                target=end_node, 
                weight=lambda u, v, _: utils.get_euclidean_dist(self.G, u, v)
            )
            return path
        except nx.NetworkXNoPath:
            return []
    
    # ========== CAPABILITY 3: COARSE 4-WAY DIRECTION (DOMINANT-AXIS) ==========
    
    def get_coarse_direction(self, start_node: str, end_node: str) -> str:
        n1 = self.G.nodes[start_node]
        n2 = self.G.nodes[end_node]
        
        # Use the utils shared math brain
        return utils.get_dominant_direction(n1['y'], n1['x'], n2['y'], n2['x'])
        
    # ========== HELPER METHODS ==========
        
    def get_node_coordinates(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get (lat, lon) of a node from the NetworkX graph."""
        if node_id not in self.G:
            return None
        node = self.G.nodes[node_id]
        # Standardize to (lat, lon)
        return (node['y'], node['x'])
    
    def get_path_length(self, path: List[str]) -> float:
        """
        Calculate total geographic distance of a path in meters.
        Uses the high-precision geodesic math from utils.
        """
        if len(path) < 2:
            return 0.0
        
        total_meters = 0.0
        for i in range(len(path) - 1):
            # Use our unified utility function
            segment_dist = utils.get_euclidean_dist(self.G, path[i], path[i+1])
            total_meters += segment_dist
            
        return total_meters
    
    def get_path_to_landmark(self, current_node: str, landmark_name: str) -> list:
        """
        Upgraded Task 3.1: Resolves landmarks using Keyword Groups or Direct Names.
        """
        import config # Ensure access to LANDMARK_GROUPS
        
        target_node = None
        clean_name = landmark_name.upper().strip()

        # 1. ATTEMPT KEYWORD RESOLUTION (The 2.5 Strategy)
        # We sort by length descending so "POST OFFICE" matches before "OFFICE"
        sorted_roots = sorted(config.LANDMARK_GROUPS.keys(), key=len, reverse=True)
        
        matched_tags = None
        for root in sorted_roots:
            if root in clean_name:
                matched_tags = config.LANDMARK_GROUPS[root]
                print(f"DEBUG: Mapping '{landmark_name}' to Category: {root}")
                break

        if matched_tags:
            # Oracle needs a method to find the closest POI node by OSM tags
            target_node = self.oracle.resolve_by_tags(current_node, matched_tags, landmark_name)

        # 2. FALLBACK: DIRECT NAME RESOLUTION
        if not target_node:
            print(f"DEBUG: No category match for '{landmark_name}'. Trying direct name match...")
            target_node = self.oracle.resolve_landmark(landmark_name)

        # 3. GRAPH VALIDATION & PATH COMPUTATION
        if not target_node or target_node not in self.G:
            print(f"⚠️ Solver: Could not resolve '{landmark_name}' to a valid graph node.")
            return []

        return self.compute_shortest_path(current_node, target_node)

    def get_search_limit(self, intended_distance: float) -> float:
        """
        Task 2.2: Implement the 'Human Error' Search Radius logic.
        Formula: max(D * 1.1, D + 80m)
        """
        # We use the global scale factor
        buffer_percent = intended_distance * config.DISTANCE_SCALE_FACTOR

        # We use the city-specific radius passed during solver initialization
        buffer_fixed = intended_distance + self.search_radius
        
        return max(buffer_percent, buffer_fixed)
    
    # --------- PHASE 5: THE MASTER CONTROLLER ---------

    def check_reachability_scc(self, start_node: str, target_node: str) -> bool:
        """
        Wrapper for the shared SCC utility. 
        Note: We pull the lookup table from the oracle instance.
        """
        
        return utils.is_reachable_fast(self.oracle.scc_lookup, start_node, target_node)

    #aaaaaaaaaaaaaaaa
    def _apply_directional_filter(
        self, candidates: list, start_node: str, target_dir: str
    ) -> list:
        """Filter candidates by direction, keeping originals on total miss (soft fallback)."""
        if not target_dir or not candidates:
            return candidates
        cand_ids = [c['node_id'] for c in candidates]
        filtered_ids = set(
            self.oracle.filter_candidates_by_direction(start_node, cand_ids, target_dir)
        )
        if not filtered_ids:
            return candidates  # soft fallback: direction killed everything, keep all
        return [c for c in candidates if c['node_id'] in filtered_ids]


    def solve(self, instruction_text: str, start_node: str) -> dict:
        """
        Master Controller: Translates text into a Symbolic State.
        Uses the 1500m 'Elbow' Horizon and O(1) SCC Reachability.
        """

        # ------ Phase A: Candidate Resolution ------

        category, raw_noun, target_dir = extract_rvs_target(instruction_text)
        tags        = config.LANDMARK_GROUPS.get(category, {})
        start_data  = self.G.nodes[start_node]
        lat, lon    = start_data['y'], start_data['x']
        horizon     = config.GLOBAL_SEARCH_HORIZON_METERS

        metadata = {
            "extracted_category": category,
            "extracted_noun":     raw_noun,
        }

        # Step 1 — Strict: category + name + direction
        candidates = self.oracle.resolve_nearby_candidates(
            tags, lat, lon, radius_m=horizon, landmark_name=raw_noun
        )
        candidates = self._apply_directional_filter(candidates, start_node, target_dir)

        # Step 2 — Philly fallback: category + direction, no name
        if not candidates:
            candidates = self.oracle.resolve_nearby_candidates(
                tags, lat, lon, radius_m=horizon, landmark_name=None
            )
            candidates = self._apply_directional_filter(candidates, start_node, target_dir)

        # Step 3 — Last resort: global fuzzy name search
        if not candidates and raw_noun:
            fallback_node = self.oracle.resolve_landmark(
                raw_noun, context_node=start_node, radius_m=horizon
            )
            if fallback_node:
                node_data = self.G.nodes[fallback_node]
                d = utils.haversine(lat, lon, node_data['y'], node_data['x'])
                candidates = [{"node_id": fallback_node,
                            "coords": (node_data['y'], node_data['x']),
                            "dist": d}]

        # Ensure every candidate has 'dist' (Steps 1/2 may omit it)
        for c in candidates:
            if 'dist' not in c:
                clat, clon = c['coords']
                c['dist'] = utils.haversine(lat, lon, clat, clon)

        # ------ Phase B: Zero-candidate exit ------

        count = len(candidates)
        metadata["candidate_count"] = count

        if count == 0:
            return {**metadata, "state": config.STATE_CONTRADICTORY}

        # ------ Phase C: RVS-Aligned Tiered Labeling ------

        if count > 1:
            sorted_cands = sorted(candidates, key=lambda x: x.get('dist', 9999))
            if sorted_cands[0]['dist'] <= 250:
                candidates = [sorted_cands[0]]          # nearest within success zone wins
            else:
                return {**metadata, "state": config.STATE_AMBIGUOUS}

        # ------ Phase D: Reachability ------

        target_node = candidates[0]['node_id']

        # trivial case: already at destination
        if target_node == start_node:
            return {**metadata, "state": config.STATE_ANSWERABLE, "target_node": target_node}

        state = (config.STATE_ANSWERABLE
                if self.check_reachability_scc(start_node, target_node)
                else config.STATE_CONTRADICTORY)
        return {**metadata, "state": state, "target_node": target_node}


    #possible delete
    def solve_OLD(self, instruction_text: str, start_node: str) -> dict:
        """
        Master Controller: Translates text into a Symbolic State.
        Uses the 1500m 'Elbow' Horizon and O(1) SCC Reachability.
        """

        # ------ Phase A: Candidate Resolution ------
        
        # 1. Extraction
        category, raw_noun, target_dir = extract_rvs_target(instruction_text)
        tags = config.LANDMARK_GROUPS.get(category, {})
        start_data = self.G.nodes[start_node]
        lat, lon = start_data['y'], start_data['x']
        horizon = config.GLOBAL_SEARCH_HORIZON_METERS

        # 2. Step 1: Strict Search (Category + Name + Direction)
        # Most instructions in Manhattan and Pitt will resolve here.
        candidates = self.oracle.resolve_nearby_candidates(
            tags, lat, lon, radius_m=horizon, landmark_name=raw_noun
        )
        if target_dir and candidates:
            cand_ids = [c['node_id'] for c in candidates]
            filtered_ids = self.oracle.filter_candidates_by_direction(start_node, cand_ids, target_dir)
            
            # --- MODIFICATION: SOFT DIRECTIONAL FALLBACK ---
            if not filtered_ids:
                # If direction killed everything, keep original candidates but log it
                # This prevents a 150m 'Success' from becoming a 'Contradictory'
                pass 
            else:
                candidates = [c for c in candidates if c['node_id'] in filtered_ids]


        # 3. Step 2: Philly Fallback (Category + Direction, NO Name)
        # If Step 1 found nothing, the description is likely "General" (e.g., "the cafe").
        if not candidates:
            candidates = self.oracle.resolve_nearby_candidates(
                tags, lat, lon, radius_m=horizon, landmark_name=None 
            )
            if target_dir and candidates:
                cand_ids = [c['node_id'] for c in candidates]
                filtered_ids = self.oracle.filter_candidates_by_direction(start_node, cand_ids, target_dir)
                candidates = [c for c in candidates if c['node_id'] in filtered_ids]

        for c in candidates:
            if 'dist' not in c:
                target_lat, target_lon = c['coords']
                c['dist'] = utils.haversine(lat, lon, target_lat, target_lon)


        # 4. Step 3: Last Resort (Global Fuzzy Name Search)
        # If Category matches failed, try a fuzzy global search for the raw text.
        if not candidates and raw_noun:
            fallback_node = self.oracle.resolve_landmark(
                raw_noun, context_node=start_node, radius_m=horizon
            )
            if fallback_node:
                node_data = self.G.nodes[fallback_node]
                # Calculate distance so the Salience Filter doesn't crash if count > 1
                d = utils.haversine(lat, lon, node_data['y'], node_data['x'])
                candidates = [{"node_id": fallback_node, "coords": (node_data['y'], node_data['x']), "dist": d}]
        
        # ------ Phase B: Label Assignment ------
        
        # Label Assignment (The Silver Standard)
        count = len(candidates)
        metadata = {
            "extracted_category": category,
            "extracted_noun": raw_noun,
            "candidate_count": count
        }

        if count == 0:
            return {**metadata, "state": config.STATE_CONTRADICTORY}
       
       
        # ------ Phase C: RVS-Aligned Tiered Labeling ------

        # Handle Candidate Density (The Salience Filter)
        if count > 1:
            # Sort by distance
            sorted_cands = sorted(candidates, key=lambda x: x.get('dist', 9999))
            d1 = sorted_cands[0]['dist']
            
            # 1. We use the 250m Coarse-grained accuracy as the 'Answerable' threshold
            # This aligns with Metric #2 from the paper.
            if d1 <= 250: 
                # Even if there's a d2, if d1 is within the 250m 'Success zone',
                # we treat it as the intended target.
                candidates = [sorted_cands[0]]
            else:
                return {**metadata, "state": config.STATE_AMBIGUOUS}
                    
        # ------ Reachability Check for the Final Candidate ------

        target_node = candidates[0]['node_id']
        if self.check_reachability_scc(start_node, target_node):
            return {**metadata, "state": config.STATE_ANSWERABLE, "target_node": target_node}
        else:
            return {**metadata, "state": config.STATE_CONTRADICTORY}