import math
import pickle
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import heapq
from matplotlib import category
import networkx as nx
import pandas as pd

# Internal Project Imports
import config
from src.oracle_engine import OracleEngine
import src.utils as utils

from src.extraction_utils import extract_rvs_target

SALIENCE_COLS = ['wikipedia', 'wikidata', 'brand', 'tourism', 'amenity', 'shop']

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
        Undirected connectivity check — physically accessible, not legally routable.
        Our project's addition to RVS methodology. O(1) via precomputed map.
        """
        return utils.is_reachable_fast(self.scc_lookup, start_node, target_node)

    
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


    # This is the main method that processes an RVS instruction and produces a symbolic state.

    def _pick_by_salience(self, candidates: list) -> dict:
        """
        Pick single best candidate using RVS paper's salience hierarchy.
        Falls back to nearest-by-distance if no salience signal found.
        """
        def salience_rank(c):
            row = self.oracle.poi_df[
                self.oracle.poi_df['graph_node_id'] == c['node_id']
            ]
            if row.empty:
                return (len(SALIENCE_COLS), c.get('dist', 9999))
            r = row.iloc[0]
            # tuple of (salience tier, distance) for sorting: lower is better
            for i, col in enumerate(SALIENCE_COLS):
                if col in r.index and pd.notna(r.get(col)) and str(r.get(col)) not in ('', 'nan', 'no'):
                    return (i, c.get('dist', 9999))
            return (len(SALIENCE_COLS), c.get('dist', 9999))

        return min(candidates, key=salience_rank)


    # Modes are: "resolve" or "label"
    def solve(self, instruction_text: str, start_node: str, mode: str) -> dict:
        """
        Master Controller: Translates text into a Symbolic State.
        Uses the 1500m 'Elbow' Horizon and O(1) SCC Reachability.
        
        Label definitions (per project proposal + RVS paper):
        Answerable   = exactly one candidate survives all filters + reachable
        Contradictory = zero candidates, or candidate unreachable
        Ambiguous    = reserved for underspecified variants (Step 4), AKA labeling mode
        """

        if mode not in {"resolve", "label"}:
            raise ValueError(f"mode must be 'resolve' or 'label', got '{mode}'")
    
        # ------ Phase A: Candidate Resolution ------
        category, raw_noun, target_dir = extract_rvs_target(instruction_text)
        tags       = config.LANDMARK_GROUPS.get(category, {})
        start_data = self.G.nodes[start_node]
        lat, lon   = start_data['y'], start_data['x']
        horizon    = config.GLOBAL_SEARCH_HORIZON_METERS

        metadata = {
            "mode":               mode,
            "extracted_category": category,
            "extracted_noun":     raw_noun,
            "extracted_direction": target_dir,
        }

        # Step 1 — Strict: category + name + direction
        candidates = self.oracle.resolve_nearby_candidates(
            tags, lat, lon, radius_m=horizon, landmark_name=raw_noun
        )
        candidates = self._apply_directional_filter(candidates, start_node, target_dir)

        # Step 2 — "Philly fallback": category + direction, no name
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

        # Ensure every candidate has 'dist'
        for c in candidates:
            if 'dist' not in c:
                clat, clon = c['coords']
                c['dist'] = utils.haversine(lat, lon, clat, clon)


        # ------ Phase B: Zero-candidate exit ------
        metadata["candidate_count"] = len(candidates)

        if not candidates:
            return {**metadata, "state": config.STATE_CONTRADICTORY}
        

        # ------ Phase C: Reachability filter ------
        # Not part of RVS methodology (which used human validation).
        # Added as a graph-based sanity check: filters nodes on physically
        # disconnected graph islands.
        # Uses undirected connectivity per RVS paper's "physical access" framing...
        # ...(not directed/legal routing).
        # O(1) via precomputed SCC map. No-op for single-component cities.
        reachable = [c for c in candidates
             if self.check_reachability_scc(start_node, c['node_id'])]
        
        metadata["reachable_candidate_count"] = len(reachable)

        if not reachable:
            return {**metadata, "state": config.STATE_CONTRADICTORY}


        # ------ Phase D: Mode-specific labeling ------
        if mode == "resolve":
            # Oracle 1: pick most salient among reachable candidates.
            # Salience breaks ties, never returns Ambiguous.
            best = self._pick_by_salience(reachable) if len(reachable) > 1 else reachable[0]
            return {
                **metadata,
                "state":       config.STATE_ANSWERABLE,
                "target_node": best['node_id'],
            }

        else:  # mode == "label"
            # Oracle 2: count reachable candidates — preserve ambiguity as signal.
            # This is the research measurement: did masking destroy uniqueness?
            if len(reachable) == 1:
                return {
                    **metadata,
                    "state":       config.STATE_ANSWERABLE,
                    "target_node": reachable[0]['node_id'],
                }
            else:
                return {
                    **metadata,
                    "state":           config.STATE_AMBIGUOUS,
                    "candidate_nodes": [c['node_id'] for c in reachable[:50]],
                }