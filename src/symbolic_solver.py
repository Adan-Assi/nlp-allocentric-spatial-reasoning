import math
import pickle
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import heapq
import networkx as nx

# Internal Project Imports
import config
from src.oracle_engine import OracleEngine
import src.utils as utils

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

        # --- Pre-compute SCC for Instant Reachability ---
        # We map every node ID to a 'Component ID' (an integer)
        # Nodes in the same component can reach each other.
        self.scc_lookup = {}
        components = list(nx.strongly_connected_components(self.G))
        for i, component in enumerate(components):
            for node in component:
                self.scc_lookup[node] = i
        
        # DELETE PRINT
        print(f"✅ Solver Initialized: Found {len(components)} isolated graph components.")

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