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
    
    def __init__(self, oracle: OracleEngine):
        """
        Initialize the solver with a pre-loaded Oracle.
        
        Args:
            oracle: An instance of OracleEngine already containing the graph and POIs.
        """
        self.oracle = oracle
        self.G = oracle.G  # Access the NetworkX graph directly from the Oracle
        
        # Salvaged properties for the Dijkstra logic
        self.nodes = self.G.nodes(data=True)
        self.edges = self.G.adj

    # ========== CAPABILITY 1: REACHABILITY ==========
    
    def check_reachability(self, start_node: str, end_node: str) -> bool:
            """
            Check if path exists between nodes using the Oracle's graph.
            
            Args:
                start_node: Starting node ID (e.g., '1#666')
                end_node: Target node ID
                
            Returns:
                True if reachable, False otherwise
            """
            # 1. Validation: Ensure both nodes actually exist in the graph
            if start_node not in self.G or end_node not in self.G:
                return False
                
            # 2. Use NetworkX built-in reachability (has_path)
            # This replaces our previous manual deque/queue logic with an optimized version
            return nx.has_path(self.G, start_node, end_node)
    
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
        Calculates a path from the agent's current node to a named landmark.
        This bridges the Oracle's NLP resolution with the Solver's math.
        """
        # 1. Resolve Name to Node ID via Oracle
        target_node = self.oracle.resolve_landmark(landmark_name)
        
        if not target_node:
            print(f"⚠️ Solver: Oracle could not resolve '{landmark_name}' to a graph node.")
            return []

        # 2. Check if the resolved node is actually in our graph
        if target_node not in self.G:
            print(f"⚠️ Solver: Resolved node {target_node} for '{landmark_name}' is not in the graph.")
            return []

        # 3. Compute the path
        return self.compute_shortest_path(current_node, target_node)