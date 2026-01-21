"""
Task 2: Symbolic Graph Operations (FINAL VERSION)
Frozen scope - supporting operations for RVS evaluation

This module provides GRAPH QUERY OPERATIONS, not end-to-end ground truth.
It does NOT parse instructions or interpret text.

Status: Week 1 - Core operations implemented and tested
"""

import json
import math
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import heapq


class SymbolicSolver:
    """
    Symbolic graph operations for RVS map queries.
    
    SCOPE (FROZEN):
    - Reachability checking (BFS)
    - Shortest path computation (Dijkstra)
    - Coarse 4-way direction (N/E/S/W, dominant-axis)
    
    EXPLICITLY EXCLUDED:
    - Language parsing
    - Instruction interpretation  
    - Ambiguity resolution
    - Landmark extraction from text
    """
    
    def __init__(self, graph_data_path: str):
        """
        Load RVS map graph.
        
        Args:
            graph_data_path: Path to graph file (.json, .pickle, .gpickle)
        """
        self.nodes = {}  # node_id -> {'lat': float, 'lon': float, ...}
        self.edges = {}  # node_id -> [neighbor_ids]
        self.graph = self._load_graph(graph_data_path)
        
    def _load_graph(self, path: str) -> Dict:
        """
        Load graph from file with proper edge filtering.
        
        Supports: JSON, pickle, NetworkX gpickle
        """
        if not Path(path).exists():
            print(f"⚠️  Graph file not found: {path}")
            print("📝 Creating sample graph for testing...")
            return self._create_sample_graph()
        
        # Load based on extension
        if path.endswith('.json'):
            with open(path, 'r') as f:
                graph = json.load(f)
        elif path.endswith(('.pickle', '.pkl', '.gpickle')):
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if NetworkX graph
            try:
                import networkx as nx
                if isinstance(data, (nx.Graph, nx.DiGraph)):
                    print("📊 Detected NetworkX graph, converting...")
                    graph = self._convert_networkx_graph(data)
                else:
                    graph = data
            except ImportError:
                print("⚠️  NetworkX not installed, treating as regular pickle")
                graph = data
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self.nodes = graph.get('nodes', {})
        self.edges = graph.get('edges', {})
        
        # Validate edge integrity
        self._validate_edges()
        
        print(f"✅ Loaded graph: {len(self.nodes)} nodes, "
              f"{sum(len(e) for e in self.edges.values())} edges")
        return graph
    
    def _convert_networkx_graph(self, G) -> Dict:
        """
        Convert NetworkX graph with proper edge filtering.
        
        CRITICAL: Only include edges between valid nodes.
        """
        # FIRST PASS: Identify nodes with valid coordinates
        valid_nodes = set()
        nodes = {}
        
        for node_id, node_data in G.nodes(data=True):
            # Try different attribute names for coordinates
            lat = node_data.get('lat') or node_data.get('latitude') or node_data.get('y')
            lon = node_data.get('lon') or node_data.get('longitude') or node_data.get('x')
            
            if lat is not None and lon is not None:
                valid_nodes.add(node_id)
                nodes[str(node_id)] = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'name': node_data.get('name', f'Node_{node_id}'),
                    **{k: v for k, v in node_data.items() 
                       if k not in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y']}
                }
        
        print(f"   Valid nodes with coordinates: {len(valid_nodes)}")
        
        # SECOND PASS: Build edges ONLY between valid nodes
        edges = {}
        skipped_edges = 0
        
        for node_id in valid_nodes:
            neighbors = []
            for neighbor in G.neighbors(node_id):
                if neighbor in valid_nodes:
                    neighbors.append(str(neighbor))
                else:
                    skipped_edges += 1
            edges[str(node_id)] = neighbors
        
        if skipped_edges > 0:
            print(f"   Skipped {skipped_edges} edges to invalid nodes")
        
        return {'nodes': nodes, 'edges': edges}
    
    def _validate_edges(self):
        """
        Validate and clean edges pointing to non-existent nodes.
        FIXED: Safer list filtering instead of remove-while-iterating.
        """
        invalid_count = 0
        
        for node_id, neighbors in list(self.edges.items()):
            filtered = [n for n in neighbors if n in self.nodes]
            invalid_count += (len(neighbors) - len(filtered))
            self.edges[node_id] = filtered
        
        if invalid_count > 0:
            print(f"⚠️  WARNING: Removed {invalid_count} edges to non-existent nodes")
    
    def _create_sample_graph(self) -> Dict:
        """Create sample graph for testing."""
        nodes = {
            'central_park': {'lat': 40.7829, 'lon': -73.9654, 'name': 'Central Park'},
            'museum': {'lat': 40.7794, 'lon': -73.9632, 'name': 'Museum'},
            'library': {'lat': 40.7532, 'lon': -73.9822, 'name': 'Library'},
            'station': {'lat': 40.7580, 'lon': -73.9855, 'name': 'Station'},
            'park_north': {'lat': 40.7950, 'lon': -73.9654, 'name': 'Park North'},
            'times_square': {'lat': 40.7580, 'lon': -73.9855, 'name': 'Times Square'},
        }
        
        # Build bidirectional edges properly
        edges_bidirectional = {node_id: [] for node_id in nodes}
        
        connections = [
            ('central_park', 'museum'),
            ('central_park', 'park_north'),
            ('museum', 'library'),
            ('library', 'station'),
            ('station', 'times_square'),
        ]
        
        for source, target in connections:
            edges_bidirectional[source].append(target)
            edges_bidirectional[target].append(source)
        
        self.nodes = nodes
        self.edges = edges_bidirectional
        
        print("📝 Using sample graph (6 nodes)")
        return {'nodes': nodes, 'edges': edges_bidirectional}

    # ========== CAPABILITY 1: REACHABILITY ==========
    
    def check_reachability(self, start_node: str, end_node: str) -> bool:
        """
        Check if path exists between nodes.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            
        Returns:
            True if reachable, False otherwise
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return False
        if start_node == end_node:
            return True
        
        visited = set([start_node])
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            for neighbor in self.edges.get(current, []):
                if neighbor == end_node:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    # ========== CAPABILITY 2: SHORTEST PATH ==========
    
    def compute_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        """
        Find shortest path using geographic distance.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            
        Returns:
            List of node IDs in path (empty if no path)
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return []
        if start_node == end_node:
            return [start_node]
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        previous = {node: None for node in self.nodes}
        pq = [(0, start_node)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_node:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return list(reversed(path))
            
            for neighbor in self.edges.get(current, []):
                if neighbor in visited:
                    continue
                
                edge_distance = self._calculate_distance(
                    self.nodes[current]['lat'], self.nodes[current]['lon'],
                    self.nodes[neighbor]['lat'], self.nodes[neighbor]['lon']
                )
                
                new_distance = current_dist + edge_distance
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        return []
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Haversine distance in meters."""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    # ========== CAPABILITY 3: COARSE 4-WAY DIRECTION (DOMINANT-AXIS) ==========
    
    def get_coarse_direction(self, coord1: Tuple[float, float], 
                            coord2: Tuple[float, float]) -> str:
        """
        Determine coarse 4-way direction (N/E/S/W) using dominant axis.
        
        This is more stable and intuitive than bearing-based quadrants.
        Returns the direction along whichever axis has the larger change.
        
        Args:
            coord1: (lat, lon) of first point
            coord2: (lat, lon) of second point
            
        Returns:
            One of: 'N', 'E', 'S', 'W'
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Use dominant axis (larger absolute change)
        if abs(dlat) >= abs(dlon):
            return 'N' if dlat > 0 else 'S'
        else:
            return 'E' if dlon > 0 else 'W'
    
    # ========== HELPER METHODS ==========
    
    def get_node_coordinates(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get (lat, lon) of a node."""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        return (node['lat'], node['lon'])
    
    def get_path_length(self, path: List[str]) -> float:
        """Calculate total geographic distance of path in meters."""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            coord1 = self.get_node_coordinates(path[i])
            coord2 = self.get_node_coordinates(path[i + 1])
            if coord1 and coord2:
                total += self._calculate_distance(
                    coord1[0], coord1[1], coord2[0], coord2[1]
                )
        return total


# ========== TESTING ==========

def run_tests():
    print("=" * 70)
    print("SYMBOLIC SOLVER TESTS (Task 2 - Final)")
    print("=" * 70)

    solver = SymbolicSolver('data/manhattan/manhattan_graph.gpickle')

    print("\n" + "="*70)
    print("TEST 0: BASIC LOAD CHECK")
    print("="*70)
    print(f"Nodes: {len(solver.nodes)}")
    print(f"Edges: {sum(len(e) for e in solver.edges.values())}")

    # -------------------------
    # TEST 1: REACHABILITY
    # -------------------------
    print("\n" + "="*70)
    print("TEST 1: REACHABILITY")
    print("="*70)

    # If sample node IDs exist, run sample tests.
    sample_nodes = ["central_park", "library", "times_square"]
    if all(n in solver.nodes for n in sample_nodes):
        tests = [
            ('central_park', 'library', True),
            ('central_park', 'times_square', True),
            ('central_park', 'nonexistent', False),
        ]
        for start, end, expected in tests:
            result = solver.check_reachability(start, end)
            status = "✅" if result == expected else "❌"
            print(f"{status} {start} → {end}: {result} (expected {expected})")
    else:
        # Real graph: pick two existing nodes from the graph
        node_ids = list(solver.nodes.keys())
        n1 = node_ids[0]
        n2 = node_ids[500]  # any index as long as < len(node_ids)

        result = solver.check_reachability(n1, n2)
        print(f"✅ Real-graph reachability test: {n1} → {n2}: {result}")
        print("ℹ️  Note: This test checks functionality, not an expected True/False value.")

    # -------------------------
    # TEST 2: SHORTEST PATH
    # -------------------------
    print("\n" + "="*70)
    print("TEST 2: SHORTEST PATH")
    print("="*70)

    if all(n in solver.nodes for n in sample_nodes):
        start, end = "central_park", "station"
    else:
        node_ids = list(solver.nodes.keys())
        start, end = node_ids[0], node_ids[500]

    path = solver.compute_shortest_path(start, end)
    if path:
        print(f"✅ Path length (nodes): {len(path)}")
        print(f"   First 5 nodes: {path[:5]}")
        print(f"   Distance: {solver.get_path_length(path):.2f}m")
    else:
        print("⚠️  No path found (graph may be disconnected OR nodes chosen are in different components).")
        print("    Try increasing the chance they’re connected by picking a nearby neighbor.")
        # Optional: try neighbor-based end node
        neighbors = solver.edges.get(start, [])
        if neighbors:
            end2 = neighbors[0]
            path2 = solver.compute_shortest_path(start, end2)
            if path2:
                print(f"✅ Neighbor path works: {start} → {end2}, length={len(path2)}")

    # -------------------------
    # TEST 3: DIRECTION (unchanged)
    # -------------------------
    print("\n" + "="*70)
    print("TEST 3: COARSE 4-WAY DIRECTION (Dominant-Axis)")
    print("="*70)

    tests = [
        ((40.7829, -73.9654), (40.7950, -73.9654), 'N'),
        ((40.7829, -73.9654), (40.7794, -73.9632), 'S'),
        ((40.7829, -73.9654), (40.7532, -73.9822), 'S'),
        ((40.7829, -73.9654), (40.7829, -73.9754), 'W'),
    ]
    for c1, c2, expected in tests:
        result = solver.get_coarse_direction(c1, c2)
        status = "✅" if result == expected else "❌"
        print(f"{status} {c1} → {c2}: {result} (expected {expected})")

    print("\n" + "="*70)
    print("✅ TESTS COMPLETE")
    print("="*70)

def test_real_graph():
    """
    Test loading real Manhattan graph.
    Uncomment and run when you have the actual .gpickle file.
    """
    print("\n" + "="*70)
    print("REAL MANHATTAN GRAPH TEST")
    print("="*70)
    
    try:
        # Update path to match your repo structure
        solver = SymbolicSolver('data/manhattan/manhattan_graph.gpickle')
        
        print(f"\n✅ Successfully loaded Manhattan graph!")
        print(f"   Nodes: {len(solver.nodes)}")
        print(f"   Total edges: {sum(len(e) for e in solver.edges.values())}")
        
        # Test with first few nodes
        node_ids = list(solver.nodes.keys())[:5]
        print(f"\n   Sample node IDs: {node_ids}")
        
        if len(node_ids) >= 2:
            n1, n2 = node_ids[0], node_ids[1]
            reachable = solver.check_reachability(n1, n2)
            print(f"\n   Reachability test: {n1} → {n2}: {reachable}")
        
    except FileNotFoundError:
        print("⚠️  Manhattan graph file not found")
        print("   Update path in test_real_graph() to match your repo structure")
    except Exception as e:
        print(f"❌ Error loading Manhattan graph: {e}")


if __name__ == '__main__':
    run_tests()
    
    # Uncomment to test real Manhattan graph:
    # test_real_graph()
