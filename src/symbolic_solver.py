"""
symbolic_solver.py
The 'How to get there' brain. Translates an extracted RVS instruction into a
symbolic state (Answerable / Ambiguous / Contradictory) using the OracleEngine
as the 'Where' / sense organ.

Improvements over the original repo version:
  1. solve() now has two modes:
       - "label"   (default): preserves ambiguity per the project's research
                              question. >1 reachable candidates → AMBIGUOUS.
       - "resolve" (legacy):  collapses to a single best candidate via salience
                              (RVS-style ground-truth recovery).
  2. Reachability is filtered FIRST, then the count is used for the 3-way
     label. Previously, ambiguity was collapsed before reachability, masking
     the case "5 reachable cafes, oracle picked one and labeled Answerable".
  3. The salience tiebreaker is fully deterministic: when two candidates have
     identical salience tier and distance, the lexicographically-smallest
     node_id wins. Without this, KDTree iteration order leaks into labels and
     reproducibility breaks across scipy/numpy versions.
"""

import math
import pickle
from typing import List, Tuple, Optional, Dict
from collections import deque
from pathlib import Path
import heapq

import networkx as nx
import pandas as pd

# Internal Project Imports
import config
from src.oracle_engine import OracleEngine
import src.utils as utils
from src.extraction_utils import extract_rvs_target


# Hierarchy mirrors RVS paper's salience signals: more authoritative columns first.
SALIENCE_COLS = ['wikipedia', 'wikidata', 'brand', 'tourism', 'amenity', 'shop']


class SymbolicSolver:
    """
    Symbolic graph operations for RVS map queries.

    Handles the 'How to get there' logic, using OracleEngine as the
    'Where - Sense Organ' to resolve landmarks and locations.
    """

    def __init__(self, oracle: 'OracleEngine', search_radius: Optional[int] = None):
        """
        Args:
            oracle: A pre-loaded OracleEngine (graph + POIs already in memory).
            search_radius: City-specific success radius in meters. If None,
                falls back to config.DISTANCE_FIXED_BUFFER.
        """
        self.oracle = oracle
        self.G = oracle.G  # NetworkX graph, by reference

        # City-specific success radius (replaces hardcoded buffer).
        self.search_radius = (
            search_radius if search_radius is not None else config.DISTANCE_FIXED_BUFFER
        )

        # Share the connectivity map already built by the Oracle. Saves memory
        # and guarantees both components see the same pedestrian topology.
        self.conn_lookup = oracle.conn_lookup

        num_comp = len(set(self.conn_lookup.values())) if self.conn_lookup else 0
        print(f"✅ Solver Initialized: Using Oracle's map with {num_comp} components.")

        # Convenience properties for direct graph access.
        self.nodes = self.G.nodes(data=True)
        self.edges = self.G.adj

    # ========== CAPABILITY 1: REACHABILITY ==========

    def check_reachability(self, start_node: str, end_node: str) -> bool:
        """
        Pedestrian-connectivity check in O(1) via the pre-computed component
        lookup. Two nodes are 'reachable' iff they share a weakly-connected
        component (see utils.get_connectivity_map for rationale).
        """
        return utils.is_reachable_fast(self.conn_lookup, start_node, end_node)

    # ========== CAPABILITY 2: SHORTEST PATH ==========

    def compute_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        """Find shortest path using actual geographic distance as edge weight."""
        if start_node not in self.G or end_node not in self.G:
            return []

        if not self.check_reachability(start_node, end_node):
            return []

        try:
            path = nx.shortest_path(
                self.G,
                source=start_node,
                target=end_node,
                weight=lambda u, v, _: utils.get_euclidean_dist(self.G, u, v),
            )
            return path
        except nx.NetworkXNoPath:
            return []

    # ========== CAPABILITY 3: COARSE 4-WAY DIRECTION ==========

    def get_coarse_direction(self, start_node: str, end_node: str) -> str:
        n1 = self.G.nodes[start_node]
        n2 = self.G.nodes[end_node]
        return utils.get_dominant_direction(n1['y'], n1['x'], n2['y'], n2['x'])

    # ========== HELPERS ==========

    def get_node_coordinates(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Get (lat, lon) of a node from the NetworkX graph."""
        if node_id not in self.G:
            return None
        node = self.G.nodes[node_id]
        return (node['y'], node['x'])

    def get_path_length(self, path: List[str]) -> float:
        """Total geographic distance of a path in meters."""
        if len(path) < 2:
            return 0.0
        total_meters = 0.0
        for i in range(len(path) - 1):
            total_meters += utils.get_euclidean_dist(self.G, path[i], path[i + 1])
        return total_meters

    def get_path_to_landmark(self, current_node: str, landmark_name: str) -> list:
        """Resolves landmarks using Keyword Groups or Direct Names, then routes."""
        target_node = None
        clean_name = landmark_name.upper().strip()

        # 1. Keyword resolution (longest match wins so "POST OFFICE" beats "OFFICE")
        sorted_roots = sorted(config.LANDMARK_GROUPS.keys(), key=len, reverse=True)
        matched_tags = None
        for root in sorted_roots:
            if root in clean_name:
                matched_tags = config.LANDMARK_GROUPS[root]
                print(f"DEBUG: Mapping '{landmark_name}' to Category: {root}")
                break

        if matched_tags:
            target_node = self.oracle.resolve_by_tags(current_node, matched_tags, landmark_name)

        # 2. Fallback: direct-name resolution
        if not target_node:
            print(f"DEBUG: No category match for '{landmark_name}'. Trying direct name match...")
            target_node = self.oracle.resolve_landmark(landmark_name)

        # 3. Validation & path computation
        if not target_node or target_node not in self.G:
            print(f"⚠️ Solver: Could not resolve '{landmark_name}' to a valid graph node.")
            return []

        return self.compute_shortest_path(current_node, target_node)

    def get_search_limit(self, intended_distance: float) -> float:
        """
        'Human Error' search radius.
        Formula: max(D * scale_factor, D + city_specific_radius)
        """
        buffer_percent = intended_distance * config.DISTANCE_SCALE_FACTOR
        buffer_fixed = intended_distance + self.search_radius
        return max(buffer_percent, buffer_fixed)

    # --------- MASTER CONTROLLER ---------

    def _apply_directional_filter(
        self, candidates: list, start_node: str, target_dir: str
    ) -> list:
        """
        Filter candidates by direction from start_node.

        Soft fallback: if the direction filter would eliminate every candidate,
        we keep the originals. The intuition is that an over-eager direction
        filter is worse than a slightly noisy one — better to surface ambiguity
        than to fabricate Contradictory.
        """
        if not target_dir or not candidates:
            return candidates

        cand_ids = [c['node_id'] for c in candidates]
        filtered_ids = set(
            self.oracle.filter_candidates_by_direction(start_node, cand_ids, target_dir)
        )
        if not filtered_ids:
            return candidates  # soft fallback
        return [c for c in candidates if c['node_id'] in filtered_ids]

    def _pick_by_salience(self, candidates: list) -> dict:
        """
        Pick a single best candidate using the RVS paper's salience hierarchy.
        Falls back to nearest-by-distance if no salience signal is found.

        Tiebreaking: when two candidates have identical (salience tier, distance),
        the node_id is used as a deterministic third key (lexicographically
        smallest wins). Without this, upstream KDTree iteration order leaks into
        the result, making labels non-reproducible across scipy/numpy versions.
        """
        def salience_rank(c):
            row = self.oracle.poi_df[
                self.oracle.poi_df['graph_node_id'] == c['node_id']
            ]
            # Deterministic tiebreaker; str() defends against non-string node IDs.
            node_id_tb = str(c.get('node_id', ''))
            if row.empty:
                return (len(SALIENCE_COLS), c.get('dist', 9999), node_id_tb)
            r = row.iloc[0]
            for i, col in enumerate(SALIENCE_COLS):
                if (
                    col in r.index
                    and pd.notna(r.get(col))
                    and str(r.get(col)) not in ('', 'nan', 'no')
                ):
                    return (i, c.get('dist', 9999), node_id_tb)
            return (len(SALIENCE_COLS), c.get('dist', 9999), node_id_tb)

        return min(candidates, key=salience_rank)

    def solve(self, instruction_text: str, start_node: str, mode: str = "label") -> dict:
        """
        Master Controller: translates an RVS instruction into a symbolic state.

        Modes:
            "label"   (default): scientific oracle mode for this project.
                      Preserves ambiguity instead of guessing.
                          0 reachable candidates  -> Contradictory
                          1 reachable candidate   -> Answerable
                          >1 reachable candidates -> Ambiguous

            "resolve": RVS-style resolver. If multiple reachable candidates
                       remain, pick one via salience. Useful for original-goal
                       recovery, but should NOT be used for underspecification
                       labels — that's exactly what we're trying to measure.

        Returns:
            dict with keys:
              state                       : Answerable | Ambiguous | Contradictory
              mode                        : "label" or "resolve"
              extracted_category          : LANDMARK_GROUPS key or "UNKNOWN"
              extracted_noun              : raw noun phrase or None
              extracted_direction         : N/NE/E/.../NW or None
              candidate_count             : POIs found before reachability filter
              reachable_candidate_count   : POIs found AND reachable
              resolution_stage            : strict | category_direction_fallback | fuzzy_name_fallback
              target_node                 : single target (Answerable only, in either mode)
              candidate_nodes             : up to 50 IDs (Ambiguous only, "label" mode)
              selection_strategy          : how the target was chosen ("resolve" mode only)
        """
        if mode not in {"label", "resolve"}:
            raise ValueError("mode must be either 'label' or 'resolve'")

        # ------ Phase A: Candidate Resolution ------
        category, raw_noun, target_dir = extract_rvs_target(instruction_text)
        tags = config.LANDMARK_GROUPS.get(category, {})
        start_data = self.G.nodes[start_node]
        lat, lon = start_data['y'], start_data['x']
        horizon = config.GLOBAL_SEARCH_HORIZON_METERS

        metadata = {
            "mode": mode,
            "extracted_category": category,
            "extracted_noun": raw_noun,
            "extracted_direction": target_dir,
        }

        # Step 1 — Strict: category + name + direction
        candidates = self.oracle.resolve_nearby_candidates(
            tags, lat, lon, radius_m=horizon, landmark_name=raw_noun
        )
        candidates = self._apply_directional_filter(candidates, start_node, target_dir)
        resolution_stage = "strict_category_name_direction"

        # Step 2 — Category + direction, no name
        if not candidates:
            candidates = self.oracle.resolve_nearby_candidates(
                tags, lat, lon, radius_m=horizon, landmark_name=None
            )
            candidates = self._apply_directional_filter(candidates, start_node, target_dir)
            resolution_stage = "category_direction_fallback"

        # Step 3 — Last resort: fuzzy name search near the start node
        if not candidates and raw_noun:
            fallback_node = self.oracle.resolve_landmark(
                raw_noun, context_node=start_node, radius_m=horizon
            )
            if fallback_node:
                node_data = self.G.nodes[fallback_node]
                d = utils.haversine(lat, lon, node_data['y'], node_data['x'])
                candidates = [{
                    "node_id": fallback_node,
                    "coords": (node_data['y'], node_data['x']),
                    "dist": d,
                }]
                resolution_stage = "fuzzy_name_fallback"

        # Ensure every candidate has a distance entry.
        for c in candidates:
            if 'dist' not in c:
                clat, clon = c['coords']
                c['dist'] = utils.haversine(lat, lon, clat, clon)

        metadata["resolution_stage"] = resolution_stage
        metadata["candidate_count"] = len(candidates)

        # ------ Phase B: Reachability-aware candidate set ------
        # IMPORTANT: We filter to reachable BEFORE counting.
        # The old code collapsed multi-candidate cases via salience first, so
        # it could not detect "5 reachable cafes, all valid → Ambiguous".
        reachable_candidates = []
        for c in candidates:
            node_id = c.get('node_id')
            if node_id == start_node or self.check_reachability(start_node, node_id):
                reachable_candidates.append(c)

        reachable_count = len(reachable_candidates)
        metadata["reachable_candidate_count"] = reachable_count

        if reachable_count == 0:
            return {**metadata, "state": config.STATE_CONTRADICTORY}

        # ------ Phase C1: label mode — preserve ambiguity ------
        if mode == "label":
            if reachable_count == 1:
                return {
                    **metadata,
                    "state": config.STATE_ANSWERABLE,
                    "target_node": reachable_candidates[0]['node_id'],
                }
            return {
                **metadata,
                "state": config.STATE_AMBIGUOUS,
                "candidate_nodes": [c['node_id'] for c in reachable_candidates[:50]],
            }

        # ------ Phase C2: resolve mode — pick one ------
        if reachable_count > 1:
            best = self._pick_by_salience(reachable_candidates)
            strategy = "salience"
        else:
            best = reachable_candidates[0]
            strategy = "single_candidate"

        return {
            **metadata,
            "state": config.STATE_ANSWERABLE,
            "target_node": best['node_id'],
            "selection_strategy": strategy,
        }
