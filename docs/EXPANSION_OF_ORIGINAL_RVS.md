## 🔬 Project Extension: Graph-Based Reachability Filter

### What RVS Does
The original RVS paper uses **human validation** as its oracle. An instruction is
"answerable" if a human follower can pin the goal within 100 meters. No graph
reachability check is performed, the paper treats navigation as a geospatial
grounding task (find the coordinate) rather than a path-planning task (find a
legal route).

### What We Add
Our project introduces a **symbolic graph-based solver** that extends RVS with an
algorithmic oracle. As part of this, we add a lightweight **undirected reachability
filter** that RVS does not perform:

> A candidate node is considered valid only if it belongs to the same **undirected
> connected component** as the start node in the city graph.

### Justification
This is consistent with the RVS paper's own framing. Section 2 describes the
environment as a graph of streets and landmarks connected for **"physical access"**, not legal turn-by-turn routing. One-way street constraints and directed edges
are never mentioned. Our undirected component check mirrors this exact definition:

- ✅ Filters nodes on genuinely disconnected graph islands (e.g. a pedestrian
  bridge with no street-level connection)
- ✅ Does not enforce driving legality or one-way street constraints
- ✅ O(1) per query via precomputed connectivity map
- ✅ No-op for single-component cities (Pittsburgh, Manhattan core), all nodes
  pass, consistent with RVS behavior

### What It Is Not
- ❌ Not a path-planning check (no Dijkstra, no A*)
- ❌ Not a legal routing check (no directed edge constraints)
- ❌ Not part of RVS methodology, this is our addition

### Implementation
```python
# Precomputed once at OracleEngine init — O(1) per query
self.scc_lookup, num_comp = utils.get_connectivity_map(self.G)

# get_connectivity_map uses undirected connected components
def get_connectivity_map(G):
    undirected_G = G.to_undirected()
    components = list(nx.connected_components(undirected_G))
    conn_map = {}
    for i, component in enumerate(components):
        for node in component:
            conn_map[node] = i
    return conn_map, len(components)

# Reachability check in solve()
reachable = [c for c in candidates
             if self.check_reachability_scc(start_node, c['node_id'])]
```

### Effect on Labels
| Scenario | Without Filter | With Filter |
|----------|---------------|-------------|
| Candidate on disconnected island | Answerable (incorrectly) | Contradictory |
| All candidates reachable | Answerable | Answerable (unchanged) |
| Single-component city | Same | Same (no-op) |

### Relation to Project Research Question
Our research question asks how LLMs handle spatially underspecified instructions
**"when answerability remains well-defined"**. The reachability filter is part of
how we define "well-defined", a rendezvous point that is physically inaccessible
from the start location is not a valid answer, regardless of how well the
instruction describes it.