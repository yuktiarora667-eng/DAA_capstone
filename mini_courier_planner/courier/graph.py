"""Graph representation + Dijkstra's shortest-path algorithm (Phase 3).

The road network is an undirected weighted graph stored as an adjacency list:
    adj[u] = [(v, weight_km), ...]
"""
import heapq
from typing import Dict, List, Optional, Tuple

INF = float("inf")


class Graph:
    """Undirected, weighted graph (adjacency list)."""

    def __init__(self) -> None:
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self.names: Dict[int, str] = {}
        self.coords: Dict[int, Tuple[float, float]] = {}

    # ---- construction -------------------------------------------------
    def add_node(self, node: int, name: Optional[str] = None,
                 x: Optional[float] = None, y: Optional[float] = None) -> None:
        self.adj.setdefault(node, [])
        self.names[node] = name if name else f"N{node}"
        if x is not None and y is not None:
            self.coords[node] = (float(x), float(y))

    def add_edge(self, u: int, v: int, w: float) -> None:
        if w < 0:
            raise ValueError("Dijkstra needs non-negative edge weights")
        for n in (u, v):
            if n not in self.adj:
                self.add_node(n)
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    # ---- queries ------------------------------------------------------
    @property
    def nodes(self) -> List[int]:
        return sorted(self.adj)

    def edges(self) -> List[Tuple[int, int, float]]:
        seen = set()
        out = []
        for u in self.adj:
            for v, w in self.adj[u]:
                key = (min(u, v), max(u, v), w)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return sorted(out)

    def num_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    def name(self, node: int) -> str:
        return self.names.get(node, f"N{node}")

    def is_connected(self) -> bool:
        if not self.adj:
            return True
        start = next(iter(self.adj))
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v, _ in self.adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(self.adj)


def dijkstra(graph: Graph, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Dijkstra with a binary min-heap (lazy deletion).

    Time  : O((V + E) log V)      Space : O(V)
    Returns (dist, prev) - prev lets us rebuild the actual path.
    """
    dist = {n: INF for n in graph.adj}
    prev: Dict[int, Optional[int]] = {n: None for n in graph.adj}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:          # stale heap entry
            continue
        for v, w in graph.adj[u]:
            nd = d + w
            if nd < dist[v]:     # relaxation
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def dijkstra_array(graph: Graph, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Classic O(V^2) Dijkstra (linear scan for the minimum).

    Kept only to compare against the heap version in Phase 4.
    """
    dist = {n: INF for n in graph.adj}
    prev: Dict[int, Optional[int]] = {n: None for n in graph.adj}
    dist[source] = 0.0
    done = set()
    for _ in range(len(graph.adj)):
        u, best = None, INF
        for n in graph.adj:
            if n not in done and dist[n] < best:
                u, best = n, dist[n]
        if u is None:
            break
        done.add(u)
        for v, w in graph.adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
    return dist, prev


def reconstruct_path(prev: Dict[int, Optional[int]], source: int, target: int) -> List[int]:
    """Walk the prev[] pointers back from target to source."""
    if target != source and prev.get(target) is None:
        return []                      # unreachable
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path
