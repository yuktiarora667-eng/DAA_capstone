"""Route construction (Phase 3).

1. build_distance_matrix : Dijkstra from every stop -> shortest road distance
                           between every pair of stops (+ the actual paths).
2. nearest_neighbor_tour : greedy TSP approximation  O(k^2)
3. two_opt               : local-search improvement of a tour (optimisation idea)
4. held_karp             : EXACT TSP by DP over subsets O(k^2 * 2^k) - used only
                           to measure how good the approximation is (k small).
"""
from typing import Dict, List, Sequence, Tuple

from .graph import INF, Graph, dijkstra, reconstruct_path

Dist = Dict[Tuple[int, int], float]
Paths = Dict[Tuple[int, int], List[int]]


def build_distance_matrix(graph: Graph, stops: Sequence[int]) -> Tuple[Dist, Paths]:
    dist: Dist = {}
    paths: Paths = {}
    for s in stops:
        d, prev = dijkstra(graph, s)
        for t in stops:
            if d[t] == INF:
                raise ValueError(f"node {t} is unreachable from node {s}")
            dist[(s, t)] = d[t]
            paths[(s, t)] = reconstruct_path(prev, s, t)
    return dist, paths


def tour_length(tour: Sequence[int], dist: Dist) -> float:
    return sum(dist[(tour[i], tour[i + 1])] for i in range(len(tour) - 1))


def nearest_neighbor_tour(depot: int, stops: Sequence[int], dist: Dist) -> List[int]:
    """Start at depot, always go to the closest unvisited stop, then return.

    Time O(k^2).  Not optimal, but easy to explain and fast.
    For metric instances the tour is at most O(log k) times optimal.
    """
    remaining = set(stops) - {depot}
    tour = [depot]
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda s: (dist[(cur, s)], s))   # tie -> lower id
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    tour.append(depot)
    return tour


def two_opt(tour: Sequence[int], dist: Dist) -> List[int]:
    """Repeatedly reverse a segment if that shortens the tour (depot stays first)."""
    route = list(tour[:-1])            # cyclic route without the repeated depot
    n = len(route)
    if n < 4:
        return list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = route[i - 1], route[i]
                c, d = route[j], route[(j + 1) % n]
                delta = dist[(a, c)] + dist[(b, d)] - dist[(a, b)] - dist[(c, d)]
                if delta < -1e-9:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
    return route + [route[0]]


def held_karp(depot: int, stops: Sequence[int], dist: Dist, max_stops: int = 12) -> Tuple[float, List[int]]:
    """Exact TSP (Held-Karp bitmask DP). Returns (optimal length, tour)."""
    pts = [s for s in stops if s != depot]
    k = len(pts)
    if k == 0:
        return 0.0, [depot, depot]
    if k > max_stops:
        raise ValueError(f"Held-Karp limited to {max_stops} stops (got {k})")
    full = 1 << k
    dp = [[INF] * k for _ in range(full)]
    parent = [[-1] * k for _ in range(full)]
    for j in range(k):
        dp[1 << j][j] = dist[(depot, pts[j])]
    for mask in range(full):
        for j in range(k):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            for nxt in range(k):
                if (mask >> nxt) & 1:
                    continue
                nmask = mask | (1 << nxt)
                cand = dp[mask][j] + dist[(pts[j], pts[nxt])]
                if cand < dp[nmask][nxt]:
                    dp[nmask][nxt] = cand
                    parent[nmask][nxt] = j
    best, last = INF, -1
    for j in range(k):
        total = dp[full - 1][j] + dist[(pts[j], depot)]
        if total < best:
            best, last = total, j
    order, mask = [], full - 1
    while last != -1:
        order.append(pts[last])
        prev = parent[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return best, [depot] + order + [depot]


def expand_route(tour: Sequence[int], paths: Paths) -> List[int]:
    """Replace each hop of the tour by the real road path found by Dijkstra."""
    full = [tour[0]]
    for a, b in zip(tour, tour[1:]):
        full.extend(paths[(a, b)][1:])
    return full


def schedule(tour: Sequence[int], dist: Dist, speed_kmph: float, service_minutes: float,
             start_time: float = 0.0) -> Tuple[Dict[int, float], float]:
    """Arrival time (hours) at each customer stop and the time the van is back at the depot."""
    t = start_time
    arrival: Dict[int, float] = {}
    hops = list(zip(tour, tour[1:]))
    for idx, (a, b) in enumerate(hops):
        t += dist[(a, b)] / speed_kmph
        if idx == len(hops) - 1:          # last hop = return to depot, no service
            break
        arrival[b] = t
        t += service_minutes / 60.0
    return arrival, t
