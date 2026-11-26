"""Integrated planner: sorting -> knapsack -> Dijkstra -> nearest-neighbour TSP.

The planner repeats the pipeline trip after trip until every deliverable
parcel is assigned. Each step is timed so Phase 4 can compare components.
"""
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .knapsack import KnapsackResult, knapsack_dp
from .models import Dataset, Parcel
from .routing import (build_distance_matrix, expand_route, held_karp, nearest_neighbor_tour,
                      schedule, tour_length, two_opt)
from .sorting import sort_parcels

STEPS = ("sorting", "knapsack", "dijkstra", "tsp")


@dataclass
class Trip:
    number: int
    parcels: List[Parcel]
    knapsack: KnapsackResult
    candidates: List[Parcel]
    stops: List[int]
    tour: List[int]
    full_path: List[int]
    length: float
    start_time: float
    end_time: float
    arrival: Dict[int, float]
    late: List[Parcel]
    dist: dict = field(repr=False, default_factory=dict)
    two_opt_tour: Optional[List[int]] = None
    two_opt_length: Optional[float] = None
    optimal_length: Optional[float] = None


@dataclass
class PlanResult:
    dataset: Dataset
    sorted_parcels: List[Parcel]
    trips: List[Trip]
    unassigned: List[Parcel]          # heavier than the van - can never be loaded
    timings: Dict[str, float]

    @property
    def total_distance(self) -> float:
        return sum(t.length for t in self.trips)

    @property
    def delivered(self) -> int:
        return sum(len(t.parcels) for t in self.trips)

    @property
    def late_count(self) -> int:
        return sum(len(t.late) for t in self.trips)


URGENT_WINDOW_H = 6.0     # parcels due within this many hours of 'now' are treated as urgent
URGENT_BOOST = 5          # their knapsack value is multiplied by this


def boosted(parcels: List[Parcel], now: float) -> List[Parcel]:
    """Urgency-aware values: parcels due soon get a larger value so the DP loads them first."""
    return [replace(p, value=p.value * URGENT_BOOST) if p.deadline <= now + URGENT_WINDOW_H else p
            for p in parcels]


def plan(ds: Dataset, sort_algo: str = "merge", improve: bool = True,
         exact_limit: int = 10, trace: bool = False, max_trips: int = 10_000,
         urgency: bool = False) -> PlanResult:
    """Run the whole pipeline.

    improve     : also run 2-opt on each tour (reported separately, NN stays the main answer)
    exact_limit : compute the optimal tour with Held-Karp when a trip has <= this many stops
                  (0 disables it; it is NOT counted in the timings)
    trace       : keep the DP table of the first trip for display
    """
    timings = {s: 0.0 for s in STEPS}

    t0 = time.perf_counter()
    ordered = sort_parcels(ds.parcels, sort_algo)
    timings["sorting"] = time.perf_counter() - t0

    unassigned = [p for p in ordered if p.weight > ds.capacity]
    remaining = [p for p in ordered if p.weight <= ds.capacity]

    trips: List[Trip] = []
    clock = 0.0
    while remaining and len(trips) < max_trips:
        t0 = time.perf_counter()
        cands = boosted(remaining, clock) if urgency else remaining
        res = knapsack_dp(cands, ds.capacity, keep_table=trace and not trips)
        timings["knapsack"] += time.perf_counter() - t0
        if not res.selected:
            break
        orig = {p.pid: p for p in remaining}          # back to the real (un-boosted) parcels
        res.selected = [orig[p.pid] for p in res.selected]

        stops = sorted({p.destination for p in res.selected})

        t0 = time.perf_counter()
        dist, paths = build_distance_matrix(ds.graph, [ds.depot] + stops)
        timings["dijkstra"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tour = nearest_neighbor_tour(ds.depot, stops, dist)
        full = expand_route(tour, paths)
        length = tour_length(tour, dist)
        timings["tsp"] += time.perf_counter() - t0

        start = clock + (ds.loading_minutes / 60.0 if trips else 0.0)
        arrival, end = schedule(tour, dist, ds.speed_kmph, ds.service_minutes, start)
        late = [p for p in res.selected if arrival[p.destination] > p.deadline + 1e-9]

        trip = Trip(len(trips) + 1, res.selected, res, list(cands), stops, tour, full,
                    length, start, end, arrival, late, dist)
        if improve:
            trip.two_opt_tour = two_opt(tour, dist)
            trip.two_opt_length = tour_length(trip.two_opt_tour, dist)
        if exact_limit and len(stops) <= exact_limit:
            trip.optimal_length, _ = held_karp(ds.depot, stops, dist, exact_limit)
        trips.append(trip)

        taken = {p.pid for p in res.selected}
        remaining = [p for p in remaining if p.pid not in taken]
        clock = end

    return PlanResult(ds, ordered, trips, unassigned, timings)
