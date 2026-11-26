"""Console report for each phase (plain ASCII so it works on any Windows/Linux terminal)."""
import time
from typing import List, Sequence

from .data_loader import dataset_to_dict
from .knapsack import format_dp_table, knapsack_bruteforce, knapsack_greedy
from .models import Dataset, Parcel
from .planner import PlanResult, Trip
from .sorting import ALGORITHMS, parcel_key


def banner(title: str) -> str:
    return "\n" + "=" * 78 + f"\n{title}\n" + "=" * 78


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
              for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join("{:<%d}" % w for w in widths) + " |"
    out = [line, fmt.format(*headers), line]
    out += [fmt.format(*r) for r in cells]
    out.append(line)
    return "\n".join(out)


def names(ds: Dataset, nodes: Sequence[int]) -> str:
    return " -> ".join(ds.graph.name(n) for n in nodes)


def hhmm(hours: float) -> str:
    m = int(round(hours * 60))
    return f"{m // 60:02d}h{m % 60:02d}m"


def parcel_rows(parcels: List[Parcel]):
    return [(p.pid, p.destination, p.weight, p.value, p.deadline, p.priority) for p in parcels]


PARCEL_HEAD = ("Parcel", "Dest", "kg", "Value", "Deadline(h)", "Prio")


# --------------------------------------------------------------- Phase 1
def phase1(ds: Dataset) -> str:
    g = ds.graph
    total_w = sum(p.weight for p in ds.parcels)
    lines = [banner("PHASE 1 - PROBLEM DESIGN AND PLANNING"),
             f"Dataset          : {ds.name}",
             f"Depot            : node {ds.depot} ({g.name(ds.depot)})",
             f"Road graph       : {len(g.nodes)} nodes, {g.num_edges()} undirected weighted edges (km), "
             f"connected = {g.is_connected()}",
             f"Parcels          : {len(ds.parcels)}  (total weight {total_w} kg)",
             f"Van capacity     : {ds.capacity} kg   speed {ds.speed_kmph:g} km/h   "
             f"service {ds.service_minutes:g} min/stop   reload {ds.loading_minutes:g} min",
             "",
             "Parcel attributes : id, destination node, weight (kg), value (importance score),",
             "                    deadline (hours after first dispatch), priority (1 low .. 5 urgent)",
             "Graph format      : adjacency list  adj[u] = [(v, km), ...]   (JSON: nodes + edges)",
             "",
             "Algorithm pipeline (one trip, repeated until all parcels are loaded):",
             "  parcels --> [MERGE SORT: deadline, priority] --> [0/1 KNAPSACK DP: best load for the van]",
             "          --> [DIJKSTRA: road distance between stops] --> [NEAREST-NEIGHBOUR TSP: visit order]",
             "          --> route + arrival times + deadline check",
             "",
             "Why these techniques",
             "  Merge sort   : O(n log n) guaranteed and stable -> ties keep their original order",
             "  Knapsack DP  : optimal load, O(n*W); greedy is faster but can miss the optimum",
             "  Dijkstra     : shortest ROAD path (not straight line) with non-negative km weights",
             "  NN TSP       : visiting order is NP-hard; nearest neighbour is a fast approximation",
             "                 (exact Held-Karp DP used only to measure the approximation error)"]
    return "\n".join(lines)


# --------------------------------------------------------------- Phase 2
def phase2(res: PlanResult, show: int = 15) -> str:
    ds = res.dataset
    lines = [banner("PHASE 2 - PACKAGE PRIORITISATION AND BOX FILLING")]
    lines.append("Step 2a: parcels sorted by (deadline asc, priority desc) using MERGE SORT")
    shown = res.sorted_parcels[:show]
    lines.append(ascii_table(PARCEL_HEAD, parcel_rows(shown)))
    if len(res.sorted_parcels) > show:
        lines.append(f"... {len(res.sorted_parcels) - show} more parcels not shown")

    # compare the sorting algorithms on THIS dataset
    timing = []
    ref = sorted(ds.parcels, key=parcel_key)
    for name, fn in ALGORITHMS.items():
        t0 = time.perf_counter()
        for _ in range(20):
            out = fn(ds.parcels, key=parcel_key)
        timing.append((name, f"{(time.perf_counter() - t0) / 20 * 1000:.4f}", "yes" if out == ref else "NO"))
    lines += ["", "All sorting algorithms on this dataset (avg of 20 runs):",
              ascii_table(("Algorithm", "ms", "Same order as reference"), timing)]

    if res.unassigned:
        lines += ["", "WARNING - parcels heavier than the van capacity (cannot be loaded): " +
                  ", ".join(f"{p.pid} ({p.weight} kg)" for p in res.unassigned)]

    if res.trips:
        t = res.trips[0]
        k = t.knapsack
        lines += ["", f"Step 2b: 0/1 KNAPSACK (Dynamic Programming) for trip 1 - capacity {ds.capacity} kg"]
        if k.table is not None and len(t.candidates) <= 20 and ds.capacity <= 100:
            lines.append(format_dp_table(t.candidates, k.table, ds.capacity, 5))
        lines += ["", f"Trip 1 optimal value = {k.best_value}   load = {k.total_weight}/{ds.capacity} kg",
                  "Selected parcels:", ascii_table(PARCEL_HEAD, parcel_rows(t.parcels))]
        if len(t.candidates) <= 20:
            bf = knapsack_bruteforce(t.candidates, ds.capacity)
            lines.append(f"Check: brute force over all 2^{len(t.candidates)} subsets gives value "
                         f"{bf.best_value}  -> DP is {'CORRECT' if bf.best_value == k.best_value else 'WRONG'}")
        gr = knapsack_greedy(t.candidates, ds.capacity)
        lines.append(f"Greedy (value/weight) would give {gr.best_value} "
                     f"({gr.best_value / k.best_value * 100:.1f} % of the DP optimum)")
    lines += ["", "Packing plan (all trips):"]
    lines.append(ascii_table(("Trip", "Parcels", "Load kg", "Value"),
                             [(t.number, " ".join(p.pid for p in t.parcels),
                               f"{t.knapsack.total_weight}/{ds.capacity}", t.knapsack.best_value)
                              for t in res.trips[:25]]))
    if len(res.trips) > 25:
        lines.append(f"... {len(res.trips) - 25} more trips")
    return "\n".join(lines)


# --------------------------------------------------------------- Phase 3
def trip_report(ds: Dataset, t: Trip, matrix: bool = True) -> str:
    g = ds.graph
    lines = [f"--- Trip {t.number} --- parcels: {', '.join(p.pid for p in t.parcels)}",
             f"Stops to visit : {t.stops}"]
    if matrix and len(t.stops) <= 8:
        pts = [ds.depot] + t.stops
        head = ("from\\to",) + tuple(str(p) for p in pts)
        rows = [(str(a),) + tuple(f"{t.dist[(a, b)]:.1f}" for b in pts) for a in pts]
        lines += ["Dijkstra shortest-distance matrix (km):", ascii_table(head, rows)]
    lines += [f"Nearest-neighbour tour : {t.tour}",
              f"   = {names(ds, t.tour)}",
              f"Full road path ({len(t.full_path)} nodes) : {t.full_path}",
              f"Route length   : {t.length:.1f} km"]
    if t.two_opt_length is not None:
        gain = (t.length - t.two_opt_length) / t.length * 100 if t.length else 0
        lines.append(f"With 2-opt     : {t.two_opt_length:.1f} km  ({gain:.1f} % shorter)")
    if t.optimal_length is not None:
        lines.append(f"Exact optimum  : {t.optimal_length:.1f} km  -> NN is "
                     f"{t.length / t.optimal_length:.3f} x optimal")
    rows = [(p.pid, g.name(p.destination), hhmm(t.arrival[p.destination]), f"{p.deadline:g}h",
             "ON TIME" if p not in t.late else "LATE") for p in t.parcels]
    lines += ["Schedule:", ascii_table(("Parcel", "Customer", "Arrival", "Deadline", "Status"), rows),
              f"Trip time: leaves {hhmm(t.start_time)}, back at depot {hhmm(t.end_time)}"]
    return "\n".join(lines)


def phase3(res: PlanResult, max_trips: int = 3) -> str:
    ds = res.dataset
    lines = [banner("PHASE 3 - ROUTE MAKER AND GRAPH PROCESSING")]
    for t in res.trips[:max_trips]:
        lines += [trip_report(ds, t), ""]
    if len(res.trips) > max_trips:
        lines.append(f"... {len(res.trips) - max_trips} more trips (see the summary)")
    return "\n".join(lines)


# --------------------------------------------------------------- Phase 5
def summary(res: PlanResult) -> str:
    ds = res.dataset
    tm = res.timings
    total_t = sum(tm.values())
    lines = [banner("FINAL SUMMARY"),
             f"Dataset              : {ds.name}",
             f"Parcels delivered    : {res.delivered} / {len(ds.parcels)}"
             + (f"   (not loadable: {', '.join(p.pid for p in res.unassigned)})" if res.unassigned else ""),
             f"Trips needed         : {len(res.trips)}",
             f"Total distance       : {res.total_distance:.1f} km",
             f"Finished all trips at: {hhmm(res.trips[-1].end_time) if res.trips else '-'}",
             f"Deadlines missed     : {res.late_count} of {res.delivered}",
             "", "Algorithm time per step:"]
    rows = [(s, f"{tm[s] * 1000:.3f} ms", f"{tm[s] / total_t * 100:.1f} %" if total_t else "-") for s in tm]
    lines.append(ascii_table(("Step", "Time", "Share"), rows))
    exact = [t for t in res.trips if t.optimal_length]
    if exact:
        avg = sum(t.length / t.optimal_length for t in exact) / len(exact)
        lines.append(f"Average nearest-neighbour / optimal ratio over {len(exact)} trips: {avg:.3f}")
    return "\n".join(lines)
