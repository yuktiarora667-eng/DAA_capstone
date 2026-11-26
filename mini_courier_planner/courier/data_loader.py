"""Load / validate / save / generate datasets (Phase 1 - data design).

JSON layout (see README):
{ "name", "depot", "capacity_kg", "speed_kmph", "service_minutes", "loading_minutes",
  "nodes":   [{"id", "name", "x", "y"}],
  "edges":   [[u, v, km], ...],
  "parcels": [{"id", "destination", "weight", "value", "deadline_h", "priority"}] }
"""
import json
import math
import random
from typing import Dict, List, Tuple

from .graph import Graph
from .models import Dataset, Parcel


class DatasetError(ValueError):
    pass


def load_dataset(path: str) -> Dataset:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise DatasetError(f"dataset file not found: {path}")
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path} is not valid JSON: {e}")
    return dataset_from_dict(raw)


def dataset_from_dict(raw: dict) -> Dataset:
    try:
        g = Graph()
        for n in raw["nodes"]:
            g.add_node(int(n["id"]), n.get("name"), n.get("x"), n.get("y"))
        for u, v, w in raw["edges"]:
            if int(u) not in g.adj or int(v) not in g.adj:
                raise DatasetError(f"edge ({u},{v}) uses an unknown node")
            g.add_edge(int(u), int(v), float(w))
        depot = int(raw.get("depot", 0))
        if depot not in g.adj:
            raise DatasetError("depot is not a node of the graph")
        parcels: List[Parcel] = []
        seen = set()
        for p in raw["parcels"]:
            parcel = Parcel(str(p["id"]), int(p["destination"]), int(p["weight"]),
                            int(p["value"]), float(p["deadline_h"]), int(p["priority"]))
            if parcel.pid in seen:
                raise DatasetError(f"duplicate parcel id {parcel.pid}")
            seen.add(parcel.pid)
            if parcel.destination not in g.adj:
                raise DatasetError(f"parcel {parcel.pid}: destination not in graph")
            if parcel.destination == depot:
                raise DatasetError(f"parcel {parcel.pid}: destination equals the depot")
            if parcel.weight <= 0 or parcel.value <= 0 or parcel.deadline <= 0:
                raise DatasetError(f"parcel {parcel.pid}: weight/value/deadline must be > 0")
            if not 1 <= parcel.priority <= 5:
                raise DatasetError(f"parcel {parcel.pid}: priority must be 1..5")
            parcels.append(parcel)
        capacity = int(raw.get("capacity_kg", 50))
        if capacity <= 0:
            raise DatasetError("capacity_kg must be positive")
        return Dataset(raw.get("name", "dataset"), g, parcels, depot, capacity,
                       float(raw.get("speed_kmph", 30)), float(raw.get("service_minutes", 5)),
                       float(raw.get("loading_minutes", 10)))
    except KeyError as e:
        raise DatasetError(f"missing field in dataset: {e}")


def dataset_to_dict(ds: Dataset) -> dict:
    g = ds.graph
    return {
        "name": ds.name, "depot": ds.depot, "capacity_kg": ds.capacity,
        "speed_kmph": ds.speed_kmph, "service_minutes": ds.service_minutes,
        "loading_minutes": ds.loading_minutes,
        "nodes": [{"id": n, "name": g.name(n),
                   "x": g.coords.get(n, (None, None))[0], "y": g.coords.get(n, (None, None))[1]}
                  for n in g.nodes],
        "edges": [[u, v, w] for u, v, w in g.edges()],
        "parcels": [{"id": p.pid, "destination": p.destination, "weight": p.weight,
                     "value": p.value, "deadline_h": p.deadline, "priority": p.priority}
                    for p in ds.parcels],
    }


def save_dataset(ds: Dataset, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset_to_dict(ds), f, indent=1)


# ------------------------------------------------------------------ generator
def generate_graph(n_nodes: int, seed: int = 1, extra_links: int = 2,
                   size: float = 30.0) -> Graph:
    """Random connected road network: Prim spanning tree + a few nearest-neighbour links.

    Edge weight = Euclidean distance * road factor (1.0 .. 1.3).
    """
    rng = random.Random(seed)
    g = Graph()
    pts: Dict[int, Tuple[float, float]] = {}
    for i in range(n_nodes):
        pts[i] = (round(rng.uniform(0, size), 1), round(rng.uniform(0, size), 1))
        g.add_node(i, "Depot" if i == 0 else f"Stop-{i}", *pts[i])

    def road(a: int, b: int) -> float:
        return round(math.dist(pts[a], pts[b]) * rng.uniform(1.0, 1.3) + 0.1, 1)

    linked = set()

    def link(a: int, b: int) -> None:
        key = (min(a, b), max(a, b))
        if a != b and key not in linked:
            linked.add(key)
            g.add_edge(a, b, road(a, b))

    # Prim's MST on coordinates -> guarantees connectivity
    in_tree = {0}
    best = {i: (math.dist(pts[0], pts[i]), 0) for i in range(1, n_nodes)}
    while len(in_tree) < n_nodes:
        v = min(best, key=lambda k: best[k][0])
        link(v, best[v][1])
        in_tree.add(v)
        del best[v]
        for u in best:
            d = math.dist(pts[v], pts[u])
            if d < best[u][0]:
                best[u] = (d, v)
    # extra links to nearest neighbours -> alternative routes (graph is not a tree)
    for i in range(n_nodes):
        near = sorted((j for j in range(n_nodes) if j != i), key=lambda j: math.dist(pts[i], pts[j]))
        for j in near[:extra_links]:
            link(i, j)
    return g


def generate_dataset(name: str, n_nodes: int, n_parcels: int, seed: int = 1,
                     capacity: int = 50) -> Dataset:
    rng = random.Random(seed + 1000)
    g = generate_graph(n_nodes, seed)
    parcels = []
    # rough planning horizon (hours): number of trips * ~7 h per trip
    horizon = max(8.0, n_parcels * 8 / capacity * 7)
    for k in range(1, n_parcels + 1):
        pr = rng.randint(1, 5)
        weight = rng.randint(1, 15)
        value = pr * 10 + rng.randint(0, 15)
        deadline = round(rng.uniform(3, horizon * 1.3) / (1 + 0.15 * (pr - 1)) * 2) / 2
        parcels.append(Parcel(f"P{k:03d}", rng.randint(1, n_nodes - 1), weight, value,
                              max(deadline, 1.0), pr))
    return Dataset(name, g, parcels, 0, capacity, 40.0, 5.0, 10.0)
