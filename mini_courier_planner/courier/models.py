"""Data models: Parcel and Dataset."""
from dataclasses import dataclass
from typing import List

from .graph import Graph


@dataclass(frozen=True)
class Parcel:
    pid: str            # unique id, e.g. "P01"
    destination: int    # graph node where it must be delivered
    weight: int         # kg  (integer -> needed for the DP table)
    value: int          # business importance / revenue score
    deadline: float     # hours after the first dispatch
    priority: int       # 1 (low) .. 5 (urgent)


@dataclass
class Dataset:
    name: str
    graph: Graph
    parcels: List[Parcel]
    depot: int = 0
    capacity: int = 50           # van capacity in kg
    speed_kmph: float = 30.0     # average city speed
    service_minutes: float = 5.0   # time spent at each customer
    loading_minutes: float = 10.0  # time to reload at the depot between trips
