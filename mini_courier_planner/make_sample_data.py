"""Creates the three sample datasets in data/ (run:  python make_sample_data.py)."""
import math
import os

from courier.data_loader import generate_dataset, save_dataset
from courier.graph import Graph
from courier.models import Dataset, Parcel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def small_gurugram() -> Dataset:
    """Hand-made dataset: schematic Gurugram map (distances are illustrative, not GPS)."""
    places = [
        (0, "KRMU Hub (Sohna Rd)", 10, 2), (1, "Badshahpur", 12, 6), (2, "Sector 49", 14, 9),
        (3, "Golf Course Ext", 18, 12), (4, "Sector 56", 20, 10), (5, "HUDA City Centre", 16, 16),
        (6, "Sector 14", 10, 18), (7, "MG Road", 20, 18), (8, "Cyber Hub", 24, 16),
        (9, "Udyog Vihar", 22, 22), (10, "Sector 29", 14, 21),
    ]
    g = Graph()
    pos = {}
    for i, name, x, y in places:
        g.add_node(i, name, x, y)
        pos[i] = (x, y)
    roads = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (4, 8), (5, 6), (5, 7),
             (6, 10), (10, 5), (10, 9), (7, 8), (7, 9), (8, 9)]
    for u, v in roads:
        g.add_edge(u, v, round(math.dist(pos[u], pos[v]) * 1.25, 1))
    g.add_edge(0, 6, 17.0)    # direct highway
    g.add_edge(1, 6, 14.0)
    rows = [  # id, dest, kg, value, deadline(h), priority
        ("P01", 5, 8, 50, 2.0, 5), ("P02", 3, 12, 35, 4.0, 3), ("P03", 7, 5, 45, 3.0, 4),
        ("P04", 9, 15, 30, 8.0, 2), ("P05", 1, 6, 20, 6.0, 2), ("P06", 8, 10, 55, 2.5, 5),
        ("P07", 2, 4, 25, 5.0, 3), ("P08", 6, 9, 28, 7.0, 2), ("P09", 10, 7, 40, 4.5, 4),
        ("P10", 4, 11, 18, 10.0, 1), ("P11", 7, 3, 22, 6.0, 3), ("P12", 5, 14, 33, 9.0, 2),
        ("P13", 9, 6, 48, 3.5, 4), ("P14", 3, 5, 15, 12.0, 1),
        ("P15", 4, 60, 10, 20.0, 1),   # deliberately heavier than the van -> edge case
    ]
    parcels = [Parcel(*r) for r in rows]
    return Dataset("small_gurugram", g, parcels, 0, 40, 30.0, 5.0, 10.0)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    save_dataset(small_gurugram(), os.path.join(DATA, "small_dataset.json"))
    save_dataset(generate_dataset("medium_random", 40, 80, seed=7, capacity=60),
                 os.path.join(DATA, "medium_dataset.json"))
    save_dataset(generate_dataset("large_random", 250, 600, seed=21, capacity=100),
                 os.path.join(DATA, "large_dataset.json"))
    print("Datasets written to", DATA)
