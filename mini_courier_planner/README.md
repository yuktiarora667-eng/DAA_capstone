# Mini-Courier Planner

**Design & Analysis of Algorithms (ETCCDA302) - Capstone Assignment**
K.R. Mangalam University, Gurugram - B.Tech CSE

> Student: ______________  Roll no: ______________  Section: ______  Faculty: Dr. Shweta Agarwal

A prototype "intelligent courier planner". Given parcels and a road network it
1. **sorts** parcels by deadline / priority (merge sort),
2. **chooses the best load** for the van (0/1 knapsack, dynamic programming),
3. **finds shortest road distances** between customers (Dijkstra),
4. **orders the stops** into a delivery route (nearest-neighbour TSP approximation; 2-opt and the exact Held-Karp DP are used to measure/improve it),
5. repeats until every parcel is delivered, checks every **deadline**, and **benchmarks** every component.

Pure Python 3.8+. The algorithms use only the standard library; `matplotlib` is needed only for charts.

## Quick start (VS Code)

```bash
# 1. open this folder in VS Code  (File > Open Folder)
# 2. open the terminal (Ctrl + `) and, optionally, create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows        (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt  # only for charts

# 3. run
python main.py                   # interactive menu
python main.py run               # full demo (Phases 1-3) on the small Gurugram dataset
python main.py run --dataset data/medium_dataset.json
python main.py benchmark         # Phase 4 runtime tests + charts (add --quick for ~2 s)
python main.py all               # regenerate every file in outputs/
python -m unittest discover -s tests -v    # 25 unit tests
```
You can also use the Run-and-Debug panel: `.vscode/launch.json` has ready-made entries.

## Project layout

```
main.py                  command-line entry point + menu
make_sample_data.py      (re)creates data/*.json
courier/
  models.py              Parcel, Dataset
  graph.py               Graph (adjacency list), Dijkstra (heap + O(V^2) version)
  sorting.py             merge / quick / heap / insertion sort
  knapsack.py            0/1 knapsack DP (2-D, 1-D), greedy, brute force
  routing.py             distance matrix, nearest-neighbour, 2-opt, Held-Karp, schedule
  planner.py             the integrated pipeline (multi-trip)
  benchmark.py           Phase 4 timing experiments
  charts.py              runtime charts + route map
  report.py              console reports for every phase
  data_loader.py         JSON load / validate / save / random generator
data/                    small (hand-made, Gurugram), medium (40 nodes/80 parcels), large (250/600)
tests/                   unit tests
docs/                    DESIGN.md (Phase 1), REPORT.md (Phase 5), VIVA_QA.md
outputs/
  traces/                text output of every run
  charts/                runtime charts + route map (PNG)
  screenshots/           console output rendered as PNG
  benchmarks/            CSV files of all timings
```

## Dataset format (JSON)

```json
{ "name": "my_city", "depot": 0, "capacity_kg": 40, "speed_kmph": 30,
  "service_minutes": 5, "loading_minutes": 10,
  "nodes":   [{"id": 0, "name": "Depot", "x": 0, "y": 0}, {"id": 1, "name": "Shop A", "x": 3, "y": 4}],
  "edges":   [[0, 1, 5.0]],
  "parcels": [{"id": "P01", "destination": 1, "weight": 8, "value": 50, "deadline_h": 2.0, "priority": 5}] }
```
`weight` is an integer (kg), `priority` 1-5, `deadline_h` is in hours after the first dispatch.
Run it with `python main.py run --dataset path/to/file.json`. Bad files give a clear error message.

## Algorithms and complexity

| Step | Algorithm | Time | Space |
|---|---|---|---|
| Prioritise | Merge sort (stable) | O(n log n) | O(n) |
| Load the van | 0/1 knapsack DP | O(n*W) | O(n*W) (O(W) if only the value is needed) |
| Shortest paths | Dijkstra + binary heap, once per stop | O(k (V+E) log V) | O(V) |
| Visit order | Nearest neighbour | O(k^2) | O(k^2) matrix |
| Improvement | 2-opt | O(k^2) per pass | O(k) |
| Exact check | Held-Karp | O(k^2 2^k) | O(k 2^k) |

n = parcels, W = capacity (kg), V/E = graph size, k = stops in a trip.

## Optional experiment: urgency-aware loading
`python main.py run --dataset data/medium_dataset.json --urgency` multiplies the knapsack value of parcels due within 6 h
by 5. On the medium dataset this cuts late parcels from 15 to 2 but adds about 3 % distance; on the overloaded
single-van large dataset it does not help. See docs/REPORT.md section 4.4.

## Notes
* Distances in the small dataset are *schematic* (not GPS-accurate); localities are only used as realistic names.
* Parcels heavier than the van (P15, 60 kg) are reported as "not loadable" instead of crashing.
* `outputs/screenshots/*.png` are console output rendered to images by the program itself. For the viva it is a good idea
  to also take your own VS Code screenshots (`python main.py run`).
