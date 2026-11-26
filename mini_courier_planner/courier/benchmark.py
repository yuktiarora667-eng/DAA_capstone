"""Performance testing (Phase 4): empirical runtime of every major component."""
import csv
import os
import random
import time
from typing import Callable, Dict, List, Optional

from .data_loader import generate_dataset, generate_graph, load_dataset
from .graph import dijkstra, dijkstra_array
from .knapsack import knapsack_bruteforce, knapsack_dp, knapsack_dp_1d, knapsack_greedy
from .models import Parcel
from .planner import STEPS, plan
from .routing import build_distance_matrix, held_karp, nearest_neighbor_tour, tour_length, two_opt
from .sorting import ALGORITHMS, parcel_key


def best_time(fn: Callable[[], object], repeat: int = 3) -> float:
    """Smallest wall-clock time (seconds) of `repeat` runs - reduces noise."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def random_parcels(n: int, seed: int = 5, nodes: int = 50) -> List[Parcel]:
    rng = random.Random(seed)
    return [Parcel(f"P{i:05d}", rng.randint(1, nodes - 1), rng.randint(1, 15), rng.randint(10, 80),
                   round(rng.uniform(1, 48), 1), rng.randint(1, 5)) for i in range(n)]


def bench_sorting(sizes: List[int], insertion_cap: int = 4000) -> Dict[str, List[Optional[float]]]:
    res: Dict[str, List[Optional[float]]] = {name: [] for name in ALGORITHMS}
    for n in sizes:
        data = random_parcels(n)
        for name, fn in ALGORITHMS.items():
            if name == "insertion" and n > insertion_cap:
                res[name].append(None)
                continue
            res[name].append(best_time(lambda: fn(data, key=parcel_key), 3 if n <= 2000 else 2))
    return res


def bench_knapsack(sizes: List[int], capacity: int = 500) -> Dict[str, List[float]]:
    res = {"DP 2-D table": [], "DP 1-D (space optimised)": [], "Greedy (ratio)": []}
    for n in sizes:
        data = random_parcels(n)
        res["DP 2-D table"].append(best_time(lambda: knapsack_dp(data, capacity)))
        res["DP 1-D (space optimised)"].append(best_time(lambda: knapsack_dp_1d(data, capacity)))
        res["Greedy (ratio)"].append(best_time(lambda: knapsack_greedy(data, capacity)))
    return res


def knapsack_quality(trials: int = 200, n: int = 14, capacity: int = 40) -> Dict[str, float]:
    """DP vs greedy vs brute force on many random small instances."""
    dp_ok = 0
    greedy_ratio = 0.0
    greedy_opt = 0
    for t in range(trials):
        data = random_parcels(n, seed=t)
        opt = knapsack_bruteforce(data, capacity).best_value
        dp = knapsack_dp(data, capacity).best_value
        gr = knapsack_greedy(data, capacity).best_value
        dp_ok += dp == opt
        greedy_ratio += gr / opt
        greedy_opt += gr == opt
    return {"trials": trials, "dp_matches_bruteforce_pct": 100 * dp_ok / trials,
            "greedy_avg_pct_of_optimal": 100 * greedy_ratio / trials,
            "greedy_optimal_pct": 100 * greedy_opt / trials}


def bench_dijkstra(sizes: List[int]) -> Dict[str, List[float]]:
    res = {"Heap Dijkstra O((V+E)logV)": [], "Array Dijkstra O(V^2)": []}
    for n in sizes:
        g = generate_graph(n, seed=3)
        res["Heap Dijkstra O((V+E)logV)"].append(best_time(lambda: dijkstra(g, 0)))
        res["Array Dijkstra O(V^2)"].append(best_time(lambda: dijkstra_array(g, 0)))
    return res


def _tsp_instance(k: int, seed: int = 4):
    g = generate_graph(k + 1, seed=seed)
    stops = list(range(1, k + 1))
    dist, _ = build_distance_matrix(g, [0] + stops)
    return stops, dist


def bench_tsp_exact(sizes: List[int]):
    """Small k: NN vs 2-opt vs exact Held-Karp (time and tour quality)."""
    times = {"Nearest neighbour": [], "NN + 2-opt": [], "Held-Karp (exact)": []}
    ratio = {"Nearest neighbour": [], "NN + 2-opt": []}
    for k in sizes:
        stops, dist = _tsp_instance(k)
        nn = nearest_neighbor_tour(0, stops, dist)
        times["Nearest neighbour"].append(best_time(lambda: nearest_neighbor_tour(0, stops, dist)))
        times["NN + 2-opt"].append(best_time(lambda: two_opt(nn, dist)))
        times["Held-Karp (exact)"].append(best_time(lambda: held_karp(0, stops, dist, 14), 1))
        opt, _ = held_karp(0, stops, dist, 14)
        ratio["Nearest neighbour"].append(tour_length(nn, dist) / opt)
        ratio["NN + 2-opt"].append(tour_length(two_opt(nn, dist), dist) / opt)
    return times, ratio


def bench_tsp_large(sizes: List[int]):
    times = {"Nearest neighbour": [], "NN + 2-opt": []}
    lengths = {"Nearest neighbour": [], "NN + 2-opt": []}
    for k in sizes:
        stops, dist = _tsp_instance(k)
        nn = nearest_neighbor_tour(0, stops, dist)
        times["Nearest neighbour"].append(best_time(lambda: nearest_neighbor_tour(0, stops, dist)))
        times["NN + 2-opt"].append(best_time(lambda: two_opt(nn, dist), 1))
        lengths["Nearest neighbour"].append(tour_length(nn, dist))
        lengths["NN + 2-opt"].append(tour_length(two_opt(nn, dist), dist))
    return times, lengths


def bench_pipeline(paths: Dict[str, str]) -> Dict[str, dict]:
    """End-to-end timing of the four major steps on small / medium / large datasets."""
    out = {}
    for label, path in paths.items():
        ds = load_dataset(path)
        runs = [plan(ds, improve=False, exact_limit=0) for _ in range(3)]
        best = min(runs, key=lambda r: sum(r.timings.values()))
        out[label] = {"parcels": len(ds.parcels), "nodes": len(ds.graph.nodes),
                      "trips": len(best.trips), "distance": best.total_distance,
                      "delivered": best.delivered, "late": best.late_count,
                      "timings": best.timings}
    return out


def save_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_ms(x: Optional[float]) -> str:
    return "   --   " if x is None else f"{x * 1000:9.3f}"


def table_text(title: str, sizes: List[int], series: Dict[str, List[Optional[float]]], unit="ms", label="n") -> str:
    names = list(series)
    lines = [title, f"{label:>7} " + " ".join(f"{n[:24]:>24}" for n in names)]
    for i, n in enumerate(sizes):
        lines.append(f"{n:>7} " + " ".join(f"{fmt_ms(series[nm][i]):>24}" for nm in names))
    lines.append(f"(times in {unit}, best of several runs)")
    return "\n".join(lines)


def run_all(data_dir: str, out_dir: str, quick: bool = False, log: Callable[[str], None] = print) -> dict:
    """Run every benchmark, write CSVs, return all results (charts are drawn by charts.py)."""
    sort_sizes = [100, 500, 1000, 2000, 4000] if quick else [100, 500, 1000, 2000, 4000, 8000, 16000]
    knap_sizes = [20, 50, 100, 200] if quick else [20, 50, 100, 200, 400, 800]
    dij_sizes = [50, 100, 200, 400] if quick else [50, 100, 200, 400, 800, 1200]
    tsp_small = [5, 7, 9, 11, 12] if quick else [5, 6, 7, 8, 9, 10, 11, 12, 13]
    tsp_big = [20, 50, 100, 150] if quick else [20, 50, 100, 200, 300]

    results = {}
    log("[1/6] sorting algorithms ...")
    results["sorting"] = (sort_sizes, bench_sorting(sort_sizes))
    log("[2/6] knapsack ...")
    results["knapsack"] = (knap_sizes, bench_knapsack(knap_sizes))
    results["knapsack_quality"] = knapsack_quality(100 if quick else 200)
    log("[3/6] Dijkstra ...")
    results["dijkstra"] = (dij_sizes, bench_dijkstra(dij_sizes))
    log("[4/6] TSP (exact vs approximation) ...")
    results["tsp_exact"] = (tsp_small,) + bench_tsp_exact(tsp_small)
    log("[5/6] TSP larger instances ...")
    results["tsp_large"] = (tsp_big,) + bench_tsp_large(tsp_big)
    log("[6/6] full pipeline on small / medium / large datasets ...")
    paths = {k: os.path.join(data_dir, f"{k}_dataset.json") for k in ("small", "medium", "large")}
    results["pipeline"] = bench_pipeline(paths)

    os.makedirs(out_dir, exist_ok=True)
    for key, title in (("sorting", "sorting"), ("knapsack", "knapsack"), ("dijkstra", "dijkstra")):
        sizes, series = results[key]
        save_csv(os.path.join(out_dir, f"runtime_{title}.csv"), ["n"] + list(series),
                 [[n] + [series[s][i] for s in series] for i, n in enumerate(sizes)])
    sizes, times, ratio = results["tsp_exact"]
    save_csv(os.path.join(out_dir, "runtime_tsp_exact.csv"),
             ["stops"] + list(times) + ["NN_ratio", "2opt_ratio"],
             [[k] + [times[s][i] for s in times] + [ratio["Nearest neighbour"][i], ratio["NN + 2-opt"][i]]
              for i, k in enumerate(sizes)])
    pl = results["pipeline"]
    save_csv(os.path.join(out_dir, "runtime_pipeline.csv"),
             ["dataset", "parcels", "nodes", "trips", "total_km"] + list(STEPS),
             [[k, v["parcels"], v["nodes"], v["trips"], round(v["distance"], 1)] +
              [v["timings"][s] for s in STEPS] for k, v in pl.items()])
    return results


def report_text(results: dict) -> str:
    parts = []
    s, ser = results["sorting"]
    parts.append(table_text("SORTING  (parcels sorted by deadline, priority)", s, ser))
    s, ser = results["knapsack"]
    parts.append(table_text("KNAPSACK  (capacity = 500 kg)", s, ser))
    q = results["knapsack_quality"]
    parts.append(f"Knapsack correctness over {q['trials']} random instances (14 parcels, 40 kg):\n"
                 f"  DP == brute-force optimum : {q['dp_matches_bruteforce_pct']:.1f} % of cases\n"
                 f"  Greedy average value      : {q['greedy_avg_pct_of_optimal']:.1f} % of optimum "
                 f"(optimal in only {q['greedy_optimal_pct']:.1f} % of cases)")
    s, ser = results["dijkstra"]
    parts.append(table_text("DIJKSTRA  (single source, V = nodes)", s, ser, label="V"))
    s, tm, ratio = results["tsp_exact"]
    parts.append(table_text("TSP small  (k = stops)", s, tm, label="k"))
    parts.append("Tour length / optimal length (1.000 = optimal)\n   k    NN      NN+2opt\n" +
                 "\n".join(f"{k:>4} {ratio['Nearest neighbour'][i]:7.3f} {ratio['NN + 2-opt'][i]:8.3f}"
                           for i, k in enumerate(s)) +
                 f"\n mean {sum(ratio['Nearest neighbour'])/len(s):7.3f} {sum(ratio['NN + 2-opt'])/len(s):8.3f}")
    s, tm, ln = results["tsp_large"]
    parts.append(table_text("TSP large  (k = stops, no exact solver)", s, tm, label="k"))
    pl = results["pipeline"]
    lines = ["FULL PIPELINE (time per major step, ms)",
             f"{'dataset':<8}{'parcels':>8}{'nodes':>7}{'trips':>7}" + "".join(f"{x:>11}" for x in STEPS) + f"{'total':>10}"]
    for k, v in pl.items():
        tot = sum(v["timings"].values())
        lines.append(f"{k:<8}{v['parcels']:>8}{v['nodes']:>7}{v['trips']:>7}" +
                     "".join(f"{v['timings'][x]*1000:>11.3f}" for x in STEPS) + f"{tot*1000:>10.3f}")
    parts.append("\n".join(lines))
    return "\n\n".join(parts)
