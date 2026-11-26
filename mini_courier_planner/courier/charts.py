"""Charts (matplotlib). Everything degrades gracefully if matplotlib is missing."""
import os
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")           # no window needed - saves PNG files
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                   # pragma: no cover
    HAVE_MPL = False

from .planner import STEPS, PlanResult

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def line_chart(path: str, title: str, xlabel: str, sizes: List[int],
               series: Dict[str, List[Optional[float]]], log: bool = False, ylabel: str = "time (ms)") -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, vals) in enumerate(series.items()):
        xs = [s for s, v in zip(sizes, vals) if v is not None]
        ys = [v * 1000 for v in vals if v is not None]
        ax.plot(xs, ys, marker="o", label=name, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def tsp_quality_chart(path: str, sizes: List[int], ratio: Dict[str, List[float]]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, vals) in enumerate(ratio.items()):
        ax.plot(sizes, vals, marker="s", label=name, color=COLORS[i], linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", label="Optimal (Held-Karp)")
    ax.set_title("TSP tour quality: length / optimal length", fontweight="bold")
    ax.set_xlabel("number of stops (k)")
    ax.set_ylabel("approximation ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def pipeline_chart(path: str, pipeline: Dict[str, dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = list(pipeline)
    width = 0.2
    for i, step in enumerate(STEPS):
        vals = [max(pipeline[l]["timings"][step] * 1000, 1e-3) for l in labels]
        ax.bar([x + i * width for x in range(len(labels))], vals, width, label=step, color=COLORS[i])
    ax.set_xticks([x + 1.5 * width for x in range(len(labels))])
    ax.set_xticklabels([f"{l}\n({pipeline[l]['parcels']} parcels)" for l in labels])
    ax.set_yscale("log")
    ax.set_ylabel("time (ms, log scale)")
    ax.set_title("Time spent in each major step", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    return _save(fig, path)


def route_map(path: str, result: PlanResult) -> str:
    """Draw road graph + every trip route (small datasets with coordinates)."""
    ds = result.dataset
    g = ds.graph
    fig, ax = plt.subplots(figsize=(9, 7))
    for u, v, _ in g.edges():
        if u in g.coords and v in g.coords:
            ax.plot([g.coords[u][0], g.coords[v][0]], [g.coords[u][1], g.coords[v][1]],
                    color="#cbd5e1", linewidth=1.2, zorder=1)
    for ti, trip in enumerate(result.trips):
        col = COLORS[ti % len(COLORS)]
        pts = [g.coords[n] for n in trip.full_path if n in g.coords]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=2.6, alpha=0.85,
                label=f"Trip {trip.number}: {trip.length:.1f} km, {len(trip.parcels)} parcels", zorder=2)
        for order, node in enumerate(trip.tour[1:-1], 1):
            x, y = g.coords[node]
            ax.annotate(str(order), (x, y), color=col, fontsize=8, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
    for n, (x, y) in g.coords.items():
        if n == ds.depot:
            ax.scatter([x], [y], marker="*", s=320, color="#111827", zorder=5)
        else:
            ax.scatter([x], [y], s=60, color="white", edgecolor="#334155", zorder=4)
        ax.annotate(g.name(n), (x, y), fontsize=7.5, xytext=(5, -11), textcoords="offset points", color="#334155")
    ax.set_title(f"Delivery routes - {ds.name}  (star = depot, numbers = visit order)", fontweight="bold")
    ax.set_xlabel("x (km, schematic)")
    ax.set_ylabel("y (km, schematic)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, path)


def draw_all(results: dict, out_dir: str, log=print) -> List[str]:
    if not HAVE_MPL:
        log("matplotlib not installed - charts skipped (pip install matplotlib)")
        return []
    files = []
    s, ser = results["sorting"]
    files.append(line_chart(os.path.join(out_dir, "chart_sorting.png"),
                            "Sorting runtime vs number of parcels", "n (parcels)", s, ser, log=True))
    s, ser = results["knapsack"]
    files.append(line_chart(os.path.join(out_dir, "chart_knapsack.png"),
                            "Knapsack runtime vs number of parcels (W = 500)", "n (parcels)", s, ser, log=True))
    s, ser = results["dijkstra"]
    files.append(line_chart(os.path.join(out_dir, "chart_dijkstra.png"),
                            "Dijkstra runtime vs number of nodes", "V (nodes)", s, ser, log=True))
    s, tm, ratio = results["tsp_exact"]
    files.append(line_chart(os.path.join(out_dir, "chart_tsp_time.png"),
                            "TSP runtime: approximation vs exact", "k (stops)", s, tm, log=True))
    files.append(tsp_quality_chart(os.path.join(out_dir, "chart_tsp_quality.png"), s, ratio))
    files.append(pipeline_chart(os.path.join(out_dir, "chart_pipeline.png"), results["pipeline"]))
    return files
