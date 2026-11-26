"""Mini-Courier Planner - DAA Capstone (K.R. Mangalam University).

Usage (from this folder):
    python main.py                 interactive menu
    python main.py run             full pipeline on data/small_dataset.json (Phases 1-3)
    python main.py run --dataset data/medium_dataset.json
    python main.py benchmark       Phase 4: runtime tests + charts   (add --quick for a fast run)
    python main.py generate --nodes 60 --parcels 120 --out data/my_data.json
    python main.py all             regenerate EVERYTHING in outputs/ (traces, charts, screenshots)
"""
import argparse
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from courier import benchmark, charts, report, screenshots                # noqa: E402
from courier.data_loader import DatasetError, generate_dataset, load_dataset, save_dataset  # noqa: E402
from courier.planner import plan                                          # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
DEFAULT_DATASET = os.path.join(DATA, "small_dataset.json")


def run_pipeline(path: str, sort_algo: str = "merge", urgency: bool = False,
                 draw: bool = True, phases: bool = True) -> str:
    """Run phases 1-3 + summary on one dataset. Returns the text that was printed."""
    ds = load_dataset(path)
    res = plan(ds, sort_algo=sort_algo, trace=True, urgency=urgency)
    parts = [report.phase1(ds), report.phase2(res), report.phase3(res), report.summary(res)]
    text = "\n".join(parts)
    print(text)
    if draw and charts.HAVE_MPL and ds.graph.coords and len(ds.graph.nodes) <= 60:
        png = charts.route_map(os.path.join(OUT, "charts", f"route_map_{ds.name}.png"), res)
        print(f"\nRoute map saved: {os.path.relpath(png, HERE)}")
    return text


def run_benchmark(quick: bool) -> str:
    results = benchmark.run_all(DATA, os.path.join(OUT, "benchmarks"), quick=quick)
    text = benchmark.report_text(results)
    print("\n" + report.banner("PHASE 4 - PERFORMANCE TESTING AND OPTIMISATION") + "\n" + text)
    files = charts.draw_all(results, os.path.join(OUT, "charts"))
    for f in files:
        print("chart saved:", os.path.relpath(f, HERE))
    return text


def capture(fn, *a, **k) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


def run_all() -> None:
    """Regenerate every file in outputs/ (used to prepare the submission)."""
    os.makedirs(os.path.join(OUT, "traces"), exist_ok=True)
    print("== Running small dataset (Phases 1-3) ==")
    ds = load_dataset(os.path.join(DATA, "small_dataset.json"))
    res = plan(ds, trace=True)
    p1, p2, p3, sm = report.phase1(ds), report.phase2(res), report.phase3(res, 3), report.summary(res)
    full = "\n".join([p1, p2, p3, sm])
    print(full)
    with open(os.path.join(OUT, "traces", "output_small_dataset.txt"), "w", encoding="utf-8") as f:
        f.write(full)
    if charts.HAVE_MPL:
        charts.route_map(os.path.join(OUT, "charts", "route_map_small_gurugram.png"), res)

    for name in ("medium", "large"):
        d = load_dataset(os.path.join(DATA, f"{name}_dataset.json"))
        r = plan(d, exact_limit=10)
        txt = "\n".join([report.phase1(d), report.phase2(r, 10), report.phase3(r, 2), report.summary(r)])
        with open(os.path.join(OUT, "traces", f"output_{name}_dataset.txt"), "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"\n[{name}] trips={len(r.trips)} delivered={r.delivered} km={r.total_distance:.1f} "
              f"late={r.late_count}")

    bench_txt = run_benchmark(quick=False)
    with open(os.path.join(OUT, "traces", "benchmark_results.txt"), "w", encoding="utf-8") as f:
        f.write(bench_txt)

    if screenshots.HAVE_MPL:
        shots = os.path.join(OUT, "screenshots")
        screenshots.text_to_png(p1, os.path.join(shots, "phase1_design.png"), "Phase 1 - Design")
        screenshots.text_to_png(p2.split("Step 2b")[0], os.path.join(shots, "phase2a_sorting.png"), "Phase 2 - Sorting")
        screenshots.text_to_png("Step 2b" + p2.split("Step 2b")[1], os.path.join(shots, "phase2b_knapsack_dp.png"),
                                "Phase 2 - Knapsack DP")
        screenshots.text_to_png(report.trip_report(ds, res.trips[0]), os.path.join(shots, "phase3_route_trip1.png"),
                                "Phase 3 - Dijkstra + TSP route")
        screenshots.text_to_png(sm, os.path.join(shots, "phase5_summary.png"), "Final summary")
        screenshots.text_to_png(bench_txt.split("\n\nKNAPSACK")[0], os.path.join(shots, "phase4_sorting_benchmark.png"),
                                "Phase 4 - Sorting benchmark")
        screenshots.text_to_png("FULL PIPELINE" + bench_txt.split("FULL PIPELINE")[1],
                                os.path.join(shots, "phase4_pipeline_benchmark.png"), "Phase 4 - Pipeline timing")
        print("\nscreenshots written to outputs/screenshots/")
    print("\nDone. All evidence files are in the outputs/ folder.")


def menu() -> None:
    items = {
        "1": "Run full demo on SMALL dataset (Phases 1-3)",
        "2": "Run on MEDIUM dataset",
        "3": "Run on LARGE dataset (summary only)",
        "4": "Run on my own JSON file",
        "5": "Phase 4 benchmark (quick) + charts",
        "6": "Generate a random dataset",
        "7": "Regenerate ALL outputs (traces, charts, screenshots)",
        "0": "Exit",
    }
    while True:
        print("\n" + "=" * 60 + "\n  MINI-COURIER PLANNER  (DAA Capstone)\n" + "=" * 60)
        for k, v in items.items():
            print(f"  {k}. {v}")
        try:
            c = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if c == "1":
                run_pipeline(os.path.join(DATA, "small_dataset.json"))
            elif c == "2":
                run_pipeline(os.path.join(DATA, "medium_dataset.json"))
            elif c == "3":
                d = load_dataset(os.path.join(DATA, "large_dataset.json"))
                print(report.summary(plan(d, exact_limit=0)))
            elif c == "4":
                run_pipeline(input("Path to JSON file: ").strip().strip('"'))
            elif c == "5":
                run_benchmark(quick=True)
            elif c == "6":
                n = int(input("Number of nodes (>=3): "))
                m = int(input("Number of parcels: "))
                cap = int(input("Van capacity kg [50]: ") or 50)
                out = input("Save as [data/my_data.json]: ").strip() or os.path.join(DATA, "my_data.json")
                save_dataset(generate_dataset("custom", n, m, 1, cap), out)
                print("saved", out)
            elif c == "7":
                run_all()
            elif c == "0":
                return
            else:
                print("Please choose a number from the menu.")
        except (DatasetError, ValueError, OSError) as e:
            print("ERROR:", e)


def main(argv=None) -> int:
    try:                                  # keep Windows consoles happy
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Mini-Courier Planner (DAA capstone)")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run phases 1-3 on a dataset")
    r.add_argument("--dataset", default=DEFAULT_DATASET)
    r.add_argument("--sort", default="merge", choices=["merge", "quick", "heap", "insertion", "timsort"])
    r.add_argument("--urgency", action="store_true",
                   help="boost knapsack value of parcels due soon (optimisation experiment)")
    b = sub.add_parser("benchmark", help="phase 4 runtime tests and charts")
    b.add_argument("--quick", action="store_true")
    g = sub.add_parser("generate", help="create a random dataset JSON")
    g.add_argument("--nodes", type=int, default=30)
    g.add_argument("--parcels", type=int, default=60)
    g.add_argument("--capacity", type=int, default=50)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", default=os.path.join(DATA, "my_data.json"))
    sub.add_parser("all", help="regenerate all outputs")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "run":
            run_pipeline(args.dataset, args.sort, args.urgency)
        elif args.cmd == "benchmark":
            run_benchmark(args.quick)
        elif args.cmd == "generate":
            if args.nodes < 3 or args.parcels < 1:
                raise ValueError("need at least 3 nodes and 1 parcel")
            save_dataset(generate_dataset("custom", args.nodes, args.parcels, args.seed, args.capacity), args.out)
            print("dataset saved to", args.out)
        elif args.cmd == "all":
            run_all()
        else:
            menu()
    except (DatasetError, ValueError) as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
