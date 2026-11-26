"""Unit tests - run with:  python -m unittest discover -s tests -v"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from courier.data_loader import (DatasetError, dataset_from_dict, dataset_to_dict, generate_dataset,
                                 generate_graph, load_dataset, save_dataset)
from courier.graph import INF, Graph, dijkstra, dijkstra_array, reconstruct_path
from courier.knapsack import knapsack_bruteforce, knapsack_dp, knapsack_dp_1d, knapsack_greedy
from courier.models import Parcel
from courier.planner import plan
from courier.routing import (build_distance_matrix, expand_route, held_karp, nearest_neighbor_tour,
                             tour_length, two_opt)
from courier.sorting import ALGORITHMS, merge_sort, parcel_key, sort_parcels

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def rand_parcels(n, seed):
    r = random.Random(seed)
    return [Parcel(f"P{i}", 1, r.randint(1, 12), r.randint(5, 60), r.randint(1, 20), r.randint(1, 5))
            for i in range(n)]


class SortingTests(unittest.TestCase):
    def test_all_algorithms_match_sorted(self):
        for seed in range(5):
            data = [random.Random(seed).randint(0, 30) for _ in range(60)]
            for name, fn in ALGORITHMS.items():
                self.assertEqual(fn(data), sorted(data), name)

    def test_edge_cases(self):
        for fn in ALGORITHMS.values():
            self.assertEqual(fn([]), [])
            self.assertEqual(fn([7]), [7])
            self.assertEqual(fn([3, 3, 3]), [3, 3, 3])
            self.assertEqual(fn([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_merge_sort_is_stable(self):
        pairs = [(1, "a"), (0, "b"), (1, "c"), (0, "d"), (1, "e")]
        self.assertEqual(merge_sort(pairs, key=lambda p: p[0]),
                         [(0, "b"), (0, "d"), (1, "a"), (1, "c"), (1, "e")])

    def test_parcel_order(self):
        ps = rand_parcels(40, 1)
        out = sort_parcels(ps, "merge")
        self.assertEqual(out, sorted(ps, key=parcel_key))
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a.deadline, b.deadline)

    def test_input_not_modified(self):
        data = [3, 1, 2]
        merge_sort(data)
        self.assertEqual(data, [3, 1, 2])


class KnapsackTests(unittest.TestCase):
    def test_dp_equals_bruteforce(self):
        for seed in range(40):
            ps = rand_parcels(10, seed)
            cap = random.Random(seed).randint(5, 50)
            dp = knapsack_dp(ps, cap)
            self.assertEqual(dp.best_value, knapsack_bruteforce(ps, cap).best_value)
            self.assertLessEqual(dp.total_weight, cap)
            self.assertEqual(sum(p.value for p in dp.selected), dp.best_value)
            self.assertEqual(knapsack_dp_1d(ps, cap), dp.best_value)

    def test_greedy_never_beats_dp(self):
        for seed in range(30):
            ps = rand_parcels(12, seed)
            self.assertLessEqual(knapsack_greedy(ps, 30).best_value, knapsack_dp(ps, 30).best_value)

    def test_known_example(self):
        ps = [Parcel("a", 1, 1, 1, 1, 1), Parcel("b", 1, 3, 4, 1, 1), Parcel("c", 1, 4, 5, 1, 1),
              Parcel("d", 1, 5, 7, 1, 1)]
        r = knapsack_dp(ps, 7)
        self.assertEqual(r.best_value, 9)         # b + c
        self.assertEqual({p.pid for p in r.selected}, {"b", "c"})

    def test_empty_and_too_heavy(self):
        self.assertEqual(knapsack_dp([], 10).best_value, 0)
        self.assertEqual(knapsack_dp([Parcel("x", 1, 99, 5, 1, 1)], 10).selected, [])


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.g = Graph()
        for a, b, w in [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]:
            self.g.add_edge(a, b, w)

    def test_dijkstra_known(self):
        d, prev = dijkstra(self.g, 0)
        self.assertEqual(d, {0: 0, 1: 3, 2: 1, 3: 4})
        self.assertEqual(reconstruct_path(prev, 0, 3), [0, 2, 1, 3])

    def test_heap_equals_array_on_random_graphs(self):
        for seed in range(8):
            g = generate_graph(30, seed)
            self.assertEqual(dijkstra(g, 0)[0], dijkstra_array(g, 0)[0])

    def test_unreachable(self):
        g = Graph()
        g.add_edge(0, 1, 1)
        g.add_node(2)
        d, prev = dijkstra(g, 0)
        self.assertEqual(d[2], INF)
        self.assertEqual(reconstruct_path(prev, 0, 2), [])
        with self.assertRaises(ValueError):
            build_distance_matrix(g, [0, 2])

    def test_negative_edge_rejected(self):
        with self.assertRaises(ValueError):
            Graph().add_edge(0, 1, -1)

    def test_generated_graph_connected(self):
        for n in (3, 10, 80):
            self.assertTrue(generate_graph(n, 2).is_connected())


class RoutingTests(unittest.TestCase):
    def instance(self, k, seed=1):
        g = generate_graph(k + 1, seed)
        stops = list(range(1, k + 1))
        dist, paths = build_distance_matrix(g, [0] + stops)
        return g, stops, dist, paths

    def test_tour_valid(self):
        _, stops, dist, _ = self.instance(9)
        t = nearest_neighbor_tour(0, stops, dist)
        self.assertEqual(t[0], 0)
        self.assertEqual(t[-1], 0)
        self.assertEqual(sorted(t[1:-1]), stops)

    def test_nn_ge_2opt_ge_optimal(self):
        for seed in range(6):
            _, stops, dist, _ = self.instance(8, seed)
            nn = nearest_neighbor_tour(0, stops, dist)
            opt, _tour = held_karp(0, stops, dist)
            two = two_opt(nn, dist)
            self.assertEqual(sorted(two[1:-1]), stops)
            self.assertLessEqual(tour_length(two, dist), tour_length(nn, dist) + 1e-9)
            self.assertLessEqual(opt, tour_length(two, dist) + 1e-9)

    def test_held_karp_matches_bruteforce(self):
        from itertools import permutations
        _, stops, dist, _ = self.instance(6, 3)
        best = min(tour_length([0, *p, 0], dist) for p in permutations(stops))
        self.assertAlmostEqual(held_karp(0, stops, dist)[0], best)

    def test_expand_route_is_walk_with_same_length(self):
        g, stops, dist, paths = self.instance(7, 2)
        t = nearest_neighbor_tour(0, stops, dist)
        full = expand_route(t, paths)
        self.assertEqual(full[0], 0)
        self.assertEqual(full[-1], 0)
        total = 0
        for a, b in zip(full, full[1:]):
            total += min(w for v, w in g.adj[a] if v == b)
        self.assertAlmostEqual(total, tour_length(t, dist), places=6)

    def test_single_stop(self):
        _, stops, dist, _ = self.instance(1)
        self.assertEqual(nearest_neighbor_tour(0, stops, dist), [0, 1, 0])


class PlannerTests(unittest.TestCase):
    def test_small_dataset_end_to_end(self):
        ds = load_dataset(os.path.join(DATA, "small_dataset.json"))
        res = plan(ds)
        ids = [p.pid for t in res.trips for p in t.parcels]
        self.assertEqual(len(ids), len(set(ids)))                  # no parcel loaded twice
        self.assertEqual(set(ids) | {p.pid for p in res.unassigned}, {p.pid for p in ds.parcels})
        self.assertEqual([p.pid for p in res.unassigned], ["P15"])  # 60 kg > 40 kg van
        for t in res.trips:
            self.assertLessEqual(t.knapsack.total_weight, ds.capacity)
            self.assertEqual(t.tour[0], ds.depot)
            self.assertAlmostEqual(t.length, tour_length(t.tour, t.dist))
            if t.optimal_length:
                self.assertLessEqual(t.optimal_length, t.length + 1e-9)
        self.assertEqual(res.trips[0].knapsack.best_value, 263)

    def test_generated_datasets(self):
        ds = generate_dataset("t", 25, 40, seed=9, capacity=30)
        res = plan(ds, exact_limit=0)
        self.assertEqual(res.delivered + len(res.unassigned), 40)

    def test_urgency_option_still_delivers_everything(self):
        ds = generate_dataset("t", 20, 30, seed=2, capacity=30)
        self.assertEqual(plan(ds, urgency=True, exact_limit=0).delivered, 30)

    def test_all_sort_choices_give_same_plan(self):
        ds = generate_dataset("t", 20, 30, seed=4, capacity=30)
        base = plan(ds, "merge", exact_limit=0).total_distance
        for algo in ("quick", "heap", "insertion", "timsort"):
            self.assertAlmostEqual(plan(ds, algo, exact_limit=0).total_distance, base)


class LoaderTests(unittest.TestCase):
    def test_roundtrip(self):
        ds = load_dataset(os.path.join(DATA, "small_dataset.json"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            save_dataset(ds, p)
            ds2 = load_dataset(p)
        self.assertEqual(dataset_to_dict(ds), dataset_to_dict(ds2))

    def test_validation_errors(self):
        base = dataset_to_dict(load_dataset(os.path.join(DATA, "small_dataset.json")))
        bad = dict(base, parcels=[dict(base["parcels"][0], destination=999)])
        with self.assertRaises(DatasetError):
            dataset_from_dict(bad)
        bad = dict(base, parcels=[dict(base["parcels"][0], priority=9)])
        with self.assertRaises(DatasetError):
            dataset_from_dict(bad)
        bad = dict(base, parcels=[base["parcels"][0], base["parcels"][0]])
        with self.assertRaises(DatasetError):
            dataset_from_dict(bad)
        with self.assertRaises(DatasetError):
            load_dataset("does_not_exist.json")


if __name__ == "__main__":
    unittest.main()
