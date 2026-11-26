"""Sorting algorithms used for parcel prioritisation (Phase 2).

All functions return a NEW sorted list and accept an optional key function,
so they can sort Parcel objects by (deadline, -priority, id).

Algorithm        Best        Average     Worst       Space   Stable
Insertion sort   O(n)        O(n^2)      O(n^2)      O(1)    yes
Merge sort       O(n log n)  O(n log n)  O(n log n)  O(n)    yes   <- chosen
Quick sort       O(n log n)  O(n log n)  O(n^2)      O(log n) no
Heap sort        O(n log n)  O(n log n)  O(n log n)  O(1)    no

WHY MERGE SORT?  Guaranteed O(n log n) and STABLE: parcels that tie on the
sort key keep their original order, which is what a courier desk expects.
"""
import random
from typing import Callable, List, TypeVar

T = TypeVar("T")
Key = Callable[[T], object]


def _identity(x):
    return x


def parcel_key(p):
    """Earliest deadline first; higher priority first; then id (deterministic)."""
    return (p.deadline, -p.priority, p.pid)


# ---------------------------------------------------------------- insertion
def insertion_sort(items: List[T], key: Key = _identity) -> List[T]:
    a = list(items)
    for i in range(1, len(a)):
        cur = a[i]
        kc = key(cur)
        j = i - 1
        while j >= 0 and key(a[j]) > kc:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = cur
    return a


# -------------------------------------------------------------------- merge
def merge_sort(items: List[T], key: Key = _identity) -> List[T]:
    """Top-down merge sort (stable)."""
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left = merge_sort(items[:mid], key)
    right = merge_sort(items[mid:], key)
    return _merge(left, right, key)


def _merge(left: List[T], right: List[T], key: Key) -> List[T]:
    out: List[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        # "<=" takes from the LEFT on ties -> stability
        if key(left[i]) <= key(right[j]):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


# -------------------------------------------------------------------- quick
def quick_sort(items: List[T], key: Key = _identity) -> List[T]:
    """Randomised quick sort, 3-way partition (safe with many duplicates)."""
    a = list(items)
    rng = random.Random(42)

    def sort(lo: int, hi: int) -> None:
        while lo < hi:
            pivot = key(a[rng.randint(lo, hi)])
            lt, i, gt = lo, lo, hi          # Dutch-national-flag partition
            while i <= gt:
                k = key(a[i])
                if k < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                elif k > pivot:
                    a[gt], a[i] = a[i], a[gt]
                    gt -= 1
                else:
                    i += 1
            # recurse into the smaller side, loop on the larger -> O(log n) stack
            if lt - lo < hi - gt:
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(a) - 1)
    return a


# --------------------------------------------------------------------- heap
def heap_sort(items: List[T], key: Key = _identity) -> List[T]:
    a = list(items)
    n = len(a)

    def sift_down(root: int, end: int) -> None:
        while True:
            child = 2 * root + 1
            if child >= end:
                return
            if child + 1 < end and key(a[child]) < key(a[child + 1]):
                child += 1
            if key(a[root]) < key(a[child]):
                a[root], a[child] = a[child], a[root]
                root = child
            else:
                return

    for start in range(n // 2 - 1, -1, -1):      # build max-heap
        sift_down(start, n)
    for end in range(n - 1, 0, -1):              # extract max repeatedly
        a[0], a[end] = a[end], a[0]
        sift_down(0, end)
    return a


def builtin_sort(items: List[T], key: Key = _identity) -> List[T]:
    """Python's Timsort - reference baseline for the benchmarks."""
    return sorted(items, key=key)


ALGORITHMS = {
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
    "insertion": insertion_sort,
    "timsort": builtin_sort,
}


def sort_parcels(parcels, algorithm: str = "merge"):
    """Prioritise parcels: earliest deadline first, then highest priority."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown sorting algorithm: {algorithm}")
    return ALGORITHMS[algorithm](list(parcels), key=parcel_key)
