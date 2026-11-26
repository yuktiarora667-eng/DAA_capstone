"""0/1 Knapsack by Dynamic Programming - choose parcels for ONE trip (Phase 2).

Items   = parcels (weight = kg, value = importance score)
Capacity = van capacity (kg)

Recurrence (i = first i parcels, w = capacity used so far):
    dp[i][w] = dp[i-1][w]                                   if weight_i > w
    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight_i] + value_i)  otherwise

Time O(n*W)   Space O(n*W)   (O(W) if we only need the best value)
"""
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

from .models import Parcel


@dataclass
class KnapsackResult:
    best_value: int
    selected: List[Parcel]
    total_weight: int
    table: Optional[List[List[int]]] = None   # kept only when keep_table=True


def knapsack_dp(parcels: List[Parcel], capacity: int, keep_table: bool = False) -> KnapsackResult:
    """Full 2-D DP table + backtracking to recover WHICH parcels were chosen."""
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wt, val = parcels[i - 1].weight, parcels[i - 1].value
        prev_row, row = dp[i - 1], dp[i]
        for w in range(capacity + 1):
            row[w] = prev_row[w]
            if wt <= w and prev_row[w - wt] + val > row[w]:
                row[w] = prev_row[w - wt] + val

    # backtrack: if the value changed between row i-1 and i, parcel i was taken
    chosen: List[Parcel] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(parcels[i - 1])
            w -= parcels[i - 1].weight
    chosen.reverse()
    return KnapsackResult(dp[n][capacity], chosen, sum(p.weight for p in chosen),
                          dp if keep_table else None)


def knapsack_dp_1d(parcels: List[Parcel], capacity: int) -> int:
    """Space-optimised DP: one row, iterate capacity backwards. Value only."""
    dp = [0] * (capacity + 1)
    for p in parcels:
        for w in range(capacity, p.weight - 1, -1):
            cand = dp[w - p.weight] + p.value
            if cand > dp[w]:
                dp[w] = cand
    return dp[capacity]


def knapsack_greedy(parcels: List[Parcel], capacity: int) -> KnapsackResult:
    """Greedy by value/weight ratio - fast but NOT optimal (for comparison)."""
    order = sorted(parcels, key=lambda p: (-p.value / p.weight, p.pid))
    chosen, used = [], 0
    for p in order:
        if used + p.weight <= capacity:
            chosen.append(p)
            used += p.weight
    return KnapsackResult(sum(p.value for p in chosen), chosen, used)


def knapsack_bruteforce(parcels: List[Parcel], capacity: int) -> KnapsackResult:
    """Try all 2^n subsets - only for validating the DP on tiny inputs."""
    if len(parcels) > 20:
        raise ValueError("brute force limited to 20 parcels")
    best = KnapsackResult(0, [], 0)
    for r in range(len(parcels) + 1):
        for combo in combinations(parcels, r):
            wt = sum(p.weight for p in combo)
            val = sum(p.value for p in combo)
            if wt <= capacity and val > best.best_value:
                best = KnapsackResult(val, list(combo), wt)
    return best


def format_dp_table(parcels: List[Parcel], table: List[List[int]], capacity: int, step: int = 5) -> str:
    """Printable DP table; columns sampled every `step` kg so it fits a console."""
    cols = list(range(0, capacity + 1, step))
    if cols[-1] != capacity:
        cols.append(capacity)
    head = f"{'parcel (w,v)':<16}" + "".join(f"{c:>6}" for c in cols)
    lines = [f"DP table (rows = parcels considered, columns = capacity in kg, every {step} kg)",
             head, "-" * len(head)]
    for i in range(len(table)):
        label = "(none)" if i == 0 else f"{parcels[i-1].pid} ({parcels[i-1].weight},{parcels[i-1].value})"
        lines.append(f"{label:<16}" + "".join(f"{table[i][c]:>6}" for c in cols))
    return "\n".join(lines)
