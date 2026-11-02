"""
local_search.py — Hill-Climbing (2-opt) sobre soluções Greedy.
IA26_P01_G02
"""

from __future__ import annotations
import random
import time
from typing import List, Tuple, Optional
try:
    from .model import Problem
    from .informed import SearchResult, solve_vehicle_route
except ImportError:
    from model import Problem
    from informed import SearchResult, solve_vehicle_route


def two_opt_swap(route: List[int], i: int, j: int) -> List[int]:
    """Inverte o segmento entre i e j (2-opt clássico)."""
    return route[:i] + route[i : j + 1][::-1] + route[j + 1 :]


def hill_climbing(
    problem: Problem,
    initial_route: List[int],
    max_iterations: int = 500,
    seed: Optional[int] = None,
) -> Tuple[SearchResult, List[float]]:
    """
    Hill-Climbing com operador 2-opt (first improvement).
    Pára assim que a vizinhança completa não traz melhoria (ótimo local).
    Devolve (SearchResult, histórico de custos).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)

    current = list(initial_route)
    current_cost, _ = problem.cost_function(current)
    expanded = 0
    history = [current_cost]
    n = len(current)
    iterations = 0

    # a vizinhança só depende de n, por isso gera-se uma vez
    all_moves = [(i, j) for i in range(1, n - 2) for j in range(i + 1, n - 1)]

    for iterations in range(1, max_iterations + 1):
        moves = all_moves[:]
        rng.shuffle(moves)

        improved = False
        for i, j in moves:
            expanded += 1
            neighbor = two_opt_swap(current, i, j)
            cost, _ = problem.cost_function(neighbor)
            if cost < current_cost - 1e-9:
                current, current_cost = neighbor, cost
                improved = True
                break  # first improvement

        if not improved:
            break  # ótimo local: repetir a mesma vizinhança seria desperdício
        history.append(current_cost)

    dist, tempo = problem.cost_function(current)
    result = SearchResult(
        path=current,
        cost_km=dist,
        cost_min=tempo,
        expanded=expanded,
        runtime_s=time.perf_counter() - t0,
        algorithm="HillClimbing",
        heuristic="2-opt",
        message=f"iters={iterations}, melhorias={len(history) - 1}",
    )
    return result, history


def improve_greedy(
    problem: Problem,
    vehicle_id: int,
    max_iterations: int = 500,
    seed: Optional[int] = 42,
) -> Tuple[SearchResult, List[float]]:
    """Constrói solução Greedy e aplica Hill-Climbing."""
    greedy_res = solve_vehicle_route(
        problem, vehicle_id, algorithm="greedy", heuristic_name="nearest"
    )
    return hill_climbing(
        problem, greedy_res.path, max_iterations=max_iterations, seed=seed
    )


if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_small")
    p = Problem(data)
    for vid in p.vehicles:
        res, hist = improve_greedy(p, vid)
        print(f"V{vid}: {res}")
        print(f"  Cost evolution (first/last): {hist[0]:.1f} -> {hist[-1]:.1f}")