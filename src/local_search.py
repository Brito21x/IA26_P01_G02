"""
local_search.py — Hill-Climbing sobre soluções Greedy.
IA26_P01_G02
"""

from __future__ import annotations
import random
import time
from typing import List, Tuple, Optional
try:
    from .model import Problem
    from .informed import greedy_search, heuristic_nearest, SearchResult
except ImportError:
    from model import Problem
    from informed import greedy_search, heuristic_nearest, SearchResult


def two_opt_swap(route: List[int], i: int, j: int) -> List[int]:
    """Inverte o segmento entre i e j (2-opt clássico)."""
    return route[:i] + route[i : j + 1][::-1] + route[j + 1 :]


def hill_climbing(
    problem: Problem,
    initial_route: List[int],
    max_iterations: int = 500,
    max_no_improve: int = 50,
) -> SearchResult:
    """
    Hill-Climbing com operador 2-opt.
    Parte de uma solução inicial (tipicamente Greedy) e tenta melhorar
    a distância total.
    """
    t0 = time.perf_counter()
    current = list(initial_route)
    current_cost, _ = problem.cost_function(current)
    best = list(current)
    best_cost = current_cost
    expanded = 0
    no_improve = 0

    history = [current_cost]

    for it in range(max_iterations):
        improved = False
        n = len(current)
        # gerar todos os vizinhos 2-opt (ou amostrar se grande)
        candidates = []
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                candidates.append((i, j))
        random.shuffle(candidates)

        for i, j in candidates:
            expanded += 1
            neighbor = two_opt_swap(current, i, j)
            cost, _ = problem.cost_function(neighbor)
            if cost < current_cost - 1e-9:
                current = neighbor
                current_cost = cost
                improved = True
                if cost < best_cost:
                    best = list(current)
                    best_cost = cost
                break  # first improvement
        history.append(current_cost)
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0

    dist, tempo = problem.cost_function(best)
    return SearchResult(
        path=best,
        cost_km=dist,
        cost_min=tempo,
        expanded=expanded,
        runtime_s=time.perf_counter() - t0,
        algorithm="HillClimbing",
        heuristic="2-opt",
        message=f"iters={it+1}, history_len={len(history)}",
    ), history


def improve_greedy(
    problem: Problem,
    vehicle_id: int,
    max_iterations: int = 500,
) -> Tuple[SearchResult, List[float]]:
    """Constrói solução Greedy e aplica Hill-Climbing."""
    try:
        from .informed import solve_vehicle_route
    except ImportError:
        from informed import solve_vehicle_route

    greedy_res = solve_vehicle_route(
        problem, vehicle_id, algorithm="greedy", heuristic_name="nearest"
    )
    hc_res, history = hill_climbing(
        problem, greedy_res.path, max_iterations=max_iterations
    )
    return hc_res, history


if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_tiny")
    p = Problem(data)
    for vid in p.vehicles:
        res, hist = improve_greedy(p, vid)
        print(f"V{vid}: {res}")
        print(f"  Cost evolution (first/last): {hist[0]:.1f} -> {hist[-1]:.1f}")
