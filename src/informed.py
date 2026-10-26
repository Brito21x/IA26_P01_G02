"""
informed.py — Pesquisa informada: Greedy Best-First e A* com duas heurísticas.
IA26_P01_G02
"""

from __future__ import annotations
import heapq
import time
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime, timedelta
try:
    from .model import Problem, Request, Vehicle, State
except ImportError:
    from model import Problem, Request, Vehicle, State


class SearchResult:
    def __init__(
        self,
        path: List[int],
        cost_km: float,
        cost_min: float,
        expanded: int,
        runtime_s: float,
        algorithm: str,
        heuristic: str = "",
        success: bool = True,
        message: str = "",
    ):
        self.path = path
        self.cost_km = cost_km
        self.cost_min = cost_min
        self.expanded = expanded
        self.runtime_s = runtime_s
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.success = success
        self.message = message

    def __repr__(self):
        return (
            f"{self.algorithm}({self.heuristic}): path={self.path}, "
            f"dist={self.cost_km:.1f}km, expanded={self.expanded}, "
            f"time={self.runtime_s*1000:.1f}ms"
        )


def heuristic_nearest(problem: Problem, current: int, remaining: List[int]) -> float:
    """h1: distância ao local restante mais próximo (admissível para TSP residual)."""
    if not remaining:
        return 0.0
    return min(problem.get_distance(current, loc) for loc in remaining)


def heuristic_mst_approx(problem: Problem, current: int, remaining: List[int]) -> float:
    """
    h2: aproximação MST — min distance to remaining + 0.5 * sum of pairwise mins.
    Mais informativa que nearest, ainda aproximada.
    """
    if not remaining:
        return 0.0
    h = min(problem.get_distance(current, loc) for loc in remaining)
    # soma das distâncias mínimas entre pares (aproximação grosseira do MST)
    extra = 0.0
    locs = list(remaining)
    for i, a in enumerate(locs):
        if i + 1 < len(locs):
            extra += min(problem.get_distance(a, b) for b in locs[i + 1 :])
    return h + 0.5 * extra


def greedy_search(
    problem: Problem,
    start: int,
    targets: List[int],
    return_to: Optional[int] = None,
    heuristic: Callable = heuristic_nearest,
) -> SearchResult:
    """
    Greedy Best-First Search: expande o nó com menor h(n).
    Constrói uma rota que visita todos os targets (ordem determinada pela heurística)
    e opcionalmente regressa a return_to.
    """
    t0 = time.perf_counter()
    remaining = set(targets)
    path = [start]
    current = start
    expanded = 0

    while remaining:
        expanded += 1
        # escolher o target com menor h
        best = min(remaining, key=lambda loc: heuristic(problem, current, [loc]))
        path.append(best)
        remaining.remove(best)
        current = best

    if return_to is not None and path[-1] != return_to:
        path.append(return_to)

    dist, tempo = problem.cost_function(path)
    return SearchResult(
        path=path,
        cost_km=dist,
        cost_min=tempo,
        expanded=expanded,
        runtime_s=time.perf_counter() - t0,
        algorithm="Greedy",
        heuristic=heuristic.__name__,
    )


def a_star_search(
    problem: Problem,
    start: int,
    targets: List[int],
    return_to: Optional[int] = None,
    heuristic: Callable = heuristic_nearest,
) -> SearchResult:
    """
    A* sobre o espaço de subconjuntos de targets visitados.
    Estado = (current_location, frozenset(remaining))
    g = custo real acumulado, h = heurística residual.
    """
    t0 = time.perf_counter()
    targets_set = frozenset(targets)
    # priority queue: (f, g, current, remaining, path)
    counter = 0
    open_set = []
    h0 = heuristic(problem, start, list(targets_set))
    heapq.heappush(open_set, (h0, 0.0, counter, start, targets_set, [start]))
    counter += 1
    closed = set()
    expanded = 0

    while open_set:
        f, g, _, current, remaining, path = heapq.heappop(open_set)
        state_key = (current, remaining)
        if state_key in closed:
            continue
        closed.add(state_key)
        expanded += 1

        if not remaining:
            final_path = list(path)
            if return_to is not None and final_path[-1] != return_to:
                final_path.append(return_to)
            dist, tempo = problem.cost_function(final_path)
            return SearchResult(
                path=final_path,
                cost_km=dist,
                cost_min=tempo,
                expanded=expanded,
                runtime_s=time.perf_counter() - t0,
                algorithm="A*",
                heuristic=heuristic.__name__,
            )

        for nxt in remaining:
            edge = problem.get_distance(current, nxt)
            if edge == float("inf"):
                continue
            new_g = g + edge
            new_rem = remaining - {nxt}
            h = heuristic(problem, nxt, list(new_rem))
            if return_to is not None and not new_rem:
                h += problem.get_distance(nxt, return_to)
            new_f = new_g + h
            new_path = path + [nxt]
            heapq.heappush(
                open_set, (new_f, new_g, counter, nxt, new_rem, new_path)
            )
            counter += 1

    return SearchResult(
        path=[],
        cost_km=float("inf"),
        cost_min=float("inf"),
        expanded=expanded,
        runtime_s=time.perf_counter() - t0,
        algorithm="A*",
        heuristic=heuristic.__name__,
        success=False,
        message="Nenhuma solução encontrada",
    )


def solve_vehicle_route(
    problem: Problem,
    vehicle_id: int,
    request_ids: Optional[List[int]] = None,
    algorithm: str = "astar",
    heuristic_name: str = "nearest",
) -> SearchResult:
    """
    Resolve a rota de recolha (produtores) para um veículo.
    Usa apenas os pedidos cujo peso total cabe no veículo (simplificação).
    """
    v = problem.vehicles[vehicle_id]
    center = v.center_loc_id
    if request_ids is None:
        # seleccionar subset de pedidos que cabem na capacidade
        selected = []
        load = 0.0
        for r in problem.requests:
            if load + r.weight_kg <= v.capacity_kg:
                selected.append(r)
                load += r.weight_kg
        targets = list({r.origin_id for r in selected})
    else:
        selected = [r for r in problem.requests if r.item_id in request_ids]
        targets = list({r.origin_id for r in selected})

    heur = heuristic_nearest if heuristic_name == "nearest" else heuristic_mst_approx

    if algorithm == "greedy":
        return greedy_search(problem, center, targets, return_to=center, heuristic=heur)
    else:
        return a_star_search(problem, center, targets, return_to=center, heuristic=heur)


if __name__ == "__main__":
    import sys, os
    # permitir correr como python -m src.informed a partir da raiz do projeto
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_tiny")
    p = Problem(data)
    print(p.summary())
    for vid in p.vehicles:
        for algo in ["greedy", "astar"]:
            for h in ["nearest", "mst"]:
                res = solve_vehicle_route(p, vid, algorithm=algo, heuristic_name=h)
                print(f"V{vid} | {res}")
