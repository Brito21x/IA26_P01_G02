"""
informed.py — Pesquisa informada: Greedy Best-First e A* com duas heurísticas.
IA26_P01_G02
"""

from __future__ import annotations
import heapq
import time
from typing import List, Optional, Callable
try:
    from .model import Problem
except ImportError:
    from model import Problem

INF = float("inf")


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


# ----------------------------------------------------------------------
# Heurísticas
# ----------------------------------------------------------------------
def heuristic_nearest(problem: Problem, current: int, remaining: List[int]) -> float:
    """h1: distância ao local restante mais próximo (admissível e consistente)."""
    if not remaining:
        return 0.0
    return min(problem.get_distance(current, loc) for loc in remaining)


def heuristic_mst_approx(problem: Problem, current: int, remaining: List[int]) -> float:
    """
    h2: peso da MST (Prim) sobre {current} ∪ remaining, com arestas simétricas
    min(d(a,b), d(b,a)). Qualquer caminho que visite todos os locais é uma
    árvore geradora, logo a MST é um limite inferior (admissível).
    Mais informativa que h1.
    """
    if not remaining:
        return 0.0
    nodes = [current] + list(remaining)
    k = len(nodes)
    in_tree = [False] * k
    best = [INF] * k
    best[0] = 0.0
    total = 0.0
    for _ in range(k):
        u = -1
        bu = INF
        for i in range(k):
            if not in_tree[i] and best[i] < bu:
                u, bu = i, best[i]
        if u == -1:  # grafo desligado: devolve o que já foi acumulado
            break
        in_tree[u] = True
        total += bu
        a = nodes[u]
        for v in range(k):
            if not in_tree[v]:
                b = nodes[v]
                w = min(problem.get_distance(a, b), problem.get_distance(b, a))
                if w < best[v]:
                    best[v] = w
    return total


# ----------------------------------------------------------------------
# Greedy
# ----------------------------------------------------------------------
def greedy_search(
    problem: Problem,
    start: int,
    targets: List[int],
    return_to: Optional[int] = None,
    heuristic: Callable = heuristic_nearest,
) -> SearchResult:
    """
    Greedy com lookahead de 1 passo: escolhe o próximo local que minimiza
    d(atual, loc) + h(loc, restantes sem loc). Opcionalmente regressa a return_to.
    """
    t0 = time.perf_counter()
    remaining = set(targets)
    path = [start]
    current = start
    expanded = 0

    while remaining:
        expanded += 1
        rem_sorted = sorted(remaining)  # determinismo em empates

        def score(loc: int) -> float:
            rest = [x for x in rem_sorted if x != loc]
            return problem.get_distance(current, loc) + heuristic(problem, loc, rest)

        best = min(rem_sorted, key=score)
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


# ----------------------------------------------------------------------
# A*
# ----------------------------------------------------------------------
def a_star_search(
    problem: Problem,
    start: int,
    targets: List[int],
    return_to: Optional[int] = None,
    heuristic: Callable = heuristic_nearest,
    max_expanded: int = 300_000,
) -> SearchResult:
    """
    A* sobre o espaço de subconjuntos de targets visitados.
    Estado = (localização atual, frozenset(locais por visitar)).
    g = distância acumulada; h = heurística residual (+ regresso no último passo).
    """
    t0 = time.perf_counter()
    targets_set = frozenset(targets)

    h_cache = {}

    def h(cur: int, rem: frozenset) -> float:
        key = (cur, rem)
        v = h_cache.get(key)
        if v is None:
            v = heuristic(problem, cur, list(rem))
            h_cache[key] = v
        return v

    start_key = (start, targets_set)
    best_g = {start_key: 0.0}
    parent = {start_key: None}
    counter = 0
    open_set = [(h(start, targets_set), 0.0, counter, start_key)]
    closed = set()
    expanded = 0

    while open_set:
        f, g, _, key = heapq.heappop(open_set)
        if key in closed:
            continue
        if g > best_g.get(key, INF) + 1e-12:
            continue  # entrada obsoleta
        closed.add(key)
        expanded += 1
        current, remaining = key

        if not remaining:
            path = []
            k = key
            while k is not None:
                path.append(k[0])
                k = parent[k]
            path.reverse()
            if return_to is not None and path[-1] != return_to:
                path.append(return_to)
            dist, tempo = problem.cost_function(path)
            return SearchResult(
                path=path,
                cost_km=dist,
                cost_min=tempo,
                expanded=expanded,
                runtime_s=time.perf_counter() - t0,
                algorithm="A*",
                heuristic=heuristic.__name__,
            )

        if expanded >= max_expanded:
            break

        for nxt in remaining:
            edge = problem.get_distance(current, nxt)
            if edge == INF:
                continue
            new_g = g + edge
            new_rem = remaining - {nxt}
            nk = (nxt, new_rem)
            if nk in closed or new_g >= best_g.get(nk, INF) - 1e-12:
                continue
            best_g[nk] = new_g
            parent[nk] = key
            hh = h(nxt, new_rem)
            if return_to is not None and not new_rem:
                hh += problem.get_distance(nxt, return_to)
            counter += 1
            heapq.heappush(open_set, (new_g + hh, new_g, counter, nk))

    msg = "Limite de nós expandidos atingido" if expanded >= max_expanded else "Nenhuma solução encontrada"
    return SearchResult(
        path=[],
        cost_km=INF,
        cost_min=INF,
        expanded=expanded,
        runtime_s=time.perf_counter() - t0,
        algorithm="A*",
        heuristic=heuristic.__name__,
        success=False,
        message=msg,
    )


# ----------------------------------------------------------------------
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
        selected = []
        load = 0.0
        for r in problem.requests:
            if load + r.weight_kg <= v.capacity_kg:
                selected.append(r)
                load += r.weight_kg
    else:
        wanted = set(request_ids)
        selected = [r for r in problem.requests if r.item_id in wanted]

    targets = sorted({r.origin_id for r in selected} - {center})

    heur = heuristic_nearest if heuristic_name == "nearest" else heuristic_mst_approx

    if algorithm == "greedy":
        return greedy_search(problem, center, targets, return_to=center, heuristic=heur)
    return a_star_search(problem, center, targets, return_to=center, heuristic=heur)


if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_small")
    p = Problem(data)
    print(p.summary())
    for vid in p.vehicles:
        for algo in ["greedy", "astar"]:
            for h in ["nearest", "mst"]:
                res = solve_vehicle_route(p, vid, algorithm=algo, heuristic_name=h)
                print(f"V{vid} | {res}")