"""
csp.py — Formulação CSP para atribuição de pedidos a veículos.
Backtracking + forward checking + consistência de nó (AC-3 simplificado),
com memoização de estados falhados e quebra de simetria (bin packing).
IA26_P01_G02
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional
try:
    from .model import Problem
except ImportError:
    from model import Problem

EPS = 1e-6


class CSPResult:
    def __init__(
        self,
        assignment: Dict[int, int],  # item_id -> vehicle_id
        feasible: bool,
        backtracks: int,
        runtime_s: float,
        message: str = "",
    ):
        self.assignment = assignment
        self.feasible = feasible
        self.backtracks = backtracks
        self.runtime_s = runtime_s
        self.message = message

    def __repr__(self):
        status = "OK" if self.feasible else "FAIL"
        return (
            f"CSP({status}): {len(self.assignment)} assignments, "
            f"backtracks={self.backtracks}, time={self.runtime_s*1000:.1f}ms"
            + (f" [{self.message}]" if self.message else "")
        )


class VehicleAssignmentCSP:
    """
    Variáveis: cada pedido (item_id)
    Domínio: veículos disponíveis
    Restrição hard: soma dos pesos atribuídos a um veículo <= capacidade.

    Ordem das variáveis: peso decrescente. É equivalente a MRV neste problema
    (o pedido mais pesado é o que tem menos veículos onde cabe) e custa O(1).
    """

    def __init__(self, problem: Problem, time_limit_s: float = 60.0):
        self.problem = problem
        self.time_limit_s = time_limit_s

        self.weights: Dict[int, float] = {r.item_id: r.weight_kg for r in problem.requests}
        self.variables: List[int] = sorted(
            self.weights, key=lambda x: self.weights[x], reverse=True
        )
        self.vehicle_ids: List[int] = list(problem.vehicles.keys())
        self.capacities: Dict[int, float] = {
            vid: v.capacity_kg for vid, v in problem.vehicles.items()
        }
        self.domains: Dict[int, List[int]] = {
            item: list(self.vehicle_ids) for item in self.variables
        }

        # suffix_weight[i] = soma dos pesos das variáveis i..n-1
        n = len(self.variables)
        self.suffix_weight = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix_weight[i] = self.suffix_weight[i + 1] + self.weights[self.variables[i]]

        self.backtracks = 0
        self._failed = set()
        self._deadline = 0.0
        self._timed_out = False

    # ------------------------------------------------------------------
    def ac3(self) -> bool:
        """
        Pré-processamento (AC-3 simplificado / consistência de nó):
        remove dos domínios os veículos onde o pedido, sozinho, já não cabe.
        """
        for var in self.variables:
            self.domains[var] = [
                v for v in self.domains[var]
                if self.weights[var] <= self.capacities[v] + EPS
            ]
            if not self.domains[var]:
                return False
        return True

    def forward_check(self, next_idx: int, free: Dict[int, float]) -> bool:
        """
        Forward checking sobre as variáveis ainda por atribuir.
        - o peso restante tem de caber na capacidade livre total;
        - a próxima variável (a mais pesada das restantes) tem de caber em
          algum veículo do seu domínio (se ela cabe, as mais leves também).
        """
        if next_idx >= len(self.variables):
            return True
        if self.suffix_weight[next_idx] > sum(free.values()) + EPS:
            return False
        nxt = self.variables[next_idx]
        w = self.weights[nxt]
        return any(free[v] + EPS >= w for v in self.domains[nxt])

    def backtrack(self, idx: int, free: Dict[int, float], assignment: Dict[int, int]) -> bool:
        if idx == len(self.variables):
            return True
        if time.perf_counter() > self._deadline:
            self._timed_out = True
            return False

        # Memoização: o futuro só depende de (idx, capacidades livres ordenadas)
        key = (idx, tuple(sorted(round(f, 6) for f in free.values())))
        if key in self._failed:
            return False

        var = self.variables[idx]
        w = self.weights[var]

        # Best-fit: primeiro os veículos com menos espaço livre onde ainda cabe
        candidates = sorted(
            (vid for vid in self.domains[var] if free[vid] + EPS >= w),
            key=lambda vid: free[vid],
        )

        tried = set()  # quebra de simetria: mesma capacidade livre => equivalentes
        for val in candidates:
            sig = round(free[val], 6)
            if sig in tried:
                continue
            tried.add(sig)

            free[val] -= w
            assignment[var] = val
            if self.forward_check(idx + 1, free) and self.backtrack(idx + 1, free, assignment):
                return True
            free[val] += w
            del assignment[var]
            self.backtracks += 1

            if self._timed_out:
                return False

        if not self._timed_out:
            self._failed.add(key)
        return False

    def solve(self) -> CSPResult:
        t0 = time.perf_counter()

        if not self.ac3():
            return CSPResult({}, False, 0, time.perf_counter() - t0, "AC-3 falhou")

        if self.suffix_weight[0] > sum(self.capacities.values()) + EPS:
            return CSPResult(
                {}, False, 0, time.perf_counter() - t0,
                "Peso total dos pedidos excede a capacidade total da frota",
            )

        self.backtracks = 0
        self._failed = set()
        self._timed_out = False
        self._deadline = t0 + self.time_limit_s

        free = dict(self.capacities)
        assignment: Dict[int, int] = {}
        ok = self.backtrack(0, free, assignment)
        runtime = time.perf_counter() - t0

        if ok:
            return CSPResult(assignment, True, self.backtracks, runtime, "Atribuição encontrada")
        if self._timed_out:
            return CSPResult({}, False, self.backtracks, runtime,
                             f"Tempo limite ({self.time_limit_s:.0f}s) atingido")
        return CSPResult({}, False, self.backtracks, runtime, "Sem atribuição viável")


def run_csp(problem: Problem, time_limit_s: float = 60.0) -> CSPResult:
    return VehicleAssignmentCSP(problem, time_limit_s=time_limit_s).solve()


if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_small")
    p = Problem(data)
    res = run_csp(p)
    print(res)

    if res.feasible:
        from collections import defaultdict
        loads = defaultdict(float)
        counts = defaultdict(int)
        weight_map = {r.item_id: r.weight_kg for r in p.requests}
        for item, vid in res.assignment.items():
            loads[vid] += weight_map[item]
            counts[vid] += 1
        for vid in sorted(loads):
            cap = p.vehicles[vid].capacity_kg
            print(f"  V{vid}: {counts[vid]:2d} items | load={loads[vid]:6.1f}kg / {cap:.1f}kg ({100*loads[vid]/cap:.1f}%)")