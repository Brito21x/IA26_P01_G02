"""
csp.py — Formulação CSP para atribuição de pedidos a veículos.
Usa backtracking com forward checking e AC-3 (simplificado).
IA26_P01_G02
"""

from __future__ import annotations
import time
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy
try:
    from .model import Problem, Request, Vehicle
except ImportError:
    from model import Problem, Request, Vehicle


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
        )


class VehicleAssignmentCSP:
    """
    Variáveis: cada pedido (item_id)
    Domínio: veículos disponíveis
    Restrições hard:
      - capacidade do veículo (soma pesos dos pedidos atribuídos <= capacity)
    Soft (penalizadas / prioritárias):
      - preferir veículos refrigerados para produtos sensíveis (não modelado aqui)
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.variables: List[int] = [r.item_id for r in problem.requests]
        self.domains: Dict[int, List[int]] = {
            r.item_id: list(problem.vehicles.keys()) for r in problem.requests
        }
        self.weights: Dict[int, float] = {
            r.item_id: r.weight_kg for r in problem.requests
        }
        self.capacities: Dict[int, float] = {
            vid: v.capacity_kg for vid, v in problem.vehicles.items()
        }
        self.backtracks = 0

    def is_consistent(self, assignment: Dict[int, int], var: int, val: int) -> bool:
        """Verifica restrição de capacidade após atribuir var=val."""
        load = self.weights[var]
        for other, v in assignment.items():
            if v == val:
                load += self.weights[other]
        return load <= self.capacities[val] + 1e-6

    def select_unassigned(self, assignment: Dict[int, int]) -> Optional[int]:
        """MRV: variável com menor domínio residual (simplificado: primeira não atribuída)."""
        for v in self.variables:
            if v not in assignment:
                return v
        return None

    def forward_check(
        self, assignment: Dict[int, int], var: int, val: int, domains: Dict[int, List[int]]
    ) -> Optional[Dict[int, List[int]]]:
        """Remove de outros domínios valores que violariam capacidade."""
        new_domains = {k: list(v) for k, v in domains.items()}
        remaining_cap = self.capacities[val] - self.weights[var]
        for other, v in assignment.items():
            if v == val:
                remaining_cap -= self.weights[other]

        for other in self.variables:
            if other in assignment or other == var:
                continue
            if val in new_domains[other]:
                if self.weights[other] > remaining_cap + 1e-6:
                    new_domains[other] = [x for x in new_domains[other] if x != val]
                    if not new_domains[other]:
                        return None  # domínio vazio
        return new_domains

    def ac3(self, domains: Dict[int, List[int]]) -> bool:
        """
        AC-3 simplificado: apenas verifica consistência de capacidade global
        (não é o AC-3 clássico de arcos, mas remove valores impossíveis).
        """
        # Para cada veículo, se o peso mínimo de um pedido > capacidade, remove
        for var in self.variables:
            new_dom = []
            for val in domains[var]:
                if self.weights[var] <= self.capacities[val] + 1e-6:
                    new_dom.append(val)
            domains[var] = new_dom
            if not new_dom:
                return False
        return True

    def backtrack(
        self,
        assignment: Dict[int, int],
        domains: Dict[int, List[int]],
    ) -> Optional[Dict[int, int]]:
        var = self.select_unassigned(assignment)
        if var is None:
            return assignment

        for val in list(domains[var]):
            if self.is_consistent(assignment, var, val):
                assignment[var] = val
                new_domains = self.forward_check(assignment, var, val, domains)
                if new_domains is not None:
                    result = self.backtrack(assignment, new_domains)
                    if result is not None:
                        return result
                del assignment[var]
                self.backtracks += 1
        return None

    def solve(self) -> CSPResult:
        t0 = time.perf_counter()
        domains = deepcopy(self.domains)
        if not self.ac3(domains):
            return CSPResult({}, False, 0, time.perf_counter() - t0, "AC-3 falhou")

        self.backtracks = 0
        solution = self.backtrack({}, domains)
        runtime = time.perf_counter() - t0
        if solution is None:
            return CSPResult({}, False, self.backtracks, runtime, "Sem atribuição viável")
        return CSPResult(solution, True, self.backtracks, runtime, "Atribuição encontrada")


def run_csp(problem: Problem) -> CSPResult:
    csp = VehicleAssignmentCSP(problem)
    return csp.solve()


if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    data = os.path.join(root, "data_tiny")
    p = Problem(data)
    res = run_csp(p)
    print(res)
    if res.feasible:
        # resumo por veículo
        from collections import defaultdict
        loads = defaultdict(float)
        counts = defaultdict(int)
        for item, vid in res.assignment.items():
            loads[vid] += p.requests[[r.item_id for r in p.requests].index(item)].weight_kg if False else 0
        # better
        weight_map = {r.item_id: r.weight_kg for r in p.requests}
        for item, vid in res.assignment.items():
            loads[vid] += weight_map[item]
            counts[vid] += 1
        for vid in sorted(loads):
            print(f"  V{vid}: {counts[vid]} items, load={loads[vid]:.1f}kg / {p.vehicles[vid].capacity_kg}kg")
