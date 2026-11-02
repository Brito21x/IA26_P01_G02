"""
model.py — Motor do sistema de planeamento de rotas (IA26_P01_G02)

Classes:
  - Request: encomenda/item com origem, destino, peso e janelas temporais
  - Vehicle: veículo com capacidade e centro base
  - State: estado durante a procura
  - Problem: carrega dados, função de custo e verificador de viabilidade de rotas
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class Request:
    """Representa um pedido (linha de order_items) com mapeamento correcto de location_ids."""

    def __init__(
        self,
        item_id: int,
        order_id: int,
        weight_kg: float,
        origin_loc_id: int,      # location_id do produtor
        dest_loc_id: int,        # location_id do local de entrega
        avail_start: datetime,
        avail_end: datetime,
        deliv_start: datetime,
        deliv_end: datetime,
        producer_id: int = None,
        delivery_ref_id: int = None,
    ):
        self.item_id = item_id
        self.order_id = order_id
        self.weight_kg = weight_kg
        self.origin_id = origin_loc_id
        self.dest_id = dest_loc_id
        self.availability_start = avail_start
        self.availability_end = avail_end
        self.delivery_start = deliv_start
        self.delivery_end = deliv_end
        self.producer_id = producer_id
        self.delivery_ref_id = delivery_ref_id

    def __repr__(self) -> str:
        return (
            f"Request(Item {self.item_id}: Loc {self.origin_id} -> Loc {self.dest_id} "
            f"| {self.weight_kg:.2f}kg)"
        )


class Vehicle:
    """Representa um veículo da frota."""

    def __init__(
        self,
        vehicle_id: int,
        center_id: int,
        capacity_kg: float,
        speed_kmh: float,
        plate: str = "",
        refrigerated: bool = False,
        center_loc_id: int = None,
    ):
        self.vehicle_id = vehicle_id
        self.center_id = center_id
        self.capacity_kg = capacity_kg
        self.speed_kmh = speed_kmh
        self.plate = plate
        self.refrigerated = refrigerated
        self.center_loc_id = center_loc_id  # location_id do centro

    def __repr__(self) -> str:
        return f"Vehicle({self.vehicle_id}, Cap: {self.capacity_kg}kg, CenterLoc: {self.center_loc_id})"


class State:
    """Estado do sistema durante a procura da rota."""

    def __init__(
        self,
        current_location: int,
        current_load: float,
        current_time: datetime,
        pending_requests: List[Request],
        path: List[int] = None,
        collected: List[int] = None,  # item_ids já recolhidos
        delivered: List[int] = None,  # item_ids já entregues
    ):
        self.current_location = current_location
        self.current_load = current_load
        self.current_time = current_time
        self.pending_requests = list(pending_requests)
        self.path = path or []
        self.collected = collected or []
        self.delivered = delivered or []

    def copy(self) -> "State":
        return State(
            self.current_location,
            self.current_load,
            self.current_time,
            list(self.pending_requests),
            list(self.path),
            list(self.collected),
            list(self.delivered),
        )


class Problem:
    """
    Classe principal (Motor).
    Carrega CSVs, mapeia IDs correctamente, fornece cost_function e
    route_feasibility_checker.
    """

    def __init__(self, data_folder: str = "data_small"):
        self.data_folder = data_folder.rstrip("/")
        self.locations: pd.DataFrame = None
        self.producer_to_loc: Dict[int, int] = {}
        self.delivery_to_loc: Dict[int, int] = {}
        self.center_to_loc: Dict[int, int] = {}
        self.distance_matrix: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        self.requests: List[Request] = []
        self.service_time_min = 10.0  # default; sobrescrito se existir rules.csv

        self._load_locations()
        self._load_distances()
        self._load_vehicles()
        self._load_requests()
        self._load_rules()

    # ------------------------------------------------------------------
    # Carregamento de dados
    # ------------------------------------------------------------------
    def _load_locations(self):
        path = f"{self.data_folder}/locations.csv"
        self.locations = pd.read_csv(path)
        for lid, ltype, ref in zip(
            self.locations["location_id"],
            self.locations["location_type"],
            self.locations["ref_id"],
        ):
            lid, ref = int(lid), int(ref)
            if ltype == "producer":
                self.producer_to_loc[ref] = lid
            elif ltype == "delivery_location":
                self.delivery_to_loc[ref] = lid
            elif ltype == "collection_center":
                self.center_to_loc[ref] = lid

    def _load_distances(self):
        df = pd.read_csv(f"{self.data_folder}/distance_matrix.csv")
        self.distance_matrix = {
            (int(o), int(d)): {"dist_km": float(km), "time_min": float(mn)}
            for o, d, km, mn in zip(
                df["origin_location_id"],
                df["destination_location_id"],
                df["distance_km"],
                df["duration_min"],
            )
        }

    def _load_vehicles(self):
        df = pd.read_csv(f"{self.data_folder}/vehicles.csv")
        veiculos = {}
        for _, row in df.iterrows():
            vid = int(row["vehicle_id"])
            cid = int(row["center_id"])
            center_loc = self.center_to_loc.get(cid)
            veiculos[vid] = Vehicle(
                vehicle_id=vid,
                center_id=cid,
                capacity_kg=float(row["capacity_kg"]),
                speed_kmh=float(row["avg_speed_kmh"]),
                plate=str(row.get("plate", "")),
                refrigerated=bool(row.get("refrigerated", 0)),
                center_loc_id=center_loc,
            )
        self.vehicles = veiculos

    def _load_requests(self):
        df_items = pd.read_csv(f"{self.data_folder}/order_items.csv")
        df_orders = pd.read_csv(f"{self.data_folder}/orders.csv")
        df_prods = pd.read_csv(f"{self.data_folder}/products.csv")

        df_merged = pd.merge(df_items, df_orders, on="order_id")
        df_merged = pd.merge(df_merged, df_prods, on="product_id")

        pedidos = []
        for _, row in df_merged.iterrows():
            peso = float(row["quantity"]) * float(row["unit_weight_kg"])
            prod_id = int(row["producer_id"])
            deliv_ref = int(row["delivery_location_id"])
            origin_loc = self.producer_to_loc.get(prod_id)
            dest_loc = self.delivery_to_loc.get(deliv_ref)
            if origin_loc is None or dest_loc is None:
                continue  # skip inconsistências
            pedidos.append(Request(
                item_id=int(row["order_item_id"]),
                order_id=int(row["order_id"]),
                weight_kg=peso,
                origin_loc_id=origin_loc,
                dest_loc_id=dest_loc,
                avail_start=pd.to_datetime(row["availability_start"]),
                avail_end=pd.to_datetime(row["availability_end"]),
                deliv_start=pd.to_datetime(row["delivery_start"]),
                deliv_end=pd.to_datetime(row["delivery_end"]),
                producer_id=prod_id,
                delivery_ref_id=deliv_ref,
            ))
        self.requests = pedidos

    def _load_rules(self):
        try:
            df = pd.read_csv(f"{self.data_folder}/rules.csv")
            for _, row in df.iterrows():
                if row["rule"] == "stop_service_min":
                    self.service_time_min = float(row["value"])
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Funções de avaliação
    # ------------------------------------------------------------------
    def get_distance(self, origin: int, dest: int) -> float:
        info = self.distance_matrix.get((origin, dest))
        if info is None:
            return float("inf")
        return info["dist_km"]

    def get_travel_time(self, origin: int, dest: int) -> float:
        info = self.distance_matrix.get((origin, dest))
        if info is None:
            return float("inf")
        return info["time_min"]

    def cost_function(self, route: List[int]) -> Tuple[float, float]:
        """Recebe lista de location_ids e devolve (distância total km, tempo total min)."""
        total_dist = 0.0
        total_time = 0.0
        dm = self.distance_matrix
        for i in range(len(route) - 1):
            info = dm.get((route[i], route[i + 1]))
            if info is None:
                return float("inf"), float("inf")
            total_dist += info["dist_km"]
            total_time += info["time_min"]
        return total_dist, total_time

    def route_feasibility_checker(
        self, route: List[int], vehicle_id: int, requests: Optional[List[Request]] = None
    ) -> Tuple[bool, str]:
        """
        Verifica se a rota é viável para o veículo:
        - existência de arcos na matriz de distâncias
        - capacidade (se requests fornecidos)
        (Janelas temporais ainda não são verificadas aqui.)
        """
        veiculo = self.vehicles.get(vehicle_id)
        if not veiculo:
            return False, "Veículo inexistente."

        for i in range(len(route) - 1):
            o, d = route[i], route[i + 1]
            if (o, d) not in self.distance_matrix:
                return False, f"Impossível ir do local {o} para o local {d}."

        if requests is not None:
            total_weight = sum(r.weight_kg for r in requests)
            if total_weight > veiculo.capacity_kg:
                return False, (
                    f"Peso total {total_weight:.1f}kg excede capacidade "
                    f"{veiculo.capacity_kg:.1f}kg do veículo {vehicle_id}."
                )

        return True, "A rota base é viável."

    def get_center_location(self, vehicle_id: int) -> Optional[int]:
        v = self.vehicles.get(vehicle_id)
        return v.center_loc_id if v else None

    def summary(self) -> str:
        lines = [
            f"Problem [{self.data_folder}]",
            f"  Locations: {len(self.locations)}",
            f"  Vehicles: {len(self.vehicles)}",
            f"  Requests: {len(self.requests)}",
            f"  Distance arcs: {len(self.distance_matrix)}",
            f"  Service time: {self.service_time_min} min",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# Teste rápido
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os
    print("A iniciar o carregamento do modelo...")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(root, "data_small")
    p = Problem(data_folder=data)
    print(p.summary())
    print("\nExemplos de requests:")
    for r in p.requests[:5]:
        print(" ", r)
    print(f"\nVeículos: {list(p.vehicles.values())}")

    rota_teste = [6, 1, 2, 6]  # centro -> produtor1 -> produtor2 -> centro
    viavel, msg = p.route_feasibility_checker(rota_teste, vehicle_id=1)
    dist, tempo = p.cost_function(rota_teste)
    print("\n--- TESTE DE ROTA ---")
    print(f"Rota: {rota_teste}")
    print(f"Viável? {viavel} ({msg})")
    print(f"Distância: {dist:.2f} km | Tempo: {tempo:.2f} min")