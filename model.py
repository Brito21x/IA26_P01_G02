import pandas as pd

class Request:
    
    #Representa um pedido/encomenda (uma linha de order_items).
    #Contém a origem (produtor), o destino (cliente), o peso total e as janelas de tempo.
    def __init__(self, item_id, order_id, weight_kg, origin_id, dest_id, 
                 avail_start, avail_end, deliv_start, deliv_end):
        self.item_id = item_id
        self.order_id = order_id
        self.weight_kg = weight_kg
        self.origin_id = origin_id
        self.dest_id = dest_id
        self.availability_start = avail_start
        self.availability_end = avail_end
        self.delivery_start = deliv_start
        self.delivery_end = deliv_end

    def __repr__(self):
        return f"Request(Item {self.item_id}: Loc {self.origin_id} -> Loc {self.dest_id} | {self.weight_kg:.2f}kg)"

class Vehicle:
    #Representa um veículo da frota e a sua capacidade de carga.
    def __init__(self, vehicle_id, center_id, capacity_kg, speed_kmh):
        self.vehicle_id = vehicle_id
        self.center_id = center_id
        self.capacity_kg = capacity_kg
        self.speed_kmh = speed_kmh

    def __repr__(self):
        return f"Vehicle({self.vehicle_id}, Cap: {self.capacity_kg}kg)"

class State:
    #Representa o estado do sistema durante a procura da rota.
    def __init__(self, current_location, current_load, current_time, pending_requests):
        self.current_location = current_location
        self.current_load = current_load
        self.current_time = current_time
        self.pending_requests = pending_requests

class Problem:
    #Classe principal (O Motor). Carrega os dados dos CSVs e avalia as rotas.
    def __init__(self, data_folder="."):
        self.data_folder = data_folder
        self.distance_matrix = self._load_distances()
        self.vehicles = self._load_vehicles()
        self.requests = self._load_requests()

    def _load_distances(self):
        #Lê o distance_matrix.csv e cria um dicionário para acesso rápido.
        df = pd.read_csv(f"{self.data_folder}/distance_matrix.csv")
        matriz = {}
        for _, row in df.iterrows():
            origem = row['origin_location_id']
            destino = row['destination_location_id']
            matriz[(origem, destino)] = {
                'dist_km': row['distance_km'], 
                'time_min': row['duration_min']
            }
        return matriz

    def _load_vehicles(self):
        #Lê o vehicles.csv e cria objetos da classe Vehicle.
        df = pd.read_csv(f"{self.data_folder}/vehicles.csv")
        veiculos = {}
        for _, row in df.iterrows():
            veiculos[row['vehicle_id']] = Vehicle(
                row['vehicle_id'], 
                row['center_id'], 
                row['capacity_kg'],
                row['avg_speed_kmh']
            )
        return veiculos

    def _load_requests(self):
        #Lê e cruza order_items, orders e products para criar os Requests completos.
        df_items = pd.read_csv(f"{self.data_folder}/order_items.csv")
        df_orders = pd.read_csv(f"{self.data_folder}/orders.csv")
        df_prods = pd.read_csv(f"{self.data_folder}/products.csv")
        
        df_merged = pd.merge(df_items, df_orders, on='order_id')
        df_merged = pd.merge(df_merged, df_prods, on='product_id')
        
        pedidos = []
        for _, row in df_merged.iterrows():
            peso_total = row['quantity'] * row['unit_weight_kg']
            req = Request(
                item_id=row['order_item_id'], 
                order_id=row['order_id'], 
                weight_kg=peso_total, 
                origin_id=row['producer_id'], 
                dest_id=row['delivery_location_id'],
                avail_start=pd.to_datetime(row['availability_start']), 
                avail_end=pd.to_datetime(row['availability_end']),
                deliv_start=pd.to_datetime(row['delivery_start']), 
                deliv_end=pd.to_datetime(row['delivery_end'])
            )
            pedidos.append(req)
        return pedidos

    def cost_function(self, route):
        #Recebe uma rota (lista de location_ids) e devolve a distância total e o tempo total.
        total_dist = 0.0
        total_time = 0.0
        
        for i in range(len(route) - 1):
            origem = route[i]
            destino = route[i+1]
            
            if (origem, destino) in self.distance_matrix:
                total_dist += self.distance_matrix[(origem, destino)]['dist_km']
                total_time += self.distance_matrix[(origem, destino)]['time_min']
            else:
                return float('inf'), float('inf')
                
        return total_dist, total_time

    def route_feasibility_checker(self, route, vehicle_id):
        #Verifica se a rota é viável para o veículo, testando a existência dos caminhos.
        veiculo = self.vehicles.get(vehicle_id)
        if not veiculo:
            return False, "Veículo inexistente."
            
        for i in range(len(route) - 1):
            origem, destino = route[i], route[i+1]
            if (origem, destino) not in self.distance_matrix:
                return False, f"Impossível ir do local {origem} para o local {destino}."
                
        return True, "A rota base é viável."

# BLOCO DE TESTE
if __name__ == "__main__":
    print("A iniciar o carregamento do modelo...")
    p = Problem(data_folder="data_tiny")
    print(f"-> {len(p.requests)} Encomendas carregadas.")
    print(f"-> {len(p.vehicles)} Veículos carregados.")
    
    rota_teste = [1, 2, 3, 1]
    viavel, mensagem = p.route_feasibility_checker(rota_teste, vehicle_id=1)
    distancia, tempo = p.cost_function(rota_teste)
    
    print("\n--- TESTE DE ROTA ---")
    print(f"Rota: {rota_teste}")
    print(f"Viável? {viavel} ({mensagem})")
    print(f"Distância Total: {distancia:.2f} km")
    print(f"Tempo de Viagem: {tempo:.2f} minutos")