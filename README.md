# IA26_P01_G02 — Projeto de Inteligência Artificial

## Elementos do Grupo (Grupo 02)

- **Edgar Brito** — 29330
- **Diogo Ribeiro** — 29321
- **Tiago Lima** — 29328
- **José Rebelo** — 31516
- **José Oliveira** — 16941

## Estrutura do Projeto

```
IA26_P01_G02/
├── data_tiny/          # Dataset pequeno
├── data_small/         # Dataset intermédio
├── src/
│   ├── model.py        # Request, Vehicle, State, Problem (custo + feasibility)
│   ├── informed.py     # Greedy + A* (2 heurísticas)
│   ├── local_search.py # Hill-Climbing 
│   └── csp.py          # CSP (backtracking + forward checking + AC-3)
├── notebooks/
│   ├── 01_Project01_G02.ipynb   # Notebook principal
│   └── 02_eda_tiny.ipynb        # Análise exploratória
└── README.md
```

## Como executar

```bash
pip install pandas matplotlib folium # em caso de não ter instalado
cd IA26_P01_G02
python -m src.model          # teste do motor
python -m src.informed       # teste Greedy/A*
python -m src.local_search   # teste Hill-Climbing
python -m src.csp            # teste CSP
```

Ou abrir `notebooks/01_Project01_G02.ipynb` no Jupyter / VS Code.

## Acesso ao Repositório

- **GitHub:** https://github.com/Brito21x/IA26_P01_G02.git
