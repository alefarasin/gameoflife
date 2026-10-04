# Game of Life – simulazione evolutiva di creature

Simulatore di una popolazione di creature su una griglia con cibo, ostacoli e (opzionalmente)
un terreno alla Conway. Ogni creatura ha un piccolo cervello (MLP) che evolve per mutazione
e, a scelta, impara durante la vita con apprendimento per rinforzo.

> **Stato attuale:** moduli di base (configurazione e cervelli) più backend, worker con metriche
> fittizie e dashboard web containerizzati. Il simulatore reale del mondo non esiste ancora.

## Struttura

| Percorso | Contenuto |
|---|---|
| `shared/gol_shared/config.py` | `SimConfig`: configurazione validata (pydantic) condivisa da backend e worker |
| `worker/gol_worker/brain.py` | `Brains`: MLP batch (uno per creatura) con TD-learning e backprop manuale (solo NumPy) |
| `worker/gol_worker/learners.py` | `DQN` e `PPO` con rete condivisa dalla popolazione (PyTorch) |

## Requisiti

- Python 3.10+
- `numpy`, `pydantic` (v2)
- `torch` solo per le modalità `dqn` e `ppo` (importato in modo lazy)

```bash
pip install numpy pydantic
pip install torch   # opzionale
```

## Uso

I pacchetti non hanno ancora un `pyproject.toml`: aggiungi le cartelle al `PYTHONPATH`.

```bash
# bash
export PYTHONPATH=shared:worker
# PowerShell
$env:PYTHONPATH = "shared;worker"
```

### Configurazione

```python
from gol_shared.config import SimConfig, default_config

cfg = SimConfig(width=128, height=128, learning="sarsa", life_layer=True)
print(cfg.model_dump())
print(default_config())            # dizionario con tutti i default
schema = SimConfig.model_json_schema()  # ogni campo ha "group" e descrizione (per generare form UI)
```

Valori fuori range o incoerenti (es. `max_population < initial_population`) sollevano `ValidationError`;
campi sconosciuti sono rifiutati.

Gruppi di parametri: World, Life layer, Creatures, Interaction, Evolution, Learning, Rewards.
A parità di `seed` e configurazione la simulazione è riproducibile.

### Modalità di apprendimento (`learning`)

| Valore | Descrizione |
|---|---|
| `none` | pura neuroevoluzione |
| `qlearning` / `sarsa` | TD-learning online, pesi propri per ogni creatura |
| `dqn` / `ppo` | rete PyTorch condivisa, addestrata sull'esperienza di tutte le creature |

`inherit_learned=True` abilita l'ereditarietà lamarckiana (solo modalità per-creatura).

### Cervelli

```python
import numpy as np
from gol_worker.brain import Brains, N_INPUTS

rng = np.random.default_rng(0)
brains = Brains(capacity=1000, hidden=12)
idx = np.arange(100)
brains.randomize(idx, rng)

x = rng.random((100, N_INPUTS)).astype(np.float32)
_, q = brains.forward(idx, x)                 # valori Q per le 5 azioni
brains.copy_mutated(np.arange(100, 110), np.arange(10), rate=0.1, std=0.15, rng=rng)  # riproduzione
```

## Docker e dashboard avanzamenti

```bash
docker compose up --build   # poi apri http://localhost:8080
```

| Servizio | Ruolo |
|---|---|
| `backend` | server HTTP stdlib + SQLite (volume `gol-data`): riceve le metriche (`POST /api/metrics`), le serve (`GET /api/runs`, `/api/runs/{id}/metrics`) e le streamma via SSE (`/api/runs/{id}/stream`) |
| `worker` | `python -m gol_worker.run`: per ora usa `FakeSim` (metriche fittizie); il simulatore reale andrà a sostituirlo. Variabili: `GOL_RUN_ID`, `GOL_CONFIG` (JSON di `SimConfig`), `GOL_REPORT_SECONDS`. PyTorch opzionale: `--build-arg WITH_TORCH=1` |
| `web` | React + Vite servita da nginx (proxy `/api` → backend): popolazione, fitness, energia in tempo reale |

Sviluppo locale: `PYTHONPATH=shared:backend python -m gol_backend.main` e `npm install && npm run dev` in `web/`.
Test backend: `pip install pytest pydantic && PYTHONPATH=shared:backend pytest backend/tests`.

## Licenza

Vedi `LICENSE`.
