"""Worker entry point: runs a simulation and reports metrics to the backend.

Env: GOL_BACKEND_URL (default http://backend:8000), GOL_RUN_ID, GOL_REPORT_SECONDS, GOL_CONFIG (JSON).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid

from gol_shared.config import SimConfig

from .fake_sim import FakeSim


def post(url: str, payload: dict) -> bool:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5).close()
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"report failed: {e}", flush=True)
        return False


def main() -> None:
    backend = os.environ.get("GOL_BACKEND_URL", "http://backend:8000").rstrip("/")
    run_id = os.environ.get("GOL_RUN_ID") or uuid.uuid4().hex[:8]
    interval = float(os.environ.get("GOL_REPORT_SECONDS", "1"))
    cfg = SimConfig(**json.loads(os.environ.get("GOL_CONFIG", "{}")))
    sim = FakeSim(cfg)
    print(f"run {run_id} ({cfg.learning}) -> {backend}", flush=True)
    while True:
        sim.step()
        post(f"{backend}/api/metrics", {"run_id": run_id, "learning": cfg.learning, **sim.metrics()})
        time.sleep(interval)


if __name__ == "__main__":
    main()
