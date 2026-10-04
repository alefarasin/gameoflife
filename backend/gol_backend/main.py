"""HTTP API: the worker posts metrics, the web app reads history and a live SSE stream."""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gol_shared.config import LEARNING_MODES

from .store import Store


class MetricIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=64)
    learning: str = "none"
    tick: int = Field(ge=0)
    generation: int = Field(ge=0)
    population: int = Field(ge=0)
    avg_energy: float
    avg_fitness: float
    max_fitness: float


def create_app(store: Store | None = None, poll_seconds: float = 1.0) -> FastAPI:
    store = store or Store(os.environ.get("GOL_DB", "gol.db"))
    app = FastAPI(title="Game of Life")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/metrics", status_code=201)
    def post_metric(m: MetricIn):
        if m.learning not in LEARNING_MODES:
            raise HTTPException(422, f"learning must be one of {LEARNING_MODES}")
        return {"id": store.add(m.run_id, m.learning, m.model_dump())}

    @app.get("/api/runs")
    def runs():
        return store.runs()

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str, after_id: int = 0):
        return store.metrics(run_id, after_id)

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, after_id: int = 0):
        async def events():
            last = after_id
            while True:
                rows = store.metrics(run_id, last)
                for r in rows:
                    last = r["id"]
                    yield f"data: {json.dumps(r)}\n\n"
                if not rows:
                    yield ": keepalive\n\n"
                await asyncio.sleep(poll_seconds)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def app_factory() -> FastAPI:  # uvicorn --factory gol_backend.main:app_factory
    return create_app()
