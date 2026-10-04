"""HTTP API (stdlib only): the worker posts metrics, the web app reads history and a live SSE stream."""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


def make_server(store: Store, host: str = "127.0.0.1", port: int = 8000,
                poll_seconds: float = 1.0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep container logs quiet
            pass

        def _json(self, status: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlparse(self.path).path != "/api/metrics":
                return self._json(404, {"detail": "not found"})
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                m = MetricIn(**json.loads(raw))
                if m.learning not in LEARNING_MODES:
                    raise ValueError(f"learning must be one of {LEARNING_MODES}")
            except (ValidationError, ValueError, TypeError) as e:
                return self._json(422, {"detail": str(e)})
            self._json(201, {"id": store.add(m.run_id, m.learning, m.model_dump())})

        def do_GET(self):
            url = urlparse(self.path)
            after = int(parse_qs(url.query).get("after_id", ["0"])[0] or 0)
            parts = url.path.strip("/").split("/")
            if parts == ["api", "health"]:
                return self._json(200, {"ok": True})
            if parts == ["api", "runs"]:
                return self._json(200, store.runs())
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "metrics":
                return self._json(200, store.metrics(parts[2], after))
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "stream":
                return self._stream(parts[2], after)
            self._json(404, {"detail": "not found"})

        def _stream(self, run_id: str, last: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    rows = store.metrics(run_id, last)
                    for r in rows:
                        last = r["id"]
                        self.wfile.write(f"data: {json.dumps(r)}\n\n".encode())
                    if not rows:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(poll_seconds)
            except (BrokenPipeError, ConnectionError, OSError):
                pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    store = Store(os.environ.get("GOL_DB", "gol.db"))
    make_server(store, "0.0.0.0", int(os.environ.get("PORT", "8000"))).serve_forever()


if __name__ == "__main__":
    main()
