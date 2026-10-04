import json
import threading
import urllib.error
import urllib.request

import pytest

from gol_backend.main import make_server
from gol_backend.store import Store


def metric(**kw):
    base = dict(run_id="r1", learning="sarsa", tick=10, generation=1, population=300,
                avg_energy=50.0, avg_fitness=1.5, max_fitness=3.0)
    return {**base, **kw}


class Client:
    def __init__(self, base):
        self.base = base

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body):
        return self.request("POST", path, body)


@pytest.fixture
def client():
    server = make_server(Store(":memory:"), port=0, poll_seconds=0.05)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Client(f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()
    server.server_close()


def test_post_and_read_back(client):
    assert client.post("/api/metrics", metric())[0] == 201
    client.post("/api/metrics", metric(tick=20, population=310))
    rows = client.get("/api/runs/r1/metrics")[1]
    assert [r["tick"] for r in rows] == [10, 20]
    assert client.get(f"/api/runs/r1/metrics?after_id={rows[0]['id']}")[1][0]["tick"] == 20


def test_runs_summary(client):
    client.post("/api/metrics", metric())
    client.post("/api/metrics", metric(run_id="r2", tick=5))
    runs = {r["run_id"]: r for r in client.get("/api/runs")[1]}
    assert runs["r1"]["reports"] == 1 and runs["r1"]["tick"] == 10
    assert set(runs) == {"r1", "r2"}


def test_validation(client):
    assert client.post("/api/metrics", metric(learning="bogus"))[0] == 422
    assert client.post("/api/metrics", metric(population=-1))[0] == 422
    assert client.post("/api/metrics", metric(extra=1))[0] == 422


def test_unknown_run_is_empty_and_unknown_path_404(client):
    assert client.get("/api/runs/nope/metrics") == (200, [])
    assert client.get("/api/nothing")[0] == 404


def test_stream_delivers_stored_metric(client):
    client.post("/api/metrics", metric())
    with urllib.request.urlopen(client.base + "/api/runs/r1/stream", timeout=5) as r:
        line = r.readline().decode()
    assert line.startswith("data: ") and json.loads(line[6:])["tick"] == 10
