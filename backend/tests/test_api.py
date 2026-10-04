import pytest
from fastapi.testclient import TestClient

from gol_backend.main import create_app
from gol_backend.store import Store


def metric(**kw):
    base = dict(run_id="r1", learning="sarsa", tick=10, generation=1, population=300,
                avg_energy=50.0, avg_fitness=1.5, max_fitness=3.0)
    return {**base, **kw}


@pytest.fixture
def client():
    return TestClient(create_app(Store(":memory:")))


def test_post_and_read_back(client):
    assert client.post("/api/metrics", json=metric()).status_code == 201
    client.post("/api/metrics", json=metric(tick=20, population=310))
    rows = client.get("/api/runs/r1/metrics").json()
    assert [r["tick"] for r in rows] == [10, 20]
    assert client.get(f"/api/runs/r1/metrics?after_id={rows[0]['id']}").json()[0]["tick"] == 20


def test_runs_summary(client):
    client.post("/api/metrics", json=metric())
    client.post("/api/metrics", json=metric(run_id="r2", tick=5))
    runs = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert runs["r1"]["reports"] == 1 and runs["r1"]["tick"] == 10
    assert set(runs) == {"r1", "r2"}


def test_validation(client):
    assert client.post("/api/metrics", json=metric(learning="bogus")).status_code == 422
    assert client.post("/api/metrics", json=metric(population=-1)).status_code == 422
    assert client.post("/api/metrics", json=metric(extra=1)).status_code == 422


def test_unknown_run_is_empty(client):
    assert client.get("/api/runs/nope/metrics").json() == []
