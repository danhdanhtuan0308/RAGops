from __future__ import annotations
from fastapi.testclient import TestClient
import sys
import types


def test_status_endpoint_ok(monkeypatch):
    # Stub pymilvus before importing app to avoid requiring the real lib
    fake_module = types.ModuleType("pymilvus")
    fake_utility = types.SimpleNamespace(has_collection=lambda name: False)

    class _FakeCollection:
        def __init__(self, *_a, **_k):
            self.num_entities = 0

    fake_module.utility = fake_utility
    fake_module.Collection = _FakeCollection
    sys.modules.setdefault("pymilvus", fake_module)

    import app as backend_app

    client = TestClient(backend_app.app)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("ok", "error")
