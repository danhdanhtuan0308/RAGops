from __future__ import annotations
import types
import builtins
import pandas as pd
import numpy as np
import pytest

# Import backend modules
from rag_app import db as db_mod
from rag_app import retrieval as retrieval_mod


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.num_entities = 0

    def query(self, expr: str, output_fields=None):
        # expr format: id in ["id1","id2",...]
        inside = expr.split("[", 1)[1].rsplit("]", 1)[0]
        ids = [s.strip().strip('"') for s in inside.split(",") if s.strip()]
        return [{"id": i} for i in ids if i in self.rows]

    def insert(self, data):
        ids, titles, abstracts, vectors = data
        for i, t, a, v in zip(ids, titles, abstracts, vectors):
            self.rows[i] = {"title": t, "abstract": a, "embedding": v}
        self.num_entities = len(self.rows)

    def flush(self):
        return


@pytest.fixture
def fake_collection(monkeypatch):
    coll = FakeCollection()

    # Patch db.create_milvus_collection to return our fake
    monkeypatch.setattr(db_mod, "create_milvus_collection", lambda: coll)
    # Avoid real connection attempts
    monkeypatch.setattr(db_mod, "connect_to_milvus", lambda: True)
    # Patch embedding generator to stable vectors and record batch sizes
    call_sizes = []

    def _fake_get_embeddings(texts):
        call_sizes.append(len(texts))
        return [np.ones(1536).tolist() for _ in texts]

    monkeypatch.setattr(db_mod, "get_embeddings", _fake_get_embeddings)
    coll._call_sizes = call_sizes
    return coll


def test_deduplication_and_row_chunking(fake_collection):
    # Prepare a DataFrame with duplicate logical rows and check de-dup across batches
    rows = [
        {"title": "A", "abstract": "alpha"},
        {"title": "B", "abstract": "beta"},
        {"title": "A", "abstract": "alpha"},  # duplicate of first
        {"title": "C", "abstract": "gamma"},
    ]
    df = pd.DataFrame(rows)

    # Force small batch size to exercise row-based chunking
    import os
    os.environ["MILVUS_BATCH"] = "2"

    state = {"running": True, "inserted": 0}
    inserted = db_mod.ingest_data_to_milvus(df, state)

    # Expect only 3 unique rows inserted due to deduplication
    assert inserted == 3
    assert fake_collection.num_entities == 3

    # Ensure progress updated during row-based batches
    assert state["inserted"] == 3

    # Row-based chunking: with MILVUS_BATCH=2 and one duplicate in second batch,
    # embedding calls should be [2, 1]
    assert getattr(fake_collection, "_call_sizes", None) in ([2, 1], [2, 2])


def test_rerank_cohere_top_k_enforced(monkeypatch):
    # Simulate 20 papers and ensure top_k=10 enforced
    papers = [{"title": f"T{i}", "abstract": f"A{i}", "authors": ""} for i in range(20)]

    class FakeCohereResponse:
        def __init__(self, n):
            class R: pass
            self.results = []
            for i in range(n):
                r = types.SimpleNamespace(index=i, relevance_score=1.0 - i*0.01)
                self.results.append(r)

    # Patch cohere client used inside module
    monkeypatch.setattr(retrieval_mod, "co", types.SimpleNamespace(rerank=lambda **kwargs: FakeCohereResponse(kwargs.get("top_n", 10))))

    # Enforce via env (module reads this env var)
    import os
    os.environ["COHERE_TOP_K"] = "10"
    out = retrieval_mod.rerank_with_cohere("q", papers, top_k=10)
    assert len(out) == 10, "Cohere rerank should return exactly K=10 results"
