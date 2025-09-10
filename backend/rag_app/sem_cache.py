from __future__ import annotations
import os
import json
from typing import Optional, Tuple, Dict, Any
import redis
import numpy as np
from .rag_embeddings import get_embedding
import hashlib


class SemanticCache:
    def __init__(self):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        self.r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        self.prefix = "rag:semcache:"
        self.threshold = float(os.getenv("SEM_CACHE_THRESHOLD", "0.8"))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if not a.size or not b.size:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _query_key(self, query: str) -> str:
        # Use stable hash across processes (avoid Python's randomized hash())
        h = hashlib.sha1(query.encode("utf-8")).hexdigest()
        return f"{self.prefix}q:{h}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        # Fetch most recent entry for this exact hash key; if not present, return None
        key = self._query_key(query)
        raw = self.r.get(key)
        if not raw:
            return None
        try:
            doc = json.loads(raw)
            # Fast path: exact same query string
            if doc.get("q") == query:
                return doc
            # Otherwise validate semantic similarity
            q_emb = np.array(get_embedding(query), dtype=float)
            c_emb = np.array(doc.get("q_embedding", []), dtype=float)
            sim = self._cosine(q_emb, c_emb) if c_emb.size else 0.0
            if sim >= self.threshold:
                return doc
        except Exception:
            return None
        return None

    def set(self, query: str, answer: str, papers: list[dict], ttl_sec: int = 21600):
        key = self._query_key(query)
        try:
            q_emb = get_embedding(query)
        except Exception:
            # Fallback: still cache without an embedding, exact-match will hit
            q_emb = []
        payload = {
            "q": query,
            "q_embedding": q_emb,
            "answer": answer,
            "papers": papers,
        }
        self.r.setex(key, ttl_sec, json.dumps(payload))
