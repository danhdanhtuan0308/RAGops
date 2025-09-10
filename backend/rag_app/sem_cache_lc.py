from __future__ import annotations
import os
import json
from typing import Optional, List, Dict, Any
import numpy as np

from langchain_community.vectorstores import Redis as LCRedis
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from .rag_embeddings import get_embedding


class OpenAIEmbeddingsCompat(Embeddings):
    # Bridge to reuse your existing OpenAI embedding setup
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Chunking not necessary here; upstream already batches; keep simple
        return [get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)


class LCSemanticCache:
    def __init__(self):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        url = f"redis://{host}:{port}"
        self.threshold = float(os.getenv("SEM_CACHE_THRESHOLD", "0.9"))
        self.index_name = os.getenv("SEM_CACHE_INDEX", "rag_semcache")
        self.embeddings = OpenAIEmbeddingsCompat()
        # Create/attach to a Redis vector index via LangChain's Redis vector store
        self.vs = LCRedis(
            redis_url=url,
            index_name=self.index_name,
            embedding=self.embeddings,
            index_schema={
                "text": {"type": "text"},
                "answer": {"type": "text"},
                "papers": {"type": "text"},
            },
        )

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        # KNN search over cached queries
        docs = self.vs.similarity_search(query, k=1)
        if not docs:
            return None
        # LangChain Redis doesn't expose the score directly in Documents; do a manual embed + similarity against the hit
        q_vec = np.array(self.embeddings.embed_query(query), dtype=float)
        try:
            meta = docs[0].metadata or {}
            stored_vec = np.array(meta.get("q_embedding", []), dtype=float)
            if stored_vec.size:
                sim = float(np.dot(q_vec, stored_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(stored_vec) + 1e-9))
            else:
                # Fallback if vector not stored in metadata (older entries)
                sim = 1.0  # already nearest by ANN; accept
            if sim < self.threshold:
                return None
            payload = json.loads(docs[0].page_content)
            return payload
        except Exception:
            return None

    def set(self, query: str, answer: str, papers: List[Dict[str, Any]], ttl_sec: int = 21600):
        payload = {"answer": answer, "papers": papers}
        # Store as the Document text (page_content) with metadata carrying the query vector for explicit sim calc
        q_vec = self.embeddings.embed_query(query)
        doc = Document(
            page_content=json.dumps(payload),
            metadata={"q_embedding": q_vec},
        )
        self.vs.add_documents([doc])
        # TTL: LangChain Redis vector store doesn't natively TTL per-doc; optional: use a parallel key to track expirations.
        # Keep simple for now; can add pruning task later.
