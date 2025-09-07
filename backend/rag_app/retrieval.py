"""Search, rerank, and related retrieval helpers."""
from __future__ import annotations
from typing import List, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pymilvus import Collection
import cohere
from .config import COLLECTION_NAME
from .rag_embeddings import get_embedding
from .db import connect_to_milvus
from .config import COHERE_API_KEY

logger = logging.getLogger(__name__)
co = cohere.Client(COHERE_API_KEY)


def search_papers(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    connect_to_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    query_embedding = get_embedding(query)
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "abstract"]
    )
    def _safe_get(hit_obj, field: str) -> str:
        try:
            return hit_obj.entity.get(field) or ""
        except Exception:
            return ""
    papers: List[Dict[str, Any]] = []
    for hits in results:
        for hit in hits:
            papers.append({
                "id": hit.id,
                "title": _safe_get(hit, "title"),
                "abstract": _safe_get(hit, "abstract"),
                "authors": "",
                "url": "",
                "categories": "",
                "published_date": "",
                "score": hit.score
            })
    collection.release()
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def rerank_with_cohere(query: str, papers: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    if not papers:
        return []
    docs = []
    for paper in papers:
        doc_text = f"Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}"
        docs.append(doc_text)
    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_k
        )
        reranked = []
        for r in response.results:
            p = papers[r.index]
            p["score"] = r.relevance_score
            reranked.append(p)
        return reranked
    except Exception as e:  # noqa
        logger.error(f"Cohere rerank error: {e}")
        return papers[:top_k]
