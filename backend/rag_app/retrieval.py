"""Search, rerank, and related retrieval helpers."""
from __future__ import annotations
from typing import List, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from pymilvus import Collection, utility
import cohere
from .config import COLLECTION_NAME
from .rag_embeddings import get_embedding
from .db import connect_to_milvus, create_milvus_collection
from .config import COHERE_API_KEY

logger = logging.getLogger(__name__)
co = cohere.Client(COHERE_API_KEY)


def search_papers(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    connect_to_milvus()
    # Ensure collection exists; if not, create (empty) so we can respond gracefully
    try:
        if not utility.has_collection(COLLECTION_NAME):
            create_milvus_collection()
        collection = Collection(COLLECTION_NAME)
    except Exception as e:  # noqa
        logger.error(f"Milvus collection access error: {e}")
        return []
    # If collection empty, return early
    try:
        if collection.is_empty:
            logger.info(
                "search.skip",
                extra={
                    "rag.query": query,
                    "reason": "collection_empty",
                    "rag.top_k": top_k,
                },
            )
            return []
        collection.load()
    except Exception as e:  # noqa
        logger.warning(f"Could not load collection (may be empty): {e}")
        return []
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
    try:
        logger.info(
            "search.results",
            extra={
                "rag.query": query,
                "rag.returned": len(papers),
                "rag.top_k": top_k,
            },
        )
    except Exception:
        pass
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def rerank_with_cohere(query: str, papers: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    if not papers:
        logger.info("rerank.skip", extra={"reason": "no_papers", "rag.query": query, "rag.top_k": top_k})
        return []
    docs = []
    for paper in papers:
        doc_text = f"Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}"
        docs.append(doc_text)
    try:
        logger.info(
            "rerank.invoke",
            extra={
                "rag.query": query,
                "rag.candidate_count": len(docs),
                "rag.top_k": top_k,
            },
        )
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
        try:
            logger.info(
                "rerank.results",
                extra={
                    "rag.query": query,
                    "rag.returned": len(reranked),
                    "rag.sample_scores": [round(x.get("score", 0.0), 4) for x in reranked[:5]],
                },
            )
        except Exception:  # pragma: no cover
            pass
        return reranked
    except Exception as e:  # noqa
        logger.error(f"Cohere rerank error: {e}")
        return papers[:top_k]
