"""FastAPI entrypoint (CLEAN) – business logic lives in rag_app modules.

This file is intentionally minimal. If you find yourself adding complex
logic here, move it into an appropriate module under rag_app/.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pymilvus import utility, Collection

from rag_app import config  # side-effect: validates env vars on import
from rag_app.models import (
    QueryRequest,
    QueryResponse,
    ResearchPaperResult,
    RankingEntry,
)
from rag_app.db import (
    fetch_data_from_bigquery,
    ingest_data_to_milvus,
    connect_to_milvus,
)
from rag_app.retrieval import search_papers, rerank_with_cohere
from rag_app.llm import handle_summarize_query, generate_answer
from rag_app.eval import evaluate_response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Paper RAG System")

# Simple in-memory ingestion tracking (could be replaced by Redis or DB if needed)
ingestion_state: Dict[str, Any] = {
    "running": False,
    "total": 0,
    "inserted": 0,
    "started_at": None,
    "ended_at": None,
    "error": None,
}


def _background_ingest(df):
    """Run ingestion in a background task."""
    ingestion_state.update(
        {
            "running": True,
            "error": None,
            "inserted": 0,
            "total": len(df),
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
        }
    )
    try:
        inserted = ingest_data_to_milvus(df, ingestion_state)
        ingestion_state["inserted"] = inserted
    except Exception as e:  # noqa
        ingestion_state["error"] = str(e)
        logger.exception("Background ingestion failed")
    finally:
        ingestion_state["running"] = False
        ingestion_state["ended_at"] = datetime.utcnow().isoformat()


@app.post("/ingest", response_model=Dict[str, Any])
async def ingest_data(background_tasks: BackgroundTasks):
    """Trigger BigQuery -> Milvus ingestion in background."""
    if ingestion_state.get("running"):
        return {
            "status": "already_running",
            "inserted": ingestion_state.get("inserted"),
            "total": ingestion_state.get("total"),
        }
    try:
        df = fetch_data_from_bigquery()
        keep = [c for c in df.columns if c in ["title", "abstract"]]
        if keep:
            df = df[keep]
        if df.empty:
            raise RuntimeError("BigQuery returned no rows")
        background_tasks.add_task(_background_ingest, df)
        return {"status": "started", "total": int(len(df))}
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")


@app.get("/ingest/status", response_model=Dict[str, Any])
async def get_ingest_status():
    """Return current ingestion progress."""
    return ingestion_state


@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Perform RAG query or summarization (if query begins with 'summarize')."""
    try:
        q = request.query.strip()
        # Summarization mode
        if q.lower().startswith("summarize"):
            summary_text, context = handle_summarize_query(q, top_k=request.top_k)
            evaluation = evaluate_response(request.query, summary_text, context)
            papers_models: List[ResearchPaperResult] = [
                ResearchPaperResult(
                    id=p.get("id", ""),
                    title=p.get("title", ""),
                    authors=p.get("authors", ""),
                    abstract=p.get("abstract", ""),
                    url=p.get("url", ""),
                    score=p.get("score", 0.0),
                )
                for p in context
            ] if context else []
            ranking = [
                RankingEntry(
                    rank=i + 1,
                    id=p.get("id", ""),
                    title=p.get("title", ""),
                    score=p.get("score", 0.0),
                )
                for i, p in enumerate(context)
            ] if context else []
            concise = (
                "\n".join([ln for ln in summary_text.split("\n") if ln.strip()][:2])[:600]
                if summary_text
                else ""
            )
            return QueryResponse(
                query=request.query,
                answer=concise,
                papers=papers_models,
                evaluation=evaluation,
                summary=summary_text,
                ranking=ranking,
            )

        # Standard RAG mode
        raw_papers = search_papers(request.query, top_k=30)
        if not raw_papers:
            raise HTTPException(status_code=404, detail="No relevant papers found")
        reranked = rerank_with_cohere(request.query, raw_papers, top_k=request.top_k)
        answer = generate_answer(request.query, reranked)
        evaluation = evaluate_response(request.query, answer, reranked)
        papers_models = [
            ResearchPaperResult(
                id=p.get("id", ""),
                title=p.get("title", ""),
                authors=p.get("authors", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                score=p.get("score", 0.0),
            )
            for p in reranked
        ]
        return QueryResponse(
            query=request.query,
            answer=answer,
            papers=papers_models,
            evaluation=evaluation,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


@app.get("/status")
async def get_status():
    """Basic health & collection info."""
    try:
        milvus_connected = False
        try:
            connect_to_milvus()
            milvus_connected = True
        except Exception:
            pass
        collection_exists = False
        record_count = 0
        try:
            if utility.has_collection(config.COLLECTION_NAME):
                collection_exists = True
                collection = Collection(config.COLLECTION_NAME)
                record_count = collection.num_entities
        except Exception:
            pass
        return {
            "status": "ok",
            "milvus_connected": milvus_connected,
            "collection_exists": collection_exists,
            "record_count": record_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:  # noqa
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


def summarize_paper_prompt():  # legacy compatibility shim (can be removed later)
    return "Summary prompt helper (migrated)."


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
