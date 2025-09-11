from __future__ import annotations

from datetime import datetime
import os
import time
import uuid
from typing import Dict, Any, List
import logging
import json
import time as _time

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _METRICS_ENABLED = True
    REQ_COUNTER = Counter("rag_requests_total", "Total /query requests", ["mode"])
    REQ_LATENCY = Histogram("rag_request_latency_seconds", "End-to-end latency per mode", ["mode"], buckets=(0.5,1,2,4,8,16,32,64))
    SEARCH_LATENCY = Histogram("rag_search_latency_seconds", "Milvus search latency seconds")
    RERANK_LATENCY = Histogram("rag_rerank_latency_seconds", "Cohere rerank latency seconds")
    ANSWER_LATENCY = Histogram("rag_answer_latency_seconds", "Answer generation latency seconds")
    EVAL_LATENCY = Histogram("rag_eval_latency_seconds", "LLM evaluation latency seconds")
    CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits", ["mode"])
    CACHE_MISSES = Counter("rag_cache_misses_total", "Semantic cache misses", ["mode"])
    INGEST_IN_PROGRESS = Gauge("rag_ingest_running", "Whether ingestion background task is running (0/1)")
    PROC_CPU_PCT = Gauge("rag_process_cpu_percent", "Process CPU percent (averaged sampling)")
    PROC_MEM_RSS_BYTES = Gauge("rag_process_memory_rss_bytes", "Resident set size in bytes")
    PROC_MEM_VMS_BYTES = Gauge("rag_process_memory_vms_bytes", "Virtual memory size in bytes")
    PROC_OPEN_FDS = Gauge("rag_process_open_fds", "Number of open file descriptors (Linux)" )
    PROC_THREADS = Gauge("rag_process_threads", "Number of active threads in the process")
except Exception:
    _METRICS_ENABLED = False

# ECS Logging Setup 
try:
    import ecs_logging  

    _ecs_handler = logging.StreamHandler()
    _ecs_handler.setFormatter(ecs_logging.StdlibFormatter())
    root_logger = logging.getLogger()
    # Replace any existing default handlers (e.g., from previous basicConfig)
    root_logger.handlers = [_ecs_handler]
    root_logger.setLevel(logging.INFO)
except Exception:  # pragma: no cover
    # Fallback to simple logging if ecs_logging is not available
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

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
# Note: import the cache lazily so failures don't break app startup.


logger = logging.getLogger(__name__)


# --- Helper for structured RAG event logging 
def log_rag_event(kind: str, rag_payload: dict):
    """Log a RAG event with ECS-compatible structure.

    kind: short message string (e.g. 'rag.query.answer') becomes the @message
    rag_payload: dict placed under top-level 'rag' key so Filebeat decode_json_fields
                 can flatten / index. Avoid dotted keys to keep structure explicit.
    """
    event_doc = {
        "event": {"dataset": "ragops.backend.query"},
        "message": kind,  # kept for stdout JSON (Filebeat parses)
        "rag": rag_payload,
    }
    # Raw JSON line (robust ingestion even if logging handlers change)
    try:
        print(json.dumps(event_doc), flush=True)
    except Exception:  # pragma: no cover
        pass
    # Also send via logger for local dev observability
    try:
        # Remove 'message' field for logger extra to avoid clash with LogRecord attributes
        extra_doc = {k: v for k, v in event_doc.items() if k != "message"}
        logger.info(kind, extra=extra_doc)
    except Exception:  # pragma: no cover
        logger.debug("Failed to emit RAG log via logger", exc_info=True)

app = FastAPI(title="Research Paper RAG System")

# Background sampler for process resource metrics (lightweight)
if _METRICS_ENABLED:
    try:  # pragma: no cover
        import psutil, threading
        _proc = psutil.Process()

        def _sample_proc_metrics():
            # Prime CPU percent measurement (first call establishes interval)
            try:
                _proc.cpu_percent(interval=None)
            except Exception:
                pass
            while True:
                try:
                    cpu = _proc.cpu_percent(interval=5.0)  # blocking sleep inside
                    mem = _proc.memory_info()
                    PROC_CPU_PCT.set(cpu)
                    PROC_MEM_RSS_BYTES.set(getattr(mem, "rss", 0))
                    PROC_MEM_VMS_BYTES.set(getattr(mem, "vms", 0))
                    # Some platforms (macOS) may not have num_fds
                    try:
                        PROC_OPEN_FDS.set(_proc.num_fds())
                    except Exception:
                        pass
                    try:
                        PROC_THREADS.set(_proc.num_threads())
                    except Exception:
                        pass
                except Exception:
                    # Swallow errors to keep thread alive
                    time.sleep(5)

        threading.Thread(target=_sample_proc_metrics, name="proc-metrics", daemon=True).start()
    except Exception:
        logger.warning("Process metrics sampler failed to start", exc_info=True)

# Lazily/defensively initialize semantic cache so failures don't break app startup
class _NoCache:
    def get(self, *_args, **_kwargs):
        return None
    def set(self, *_args, **_kwargs):
        return None

def _make_sem_cache():
    # Single implementation: simple Redis-backed semantic cache
    try:
        from rag_app.sem_cache import SemanticCache  # lazy import
        return SemanticCache()
    except Exception as _e:  # pragma: no cover
        logger.warning("Semantic cache init failed, disabling cache: %s", _e)
        return _NoCache()

sem_cache = _make_sem_cache()

# Getting logs from https request
@app.middleware("http")
async def log_http_requests(request, call_next):  # pragma: no cover (runtime concern)
    start = time.time()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    path = request.url.path
    method = request.method
    client = getattr(request.client, "host", None)
    try:
        response = await call_next(request)
        status = response.status_code
        duration_ms = round((time.time() - start) * 1000, 2)
        event_doc = {
            "event": {"dataset": "ragops.backend.http"},
            "message": "http.request",
            "http": {
                "request": {"id": request_id, "method": method, "path": path},
                "response": {"status_code": status, "duration_ms": duration_ms},
            },
            "client": {"ip": client},
            "service": {"name": "ragops-backend"},
        }
        try:
            print(json.dumps(event_doc), flush=True)
        except Exception:
            pass
        # Avoid 'message' overwrite in logging extra
        extra_doc = {k: v for k, v in event_doc.items() if k != "message"}
        logger.info("http.request", extra=extra_doc)
        return response
    except Exception as exc:  # noqa
        duration_ms = round((time.time() - start) * 1000, 2)
        event_doc = {
            "event": {"dataset": "ragops.backend.http"},
            "message": "http.request.error",
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "http": {
                "request": {"id": request_id, "method": method, "path": path},
                "response": {"status_code": 500, "duration_ms": duration_ms},
            },
            "client": {"ip": client},
            "service": {"name": "ragops-backend"},
        }
        try:
            print(json.dumps(event_doc), flush=True)
        except Exception:
            pass
    extra_doc = {k: v for k, v in event_doc.items() if k != "message"}
    logger.exception("http.request.error", extra=extra_doc)
    raise

# Simple in-memory ingestion tracking (could be replaced by Redis or DB if needed)
ingestion_state: Dict[str, Any] = {
    "running": False,
    "total": 0,
    "inserted": 0,
    "started_at": None,
    "ended_at": None,
    "error": None,
}

# On startup try to reflect existing Milvus collection count into ingestion_state
try:  # pragma: no cover - defensive
    connect_to_milvus()
    if utility.has_collection(config.COLLECTION_NAME):
        existing_collection = Collection(config.COLLECTION_NAME)
        existing = existing_collection.num_entities
        ingestion_state["inserted"] = existing
        ingestion_state["total"] = existing
except Exception:
    pass


@app.on_event("startup")
async def _init_ingestion_counts():  # pragma: no cover
    """Populate ingestion_state with existing Milvus entity count at startup (post Milvus readiness)."""
    try:
        connect_to_milvus()
        if utility.has_collection(config.COLLECTION_NAME):
            collection = Collection(config.COLLECTION_NAME)
            n = collection.num_entities
            ingestion_state["inserted"] = n
            ingestion_state["total"] = n
    except Exception:
        logger.warning("Startup ingestion count init failed", exc_info=True)
    # Normalize uvicorn loggers to propagate to root ECS handler
    for _name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        _l = logging.getLogger(_name)
        _l.handlers = []  # remove uvicorn's default plain handlers
        _l.propagate = True


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
        if _METRICS_ENABLED:
            try:
                INGEST_IN_PROGRESS.set(0)
            except Exception:
                pass


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
        if _METRICS_ENABLED:
            try:
                INGEST_IN_PROGRESS.set(1)
            except Exception:
                pass
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
            # Try semantic cache first for summarize queries
            cached = None
            try:
                cached = sem_cache.get(request.query)
            except Exception:
                cached = None
            mode = "summarize"
            req_start = _time.time()
            if cached:
                summary_text = cached.get("answer", "")
                context = cached.get("papers", [])
                log_rag_event(
                    "rag.cache.hit",
                    {
                        "mode": "summarize",
                        "query": request.query,
                        "top_k": request.top_k,
                        "papers": {"count": len(context or [])},
                    },
                )
                if _METRICS_ENABLED:
                    try:
                        CACHE_HITS.labels(mode=mode).inc()
                    except Exception:
                        pass
            else:
                if _METRICS_ENABLED:
                    try:
                        CACHE_MISSES.labels(mode=mode).inc()
                    except Exception:
                        pass
                summary_text, context = handle_summarize_query(q, top_k=request.top_k)
            if _METRICS_ENABLED:
                try:
                    REQ_COUNTER.labels(mode=mode).inc()
                except Exception:
                    pass
            evaluation = evaluate_response(request.query, summary_text, context)
            if _METRICS_ENABLED:
                try:
                    REQ_LATENCY.labels(mode=mode).observe(_time.time() - req_start)
                except Exception:
                    pass
            log_rag_event(
                "rag.query.summary",
                {
                    "mode": "summarize",
                    "query": request.query,
                    "top_k": request.top_k,
                    "answer_preview": (summary_text or "")[:200],
                    "answer_length": len(summary_text or ""),
                    "papers": {"count": len(context or [])},
                    "ranking": [
                        {"rank": i + 1, "id": p.get("id", ""), "score": p.get("score", 0.0)}
                        for i, p in enumerate(context or [])
                    ],
                    "eval": evaluation if isinstance(evaluation, dict) else {},
                },
            )
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
            # Store summarize result in semantic cache for subsequent hits
            try:
                if summary_text:
                    sem_cache.set(request.query, summary_text, context or [])
            except Exception:
                pass
            return QueryResponse(
                query=request.query,
                answer=concise,
                papers=papers_models,
                evaluation=evaluation,
                summary=summary_text,
                ranking=ranking,
            )

        # Standard RAG mode
        # First: semantic cache lookup; if present, skip expensive retrieval/rerank
        cached = None
        try:
            cached = sem_cache.get(request.query)
        except Exception:
            cached = None
        mode = "rag"
        req_start = _time.time()
        if cached:
            answer = cached.get("answer", "")
            reranked = cached.get("papers", [])
            log_rag_event(
                "rag.cache.hit",
                {
                    "mode": "rag",
                    "query": request.query,
                    "top_k": request.top_k,
                    "papers": {"count": len(reranked or [])},
                },
            )
            if _METRICS_ENABLED:
                try:
                    CACHE_HITS.labels(mode=mode).inc()
                except Exception:
                    pass
        else:
            if _METRICS_ENABLED:
                try:
                    CACHE_MISSES.labels(mode=mode).inc()
                except Exception:
                    pass
            raw_papers = search_papers(request.query, top_k=30)
            if not raw_papers:
                raise HTTPException(status_code=404, detail="No relevant papers found")
            # Measure search + rerank + answer generation latencies
            if _METRICS_ENABLED:
                search_start = _time.time()
            reranked = rerank_with_cohere(request.query, raw_papers, top_k=request.top_k)
            if _METRICS_ENABLED:
                try:
                    SEARCH_LATENCY.observe(_time.time() - search_start)
                except Exception:
                    pass
            if _METRICS_ENABLED:
                rerank_start = _time.time()
            # Cohere rerank already happened inside function; treat time inside above
            if _METRICS_ENABLED:
                try:
                    RERANK_LATENCY.observe(_time.time() - rerank_start)
                except Exception:
                    pass
            if _METRICS_ENABLED:
                ans_start = _time.time()
            answer = generate_answer(request.query, reranked)
            if _METRICS_ENABLED:
                try:
                    ANSWER_LATENCY.observe(_time.time() - ans_start)
                except Exception:
                    pass
        if _METRICS_ENABLED:
            try:
                REQ_COUNTER.labels(mode=mode).inc()
            except Exception:
                pass
        evaluation = evaluate_response(request.query, answer, reranked)
        if _METRICS_ENABLED:
            try:
                EVAL_LATENCY.observe( _time.time() - req_start )
            except Exception:
                pass
            try:
                REQ_LATENCY.labels(mode=mode).observe(_time.time() - req_start)
            except Exception:
                pass
        # Save to semantic cache only on successful evaluation
        try:
            if answer:
                sem_cache.set(request.query, answer, reranked)
        except Exception:
            pass
        log_rag_event(
            "rag.query.answer",
            {
                "mode": "rag",
                "query": request.query,
                "top_k": request.top_k,
                "answer_preview": (answer or "")[:200],
                "answer_length": len(answer or ""),
                "papers": {"count": len(reranked or [])},
                "ranking": [
                    {
                        "rank": i + 1,
                        "id": p.get("id", ""),
                        "cohere_score": p.get("score", 0.0),
                    }
                    for i, p in enumerate(reranked or [])
                ],
                "raw_scores": {"sample": [
                    {"id": p.get("id", ""), "score": p.get("score", 0.0)} for p in (raw_papers[:5] or [])
                ]},
                "eval": evaluation if isinstance(evaluation, dict) else {},
            },
        )
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


# Prometheus metrics endpoint
if _METRICS_ENABLED:
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():  # pragma: no cover
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
