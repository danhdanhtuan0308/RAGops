# Observability Stack (Prometheus + Grafana)

## Overview
This directory provides a lightweight local stack to collect and visualize metrics from the RAG backend.

Components:
- Prometheus (scrapes FastAPI `/metrics` every 15s)
- Grafana (dashboards & visualization)

The backend now exposes Prometheus metrics at `/metrics` including:
- `rag_requests_total{mode}` – total requests per mode (`rag`, `summarize`)
- `rag_request_latency_seconds{mode}` – end‑to‑end latency histogram
- `rag_search_latency_seconds` – Milvus search latency
- `rag_rerank_latency_seconds` – Cohere rerank latency (wrapper timing)
- `rag_answer_latency_seconds` – LLM answer generation latency
- `rag_eval_latency_seconds` – LLM evaluation latency
- `rag_cache_hits_total{mode}` / `rag_cache_misses_total{mode}` – cache effectiveness
- `rag_ingest_running` – ingestion background task gauge (1 while running)
- `rag_process_cpu_percent` – sampled process CPU usage (%)
- `rag_process_memory_rss_bytes` – resident memory
- `rag_process_memory_vms_bytes` – virtual memory size
- `rag_process_open_fds` – open file descriptors
- `rag_process_threads` – thread count

## Prerequisites
Start the main application stack first (so service name `backend` is reachable on the shared network):

```
docker compose up -d --build
```

Then start observability services:

```
docker compose -f observability/docker-compose.observability.yml up -d
```

## Access
- Prometheus: http://localhost:9090/targets (verify `rag-backend` is UP)
- Grafana: http://localhost:3000  (login: admin / admin)

## Add Prometheus Datasource in Grafana
1. Open Grafana → Settings (gear) → Data Sources → Add data source
2. Select Prometheus
3. URL: http://prometheus:9090
4. Save & Test

## Suggested Panels
- Graph (Time series): `sum(rate(rag_requests_total[5m])) by (mode)`
- Histogram heatmap: `rag_request_latency_seconds_bucket`
- Single Stat: `sum(rag_cache_hits_total) / (sum(rag_cache_hits_total)+sum(rag_cache_misses_total))`
- Table (Latency breakdown): use recording rules or direct queries for p95: `histogram_quantile(0.95, sum(rate(rag_request_latency_seconds_bucket[5m])) by (le, mode))`
- CPU (%): `rag_process_cpu_percent`
- RSS MB: `rag_process_memory_rss_bytes / 1024 / 1024`
- Open FDs: `rag_process_open_fds`

## Troubleshooting
- No metrics: Hit the API at least once (Prometheus scrapes passively)
- Target DOWN: Ensure both compose projects share network `ragops-network` (created by main stack)
- Empty latency buckets: Generate traffic (curl loop or load test)

## Cleanup
```
docker compose -f observability/docker-compose.observability.yml down -v
```

