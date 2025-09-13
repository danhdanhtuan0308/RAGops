
## Features & Pipeline for RAG system 

- **Data Ingestion**: Retrieves research paper data from Google BigQuery
- **Vector Storage**: Uses Milvus as a vector database
- **Embedding**: Uses OpenAI's text-embedding-3-small 
- **Retrieval**: Fast vector similarity search using Milvus
- **Reranking**: Semantic reranking with Cohere
- **Answer Generation**: Synthesizes information with OpenAI's GPT-4.1
- **Evaluation**: Automatic evaluation of responses for hallucination, truthfulness, accuracy, and relevancy
- **Metadata Enrichment**: Extracts additional paper metadata from URLs 
- **Implement Cache** : Saving cost + time 

## Cloud Run Deployment (FastAPI only)
- **Goal:** Deploy only the FastAPI backend to Google Cloud Run as a public HTTPS endpoint. Redis/Milvus remain external/optional and can be configured via environment variables.

### Prerequisites
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Select project and region:
	- `export PROJECT_ID=<your-project>`
	- `export REGION=us-central1` (or preferred region)
- Enable required APIs:
	- `gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID`
- API keys in your shell:
	- `export OPENAI_API_KEY=...`
	- `export COHERE_API_KEY=...`

Optional (if using hosted services):
- Redis: `export REDIS_HOST=... REDIS_PORT=6379`
- Milvus (e.g., Zilliz Cloud): `export MILVUS_URI=... MILVUS_TOKEN=... MILVUS_SECURE=true`
- Or self-hosted Milvus: `export MILVUS_HOST=... MILVUS_PORT=19530`

### Deploy
From this folder (`backend/`):

```bash
bash deploy/cloud-run.sh
```

This builds the container with Cloud Build and deploys to Cloud Run. On success it prints the service URL, e.g., `https://ragops-backend-xxxx-uc.a.run.app`.

### Test
```bash
SERVICE_URL=$(gcloud run services describe ragops-backend --region ${REGION:-us-central1} --format='value(status.url)')
curl -s "$SERVICE_URL/status" | jq .
```

### Notes
- The container binds to `$PORT` automatically as required by Cloud Run.
- If Redis/Milvus are not available, the app will still start; vector search will return empty results and semantic cache will be disabled gracefully.
- To use Secret Manager instead of inline env vars, create secrets and bind them as env vars via `gcloud run deploy --set-secrets OPENAI_API_KEY=projects/$PROJECT_ID/secrets/OPENAI_API_KEY:latest,...`.

### Local Docker compose tips
- Ensure the BigQuery credential file is mounted as a file, not a folder. The compose file binds `./gcp-credentials.json:/app/gcp-credentials.json:ro` and sets `GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json`.
- Backend now waits for Milvus health (`depends_on.condition: service_healthy`), reducing early connection retries.
- If ingestion appears to stall, check logs: `docker compose logs -f backend`. You should see batch progress like `Ingest batch X/Y...`. OpenAI embedding calls have a 60s timeout and retries; sustained failures will fall back to zero vectors so ingestion continues.
