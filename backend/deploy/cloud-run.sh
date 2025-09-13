#!/usr/bin/env bash
set -euo pipefail

# Deploy the FastAPI backend to Google Cloud Run (FastAPI only)
# Requirements:
# - gcloud CLI installed and authenticated
# - APIs enabled: Cloud Run, Cloud Build, Artifact Registry
# - Env vars OPENAI_API_KEY and COHERE_API_KEY set in your shell

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-ragops-backend}
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required (export PROJECT_ID=your-gcp-project)" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" || -z "${COHERE_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY and COHERE_API_KEY must be exported in your shell." >&2
  echo "For Secret Manager based deployment, see README instructions." >&2
  exit 1
fi

echo "Building container with Cloud Build..."
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  "$(dirname "$0")/.."  # build context = backend/

echo "Deploying to Cloud Run service: ${SERVICE} in ${REGION}..."
DEPLOY_ARGS=(
  --image "${IMAGE}"
  --project "${PROJECT_ID}"
  --region "${REGION}"
  --platform managed
  --allow-unauthenticated
  --port 8000
  --timeout 900
  --concurrency 80
  --cpu 1
  --memory 1Gi
)

# Optional: use a custom service account if provided
if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  DEPLOY_ARGS+=(--service-account "${SERVICE_ACCOUNT}")
fi

# Set required API keys; other services (Milvus/Redis) are optional and can be empty
ENV_VARS=(
  "OPENAI_API_KEY=${OPENAI_API_KEY}"
  "COHERE_API_KEY=${COHERE_API_KEY}"
)

[[ -n "${REDIS_HOST:-}" ]] && ENV_VARS+=("REDIS_HOST=${REDIS_HOST}")
[[ -n "${REDIS_PORT:-}" ]] && ENV_VARS+=("REDIS_PORT=${REDIS_PORT}")
[[ -n "${MILVUS_URI:-}" ]] && ENV_VARS+=("MILVUS_URI=${MILVUS_URI}")
[[ -n "${MILVUS_TOKEN:-}" ]] && ENV_VARS+=("MILVUS_TOKEN=${MILVUS_TOKEN}")
[[ -n "${MILVUS_SECURE:-}" ]] && ENV_VARS+=("MILVUS_SECURE=${MILVUS_SECURE}")
[[ -n "${MILVUS_HOST:-}" ]] && ENV_VARS+=("MILVUS_HOST=${MILVUS_HOST}")
[[ -n "${MILVUS_PORT:-}" ]] && ENV_VARS+=("MILVUS_PORT=${MILVUS_PORT}")
[[ -n "${MILVUS_COLLECTION:-}" ]] && ENV_VARS+=("MILVUS_COLLECTION=${MILVUS_COLLECTION}")

DEPLOY_ARGS+=(--set-env-vars "$(IFS=, ; echo "${ENV_VARS[*]}")")

gcloud run deploy "${SERVICE}" "${DEPLOY_ARGS[@]}"

URL=$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')
echo "Deployed: ${URL}"
echo "Try: curl -s ${URL}/status | jq ."
