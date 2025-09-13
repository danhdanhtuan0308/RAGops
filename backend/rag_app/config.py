"""Configuration and environment loading for the RAG application."""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# Optional hosted Milvus connection settings (e.g., Zilliz Cloud)
MILVUS_URI = os.getenv("MILVUS_URI", "")  # e.g., https://xxxxx.zillizcloud.com:19530
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_SECURE = os.getenv("MILVUS_SECURE", "true").lower() in ("1", "true", "yes")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "research_papers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DIMENSION = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
OPENAI_EMBED_MAX_RETRIES = int(os.getenv("OPENAI_EMBED_MAX_RETRIES", "5"))
OPENAI_EMBED_RETRY_BASE_SLEEP = float(os.getenv("OPENAI_EMBED_RETRY_BASE_SLEEP", "2"))

PROJECT_ID = "ragops"
DATASET_ID = "Research_paper"
TABLE_ID = "final_table"

# (Removed dynamic evaluation model configuration; using fixed model in eval.py)
