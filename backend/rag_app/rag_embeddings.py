"""Embedding related utilities."""
from __future__ import annotations
import time
import logging
from typing import List
import openai
from .config import EMBED_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_EMBED_MAX_RETRIES, OPENAI_EMBED_RETRY_BASE_SLEEP, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
logger = logging.getLogger(__name__)


def _call_openai_embeddings(payload: List[str]) -> List[List[float]]:
    attempt = 0
    while True:
        try:
            resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=payload)
            return [d["embedding"] for d in resp["data"]]
        except Exception as e:  # noqa
            attempt += 1
            if attempt >= OPENAI_EMBED_MAX_RETRIES:
                raise
            sleep_s = OPENAI_EMBED_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            time.sleep(sleep_s)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    cleaned: List[str] = []
    for t in texts:
        t = (t or "").replace("\n", " ").strip() or "empty"
        cleaned.append(t)
    vectors: List[List[float]] = []
    for i in range(0, len(cleaned), EMBED_BATCH_SIZE):
        slice_batch = cleaned[i:i+EMBED_BATCH_SIZE]
        vectors.extend(_call_openai_embeddings(slice_batch))
        time.sleep(0.1)
    return vectors


def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]
