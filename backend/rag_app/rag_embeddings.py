"""Embedding related utilities with graceful fallbacks.

Supports both legacy openai.Embedding.create and new OpenAI client API.
If the environment variable RAG_FAKE_EMBED is set, returns zero vectors (for offline tests).
"""
from __future__ import annotations
import time
import logging
from typing import List
import os

from .config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    OPENAI_EMBED_MAX_RETRIES,
    OPENAI_EMBED_RETRY_BASE_SLEEP,
    OPENAI_API_KEY,
    DIMENSION,
)

logger = logging.getLogger(__name__)

_OPENAI_NEW_CLIENT = None
_OPENAI_LEGACY = None
try:  # Detect API style
    import openai  # type: ignore
    # New SDK (>=1.0) removed global create in favor of client
    if hasattr(openai, "OpenAI"):
        from openai import OpenAI  # type: ignore
        _OPENAI_NEW_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    else:
        # Legacy style
        openai.api_key = OPENAI_API_KEY
        _OPENAI_LEGACY = openai
except Exception as _e:  # pragma: no cover
    logger.warning("OpenAI import failed: %s", _e)


def _call_openai_embeddings(payload: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings with retries, supporting both API styles.

    Returns list of embedding vectors. Raises last exception after retries.
    """
    if os.getenv("RAG_FAKE_EMBED"):
        return [[0.0] * DIMENSION for _ in payload]  # offline deterministic vectors
    attempt = 0
    while True:
        try:
            if _OPENAI_NEW_CLIENT:  # New style
                resp = _OPENAI_NEW_CLIENT.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=payload,
                    timeout=60,
                )
                return [d.embedding for d in resp.data]
            elif _OPENAI_LEGACY:  # Legacy style
                resp = _OPENAI_LEGACY.Embedding.create(model=EMBEDDING_MODEL, input=payload, request_timeout=60)
                return [d["embedding"] for d in resp["data"]]
            else:
                raise RuntimeError("OpenAI client not initialized")
        except Exception as e:  # noqa
            attempt += 1
            if attempt >= OPENAI_EMBED_MAX_RETRIES:
                logger.error("Embedding attempts exhausted: %s", e)
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
