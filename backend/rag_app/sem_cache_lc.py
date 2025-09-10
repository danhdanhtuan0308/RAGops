"""
Deprecated: LangChain-based semantic cache stub.

This file is intentionally left as a no-op to avoid import errors in old references.
Production uses only `rag_app.sem_cache.SemanticCache`.
"""

from __future__ import annotations

class LCSemanticCache:  # pragma: no cover
    def __init__(self, *_, **__):
        raise RuntimeError(
            "LCSemanticCache is deprecated and disabled. Use rag_app.sem_cache.SemanticCache instead."
        )
