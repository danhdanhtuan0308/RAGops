import os
import sys
import pathlib
import pytest


# Ensure API keys exist before importing backend modules during collection
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")

# Ensure backend package is importable when running from repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))


@pytest.fixture(autouse=True)
def _env_keys(monkeypatch):
    # Provide dummy keys so config import doesn't fail
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "test-key"))
    monkeypatch.setenv("COHERE_API_KEY", os.getenv("COHERE_API_KEY", "test-key"))
    # Keep Milvus local defaults but avoid accidental real connections in unit tests
    monkeypatch.setenv("MILVUS_HOST", os.getenv("MILVUS_HOST", "localhost"))
    monkeypatch.setenv("MILVUS_PORT", os.getenv("MILVUS_PORT", "19530"))
    yield
