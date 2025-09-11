from __future__ import annotations
import sys
import types
import pytest


def test_library_imports():
    # Basic integration: key libs importable
    __import__("fastapi")
    __import__("pymilvus")
    __import__("cohere")
    __import__("openai")
    __import__("redis")


def test_openai_generation_invocation(monkeypatch):
    # Verify our generate_answer calls OpenAI with expected args without real network
    from rag_app import llm as llm_mod

    called = {}

    class FakeOpenAI:
        def ChatCompletion(self):
            return self

        @staticmethod
        def create(model, messages, temperature, max_tokens):  # signature used in code
            called["model"] = model
            called["messages_len"] = len(messages)
            called["temperature"] = temperature
            called["max_tokens"] = max_tokens
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm_mod, "openai", types.SimpleNamespace(ChatCompletion=FakeOpenAI))

    out = llm_mod.generate_answer("test", papers=[{"title":"t","abstract":"a","authors":""}])
    assert out == "ok"
    assert called.get("model") == "gpt-4.1-mini"
    assert called.get("messages_len") >= 2
    assert called.get("max_tokens") == 1000
