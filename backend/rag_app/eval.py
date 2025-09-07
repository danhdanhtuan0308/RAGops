"""Evaluation utilities."""
from __future__ import annotations
import logging
import json
import re
from typing import List, Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import OPENAI_API_KEY
from .models import EvaluationMetrics

openai.api_key = OPENAI_API_KEY
logger = logging.getLogger(__name__)


def extract_score(text: str, dimension: str, default: float = 5.0) -> float:
    try:
        pattern = rf"{dimension}[^0-9]*(\d+(?:\.\d+)?)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val < 0: val = 0.0
            if val > 10: val = 10.0
            return val
        return default
    except Exception:
        return default

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def evaluate_response(query: str, answer: str, papers: List[Dict[str, Any]]) -> EvaluationMetrics:
    if not papers or not any(p.get('abstract','').strip() for p in papers):
        return EvaluationMetrics(
            hallucination_score=5.0,
            truthfulness_score=5.0,
            accuracy_score=5.0,
            relevancy_score=5.0,
            explanation="Evaluation skipped: no abstract text in retrieved documents."
        )
    context = ""
    for i, paper in enumerate(papers[:3]):
        context += f"\nPaper {i+1}:\nTitle: {paper.get('title','')}\nAbstract: {paper.get('abstract','')}\n"
    system_prompt = """You are an expert RAG evaluator. Return STRICT minified JSON with keys Hallucination, Truthfulness, Accuracy, Relevancy (each has score + explanation) and OverallComment. Scores 0-10. No markdown fences."""
    user_prompt = f"Query: {query}\nDocuments:{context}\nAnswer:{answer}\nReturn JSON only." 
    try:
        resp = openai.ChatCompletion.create(model="gpt-4.1",messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],temperature=0.0,max_tokens=500)
        raw = resp['choices'][0]['message']['content'].strip()
        candidate = raw
        if '```' in raw:
            candidate = re.sub(r"```(json)?", "", raw).strip()
        parsed = None
        try:
            start = candidate.find('{'); end = candidate.rfind('}') + 1
            if start != -1 and end != -1:
                parsed = json.loads(candidate[start:end])
        except Exception as je:  # noqa
            logger.warning(f"Eval JSON parse failed: {je}")
        def _safe(obj, key):
            try:
                if obj is None: return None
                v = obj.get(key)
                if isinstance(v, dict): v = v.get('score')
                if v is None: return None
                f = float(v); f = max(0.0, min(10.0, f)); return f
            except Exception: return None
        hs = _safe(parsed, 'Hallucination')
        ts = _safe(parsed, 'Truthfulness')
        acc = _safe(parsed, 'Accuracy')
        rel = _safe(parsed, 'Relevancy')
        if None in [hs, ts, acc, rel]:
            hs = hs or extract_score(raw, 'Hallucination')
            ts = ts or extract_score(raw, 'Truthfulness')
            acc = acc or extract_score(raw, 'Accuracy')
            rel = rel or extract_score(raw, 'Relevancy')
        def _final(v): return 5.0 if v is None else v
        return EvaluationMetrics(hallucination_score=_final(hs), truthfulness_score=_final(ts), accuracy_score=_final(acc), relevancy_score=_final(rel), explanation=raw)
    except Exception as e:  # noqa
        return EvaluationMetrics(hallucination_score=5.0, truthfulness_score=5.0, accuracy_score=5.0, relevancy_score=5.0, explanation=f"Eval error: {e}")
