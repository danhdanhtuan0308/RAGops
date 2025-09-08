"""LLM generation, summarization, PDF utilities."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import httpx
import fitz
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import OPENAI_API_KEY
from .retrieval import search_papers, rerank_with_cohere

openai.api_key = OPENAI_API_KEY
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(query: str, papers: List[Dict[str, Any]]) -> str:
    context = ""
    for i, paper in enumerate(papers[:5]):
        context += f"\nPaper {i+1}:\nTitle: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}\nURL: {paper.get('url', '')}\n"
    system_prompt = """
You are a research assistant specialized in summarizing scientific papers and answering questions based on research papers.
Guidelines: cite papers, ground statements, be concise, no fabrication.
""".strip()
    user_prompt = f"Query: {query}\n\nPapers:\n{context}\nProvide answer." 
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:  # noqa
        logger.error(f"Answer gen error: {e}")
        return "Error generating answer."


def fetch_pdf_text(url: str, max_pages: int = 6, max_chars: int = 20000) -> Optional[str]:
    try:
        if 'arxiv.org/pdf/' in url and not url.endswith('.pdf'):
            url = url + '.pdf'
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(url)
            if r.status_code != 200 or 'application/pdf' not in r.headers.get('Content-Type',''):
                return None
            data = r.content
        doc = fitz.open(stream=data, filetype='pdf')
        texts = []
        for i in range(min(len(doc), max_pages)):
            page = doc.load_page(i)
            texts.append(page.get_text())
        full = '\n'.join(texts).strip()[:max_chars]
        return full
    except Exception:
        return None


def extract_arxiv_or_pdf_url(text: str) -> Optional[str]:
    pattern = r'https?://arxiv\.org/pdf/\d{4}\.\d{5}(v\d+)?(?:\.pdf)?'
    m = re.search(pattern, text)
    if m:
        return m.group(0)
    g = re.search(r'https?://\S+\.pdf', text)
    if g:
        return g.group(0)
    return None


def summarize_from_text(title: str, raw: str) -> str:
    sys = """You are an expert multi-disciplinary research assistant. Summarize content grounded only on the provided text."""
    snippet = raw[:12000]
    user = f"TITLE: {title}\nCONTENT:\n{snippet}\nSummary:" 
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            max_tokens=450,
            temperature=0.25
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:  # noqa
        return "Error generating full-text summary."


def generate_summary(paper: Dict[str, Any], context_papers: List[Dict[str, Any]]) -> str:
    main_title = paper.get('title','')
    main_abs = paper.get('abstract','')
    other = ""
    for i,p in enumerate(context_papers[:2]):
        if p.get('id') == paper.get('id'): continue
        other += f"\nContext {i+1}: {p.get('title','')}\nAbstract: {p.get('abstract','')[:800]}\n"
    sys = """Expert research assistant. Produce precise, math-aware summary. Sections: Problem, Method, Results, Significance, Limitations & Open Questions, Key Parameters & Notation (table)."""
    user = f"TITLE: {main_title}\nABSTRACT:{main_abs}\nCONTEXT:{other or 'None'}\nReturn summary only." 
    try:
        resp = openai.ChatCompletion.create(model="gpt-4.1",messages=[{"role":"system","content":sys},{"role":"user","content":user}],temperature=0.3,max_tokens=400)
        return resp['choices'][0]['message']['content'].strip()
    except Exception:
        return "Error generating summary."


def handle_summarize_query(raw_query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    cleaned = re.sub(r'^\s*summarize\s+', '', raw_query, flags=re.IGNORECASE).strip()
    url = extract_arxiv_or_pdf_url(cleaned)
    if url:
        cleaned = cleaned.replace(url, '').strip().strip('()').strip()
    initial = search_papers(cleaned or raw_query, top_k=30)
    context = rerank_with_cohere(cleaned or raw_query, initial, top_k=top_k) if initial else []
    target = context[0] if context else {"title": cleaned, "abstract": ""}
    abstract = target.get('abstract','')
    if abstract and len(abstract.strip()) > 40:
        summary = generate_summary(target, context)
    else:
        pdf_text = fetch_pdf_text(url) if url else None
        if pdf_text:
            target['abstract'] = pdf_text[:8000]
            # Provide consistent fields for downstream logging/UI
            if not target.get('id'):
                target['id'] = 'pdf_direct'
            target.setdefault('score', 1.0)  # Give a non-zero placeholder so user sees "ranked"
            target['source'] = 'pdf_fetch'
            if context:
                context[0] = target
            else:
                context = [target]
            summary = summarize_from_text(target.get('title',''), pdf_text)
        else:
            summary = "Unable to fetch abstract or PDF. Provide a direct PDF URL or ensure paper exists in index."
    # If we only got a single context doc (e.g. PDF fetched) the reranker had no effect; if >1 try a light second-pass rerank
    try:
        if context and len(context) > 1:
            context = rerank_with_cohere(cleaned or raw_query, context, top_k=min(top_k, len(context)))
    except Exception:
        pass
    return summary, context
