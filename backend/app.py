"""
RAG Application for Research Paper Retrieval and Summarization
- Ingests data from Google BigQuery 
- Stores vectors in Milvus
- Uses OpenAI for embeddings and LLM
- Uses Cohere for reranking
- Provides evaluation metrics
"""

import os
import json
import logging
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Data processing and vector operations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import fitz  # PyMuPDF

# Google Cloud
from google.cloud import bigquery
from google.oauth2 import service_account

# Vector Database
from pymilvus import (
    connections, 
    utility,
    Collection,
    CollectionSchema, 
    FieldSchema,
    DataType
)

# LLM Providers
import openai
import cohere

# LlamaIndex
from llama_index import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

# API and utilities
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set")

# Initialize clients
openai.api_key = OPENAI_API_KEY
co = cohere.Client(COHERE_API_KEY)

# Milvus / Embedding configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "research_papers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
DIMENSION = int(os.getenv("EMBEDDING_DIM", "1536"))  # keep configurable, must match model
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
OPENAI_EMBED_MAX_RETRIES = int(os.getenv("OPENAI_EMBED_MAX_RETRIES", "5"))
OPENAI_EMBED_RETRY_BASE_SLEEP = float(os.getenv("OPENAI_EMBED_RETRY_BASE_SLEEP", "2"))

# BigQuery configuration
PROJECT_ID = "ragops"
DATASET_ID = "Research_paper"
TABLE_ID = "final_table"

# FastAPI app
app = FastAPI(title="Research Paper RAG System")

# Ingestion job state (very simple in-memory tracking)
ingestion_state: Dict[str, Any] = {
    "running": False,
    "total": 0,
    "inserted": 0,
    "started_at": None,
    "ended_at": None,
    "error": None
}

#------------------------------------------------------------------------
# Data Models
#------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to search for papers")
    top_k: int = Field(10, description="Number of results to return")
    get_metadata: bool = Field(True, description="Whether to get additional metadata from the URL")

class EvaluationMetrics(BaseModel):
    hallucination_score: float
    truthfulness_score: float 
    accuracy_score: float
    relevancy_score: float
    explanation: str

class ResearchPaperResult(BaseModel):
    id: str
    title: str
    authors: str
    abstract: str
    url: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class RankingEntry(BaseModel):
    rank: int
    id: str
    title: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    papers: List[ResearchPaperResult]
    evaluation: EvaluationMetrics
    summary: Optional[str] = None  # Detailed summary if summarization requested
    ranking: Optional[List[RankingEntry]] = None  # Ordered ranking list

class SummarizeRequest(BaseModel):  # retained for potential internal use
    title: str = Field(..., description="Exact or partial paper title to summarize")
    top_k_context: int = Field(3, description="How many nearest papers to pull for context")

#------------------------------------------------------------------------
# Database Connection Functions
#------------------------------------------------------------------------

def connect_to_milvus():
    """Establish connection to Milvus server"""
    max_retries = 5
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            connections.connect(
                alias="default", 
                host=MILVUS_HOST, 
                port=MILVUS_PORT
            )
            logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} to connect to Milvus failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {str(e)}")
                raise

def create_milvus_collection():
    """Create Milvus collection if it doesn't exist (id,title,abstract,embedding)."""
    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Collection {COLLECTION_NAME} already exists – keeping existing data")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields=fields, description="Research papers collection (minimal schema)")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info(f"Created collection {COLLECTION_NAME} (dim={DIMENSION}, model={EMBEDDING_MODEL})")
    return collection

def get_bigquery_client():
    """Get BigQuery client with default credentials"""
    try:
        # First try using environment variable
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                client = bigquery.Client(project=PROJECT_ID)
                logger.info("Connected to BigQuery using GOOGLE_APPLICATION_CREDENTIALS")
                return client
            except Exception as e:
                logger.warning(f"Failed to connect using GOOGLE_APPLICATION_CREDENTIALS: {str(e)}")
        
        # Try alternative authentication methods
        try:
            # Try application default credentials
            client = bigquery.Client(project=PROJECT_ID)
            logger.info("Connected to BigQuery using application default credentials")
            return client
        except Exception as e:
            logger.warning(f"Failed to connect using application default credentials: {str(e)}")
            
        # If all authentication methods fail, raise exception
        raise Exception("All authentication methods failed. Please check your credentials.")
    except Exception as e:
        logger.error(f"Failed to connect to BigQuery: {str(e)}")
        raise

#------------------------------------------------------------------------
# Data Ingestion Functions
#------------------------------------------------------------------------

def fetch_data_from_bigquery():
    """Fetch research paper data from BigQuery"""
    client = get_bigquery_client()
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    
    logger.info(f"Executing BigQuery: {query}")
    query_job = client.query(query)
    results = query_job.result()
    
    # Convert to DataFrame
    df = results.to_dataframe()
    logger.info(f"Retrieved {len(df)} rows from BigQuery")
    return df

def _call_openai_embeddings(payload: List[str]) -> List[List[float]]:
    """Low-level OpenAI embedding call with retries & backoff."""
    attempt = 0
    while True:
        try:
            resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=payload)
            vectors = [d["embedding"] for d in resp["data"]]
            return vectors
        except Exception as e:
            attempt += 1
            if attempt >= OPENAI_EMBED_MAX_RETRIES:
                logger.error(f"OpenAI embedding failed after {attempt} attempts: {e}")
                raise
            sleep_s = OPENAI_EMBED_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(f"OpenAI embed attempt {attempt} failed: {e} – retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch embedding with sanitization & chunking (resilient)."""
    cleaned: List[str] = []
    for t in texts:
        t = (t or "").replace('\n', ' ').strip()
        if not t:
            t = "empty"
        cleaned.append(t)

    vectors: List[List[float]] = []
    for i in range(0, len(cleaned), EMBED_BATCH_SIZE):
        slice_batch = cleaned[i:i+EMBED_BATCH_SIZE]
        vecs = _call_openai_embeddings(slice_batch)
        vectors.extend(vecs)
        # Small pacing delay to reduce chance of disconnect / rate limit
        time.sleep(0.1)
    return vectors

def get_embedding(text: str) -> List[float]:
    """Single text helper (wraps batch pathway)."""
    return get_embeddings([text])[0]

def create_paper_embedding(paper: Dict[str, Any]) -> List[float]:
    """Create embedding for a research paper by combining relevant fields"""
    # Combine fields for comprehensive embedding
    combined_text = f"Title: {paper.get('title', '')}\n"
    combined_text += f"Authors: {paper.get('authors', '')}\n"
    combined_text += f"Abstract: {paper.get('abstract', '')}\n"
    combined_text += f"Categories: {paper.get('categories', '')}"
    
    return get_embedding(combined_text)

def _deterministic_id(title: str, abstract: str) -> str:
    h = hashlib.sha1((title + "||" + abstract).encode("utf-8")).hexdigest()[:32]
    return f"doc_{h}"

def ingest_data_to_milvus(df: pd.DataFrame):
    """Robust ingestion: batch embeddings, resume-friendly, no collection drop."""
    connect_to_milvus()
    collection = create_milvus_collection()
    total_records = len(df)
    if total_records == 0:
        raise ValueError("No rows to ingest")

    # Normalize columns
    if "title" not in df.columns:
        raise ValueError("DataFrame missing required column 'title'")
    if "abstract" not in df.columns:
        df["abstract"] = ""

    # Use larger logical batch for embeddings than for Milvus insert if needed
    MILVUS_BATCH = int(os.getenv("MILVUS_BATCH", "256"))

    processed = 0
    inserted = 0
    start_time = time.time()

    # Preload existing IDs for resume (optional – query per batch to limit memory)
    def fetch_existing(ids_chunk: List[str]) -> set:
        if not ids_chunk:
            return set()
        expr_ids = ",".join([f'"{i}"' for i in ids_chunk])
        expr = f"id in [{expr_ids}]"
        try:
            res = collection.query(expr=expr, output_fields=["id"])
            return {r["id"] for r in res}
        except Exception:
            return set()

    for start in range(0, total_records, MILVUS_BATCH):
        end = min(start + MILVUS_BATCH, total_records)
        batch_df = df.iloc[start:end].copy()

        titles = batch_df["title"].astype(str).str.slice(0, 500).tolist()
        abstracts = batch_df["abstract"].astype(str).str.slice(0, 10000).tolist()
        ids = [_deterministic_id(t, a) for t, a in zip(titles, abstracts)]

        # Skip IDs that already exist
        existing = fetch_existing(ids)
        if existing:
            mask = [i not in existing for i in ids]
            titles = [t for t, m in zip(titles, mask) if m]
            abstracts = [a for a, m in zip(abstracts, mask) if m]
            ids = [i for i, m in zip(ids, mask) if m]
            if not ids:
                logger.info(f"Batch {start}-{end}: all {len(existing)} docs already ingested – skipping")
                continue

        # Prepare texts for embeddings
        embed_texts = [f"Title: {t}\nAbstract: {a}" for t, a in zip(titles, abstracts)]

        try:
            vectors = get_embeddings(embed_texts)
        except Exception as embed_err:
            logger.error(f"Batch embedding failed (batch start={start}): {embed_err} – will fallback to zero vectors")
            vectors = [[0.0] * DIMENSION for _ in embed_texts]

        # Sanity check
        good_rows = [(i, t, a, v) for i, t, a, v in zip(ids, titles, abstracts, vectors) if len(v) == DIMENSION]
        if not good_rows:
            logger.warning(f"No valid rows to insert for batch {start}-{end}")
            continue

        ins_ids, ins_titles, ins_abstracts, ins_vectors = zip(*good_rows)
        data = [list(ins_ids), list(ins_titles), list(ins_abstracts), list(ins_vectors)]

        try:
            collection.insert(data)
            inserted += len(ins_ids)
        except Exception as insert_err:
            logger.error(f"Insert error batch {start}-{end}: {insert_err}")
            continue  # Skip failing batch – keep going

        processed += (end - start)
        if inserted % 1000 < len(ins_ids):  # every ~1000 new inserts
            collection.flush()
        elapsed = time.time() - start_time
        logger.info(f"Progress: inserted={inserted} processed_rows={processed}/{total_records} elapsed={elapsed:.1f}s")
        # Cooperative yield so event loop can serve /status when called from thread
        time.sleep(0)
        # Update global ingestion state if used
        if ingestion_state.get("running"):
            ingestion_state["inserted"] = inserted

    collection.flush()
    logger.info(f"Ingestion finished: inserted={inserted} total_rows={total_records}")
    return inserted

#------------------------------------------------------------------------
# Search and Retrieval Functions
#------------------------------------------------------------------------

def search_papers(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    """Search for papers using simplified schema (id,title,abstract)."""
    connect_to_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()

    query_embedding = get_embedding(query)
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "abstract"]
    )

    def _safe_get(hit_obj, field: str) -> str:
        try:
            val = hit_obj.entity.get(field)  # Milvus Hit.entity.get(field) has no default param
            return val if val is not None else ""
        except Exception:
            return ""

    papers: List[Dict[str, Any]] = []
    for hits in results:
        for hit in hits:
            papers.append({
                "id": hit.id,
                "title": _safe_get(hit, "title"),
                "abstract": _safe_get(hit, "abstract"),
                "authors": "",
                "url": "",
                "categories": "",
                "published_date": "",
                "score": hit.score
            })
    collection.release()
    return papers

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def rerank_with_cohere(query: str, papers: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Rerank papers using Cohere API"""
    if not papers:
        return []
    
    # Prepare documents for reranking
    docs = []
    for paper in papers:
        # Create a comprehensive document for reranking
        doc_text = f"Title: {paper.get('title', '')}\n"
        doc_text += f"Authors: {paper.get('authors', '')}\n"
        doc_text += f"Abstract: {paper.get('abstract', '')}"
        docs.append(doc_text)
    
    # Call Cohere rerank API
    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_k
        )
        
        # Process results
        reranked_papers = []
        for result in response.results:
            index = result.index
            reranked_paper = papers[index]
            reranked_paper["score"] = result.relevance_score
            reranked_papers.append(reranked_paper)
        
        return reranked_papers
    except Exception as e:
        logger.error(f"Error during Cohere reranking: {str(e)}")
        # Fallback to original order but limit to top_k
        return papers[:top_k]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_url_metadata(url: str) -> Dict[str, Any]:
    """Get additional metadata from the paper URL using OpenAI"""
    try:
        system_prompt = """
        You are a research assistant that extracts key metadata from academic paper URLs. 
        Please extract and structure the following information if available:
        - Publication venue (journal/conference)
        - Publication year
        - Citation count (if visible)
        - Key topics or themes
        - Main contributions
        """
        
        user_prompt = f"Please extract metadata from this research paper URL: {url}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse response as JSON if possible, otherwise return as text
        metadata_text = response['choices'][0]['message']['content']
        try:
            # Try to extract structured data from the response
            metadata = json.loads(metadata_text)
            return metadata
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {"extracted_text": metadata_text}
            
    except Exception as e:
        logger.error(f"Error getting URL metadata: {str(e)}")
        return {"error": str(e)}

#------------------------------------------------------------------------
# RAG and LLM Functions
#------------------------------------------------------------------------

def create_llama_index(papers: List[Dict[str, Any]]):
    """Create LlamaIndex from papers for advanced retrieval"""
    # Create documents
    documents = []
    for paper in papers:
        # Create content combining all relevant fields
        content = f"Title: {paper.get('title', '')}\n"
        content += f"Authors: {paper.get('authors', '')}\n"
        content += f"Abstract: {paper.get('abstract', '')}\n"
        content += f"Categories: {paper.get('categories', '')}\n"
        content += f"URL: {paper.get('url', '')}\n"
        content += f"Published: {paper.get('published_date', '')}"
        
        # Create document with metadata
        doc = Document(
            text=content,
            metadata={
                "id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "url": paper.get("url", ""),
                "score": paper.get("score", 0.0)
            }
        )
        documents.append(doc)
    
    # Create embedding model
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
        dimensions=DIMENSION
    )
    
    # Create vector store index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )
    
    return index

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(query: str, papers: List[Dict[str, Any]]) -> str:
    """Generate answer based on retrieved papers using OpenAI"""
    # Create context from papers
    context = ""
    for i, paper in enumerate(papers[:5]):  # Use top 5 papers for context
        context += f"\nPaper {i+1}:\n"
        context += f"Title: {paper.get('title', '')}\n"
        context += f"Authors: {paper.get('authors', '')}\n"
        context += f"Abstract: {paper.get('abstract', '')}\n"
        context += f"URL: {paper.get('url', '')}\n"
    
    # Create prompt
    system_prompt = """
    You are a research assistant specialized in summarizing scientific papers and answering questions based on research papers.
    Your task is to synthesize information from the provided papers to answer the user's query.
    
    Guidelines:
    1. Focus on information directly supported by the papers
    2. Cite specific papers when providing information
    3. If the papers don't contain enough information to answer completely, acknowledge the limitations
    4. Provide a concise but comprehensive answer
    5. Structure your response logically with clear sections
    6. Do not fabricate information not present in the papers
    """
    
    user_prompt = f"""
    Query: {query}
    
    Based on the following research papers, please provide a comprehensive answer:
    
    {context}
    """
    
    # Generate answer
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Error generating answer. Please try again later."

def generate_summary(paper: Dict[str, Any], context_papers: List[Dict[str, Any]]) -> str:
    """Generate a focused, math-aware summary for a single paper using its abstract plus light context."""
    main_title = paper.get("title", "")
    main_abstract = paper.get("abstract", "")
    # Build context block
    other_context = ""
    for i, p in enumerate(context_papers[:2]):
        if p.get("id") == paper.get("id"):  # skip self duplicate in context list
            continue
        other_context += f"\nContext Paper {i+1}: {p.get('title','')}\nAbstract: {p.get('abstract','')[:800]}\n"

    system_prompt = """
You are an expert researcher scientific assistant at multiple field such as bio, physics, math, stats , quant finance , econ , computer science, ees. Produce a precise, information-dense, math-aware summary of the target paper.
Sections REQUIRED (use Markdown headings):
1. Problem
2. Method (include key equations / variables if present; represent equations in LaTeX inline $...$ or block $$...$$; do NOT invent symbols).
3. Results (quantitative findings; trends; comparative performance)
4. Physical / Theoretical Significance
5. Limitations & Open Questions
6. Key Parameters & Notation (table)

Guidelines:
- Base ONLY on provided abstract/context; if math not explicit, state "No explicit equations provided in abstract".
- Aim 180-260 words.
- No hallucinated numerical values.
- Provide a concise Markdown table with columns: Symbol | Meaning | Notes (only if symbols exist).
- If insufficient data for a section, write: "Insufficient detail in abstract".
"""

    user_prompt = f"""
TARGET TITLE: {main_title}\n
TARGET ABSTRACT:\n{main_abstract}\n
OPTIONAL CONTEXT:\n{other_context if other_context else 'None'}\n
Return ONLY the summary paragraphs (no preface, no JSON)."""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa
        logger.error(f"Summary generation error: {e}")
        return "Error generating summary."

def fetch_pdf_text(url: str, max_pages: int = 6, max_chars: int = 20000) -> Optional[str]:
    """Download PDF and extract text (first pages) using PyMuPDF."""
    try:
        # Normalize arXiv URL (add .pdf if missing)
        if 'arxiv.org/pdf/' in url and not url.endswith('.pdf'):
            url = url + '.pdf'
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(url)
            if r.status_code != 200 or 'application/pdf' not in r.headers.get('Content-Type',''):
                logger.warning(f"PDF fetch failed or not PDF content-type: status={r.status_code}")
                return None
            data = r.content
        doc = fitz.open(stream=data, filetype='pdf')
        texts = []
        for i in range(min(len(doc), max_pages)):
            page = doc.load_page(i)
            texts.append(page.get_text())
        full_text = '\n'.join(texts).strip()
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]
        return full_text
    except Exception as e:  # noqa
        logger.error(f"PDF fetch/extract error: {e}")
        return None

def summarize_from_text(title: str, raw_text: str) -> str:
    """Summarize raw full-text (truncated) when abstract is missing."""
    system_prompt = """
You are an expert multi-disciplinary research assistant (covering CS, physics, biology, math, statistics, economics, finance, engineering). Summarize the paper content provided WITHOUT assuming any single domain unless explicitly evident from the text.
Required Sections (Markdown headings):
1. Problem
2. Methods
3. Key Results
4. Interpretation / Significance
5. Limitations & Future Work

Guidelines:
- 180-230 words.
- No fabrication beyond provided text; if numerical values or equations are absent, do not invent them.
- If the supplied text appears partial or truncated, add a final line: "Note: Source text appears partial/truncated.".
"""
    snippet = raw_text[:12000]
    user_prompt = f"TITLE: {title}\n\nCONTENT (truncated if long):\n{snippet}\n\nProvide the summary now."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
            max_tokens=450,
            temperature=0.25
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:  # noqa
        logger.error(f"Full-text summary error: {e}")
        return "Error generating full-text summary."

def extract_arxiv_or_pdf_url(text: str) -> Optional[str]:
    """Extract first arXiv/pdf-like URL from text."""
    pattern = r'https?://arxiv\.org/pdf/\d{4}\.\d{5}(v\d+)?(?:\.pdf)?'
    m = re.search(pattern, text)
    if m:
        return m.group(0)
    # generic pdf URL
    generic = re.search(r'https?://\S+\.pdf', text)
    if generic:
        return generic.group(0)
    return None

def handle_summarize_query(raw_query: str, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Process a 'summarize <title> (url?)' style query returning a summary answer and context papers."""
    # Strip leading keyword
    cleaned = re.sub(r'^\s*summarize\s+', '', raw_query, flags=re.IGNORECASE).strip()
    url = extract_arxiv_or_pdf_url(cleaned)
    if url:
        cleaned = cleaned.replace(url, '').strip().strip('()').strip()
    # First stage: broad vector search (fixed, e.g. 30) then Cohere rerank to requested top_k
    initial_candidates = search_papers(cleaned or raw_query, top_k=30)
    context_papers = rerank_with_cohere(cleaned or raw_query, initial_candidates, top_k=top_k) if initial_candidates else []
    target = context_papers[0] if context_papers else {"title": cleaned, "abstract": ""}
    abstract = target.get('abstract','') if target else ''
    summary_answer = None
    # If we have abstract text of reasonable length use abstract summary; else attempt PDF
    if abstract and len(abstract.strip()) > 40:
        summary_answer = generate_summary(target, context_papers)
    else:
        pdf_text = None
        if url:
            pdf_text = fetch_pdf_text(url)
        if pdf_text:
            summary_answer = summarize_from_text(target.get('title',''), pdf_text)
        else:
            summary_answer = ("Unable to locate sufficient abstract text or fetch PDF content for the requested title. "
                              "Provide a direct PDF URL or ensure the paper exists in the index.")
    return summary_answer, context_papers

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def evaluate_response(query: str, answer: str, papers: List[Dict[str, Any]]) -> EvaluationMetrics:
    """Evaluate the response using LLM as judge"""
    # Create context from papers
    context = ""
    for i, paper in enumerate(papers[:3]):  # Use top 3 papers for evaluation context
        context += f"\nPaper {i+1}:\n"
        context += f"Title: {paper.get('title', '')}\n"
        context += f"Abstract: {paper.get('abstract', '')}\n"
    
    # Create evaluation prompt
        system_prompt = """
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. Evaluate the answer strictly with ONLY the provided documents as ground truth.

Return STRICT minified JSON (no markdown, no prose before/after) with this exact schema:
{
    "Hallucination": {"score": <0-10 number>, "explanation": "..."},
    "Truthfulness": {"score": <0-10 number>, "explanation": "..."},
    "Accuracy": {"score": <0-10 number>, "explanation": "..."},
    "Relevancy": {"score": <0-10 number>, "explanation": "..."},
    "OverallComment": "... concise overall assessment ..."
}

Scoring guidance (0–10 integers or decimals allowed):
- Hallucination: 10 = entirely grounded; 0 = mostly fabricated.
- Truthfulness: 10 = factually correct; 0 = major factual errors.
- Accuracy: 10 = faithfully represents documents; 0 = misrepresents.
- Relevancy: 10 = directly answers query; 0 = off-topic.
Keep explanations short (<=30 words each). Do not add fields. Do not wrap in ```.
"""
    
    user_prompt = f"""
User Query: {query}

Retrieved Documents:
{context}

Generated Answer:
{answer}

Return ONLY the JSON object now."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        evaluation_text = response['choices'][0]['message']['content'].strip()

        # Attempt to isolate JSON (in case model adds stray text)
        json_candidate = evaluation_text
        if '```' in evaluation_text:
            # Strip code fences if present
            json_candidate = re.sub(r"```(json)?", "", evaluation_text).strip()

        # Try JSON parsing first
        parsed = None
        try:
            # Find first '{' ... last '}' span to be safe
            start = json_candidate.find('{')
            end = json_candidate.rfind('}') + 1
            if start != -1 and end != -1:
                parsed = json.loads(json_candidate[start:end])
        except Exception as je:  # noqa
            logger.warning(f"Evaluation JSON parse failed: {je}; will fallback to regex extraction")

        def _safe_score(obj, key) -> Optional[float]:
            try:
                if obj is None:
                    return None
                val = obj.get(key)
                if isinstance(val, dict):
                    val = val.get('score')
                if val is None:
                    return None
                f = float(val)
                if f < 0: f = 0.0
                if f > 10: f = 10.0
                return f
            except Exception:
                return None

        hallucination_score = _safe_score(parsed, 'Hallucination')
        truthfulness_score = _safe_score(parsed, 'Truthfulness')
        accuracy_score = _safe_score(parsed, 'Accuracy')
        relevancy_score = _safe_score(parsed, 'Relevancy')

        # Fallback to regex extraction if any missing
        if None in [hallucination_score, truthfulness_score, accuracy_score, relevancy_score]:
            logger.info("Falling back to regex score extraction for evaluation response")
            hallucination_score = hallucination_score or extract_score(evaluation_text, "Hallucination")
            truthfulness_score = truthfulness_score or extract_score(evaluation_text, "Truthfulness")
            accuracy_score = accuracy_score or extract_score(evaluation_text, "Accuracy")
            relevancy_score = relevancy_score or extract_score(evaluation_text, "Relevancy")

        # Final guard: if still None, assign NaN-like neutral mid value 5.0 and log
        def _finalize(v, name):
            if v is None:
                logger.warning(f"Evaluation missing {name} score; defaulting to 5.0")
                return 5.0
            return v

        hallucination_score = _finalize(hallucination_score, 'Hallucination')
        truthfulness_score = _finalize(truthfulness_score, 'Truthfulness')
        accuracy_score = _finalize(accuracy_score, 'Accuracy')
        relevancy_score = _finalize(relevancy_score, 'Relevancy')

        return EvaluationMetrics(
            hallucination_score=float(hallucination_score),
            truthfulness_score=float(truthfulness_score),
            accuracy_score=float(accuracy_score),
            relevancy_score=float(relevancy_score),
            explanation=evaluation_text
        )
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        # Return default metrics in case of error
        return EvaluationMetrics(
            hallucination_score=5.0,
            truthfulness_score=5.0,
            accuracy_score=5.0,
            relevancy_score=5.0,
            explanation=f"Error evaluating response: {str(e)}"
        )

def extract_score(text: str, dimension: str, default: float = 5.0) -> float:
    """Regex-based extraction of a dimension score (0-10) from free-form evaluation text."""
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

#------------------------------------------------------------------------
# API Endpoints
#------------------------------------------------------------------------

def _background_ingest(df: pd.DataFrame):
    ingestion_state.update({
        "running": True,
        "error": None,
        "inserted": 0,
        "total": len(df),
        "started_at": datetime.utcnow().isoformat(),
        "ended_at": None
    })
    try:
        inserted = ingest_data_to_milvus(df)
        ingestion_state["inserted"] = inserted
    except Exception as e:  # noqa
        ingestion_state["error"] = str(e)
        logger.error(f"Background ingestion failed: {e}")
    finally:
        ingestion_state["running"] = False
        ingestion_state["ended_at"] = datetime.utcnow().isoformat()

@app.post("/ingest", response_model=Dict[str, Any])
async def ingest_data(background_tasks: BackgroundTasks):
    """Start ingestion (non-blocking). If large table, run in background so /status isn't blocked."""
    if ingestion_state.get("running"):
        return {"status": "already_running", "inserted": ingestion_state.get("inserted"), "total": ingestion_state.get("total")}
    try:
        # Fetch ONLY real data from BigQuery; no synthetic fallback to avoid polluting the index
        bq_df = fetch_data_from_bigquery()
        keep_cols = [c for c in bq_df.columns if c in ["title", "abstract"]]
        df = bq_df[keep_cols] if keep_cols else bq_df
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="BigQuery returned no rows; ingestion aborted (no fallback sample).")
        background_tasks.add_task(_background_ingest, df)
        return {"status": "started", "total": int(len(df))}
    except Exception as e:
        logger.error(f"Ingestion start error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

@app.get("/ingest/status", response_model=Dict[str, Any])
async def get_ingest_status():
    return ingestion_state

"""/summarize endpoint removed from schema; functionality merged into /query."""

@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Query for papers and generate answer"""
    try:
        # Detect summarize directive inside query
        normalized_q = request.query.strip()
        lower_q = normalized_q.lower()
        if lower_q.startswith("summarize"):
            logger.info(f"Summarize path triggered for query='{normalized_q[:120]}'")
            summary_text, context = handle_summarize_query(normalized_q, top_k=request.top_k)
            if not summary_text:
                summary_text = "No summary generated (empty result)."
            # Evaluate the summary as the 'answer'
            evaluation = evaluate_response(request.query, summary_text, context)
            result_papers: List[ResearchPaperResult] = []
            for paper in context:
                result_papers.append(ResearchPaperResult(
                    id=paper.get("id",""),
                    title=paper.get("title",""),
                    authors=paper.get("authors",""),
                    abstract=paper.get("abstract",""),
                    url=paper.get("url",""),
                    score=paper.get("score",0.0)
                ))
            ranking_entries = [RankingEntry(rank=i+1, id=p.get("id",""), title=p.get("title",""), score=p.get("score",0.0)) for i, p in enumerate(context)]
            # Provide concise answer separate from full summary (first 2 sentences)
            concise_answer = '\n'.join([ln for ln in summary_text.split('\n') if ln.strip()][:2])[:600] if summary_text else ""
            return QueryResponse(
                query=request.query,
                answer=concise_answer,
                papers=result_papers,
                evaluation=evaluation,
                summary=summary_text,
                ranking=ranking_entries
            )
        # Step 1: Search for relevant papers
        papers = search_papers(request.query, top_k=30)
        if not papers:
            raise HTTPException(status_code=404, detail="No relevant papers found")
        
        # Step 2: Rerank results with Cohere
        reranked_papers = rerank_with_cohere(request.query, papers, top_k=request.top_k)
        
        # Step 3: Generate answer
        answer = generate_answer(request.query, reranked_papers)
        
        # Step 4: Evaluate response
        evaluation = evaluate_response(request.query, answer, reranked_papers)
        
        # Step 5: Get metadata for top papers if requested
        result_papers = []
        for paper in reranked_papers:
            paper_result = ResearchPaperResult(
                id=paper.get("id", ""),
                title=paper.get("title", ""),
                authors=paper.get("authors", ""),
                abstract=paper.get("abstract", ""),
                url=paper.get("url", ""),
                score=paper.get("score", 0.0)
            )
            
            # Get metadata if requested
            if request.get_metadata and paper.get("url"):
                paper_result.metadata = get_url_metadata(paper.get("url"))
            
            result_papers.append(paper_result)
        
        # Return response
        return QueryResponse(
            query=request.query,
            answer=answer,
            papers=result_papers,
            evaluation=evaluation
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        # Check Milvus connection
        milvus_connected = False
        try:
            connect_to_milvus()
            milvus_connected = True
        except:
            pass
        
        # Check collection exists and count
        collection_exists = False
        record_count = 0
        try:
            if utility.has_collection(COLLECTION_NAME):
                collection_exists = True
                collection = Collection(COLLECTION_NAME)
                record_count = collection.num_entities
        except:
            pass
        
        return {
            "status": "ok",
            "milvus_connected": milvus_connected,
            "collection_exists": collection_exists,
            "record_count": record_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

#------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------

def summarize_paper_prompt():
    """
    Returns a prompt template for paper summarization
    """
    return """
    Please provide a comprehensive summary of the following research paper:
    
    Title: {title}
    Authors: {authors}
    
    Focus on:
    1. The main problem or research question addressed
    2. The key methodologies used
    3. The most significant results and findings
    4. The important implications or applications
    5. Any limitations mentioned
    
    Make the summary detailed enough to understand the core contributions of the paper,
    but concise enough to be readable in a few minutes. Use clear, straightforward language
    and maintain scientific accuracy.
    """

if __name__ == "__main__":
    import uvicorn
    # Run with hot reload during development
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
