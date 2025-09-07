"""Pydantic data models."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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
    summary: Optional[str] = None
    ranking: Optional[List[RankingEntry]] = None

class SummarizeRequest(BaseModel):
    title: str = Field(..., description="Exact or partial paper title to summarize")
    top_k_context: int = Field(3, description="How many nearest papers to pull for context")
