"""Milvus and BigQuery related functions."""
from __future__ import annotations
import time
import hashlib
import os
import logging
from typing import List, Dict, Any
import pandas as pd
from google.cloud import bigquery
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

from .config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIMENSION,
    PROJECT_ID, DATASET_ID, TABLE_ID, MILVUS_URI, MILVUS_TOKEN, MILVUS_SECURE
)
from .config import EMBED_BATCH_SIZE  # re-export if needed
from .rag_embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Connection helpers

def connect_to_milvus():
    max_retries = 5
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            if MILVUS_URI:
                # Hosted Milvus (e.g., Zilliz) via URI + token
                connections.connect(
                    alias="default",
                    uri=MILVUS_URI,
                    token=MILVUS_TOKEN or None,
                    secure=MILVUS_SECURE,
                )
                logger.info(f"Connected to Milvus via URI {MILVUS_URI}")
            else:
                connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
                logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:  # noqa
            logger.warning(f"Attempt {attempt+1}/{max_retries} to connect to Milvus failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def create_milvus_collection():
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(fields=fields, description="Research papers collection (minimal schema)")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def get_bigquery_client():
    return bigquery.Client(project=PROJECT_ID)


def fetch_data_from_bigquery():
    client = get_bigquery_client()
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    job = client.query(query)
    df = job.result().to_dataframe()
    logger.info(f"Retrieved {len(df)} rows from BigQuery")
    return df


def _deterministic_id(title: str, abstract: str) -> str:
    h = hashlib.sha1((title + "||" + abstract).encode("utf-8")).hexdigest()[:32]
    return f"doc_{h}"


def ingest_data_to_milvus(df: pd.DataFrame, ingestion_state: Dict[str, Any]):
    connect_to_milvus()
    collection = create_milvus_collection()
    total_records = len(df)
    if total_records == 0:
        raise ValueError("No rows to ingest")
    if "title" not in df.columns:
        raise ValueError("DataFrame missing required column 'title'")
    if "abstract" not in df.columns:
        df["abstract"] = ""
    MILVUS_BATCH = int(os.getenv("MILVUS_BATCH", "256"))
    inserted = 0
    start_time = time.time()

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
        existing = fetch_existing(ids)
        if existing:
            mask = [i not in existing for i in ids]
            titles = [t for t, m in zip(titles, mask) if m]
            abstracts = [a for a, m in zip(abstracts, mask) if m]
            ids = [i for i, m in zip(ids, mask) if m]
            if not ids:
                continue
        embed_texts = [f"Title: {t}\nAbstract: {a}" for t, a in zip(titles, abstracts)]
        try:
            vectors = get_embeddings(embed_texts)
        except Exception:
            vectors = [[0.0] * DIMENSION for _ in embed_texts]
        good_rows = [(i, t, a, v) for i, t, a, v in zip(ids, titles, abstracts, vectors) if len(v) == DIMENSION]
        if not good_rows:
            continue
        ins_ids, ins_titles, ins_abstracts, ins_vectors = zip(*good_rows)
        data = [list(ins_ids), list(ins_titles), list(ins_abstracts), list(ins_vectors)]
        try:
            collection.insert(data)
            inserted += len(ins_ids)
        except Exception:
            continue
        if inserted % 1000 < len(ins_ids):
            collection.flush()
        if ingestion_state.get("running"):
            ingestion_state["inserted"] = inserted
    collection.flush()
    return inserted
