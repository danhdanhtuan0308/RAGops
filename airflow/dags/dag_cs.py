from datetime import datetime, timedelta
import logging
import sys
import os
import csv
from pathlib import Path

# Add ingestion module to Python path
ingestion_path = Path("/opt/airflow/ingestion")
if ingestion_path.exists():
    p = str(ingestion_path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

# Airflow imports
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def _iso(s: str) -> datetime:
    """Parse ISO date or datetime (naive)."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None)

@dag(
    dag_id="cs_ingestion_monthly",
    description="Fetch arXiv CS papers monthly and store in GCS",
    schedule="0 0 1 * *",
    start_date=datetime(2025, 3, 1),
    catchup=False,
    default_args=default_args,
    tags=["arxiv", "cs", "ingestion"],
)
def cs_ingestion_monthly():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="fetch_for_interval")
    def run_ingestion_for_window():
        logging.info("Starting ingestion task")
        
        try:
            ctx = get_current_context()
            dr = ctx.get("dag_run")
            conf = (dr.conf or {}) if dr else {}

            # Handle window configuration
            if "start" in conf and "end" in conf:
                wstart = _iso(conf["start"])
                wend_exclusive = _iso(conf["end"])
            else:
                ds = ctx.get("data_interval_start")
                de = ctx.get("data_interval_end")
                if ds and de:
                    wstart = ds.replace(tzinfo=None)
                    wend_exclusive = de.replace(tzinfo=None)
                else:
                    logical_date = ctx.get("logical_date", datetime.now())
                    wstart = logical_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    if wstart.month == 12:
                        wend_exclusive = wstart.replace(year=wstart.year + 1, month=1)
                    else:
                        wend_exclusive = wstart.replace(month=wstart.month + 1)

            logging.info(f"Processing window: {wstart} -> {wend_exclusive}")

            # Robust import for ingestion functions
            try:
                from ingestion.ingestion_cs import build_cs_query, fetch_window
                logging.info("Imported ingestion.ingestion_cs functions")
            except Exception as e1:
                logging.warning(f"Package function import failed ({e1}); trying package module")
                try:
                    from ingestion import ingestion_cs as _mod
                    build_cs_query = getattr(_mod, "build_cs_query")
                    fetch_window = getattr(_mod, "fetch_window")
                    logging.info("Imported ingestion package module")
                except Exception as e2:
                    logging.warning(f"Package module import failed ({e2}); trying top-level module")
                    import importlib
                    _mod2 = importlib.import_module("ingestion_cs")
                    build_cs_query = getattr(_mod2, "build_cs_query")
                    fetch_window = getattr(_mod2, "fetch_window")
                    logging.info("Imported top-level ingestion_cs module")

            # Fetch using ingestion primitives
            q_base = build_cs_query()
            wend_inclusive = (wend_exclusive - timedelta(minutes=1))
            rows = fetch_window(
                q_base,
                wstart,
                wend_inclusive,
                page_size=200,
                delay=3.0,
                sort_by="submittedDate",
                sort_order="ascending",
            )

            # Convert to DataFrame and write to both local CSV and GCS Parquet
            try:
                import pandas as pd
            except Exception as e:
                logging.error(f"pandas not available: {e}")
                raise

            cols = [
                "title",
                "authors",
                "summary",
                "published",
                "updated",
                "link",
                "pdf_url",
                "categories",
            ]
            df = pd.DataFrame(rows, columns=cols)

            # Local CSV for debugging/visibility
            out_dir = "/opt/airflow/logs/data"
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(
                out_dir,
                f"arxiv_cs_{wstart.strftime('%Y%m%d')}_{wend_exclusive.strftime('%Y%m%d')}.csv",
            )
            df.to_csv(out_csv, index=False)
            logging.info(f"Saved CSV with {len(df)} rows to {out_csv}")

            # GCS Parquet using ADC (google_default)
            gcs_prefix = "gs://research-paper857/research"

            dated_uri  = f"{gcs_prefix}/cs_{wstart.strftime('%Y%m%d')}_{wend_exclusive.strftime('%Y%m%d')}.parquet"
            stable_uri = f"{gcs_prefix}/cs_latest.parquet"  # <- constant name for BigQuery

            try:
                # Write historical, date-stamped object
                df.to_parquet(
                    dated_uri,
                    index=False,
                    storage_options={"token": "google_default"},
                )
                logging.info(f"Saved Parquet to GCS (dated): {dated_uri}")

                # Overwrite the stable object every run
                df.to_parquet(
                    stable_uri,
                    index=False,
                    storage_options={"token": "google_default"},
                )
                logging.info(f"Saved Parquet to GCS (stable): {stable_uri}")
            except Exception as e:
                logging.error(f"Failed to write Parquet to GCS: {e}")
                raise

            return {"rows": len(rows), "csv": out_csv, "gcs_dated": dated_uri, "gcs_stable": stable_uri}


        except Exception as e:
            logging.error(f"Task failed with error: {e}")
            raise

    # Define task dependencies
    start >> run_ingestion_for_window() >> end

# Create DAG instance
dag = cs_ingestion_monthly()
