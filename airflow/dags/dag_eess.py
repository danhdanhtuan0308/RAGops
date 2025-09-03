# dag_eess.py
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path
# Airflow imports
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.empty import EmptyOperator


#GCS config (edit bucket/prefix here)
GCS_BUCKET = "research-paper857"
GCS_PREFIX = "research"  # folder inside the bucket
# -----------------------------------------------

# Add ingestion module to Python path
ingestion_path = Path("/opt/airflow/ingestion")
if ingestion_path.exists():
    p = str(ingestion_path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def _iso(s: str) -> datetime:
    """Parse ISO date/datetime to naive datetime."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None)

@dag(
    dag_id="eess_ingestion_monthly",
    description="Fetch arXiv EESS (eess.*) papers monthly and store in GCS",
    schedule="0 0 1 * *",                # monthly at 00:00 on the 1st
    start_date=datetime(2025, 3, 1),     # bump to 2025-10-01 when ready
    catchup=False,
    default_args=default_args,
    tags=["arxiv", "eess", "ingestion"],
)
def eess_ingestion_monthly():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="fetch_for_interval")
    def run_ingestion_for_window():
        logging.info("Starting EESS ingestion task")
        try:
            ctx = get_current_context()
            dr = ctx.get("dag_run")
            conf = (dr.conf or {}) if dr else {}

            # Window resolution: conf → data_interval → logical_date fallback
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

            # Robust import from your ingestion_eess module (no codebase changes)
            try:
                from ingestion.ingestion_eess import build_eess_query, fetch_window
                logging.info("Imported ingestion.ingestion_eess functions")
            except Exception as e1:
                logging.warning(f"Package function import failed ({e1}); trying package module")
                try:
                    from ingestion import ingestion_eess as _mod
                    build_eess_query = getattr(_mod, "build_eess_query")
                    fetch_window = getattr(_mod, "fetch_window")
                    logging.info("Imported ingestion package module")
                except Exception as e2:
                    logging.warning(f"Package module import failed ({e2}); trying top-level module")
                    import importlib
                    _mod2 = importlib.import_module("ingestion_eess")
                    build_eess_query = getattr(_mod2, "build_eess_query")
                    fetch_window = getattr(_mod2, "fetch_window")
                    logging.info("Imported top-level ingestion_eess module")

            # Fetch using your ingestion primitives
            q_base = build_eess_query()
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

            # DataFrame
            try:
                import pandas as pd
            except Exception as e:
                logging.error(f"pandas not available: {e}")
                raise

            cols = ["title","authors","summary","published","updated","link","pdf_url","categories"]
            df = pd.DataFrame(rows, columns=cols)

            # Local CSV (debug/visibility)
            out_dir = "/opt/airflow/logs/data"
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(
                out_dir,
                f"arxiv_eess_{wstart:%Y%m%d}_{wend_exclusive:%Y%m%d}.csv",
            )
            df.to_csv(out_csv, index=False)
            logging.info(f"Saved CSV with {len(df)} rows to {out_csv}")

            # GCS Parquet: dated + stable (external table can point to the stable one)
            gcs_prefix = f"gs://{GCS_BUCKET}/{GCS_PREFIX}"
            dated_uri  = f"{gcs_prefix}/eess_{wstart:%Y%m%d}_{wend_exclusive:%Y%m%d}.parquet"
            stable_uri = f"{gcs_prefix}/eess_latest.parquet"

            try:
                # historical, date-stamped object
                df.to_parquet(dated_uri, index=False, storage_options={"token": "google_default"})
                logging.info(f"Saved Parquet to GCS (dated): {dated_uri}")

                # constant name for BigQuery
                df.to_parquet(stable_uri, index=False, storage_options={"token": "google_default"})
                logging.info(f"Saved Parquet to GCS (stable): {stable_uri}")
            except Exception as e:
                logging.error(f"Failed to write Parquet to GCS: {e}")
                raise

            return {
                "rows": len(rows),
                "csv": out_csv,
                "gcs_dated": dated_uri,
                "gcs_stable": stable_uri,
            }

        except Exception as e:
            logging.error(f"Task failed with error: {e}")
            raise

    start >> run_ingestion_for_window() >> end

dag = eess_ingestion_monthly()
