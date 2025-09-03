# dag_physics.py
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Add ingestion module to Python path (same pattern as CS)
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
    """Parse ISO date/datetime to naive datetime."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None)

@dag(
    dag_id="physics_ingestion_monthly",
    description="Fetch arXiv physics papers monthly and store in GCS",
    schedule="0 0 1 * *",                # run monthly at 00:00 on the 1st
    start_date=datetime(2025, 3, 1),     # change to 2025-10-01 when you’re ready
    catchup=False,
    default_args=default_args,
    tags=["arxiv", "physics", "ingestion"],
)
def physics_ingestion_monthly():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task(task_id="fetch_for_interval")
    def run_ingestion_for_window():
        logging.info("Starting physics ingestion task")
        try:
            ctx = get_current_context()
            dr = ctx.get("dag_run")
            conf = (dr.conf or {}) if dr else {}

            # Determine window: conf → data_interval → logical_date fallback
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

            # Robust imports from your ingestion_physic module
            try:
                from ingestion.ingestion_physic import build_physics_query, fetch_window
                logging.info("Imported ingestion.ingestion_physic functions")
            except Exception as e1:
                logging.warning(f"Package function import failed ({e1}); trying package module")
                try:
                    from ingestion import ingestion_physic as _mod
                    build_physics_query = getattr(_mod, "build_physics_query")
                    fetch_window = getattr(_mod, "fetch_window")
                    logging.info("Imported ingestion package module")
                except Exception as e2:
                    logging.warning(f"Package module import failed ({e2}); trying top-level module")
                    import importlib
                    _mod2 = importlib.import_module("ingestion_physic")
                    build_physics_query = getattr(_mod2, "build_physics_query")
                    fetch_window = getattr(_mod2, "fetch_window")
                    logging.info("Imported top-level ingestion_physic module")

            # Fetch using your physics ingestion primitives
            q_base = build_physics_query()
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

            # DataFrame + outputs
            try:
                import pandas as pd
            except Exception as e:
                logging.error(f"pandas not available: {e}")
                raise

            cols = [
                "title", "authors", "summary",
                "published", "updated", "link", "pdf_url", "categories",
            ]
            df = pd.DataFrame(rows, columns=cols)

            # Local CSV (for debug/visibility)
            out_dir = "/opt/airflow/logs/data"
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(
                out_dir,
                f"arxiv_physics_{wstart.strftime('%Y%m%d')}_{wend_exclusive.strftime('%Y%m%d')}.csv",
            )
            df.to_csv(out_csv, index=False)
            logging.info(f"Saved CSV with {len(df)} rows to {out_csv}")

            # GCS Parquet: write both a dated file AND a stable file for BigQuery
            gcs_prefix = "gs://research-paper857/research"
            dated_uri  = f"{gcs_prefix}/physics_{wstart.strftime('%Y%m%d')}_{wend_exclusive.strftime('%Y%m%d')}.parquet"
            stable_uri = f"{gcs_prefix}/physics_latest.parquet"

            try:
                # historical, date-stamped object
                df.to_parquet(
                    dated_uri,
                    index=False,
                    storage_options={"token": "google_default"},
                )
                logging.info(f"Saved Parquet to GCS (dated): {dated_uri}")

                # constant name for BigQuery external table
                df.to_parquet(
                    stable_uri,
                    index=False,
                    storage_options={"token": "google_default"},
                )
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

    # Wire up dependencies
    start >> run_ingestion_for_window() >> end

dag = physics_ingestion_monthly()
