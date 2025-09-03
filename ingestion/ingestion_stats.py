import requests
import feedparser
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "danhdanhtuan0308@gmail.com"

# Statistics categories (stat.*)
STAT_CATEGORIES = [
    "stat.AP",  # Applications
    "stat.CO",  # Computation
    "stat.ME",  # Methodology
    "stat.ML",  # Machine Learning
    "stat.OT",  # Other
    "stat.TH",  # Theory
]

def month_windows(start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    """Generate inclusive monthly windows [start_of_month, end_of_month (minute precision)]."""
    windows = []
    cur = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end_date:
        nxt = (cur.replace(year=cur.year + 1, month=1) if cur.month == 12
               else cur.replace(month=cur.month + 1))
        wnd_end = min(end_date, nxt - timedelta(minutes=1))  # inclusive to the minute
        windows.append((cur, wnd_end))
        cur = nxt
    logging.info("Generated %d monthly windows from %s to %s.", len(windows), start_date, end_date)
    return windows

def fmt_arxiv_dt(dt: datetime) -> str:
    """arXiv expects YYYYMMDDHHMM (UTC, minute precision)."""
    logging.info("Formatting datetime %s to arXiv format.", dt)
    return dt.strftime("%Y%m%d%H%M")

def build_stat_query(categories: Optional[List[str]] = None) -> str:
    cats = categories or STAT_CATEGORIES
    ors = " OR ".join(f"cat:{c}" for c in cats)
    return f"({ors})"

def atom_date_ymd(entry, attr: str) -> str:
    """
    Return YYYY-MM-DD from arXiv Atom date fields.
    Prefers feedparser's *_parsed (time.struct_time), else trims ISO string.
    """
    parsed = getattr(entry, f"{attr}_parsed", None)
    if parsed:
        return time.strftime("%Y-%m-%d", parsed)
    raw = getattr(entry, attr, "") or ""
    return raw.split("T", 1)[0][:10] if len(raw) >= 10 else ""

def fetch_window(
    q_base: str,
    wstart: datetime,
    wend: datetime,
    page_size: int = 200,
    delay: float = 3.0,
    sort_by: str = "submittedDate",
    sort_order: str = "ascending",
    max_empty_retries: int = 2,
    max_http_retries: int = 3,
) -> List[dict]:
    """Fetch one [wstart, wend] window with pagination and light retries."""
    date_part = f"submittedDate:[{fmt_arxiv_dt(wstart)} TO {fmt_arxiv_dt(wend)}]"
    q = f"{q_base} AND {date_part}"

    headers = {"User-Agent": USER_AGENT}
    results: List[dict] = []
    start_index = 0
    empty_retries = 0

    while True:
        params = {
            "search_query": q,
            "start": start_index,
            "max_results": page_size,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        http_retries = 0
        while True:
            try:
                r = requests.get(ARXIV_API, params=params, headers=headers, timeout=60)
                if r.status_code == 200:
                    break
                http_retries += 1
                if http_retries > max_http_retries:
                    logging.info(f"[{wstart:%Y-%m}] HTTP {r.status_code} at start={start_index}; giving up this window page.")
                    return results
                backoff = 2 ** http_retries
                logging.info(f"[{wstart:%Y-%m}] HTTP {r.status_code} at start={start_index}; retry {http_retries} in {backoff}s.")
                time.sleep(backoff)
            except requests.RequestException as e:
                http_retries += 1
                if http_retries > max_http_retries:
                    logging.info(f"[{wstart:%Y-%m}] Request error {e!r} at start={start_index}; giving up this window page.")
                    return results
                backoff = 2 ** http_retries
                logging.info(f"[{wstart:%Y-%m}] Request error {e!r}; retry {http_retries} in {backoff}s.")
                time.sleep(backoff)

        feed = feedparser.parse(r.text)
        entries = getattr(feed, "entries", []) or []
        if not entries:
            if empty_retries < max_empty_retries:
                empty_retries += 1
                logging.info(f"[{wstart:%Y-%m}] Empty page at start={start_index}; retry {empty_retries} after short backoff.")
                time.sleep(2 + empty_retries)
                continue
            logging.info(f"[{wstart:%Y-%m}] No more entries at start={start_index}.")
            break

        empty_retries = 0  # reset after success

        for entry in entries:
            authors = ", ".join(getattr(a, "name", "").strip() for a in getattr(entry, "authors", []))
            tags = getattr(entry, "tags", []) or []
            categories = ", ".join(getattr(t, "term", "").strip() for t in tags) if tags else "N/A"

            pdf_url = None
            for link in getattr(entry, "links", []):
                typ = getattr(link, "type", "")
                href = getattr(link, "href", None)
                if typ == "application/pdf" and href:
                    pdf_url = href
                    break
            if not pdf_url:
                entry_id = getattr(entry, "id", "") or getattr(entry, "link", "")
                if entry_id:
                    pdf_url = entry_id.replace("/abs/", "/pdf/") + ("" if entry_id.endswith(".pdf") else ".pdf")

            paper = {
                "title": getattr(entry, "title", "").strip().replace("\n", " "),
                "authors": authors,
                "summary": getattr(entry, "summary", "").strip().replace("\n", " "),
                "published": atom_date_ymd(entry, "published"),
                "updated": atom_date_ymd(entry, "updated"),
                "link": getattr(entry, "link", ""),
                "pdf_url": pdf_url,
                "categories": categories,
            }
            results.append(paper)

        logging.info(f"[{wstart:%Y-%m}] got {len(entries)} (start={start_index})")
        start_index += page_size
        if delay:
            time.sleep(delay)

    logging.info("Result: %d entries fetched for window %s to %s.", len(results), wstart, wend)
    return results

def fetch_statistics(
    to_date: Optional[datetime] = None,
    page_size: int = 200,
    delay: float = 3.0,
    out_parquet: Optional[str] = None,
    return_dataframe: bool = True,
):
    """
    Harvest arXiv Statistics (stat.*) papers from 2023-01-01 up to 'to_date' (default: 2025-03-01).
    Saves to Parquet (GCS) and returns a pandas DataFrame.
    """
    start = datetime(2024, 7, 1, 0, 0)
    end_dt = datetime(2025, 3, 1, 0, 0) if to_date is None else to_date
    end = end_dt.replace(second=0, microsecond=0).replace(tzinfo=None)

    q_base = build_stat_query()
    all_rows: List[dict] = []

    windows = month_windows(start, end)
    logging.info(f"Harvesting {len(windows)} monthly windows from {start} to {end} (stat.* categories).")

    for wstart, wend in windows:
        rows = fetch_window(
            q_base, wstart, wend,
            page_size=page_size,
            delay=delay,
            sort_by="submittedDate",
            sort_order="ascending",
        )
        logging.info(f"[{wstart:%Y-%m}] total accumulated this month: {len(rows)}")
        all_rows.extend(rows)

    # DataFrame
    cols = ["title","authors","summary","published","updated","link","pdf_url","categories"]
    df = pd.DataFrame(all_rows, columns=cols)

    # Parquet: write to your GCS bucket; also set a local filename for logs
    if out_parquet is None:
        out_parquet = f"arxiv_statistics_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    gcs_uri = f"gs://research-paper857/research/statistics_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

    try:
        # Requires pyarrow or fastparquet and gcsfs installed; uses ADC via storage_options
        df.to_parquet(
            gcs_uri,
            index=False,
            storage_options={"token": "google_default"}
        )
    except ImportError as e:
        raise ImportError(
            "Saving to Parquet requires 'pyarrow' or 'fastparquet' (and 'gcsfs' for GCS). "
            "Install them, e.g.: pip install pyarrow gcsfs"
        ) from e

    logging.info("Done. Saved %d rows to %s.", len(df), out_parquet)
    print(f"Done. Saved {len(df)} rows to {out_parquet}.")

    return df if return_dataframe else None

if __name__ == "__main__":
    fetch_statistics(page_size=200, delay=3.0)
