"""
data/bootstrap_fetcher.py — One-time historical OHLCV bootstrap fetch.

Downloads 5-year daily price history for all S&P 500 + S&P 400 constituents
(~900 tickers) via yfinance and saves each ticker as a parquet file. Optionally
uploads to S3 for use by training jobs running on SageMaker or EC2.

Usage:
    python data/bootstrap_fetcher.py [--output-dir data/cache] [--upload] [--tickers AAPL MSFT ...]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Download constants
_BATCH_SIZE = 50          # tickers per yfinance batch call
_DOWNLOAD_WORKERS = 4     # parallel batch workers
_PRICE_PERIOD = "5y"      # 5-year history for training
_RETRY_DELAY = 2.0        # seconds between retries


# ── Ticker universe ───────────────────────────────────────────────────────────

def fetch_sp500_sp400_tickers() -> list[str]:
    """
    Fetch S&P 500 and S&P 400 constituent tickers from Wikipedia.
    Falls back to a local cache CSV if the network request fails.
    Returns a deduplicated list of ticker symbols.
    """
    import requests

    cache_path = Path(__file__).parent.parent / "data" / "constituents_cache.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    urls = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    }
    headers = {"User-Agent": "alpha-engine-predictor/1.0 (bootstrap-fetcher)"}

    tickers: list[str] = []

    try:
        for index_name, url in urls.items():
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]

            # Flatten multi-level columns if Wikipedia returns them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

            col = next(
                (c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()),
                df.columns[0],
            )
            batch = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)  # BRK.B → BRK-B for yfinance
                .tolist()
            )
            batch = [t for t in batch if t and t != "nan" and len(t) <= 6]
            tickers.extend(batch)
            log.info("Fetched %d tickers from %s", len(batch), index_name)

        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order
        pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
        log.info("Total universe: %d tickers (saved to %s)", len(tickers), cache_path)
        return tickers

    except Exception as exc:
        log.warning("Wikipedia fetch failed (%s); trying local cache...", exc)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)["ticker"].tolist()
            log.info("Loaded %d tickers from cache at %s", len(cached), cache_path)
            return cached
        log.error("No cache found — cannot build universe. Run with network access first.")
        return []


# ── Batch download ────────────────────────────────────────────────────────────

def _download_batch(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for one batch of tickers. Returns a dict of ticker → DataFrame.
    Empty DataFrames are returned for tickers that failed or had no data.
    """
    if not tickers:
        return {}

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Close"])
        result[ticker] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].copy()
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["Close"])
                if not df.empty and len(df) >= 30:
                    result[ticker] = df
                else:
                    result[ticker] = pd.DataFrame()
            except (KeyError, AttributeError):
                result[ticker] = pd.DataFrame()

    return result


# ── Save / upload ─────────────────────────────────────────────────────────────

def _save_parquet(ticker: str, df: pd.DataFrame, output_dir: Path) -> Path:
    """Save a ticker's DataFrame to a parquet file. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}.parquet"
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def _upload_to_s3(ticker: str, local_path: Path, s3_bucket: str, s3_key_template: str) -> None:
    """Upload a parquet file to S3. Logs but does not raise on failure."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = s3_key_template.format(ticker=ticker)
        s3.upload_file(str(local_path), s3_bucket, key)
        log.debug("Uploaded s3://%s/%s", s3_bucket, key)
    except Exception as exc:
        log.warning("S3 upload failed for %s: %s", ticker, exc)


# ── Main orchestration ────────────────────────────────────────────────────────

def run_bootstrap(
    tickers: list[str],
    output_dir: Path,
    upload: bool = False,
    s3_bucket: str = "alpha-engine-research",
    s3_key_template: str = "predictor/price_cache/{ticker}.parquet",
    period: str = _PRICE_PERIOD,
) -> None:
    """
    Download and save OHLCV history for all tickers.

    Parameters
    ----------
    tickers:         List of ticker symbols to download.
    output_dir:      Local directory to save parquet files.
    upload:          If True, upload each file to S3 after saving locally.
    s3_bucket:       S3 bucket name (only used if upload=True).
    s3_key_template: S3 key template with {ticker} placeholder.
    period:          yfinance period string (e.g. "5y", "3y").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(tickers)
    completed = 0
    failed: list[str] = []

    batches = [tickers[i : i + _BATCH_SIZE] for i in range(0, total, _BATCH_SIZE)]
    log.info(
        "Starting bootstrap: %d tickers / %d batches (batch_size=%d)",
        total,
        len(batches),
        _BATCH_SIZE,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        future_to_batch = {
            executor.submit(_download_batch, batch, period): batch for batch in batches
        }
        failed_batches: list[list[str]] = []

        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
            except Exception as exc:
                log.warning("Batch failed (%s), will retry: %s", batch[:3], exc)
                failed_batches.append(batch)
                continue

            for ticker, df in batch_results.items():
                if df.empty:
                    log.warning("No data for %s — skipping", ticker)
                    failed.append(ticker)
                else:
                    local_path = _save_parquet(ticker, df, output_dir)
                    if upload:
                        _upload_to_s3(ticker, local_path, s3_bucket, s3_key_template)
                    completed += 1

            pct = 100 * completed / total
            log.info("Progress: %d / %d (%.1f%%)  failed so far: %d", completed, total, pct, len(failed))

    # ── Retry failed batches sequentially ────────────────────────────────────
    if failed_batches:
        log.info("Retrying %d failed batches sequentially...", len(failed_batches))
        for batch in failed_batches:
            time.sleep(_RETRY_DELAY)
            try:
                batch_results = _download_batch(batch, period)
                for ticker, df in batch_results.items():
                    if df.empty:
                        failed.append(ticker)
                    else:
                        local_path = _save_parquet(ticker, df, output_dir)
                        if upload:
                            _upload_to_s3(ticker, local_path, s3_bucket, s3_key_template)
                        completed += 1
            except Exception as exc:
                log.error("Retry batch failed: %s — adding to failures: %s", exc, batch[:3])
                failed.extend(batch)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("Bootstrap complete: %d / %d succeeded, %d failed", completed, total, len(failed))
    if failed:
        failed_path = output_dir / "failed_tickers.txt"
        failed_path.write_text("\n".join(failed))
        log.warning("Failed tickers written to %s", failed_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap 5-year OHLCV history for S&P 500 + S&P 400 (~900 tickers)."
    )
    parser.add_argument(
        "--output-dir",
        default="data/cache",
        help="Local directory to save parquet files (default: data/cache)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload each parquet file to S3 after saving locally",
    )
    parser.add_argument(
        "--s3-bucket",
        default="alpha-engine-research",
        help="S3 bucket name (only used with --upload)",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance period string, e.g. 5y, 3y, 1y (default: 5y)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override ticker list (space-separated). Fetches full S&P 500+400 if omitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N tickers (useful for testing)",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
        log.info("Using provided ticker list: %d tickers", len(tickers))
    else:
        tickers = fetch_sp500_sp400_tickers()
        if not tickers:
            log.error("No tickers found — aborting.")
            sys.exit(1)

    if args.limit:
        tickers = tickers[: args.limit]
        log.info("Limiting to %d tickers", len(tickers))

    run_bootstrap(
        tickers=tickers,
        output_dir=Path(args.output_dir),
        upload=args.upload,
        s3_bucket=args.s3_bucket,
        period=args.period,
    )


if __name__ == "__main__":
    main()
