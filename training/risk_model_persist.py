"""training/risk_model_persist.py — C.2b production-persistence layer.

ROADMAP C.2b — builds the Barra-style structural factor risk model
(F = factor-return covariance, D = idiosyncratic variance) from the
universe library's price + factor-loading columns, and persists the
two parquet artifacts weekly to
``s3://{bucket}/risk_model/{date}/{F,D,metadata}.parquet`` from the
``PredictorTraining`` Saturday SF stage.

The pure-math primitives live in ``risk_model/__init__.py`` (C.2a,
PR #200). This module is the wiring layer — local-parquet read,
DataFrame pivoting, S3 write.

**Hard sequencing constraint (per C.3):** Σ = B · F · Bᵀ + D is NOT
wired into ``executor.portfolio_optimizer.solve_target_weights`` until
B.5 (α̂-uncertainty cutover gate) passes. Two simultaneous Σ-substrate
changes make backtester regressions untraceable. C.2b ships the
weekly persistence so by the time C.3 reads ``risk_model/{date}/``,
there are ≥4 weeks of F + D accumulated.

**Graceful degradation.** The 8 factor-loading ``*_zscore`` columns
shipped 2026-05-26 evening in alpha-engine-data #324 — older universe-
library rows don't carry them. The builder is best-effort: if zero
loading columns are present in the local cache, it returns
``status=skipped`` with a clear reason rather than producing an
all-NaN F. The Saturday SF will start emitting real F + D once a
weeks' worth of post-#324 universe data exists.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import build_factor_risk_model

log = logging.getLogger(__name__)

# Pinned factor-loading column registry. Mirrors the
# alpha-engine-data ``features/registry.py`` ``factor_loading``
# entries (PR #324). Order is the canonical Barra-style factor order:
# momentum → return → beta → idio_vol → realized_vol → dist_high → value → quality.
FACTOR_LOADING_COLUMNS: tuple[str, ...] = (
    "momentum_20d_zscore",
    "return_60d_zscore",
    "beta_60d_zscore",
    "idio_vol_60d_zscore",
    "realized_vol_63d_zscore",
    "dist_from_52w_high_zscore",
    "pe_ratio_zscore",
    "roe_zscore",
)

# Minimum data thresholds before we attempt the F+D build. If fewer
# tickers carry loading columns, the cross-sectional regression at
# each date will be rank-deficient (K=8 + intercept needs >>9 obs to
# be meaningful). 30 is the same floor risk_model.estimate_factor_
# covariance defaults to for cov_obs.
_MIN_TICKERS_WITH_LOADINGS = 30
_MIN_DATES_WITH_RETURNS = 60


def _load_universe_panels(
    data_dir: str | os.PathLike,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    """Read every ticker parquet under ``data_dir`` and pivot into:

      * returns_panel: (T × N) DataFrame of close-to-close log returns,
        DatetimeIndex × ticker columns. NaNs preserved (tickers with
        gaps for a date don't get filled).
      * loadings_by_date: ``{date: (N × K) DataFrame}`` — for each
        date, the per-ticker factor-loading vector (one row per
        ticker that has all ``FACTOR_LOADING_COLUMNS`` non-NaN on
        that date).

    Skips macro files (no ``Close`` column expected to behave like a
    ticker — macros are not part of the universe).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"data_dir not found: {data_dir}")

    returns_by_ticker: dict[str, pd.Series] = {}
    loadings_rows: list[tuple[pd.Timestamp, str, np.ndarray]] = []

    parquets = sorted(data_dir.glob("*.parquet"))
    log.info(
        "risk_model_persist: scanning %d parquet files in %s",
        len(parquets), data_dir,
    )

    for path in parquets:
        ticker = path.stem
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.debug("Skip %s: read failed (%s)", ticker, exc)
            continue
        if "Close" not in df.columns or df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                log.debug("Skip %s: index not DatetimeIndex", ticker)
                continue

        # Close → log returns. NaN on first row by construction.
        close = df["Close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        returns_by_ticker[ticker] = log_ret

        # Extract factor-loading columns. If any of the 8 are missing
        # in this ticker's parquet, skip the ticker entirely (rank-
        # deficient row at every date).
        available = [c for c in FACTOR_LOADING_COLUMNS if c in df.columns]
        if len(available) != len(FACTOR_LOADING_COLUMNS):
            continue
        loadings = df[list(FACTOR_LOADING_COLUMNS)].astype(float)
        for date, row in loadings.iterrows():
            if row.isna().any():
                continue
            loadings_rows.append((date, ticker, row.values))

    returns_panel = pd.DataFrame(returns_by_ticker).sort_index()

    # Pivot loadings_rows into a per-date dict of N×K DataFrames.
    loadings_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    by_date_buckets: dict[pd.Timestamp, list[tuple[str, np.ndarray]]] = {}
    for date, ticker, row in loadings_rows:
        by_date_buckets.setdefault(date, []).append((ticker, row))
    for date, items in by_date_buckets.items():
        tickers = [t for t, _ in items]
        mat = np.vstack([r for _, r in items])
        loadings_by_date[date] = pd.DataFrame(
            mat,
            index=tickers,
            columns=list(FACTOR_LOADING_COLUMNS),
        )

    return returns_panel, loadings_by_date


def _write_parquet_to_s3(s3_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    """Serialize ``df`` to in-memory parquet and put it to S3 at
    ``s3://{bucket}/{key}``. Caller wraps in best-effort try/except."""
    import io

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )


def build_and_persist_risk_model(
    data_dir: str | os.PathLike,
    bucket: str,
    date_str: str,
    *,
    dry_run: bool = False,
    s3_client=None,
) -> dict:
    """Build F + D from the universe-library local cache and persist
    to S3.

    Returns a status dict shaped like the sweep stages in
    ``alpha-engine-backtester/backtest.py`` (status=ok|skipped, plus
    metadata). Non-fatal: callers wrap in try/except so a failure
    here doesn't abort the training pipeline.

    Args:
        data_dir: local directory of per-ticker parquets (the same
            ``tmp_cache`` ``train_handler.main`` populated from
            ``download_from_arctic``).
        bucket: S3 bucket for the persistence layer.
        date_str: Saturday SF run date in ``YYYY-MM-DD``. Used as
            the per-week S3 prefix.
        dry_run: when True, build the model but skip the S3 write.
        s3_client: optional boto3 client override (for tests).
    """
    t0 = time.time()
    log.info("risk_model_persist: starting (date=%s, bucket=%s)", date_str, bucket)

    returns_panel, loadings_by_date = _load_universe_panels(data_dir)

    # Sanity / coverage gates — return skipped on insufficient data
    # rather than producing an all-NaN F that pollutes the weekly
    # accumulation series.
    if returns_panel.empty or returns_panel.shape[0] < _MIN_DATES_WITH_RETURNS:
        reason = (
            f"returns_panel has only {returns_panel.shape[0]} dates "
            f"(< {_MIN_DATES_WITH_RETURNS} threshold) — universe cache "
            f"may be empty or freshly bootstrapped"
        )
        log.warning("risk_model_persist: skipped — %s", reason)
        return {
            "status": "skipped",
            "date": date_str,
            "reason": reason,
            "n_dates": int(returns_panel.shape[0]),
            "n_tickers": int(returns_panel.shape[1]),
            "n_dates_with_loadings": 0,
        }

    n_tickers_with_loadings = (
        max(df.shape[0] for df in loadings_by_date.values())
        if loadings_by_date else 0
    )
    if n_tickers_with_loadings < _MIN_TICKERS_WITH_LOADINGS:
        reason = (
            f"only {n_tickers_with_loadings} tickers carry all "
            f"{len(FACTOR_LOADING_COLUMNS)} factor-loading columns "
            f"(< {_MIN_TICKERS_WITH_LOADINGS} threshold) — likely "
            "pre-#324 universe cache; activates automatically once "
            "alpha-engine-data PR #324's loadings have accumulated"
        )
        log.warning("risk_model_persist: skipped — %s", reason)
        return {
            "status": "skipped",
            "date": date_str,
            "reason": reason,
            "n_dates": int(returns_panel.shape[0]),
            "n_tickers": int(returns_panel.shape[1]),
            "n_dates_with_loadings": len(loadings_by_date),
            "max_tickers_per_loading_date": n_tickers_with_loadings,
        }

    log.info(
        "risk_model_persist: building model — "
        "%d dates × %d tickers, %d loading-dates",
        returns_panel.shape[0], returns_panel.shape[1],
        len(loadings_by_date),
    )

    model = build_factor_risk_model(
        returns_panel=returns_panel,
        loadings_by_date=loadings_by_date,
    )

    if not dry_run:
        import boto3

        s3 = s3_client or boto3.client("s3")
        prefix = f"risk_model/{date_str}"
        try:
            _write_parquet_to_s3(s3, bucket, f"{prefix}/F.parquet", model["F"])
            _write_parquet_to_s3(
                s3, bucket, f"{prefix}/D.parquet",
                model["D"].to_frame(name="idiosyncratic_variance"),
            )
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}/metadata.json",
                Body=json.dumps(model["metadata"], indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            log.info("risk_model_persist: persisted s3://%s/%s/", bucket, prefix)
        except Exception as exc:
            log.warning(
                "risk_model_persist: S3 persist failed (non-fatal): %s", exc,
            )

    elapsed = time.time() - t0
    log.info(
        "risk_model_persist: ok — F shape=%s, D length=%d, %.1fs",
        tuple(model["F"].shape), len(model["D"]), elapsed,
    )
    return {
        "status": "ok",
        "date": date_str,
        "elapsed_seconds": round(elapsed, 1),
        "n_dates": int(returns_panel.shape[0]),
        "n_tickers": int(returns_panel.shape[1]),
        "n_dates_with_loadings": len(loadings_by_date),
        "F_shape": list(model["F"].shape),
        "D_length": int(len(model["D"])),
        "metadata": model["metadata"],
    }
