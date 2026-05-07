"""
training/meta_trainer.py — Meta-model training pipeline (v3.0).

Trains 4 specialized Layer 1 models + 1 ridge meta-model using walk-forward
validation with out-of-fold stacking to prevent leakage.

Layer 1 models:
  - Momentum Model (GBM, 6 price features)
  - Volatility Model (GBM, 6 vol features)
  - Regime Predictor (logistic regression, 6 macro features)
  - Research Calibrator v0 (lookup table, score → hit rate)

Layer 2:
  - Meta-Model (ridge regression on Layer 1 outputs + research context)
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


_MOMENTUM_PARAMS_S3_KEY = "config/predictor_momentum_params.json"

# Horizons for the post-training Spearman IC diagnostic. Training still
# uses cfg.FORWARD_DAYS (5d today) as the label; these are *sidecar*
# metrics that measure how well the trained model's rank-ordering
# predicts alpha at alternative horizons. Answers "is 5d the right
# horizon for this signal?" autonomously.
#
# 2026-04-15 extension: added 60d and 90d. First diagnostic run showed
# IC climbing monotonically 5d → 40d (0.0099 → 0.0300, 3.0× lift, no
# peak in range). Extending to 60d/90d to find where the curve stops
# rising — that peak is the target horizon for any parallel training
# stack we build next. If IC continues climbing past 90d we're looking
# at a multi-month momentum regime and should consider longer still.
_DIAGNOSTIC_HORIZONS = [5, 10, 15, 21, 40, 60, 90]


def _nonoverlapping_date_mask(dates: list, horizon_trading_days: int) -> list[bool]:
    """Return a boolean mask selecting one date per non-overlapping h-day window.

    Why this matters: the existing horizon IC diagnostic in this trainer
    samples (date, ticker) rows with a forward window that overlaps heavily
    across consecutive dates — at h=21d, today's training row and tomorrow's
    share 20 of 21 days of forward return. That autocorrelation inflates IC
    point estimates compared to honest non-overlapping samples and makes
    cross-horizon IC ratios (e.g. "21d IC = 2.2× 5d IC") harder to interpret
    cleanly. Per the 2026-05-07 predictor audit Track B, the right fix is
    to compute IC on both overlapping (current) and non-overlapping samples
    so the autocorrelation contribution is visible.

    Strategy: walk the sorted unique dates ascending; greedily keep a date
    if it's at least ``horizon_trading_days`` calendar days past the most
    recently kept date. Calendar-day spacing is a good proxy for trading-day
    spacing at these horizons (5/10/15/21/40/60/90) and avoids needing the
    NYSE calendar in this hot path. Returns a mask aligned to ``dates``.

    The greedy "keep-if-far-enough" pattern minimizes overlap pessimistically:
    dates dropped because their forward window overlapped with a kept date's
    are simply not used for IC computation at that horizon. The cost is
    sample-size; the benefit is honest IC on independent observations.
    """
    if not dates:
        return []
    # Convert to pandas Timestamps for comparable arithmetic without losing
    # precision. The training code stores dates as pd.Timestamp objects on
    # rows (per the row construction in Step 6); accept any sortable type.
    import pandas as pd
    dates_ts = [pd.Timestamp(d) for d in dates]
    # Sort positions by date; we must build the mask in original order.
    sort_order = sorted(range(len(dates_ts)), key=lambda i: dates_ts[i])
    mask = [False] * len(dates_ts)
    last_kept_ts: pd.Timestamp | None = None
    horizon_delta = pd.Timedelta(days=horizon_trading_days)
    for i in sort_order:
        ts = dates_ts[i]
        if last_kept_ts is None or (ts - last_kept_ts) >= horizon_delta:
            mask[i] = True
            last_kept_ts = ts
    return mask


def _bootstrap_ic_ci_by_date(
    predictions,
    actuals,
    dates: list,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """1000-iter bootstrap of Spearman IC by resampling unique dates with replacement.

    Why bootstrap-by-date rather than bootstrap-by-row: at h=21d, two rows
    on consecutive dates share 20 of 21 days of forward return. Resampling
    rows treats them as independent observations — that's the same
    autocorrelation problem the non-overlapping subsample addresses for the
    point estimate. The right unit of independence is the date. Resampling
    dates with replacement, then including ALL rows whose date is in the
    resample, gives a CI that respects the within-date cross-section but
    properly accounts for between-date uncertainty.

    Returns ``(ic_lo, ic_hi)`` at the specified confidence level (default
    95% = 2.5th / 97.5th percentiles). Returns ``(NaN, NaN)`` when:
      - ``dates`` is empty or fewer than 3 unique dates available (CI is
        meaningless on <3 independent observations);
      - ``predictions`` and ``actuals`` lengths disagree;
      - every bootstrap iteration produces a non-finite IC (degenerate input).

    NaN-tolerant on actuals: rows with NaN actual are excluded from each
    bootstrap iteration's IC computation, just like the point estimate.

    Seeded RNG (default seed=42) so the CI is deterministic across training
    runs holding training input constant. Lets operators eyeball "did the
    CI shrink because of more data, or move because of new data."
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    if not dates or len(predictions) != len(actuals) or len(predictions) != len(dates):
        return float("nan"), float("nan")

    dates_ts = [pd.Timestamp(d) for d in dates]
    unique_dates = sorted(set(dates_ts))
    if len(unique_dates) < 3:
        return float("nan"), float("nan")

    # Pre-build date → row-index list once. Each bootstrap iteration walks
    # this dict; avoids repeated O(N) date scans inside the inner loop.
    by_date: dict = {}
    for i, d in enumerate(dates_ts):
        by_date.setdefault(d, []).append(i)

    preds_arr = np.asarray(predictions)
    actuals_arr = np.asarray(actuals)
    rng = np.random.default_rng(seed)
    n_dates = len(unique_dates)
    ics: list[float] = []

    for _ in range(n_iter):
        sampled_date_idx = rng.integers(0, n_dates, size=n_dates)
        sampled_indices: list[int] = []
        for di in sampled_date_idx:
            sampled_indices.extend(by_date[unique_dates[di]])
        if not sampled_indices:
            continue
        idx_arr = np.array(sampled_indices)
        p = preds_arr[idx_arr]
        a = actuals_arr[idx_arr]
        finite = np.isfinite(a) & np.isfinite(p)
        if finite.sum() < 5:
            continue
        sp = spearmanr(p[finite], a[finite])
        if np.isfinite(sp.correlation):
            ics.append(float(sp.correlation))

    if len(ics) < 10:
        # Too many degenerate iterations to trust the CI shape.
        return float("nan"), float("nan")

    ics_arr = np.array(ics)
    alpha = 1.0 - ci
    lo = float(np.percentile(ics_arr, 100 * alpha / 2))
    hi = float(np.percentile(ics_arr, 100 * (1 - alpha / 2)))
    return lo, hi


def _load_momentum_params_from_s3(bucket: str) -> dict | None:
    """Read momentum GBM params from S3 if present.

    Written by the backtester's hyperparam sweep (future). Schema:
      {"n_estimators": int, "early_stopping_rounds": int,
       "tuned_params": {"num_leaves": int, ...}}

    Returns None when the key is absent or unreadable — callers fall back
    to YAML defaults (config.MOMENTUM_GBM_*).
    """
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=_MOMENTUM_PARAMS_S3_KEY)
        data = json.loads(obj["Body"].read())
        if not isinstance(data, dict):
            log.warning("Momentum params S3 override is not a dict — ignoring")
            return None
        return data
    except Exception as e:
        # NoSuchKey / AccessDenied / parse errors all fall back silently
        log.debug("No momentum params override on S3 (%s): %s",
                  _MOMENTUM_PARAMS_S3_KEY, e)
        return None


# ── Research-signal feature lookup (added 2026-04-28) ────────────────────────
# Pre-2026-04-28 the meta-trainer hardcoded the four per-ticker research
# features (`research_calibrator_prob`, `research_composite_score`,
# `research_conviction`, `sector_macro_modifier`) to constants — see the
# row construction in run_meta_training. With zero training-time variance
# Ridge correctly zeroed those coefficients, so per-ticker variance in
# meta-model output came almost exclusively from `expected_move`. The
# isotonic calibrator's plateau over [+0.002, +0.005] then collapsed every
# ticker on a typical day to a single p_up bin (~0.5119), which surfaced
# 2026-04-28 as 27 UP / 0 DOWN with identical confidence. Helpers below
# load the historical weekly signals snapshots from S3 and populate real
# values per (test_date, ticker) tuple.

# Per-ticker feature extraction + sector-name canonical map live in
# ``model/research_features.py`` so both training and inference call
# the same code path (inference's ``run_inference.py:293`` had the same
# top-level-vs-per-ticker bug as the original placeholder this PR closes).
# The local underscore-prefixed names are kept as private aliases so the
# existing test imports keep working.
from model.research_features import (
    SECTOR_NAME_CANONICAL as _SECTOR_NAME_CANONICAL,
    extract_research_features as _extract_research_features,
)


def _load_signals_history(s3, bucket: str) -> dict[str, dict]:
    """Load every ``signals/{date}/signals.json`` snapshot from S3.

    Returns ``{date_str: signals_payload_dict}`` covering the full S3 history.
    Each payload preserves the original JSON: ``universe`` list, top-level
    ``sector_modifiers`` and ``sector_ratings``, ``buy_candidates``, etc.
    Failures on individual dates log + skip; an empty result raises (per
    no-silent-fails — training without research history would silently
    revert to the bug we're trying to fix).
    """
    paginator = s3.get_paginator("list_objects_v2")
    date_prefixes: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix="signals/", Delimiter="/"):
        for cp in page.get("CommonPrefixes") or []:
            # CommonPrefix["Prefix"] looks like "signals/2026-03-05/"
            parts = cp["Prefix"].rstrip("/").split("/")
            if len(parts) != 2:
                continue
            date_str = parts[1]
            if (
                len(date_str) == 10
                and date_str[4] == "-"
                and date_str[7] == "-"
            ):
                date_prefixes.append(date_str)

    history: dict[str, dict] = {}
    for date_str in sorted(date_prefixes):
        try:
            obj = s3.get_object(
                Bucket=bucket, Key=f"signals/{date_str}/signals.json"
            )
            history[date_str] = json.loads(obj["Body"].read())
        except Exception as exc:
            log.warning(
                "Could not load signals/%s/signals.json: %s — skipping",
                date_str, exc,
            )

    if not history:
        raise RuntimeError(
            "No signals.json snapshots found under "
            f"s3://{bucket}/signals/. Meta-trainer requires real research "
            "history to populate per-ticker research features. Either the "
            "bucket is wrong or no Research run has ever produced output."
        )

    log.info(
        "Loaded %d signals snapshots from s3://%s/signals/ (range: %s → %s)",
        len(history), bucket, min(history), max(history),
    )
    return history


def _build_signals_lookup_by_test_date(
    signals_history: dict[str, dict],
    test_dates: list,
) -> dict[str, dict | None]:
    """Map every test_date to the most-recent-prior signals snapshot.

    Signals are written weekly (Saturday SF). On any weekday the "active"
    research view is the most recent prior Saturday's signals.json. This
    function is called once before the walk-forward loop so per-row
    lookups are O(1) dict reads.

    Returns ``{test_date_str: signals_payload | None}`` — None for dates
    that predate the first available signals snapshot.
    """
    import bisect

    snapshot_dates = sorted(signals_history.keys())
    lookup: dict[str, dict | None] = {}
    for td in set(test_dates):
        td_str = td if isinstance(td, str) else str(td)
        i = bisect.bisect_right(snapshot_dates, td_str)
        if i == 0:
            lookup[td_str] = None
        else:
            lookup[td_str] = signals_history[snapshot_dates[i - 1]]
    return lookup


def run_meta_training(
    data_dir: str,
    bucket: str,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """
    Train all Layer 1 + meta-model, validate with walk-forward, promote to S3.

    Parameters
    ----------
    data_dir : Local directory with *.parquet price cache + sector_map.json.
    bucket   : S3 bucket for model uploads.
    date_str : Training date (YYYY-MM-DD).
    dry_run  : Skip S3 writes if True.

    Returns
    -------
    dict with per-model ICs, meta-model IC, walk-forward results.
    """
    import config as cfg
    from model.gbm_scorer import GBMScorer
    from model.regime_predictor import RegimePredictor
    from model.research_calibrator import ResearchCalibrator
    from model.meta_model import MetaModel, META_FEATURES, MACRO_FEATURE_META_MAP

    start_ts = datetime.now(timezone.utc)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    log.info("Meta-training: loading data from %s", data_dir)
    data_path = Path(data_dir)

    from data.dataset import _load_ticker_parquet, cross_sectional_rank_normalize
    from data.label_generator import compute_labels
    # compute_features is intentionally NOT imported — training reads
    # pre-computed features from ArcticDB via the parquet cache written
    # by store/arctic_reader.py:download_from_arctic. Source of truth is
    # alpha-engine-data (features/feature_engineer.py). Inline compute
    # was deleted 2026-04-15 per ROADMAP P2 to eliminate the predictor/
    # data-module feature logic duplication. See PR #NN.

    # Load reference series
    def _load_close(fn):
        p = data_path / fn
        if not p.exists():
            return None
        d = _load_ticker_parquet(p)
        if d.empty or "Close" not in d.columns:
            return None
        return d["Close"].astype(float)

    spy_series = _load_close("SPY.parquet")
    vix_series = _load_close("VIX.parquet")
    vix3m_series = _load_close("VIX3M.parquet")
    tnx_series = _load_close("TNX.parquet")
    irx_series = _load_close("IRX.parquet")
    gld_series = _load_close("GLD.parquet")
    uso_series = _load_close("USO.parquet")

    if spy_series is None:
        raise RuntimeError("SPY.parquet not found in price cache")

    # Load sector map
    sector_map = {}
    sector_map_path = data_path / "sector_map.json"
    if sector_map_path.exists():
        sector_map = json.loads(sector_map_path.read_text())

    sector_etf_cache = {}
    for etf_sym in set(sector_map.values()):
        s = _load_close(f"{etf_sym}.parquet")
        if s is not None:
            sector_etf_cache[etf_sym] = s

    # Load all ticker close prices for regime breadth + feature computation
    _SKIP = {
        "SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }
    all_close_prices = {}
    all_parquets = sorted(data_path.glob("*.parquet"))

    # ── Step 2: Read pre-computed features from ArcticDB-populated parquets ──
    # alpha-engine-data's features/feature_engineer.py writes 53 features
    # to ArcticDB's universe library (per ticker, per date). store/
    # arctic_reader.py:download_from_arctic has already materialized those
    # to the local parquet cache at data_dir. We read them directly here
    # and compute labels. No inline feature compute — the predictor owns
    # labels (future-looking, not in ArcticDB), the data module owns
    # features (point-in-time, in ArcticDB).
    #
    # Per feedback_hard_fail_until_stable: any ticker missing required
    # feature columns is counted and, if the total accepted is zero, the
    # error surfaces with a full reject summary. Compare to pre-2026-04-15
    # behavior where compute_features was called inline and silent
    # except-Exception-continue discarded 909/909 tickers.
    _required_features = list(set(cfg.MOMENTUM_FEATURES) | set(cfg.VOLATILITY_FEATURES))
    log.info(
        "Reading pre-computed features from ArcticDB-populated parquets "
        "(required: %d columns)...",
        len(_required_features),
    )
    # ── Step 2 + Step 3: streaming parquet → arrays (rewritten 2026-04-28) ────
    # Pre-rewrite this loop populated ``ticker_features: dict[str, DataFrame]``
    # holding all ~900 labeled DataFrames (~700 KB each ≈ 630 MB) AND a
    # subsequent loop built a ``all_rows: list[tuple]`` of 1.77M Python
    # tuples on top of that ≈ another 265 MB AND `np.stack` over the
    # 1.77M tuples allocated the final arrays while everything was still
    # alive. Peak RSS ~1.5-2.5 GB plus pandas allocator burst overhead
    # OOM'd c5.large (4 GB) on 2026-04-28.
    #
    # Streaming version: per ticker we read the parquet, label, extract
    # numpy chunks, append to lightweight per-ticker chunk lists, then
    # drop the DataFrame so the next iteration's allocations have
    # headroom. After the loop a single ``np.concatenate`` glues the
    # chunks into the final arrays. Peak memory: ~one DF (~700 KB) +
    # the chunk list (~140 MB across all tickers) ≈ ~600 MB total
    # working set. Functionally identical to the previous code: same
    # rows, same sort order (by date, ascending), same shapes.
    reject_too_short: list[tuple[str, int]] = []
    reject_empty_raw: list[str] = []
    reject_missing_features: list[tuple[str, str]] = []
    reject_label_error: list[tuple[str, str]] = []
    reject_empty_labeled: list[str] = []
    n_candidates = 0
    n_accepted = 0
    _MIN_HISTORY_ROWS = 265  # ~1 year of trading days — required for walk-forward folds

    # Per-ticker chunk accumulators. Each list element is the per-ticker
    # numpy slice; final concat happens after the loop.
    date_chunks: list[np.ndarray] = []
    ticker_chunks: list[np.ndarray] = []
    mom_chunks: list[np.ndarray] = []
    vol_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    fwd_horizons_chunks: list[np.ndarray] = []

    for path in all_parquets:
        ticker = path.stem
        if ticker in _SKIP:
            continue
        n_candidates += 1
        raw_df = _load_ticker_parquet(path)
        if raw_df.empty:
            reject_empty_raw.append(ticker)
            continue
        if len(raw_df) < _MIN_HISTORY_ROWS:
            reject_too_short.append((ticker, len(raw_df)))
            continue
        if "Close" in raw_df.columns:
            all_close_prices[ticker] = raw_df["Close"].astype(float)
        missing = [c for c in _required_features if c not in raw_df.columns]
        if missing:
            reject_missing_features.append((ticker, ",".join(missing[:3])))
            continue
        sector_etf_sym = sector_map.get(ticker)
        sector_etf_s = sector_etf_cache.get(sector_etf_sym) if sector_etf_sym else None
        try:
            labeled = compute_labels(
                raw_df, forward_days=cfg.FORWARD_DAYS,
                up_threshold=cfg.UP_THRESHOLD, down_threshold=cfg.DOWN_THRESHOLD,
                benchmark_returns=sector_etf_s if sector_etf_s is not None else spy_series,
            )
        except Exception as exc:
            reject_label_error.append((ticker, f"{type(exc).__name__}: {exc}"))
            continue
        if labeled.empty:
            reject_empty_labeled.append(ticker)
            continue

        # v3.1 diagnostic (ROADMAP Predictor P2): compute multi-horizon
        # forward sector-neutral alpha alongside the 5d training label.
        # Used ONLY to evaluate the trained meta-model's IC across
        # horizons — tells us autonomously where the rank signal is
        # strongest. 2026-04-15 first measurement showed 21d IC was 2.2×
        # 5d IC; this sidecar extends to 10d / 15d / 21d / 40d so we
        # can see the full curve rather than two endpoints. Not used
        # for training.
        #
        # Shift -N means "close N rows ahead". At the tail of the
        # history the value is NaN; rows with NaN are excluded from
        # the corresponding-horizon IC only.
        close_for_horizon = raw_df["Close"].astype(float)
        bench_for_horizon = sector_etf_s if sector_etf_s is not None else spy_series
        if bench_for_horizon is not None:
            bench_aligned = bench_for_horizon.reindex(close_for_horizon.index, method="ffill")
            for h in _DIAGNOSTIC_HORIZONS:
                stock_fwd = (close_for_horizon.shift(-h) / close_for_horizon) - 1.0
                bench_fwd = (bench_aligned.shift(-h) / bench_aligned) - 1.0
                labeled[f"forward_return_{h}d"] = (stock_fwd - bench_fwd).reindex(labeled.index)
        else:
            for h in _DIAGNOSTIC_HORIZONS:
                labeled[f"forward_return_{h}d"] = float("nan")

        # Extract this ticker's slice into numpy chunks; the DataFrame
        # is then garbage-collectable when the loop iterates.
        n_rows = len(labeled.index)
        date_chunks.append(labeled.index.to_numpy())
        ticker_chunks.append(np.full(n_rows, ticker, dtype=object))
        mom_chunks.append(
            labeled[cfg.MOMENTUM_FEATURES].to_numpy(dtype=np.float32)
        )
        vol_chunks.append(
            labeled[cfg.VOLATILITY_FEATURES].to_numpy(dtype=np.float32)
        )
        fwd_chunks.append(
            labeled["forward_return_5d"].to_numpy(dtype=np.float32)
        )
        fwd_horizons_chunks.append(np.column_stack([
            labeled.get(
                f"forward_return_{h}d",
                pd.Series(float("nan"), index=labeled.index),
            ).to_numpy(dtype=np.float32)
            for h in _DIAGNOSTIC_HORIZONS
        ]))
        n_accepted += 1

        # Explicit cleanup — the per-ticker DataFrame and intermediate
        # views are no longer referenced; freeing them here keeps the
        # next iteration's allocations bounded by one DF's worth of
        # peak memory rather than the cumulative sum across 900 tickers
        # (the 2026-04-28 OOM root cause).
        del labeled, raw_df, close_for_horizon, bench_for_horizon

    log.info(
        "Feature read complete: %d accepted / %d candidates  "
        "(rejected: too_short=%d  empty_raw=%d  missing_features=%d  "
        "label_error=%d  empty_labeled=%d)",
        n_accepted, n_candidates,
        len(reject_too_short), len(reject_empty_raw),
        len(reject_missing_features), len(reject_label_error), len(reject_empty_labeled),
    )

    if n_accepted == 0:
        # Hard-fail with diagnostics rather than letting np.concatenate
        # emit its cryptic "need at least one array" error on the empty list.
        def _preview(rows, limit=5):
            return ", ".join(str(r) for r in rows[:limit]) + (
                f"  ... (+{len(rows) - limit} more)" if len(rows) > limit else ""
            )
        raise RuntimeError(
            f"All {n_candidates} candidate tickers were rejected during feature "
            f"read. Training cannot proceed on an empty dataset. "
            f"If missing_features > 0, alpha-engine-data's backfill must be "
            f"re-run against ArcticDB to populate the required columns. "
            f"Rejects — "
            f"too_short (<{_MIN_HISTORY_ROWS} rows): {len(reject_too_short)} "
            f"[{_preview(reject_too_short)}]  "
            f"empty_raw: {len(reject_empty_raw)} [{_preview(reject_empty_raw)}]  "
            f"missing_features: {len(reject_missing_features)} "
            f"[{_preview(reject_missing_features)}]  "
            f"label_error: {len(reject_label_error)} "
            f"[{_preview(reject_label_error)}]  "
            f"empty_labeled: {len(reject_empty_labeled)} "
            f"[{_preview(reject_empty_labeled)}]"
        )

    # Concatenate per-ticker chunks → single arrays in date-then-ticker
    # order, then sort by date so walk-forward iteration matches the
    # legacy behavior. argsort over a numpy datetime array is O(N log N)
    # in C; the prior code did Python-level sort over 1.77M tuples.
    date_unsorted = np.concatenate(date_chunks)
    ticker_unsorted = np.concatenate(ticker_chunks)
    X_mom_unsorted = np.concatenate(mom_chunks, axis=0)
    X_vol_unsorted = np.concatenate(vol_chunks, axis=0)
    y_fwd_unsorted = np.concatenate(fwd_chunks)
    y_fwd_horizons_unsorted = np.concatenate(fwd_horizons_chunks, axis=0)

    # Free the chunk lists — referenced data has been copied into the
    # contiguous arrays above. Without this `del` the chunks linger
    # through walk-forward and double the RSS we just reduced.
    del date_chunks, ticker_chunks, mom_chunks, vol_chunks
    del fwd_chunks, fwd_horizons_chunks

    # Stable sort by date so ties keep ticker-order deterministic for
    # snapshot tests + reproducibility.
    sort_idx = np.argsort(date_unsorted, kind="stable")
    # Wrap the sorted dates in ``pd.DatetimeIndex(...).tolist()`` so
    # ``all_dates`` is a list of ``pd.Timestamp`` objects, matching the
    # pre-streaming behavior. Bare ``.tolist()`` on a ``datetime64[ns]``
    # numpy array returns nanosecond-since-epoch ints (Python int can
    # represent ns precision; datetime.datetime cannot, so numpy
    # downgrades to int rather than truncate). Surfaced 2026-04-28 in
    # the v2 retraining run where ``str(int_ns)`` produced strings like
    # "1524009600000000000", which downstream sites consume in two
    # ways:
    #   1. ``signals_lookup`` keys, which compare lexicographically
    #      against snapshot keys like "2026-03-05" — a 1-prefixed int
    #      string sorts before any 2-prefixed date string, so bisect
    #      always returned 0 → every row dropped from the L2 training
    #      set (1,755,798 rows in the failing run).
    #   2. The walk-forward calibrator's ``np.datetime64(int_str, "D")``
    #      reparses the 19-digit string as a year, returning bizarre
    #      datetimes like year 8.8e15 AD. The resulting comparison
    #      flipped True/False unpredictably across folds (Fold 1 = 42,
    #      Fold 3 = 0, Fold 4 = 42, Fold 6 = 0, …).
    # ``pd.DatetimeIndex(arr).tolist()`` returns ``list[pd.Timestamp]``
    # which preserves nanosecond precision *and* round-trips cleanly to
    # the ISO string forms downstream code expects.
    all_dates = pd.DatetimeIndex(date_unsorted[sort_idx]).tolist()
    all_tickers = ticker_unsorted[sort_idx].tolist()
    X_mom = X_mom_unsorted[sort_idx]
    X_vol = X_vol_unsorted[sort_idx]
    y_fwd = y_fwd_unsorted[sort_idx]
    y_fwd_horizons = y_fwd_horizons_unsorted[sort_idx]

    del date_unsorted, ticker_unsorted
    del X_mom_unsorted, X_vol_unsorted, y_fwd_unsorted, y_fwd_horizons_unsorted

    # Winsorize
    if cfg.LABEL_CLIP:
        y_fwd = np.clip(y_fwd, -cfg.LABEL_CLIP, cfg.LABEL_CLIP)

    # Capture NaN masks BEFORE rank-normalization erases them (NaN inputs
    # become rank 1.0 after argsort → numpy puts NaN at the end). Used by
    # the short-history subsample IC gate at promotion time — see
    # ``model.subsample_validator`` and the gate block below the
    # production-train step.
    X_mom_raw = X_mom.copy()
    X_vol_raw = X_vol.copy()
    mom_nan_mask = np.any(np.isnan(X_mom_raw), axis=1)
    vol_nan_mask = np.any(np.isnan(X_vol_raw), axis=1)

    # Cross-sectional rank normalize (per-date, per-feature)
    X_mom = cross_sectional_rank_normalize(X_mom, all_dates)
    X_vol = cross_sectional_rank_normalize(X_vol, all_dates)

    N = len(y_fwd)
    log.info("Arrays built: %d samples, mom=%d features, vol=%d features",
             N, X_mom.shape[1], X_vol.shape[1])

    # ── Step 4: Build regime features + labels ───────────────────────────────
    regime_predictor = RegimePredictor()
    regime_features_df = regime_predictor.build_features(
        spy_series, vix_series, vix3m_series, tnx_series, irx_series,
        all_close_prices,
    )
    regime_labels = regime_predictor.build_labels(spy_series)

    # Align regime features with the training sample dates
    regime_dates = regime_features_df.index
    regime_X = regime_features_df[RegimePredictor.FEATURE_NAMES].to_numpy()
    regime_y = regime_labels.reindex(regime_dates).dropna()
    # Keep only dates present in both
    common_dates = regime_features_df.index.intersection(regime_y.index)
    regime_X_aligned = regime_features_df.loc[common_dates, RegimePredictor.FEATURE_NAMES].to_numpy()
    regime_y_aligned = regime_y.loc[common_dates].to_numpy().astype(int)

    log.info("Regime data: %d dates with features+labels", len(common_dates))

    # ── Step 5: Load research calibrator data from S3 ────────────────────────
    research_scores = np.array([])
    research_beat_spy = np.array([])
    research_score_dates = np.array([], dtype="datetime64[D]")
    try:
        import boto3
        s3 = boto3.client("s3")
        # Try to load research.db score_performance table
        import sqlite3
        db_tmp = Path(tempfile.gettempdir()) / "research_train.db"
        s3.download_file(bucket, "research.db", str(db_tmp))
        conn = sqlite3.connect(str(db_tmp))
        # score_date pulled in 2026-04-28 (PR #56) so the walk-forward
        # calibrator below can filter to entries dated <= train_end_date
        # per fold and avoid leaking forward beat_spy_10d outcomes into
        # earlier folds' research_calibrator_prob lookups.
        sp_df = pd.read_sql_query(
            "SELECT score, beat_spy_10d, score_date FROM score_performance "
            "WHERE beat_spy_10d IS NOT NULL AND score IS NOT NULL "
            "AND score_date IS NOT NULL",
            conn,
        )
        conn.close()
        if not sp_df.empty:
            research_scores = sp_df["score"].to_numpy()
            research_beat_spy = sp_df["beat_spy_10d"].to_numpy().astype(int)
            research_score_dates = pd.to_datetime(
                sp_df["score_date"]
            ).to_numpy().astype("datetime64[D]")
            log.info(
                "Research calibrator data: %d rows from score_performance "
                "(date range: %s → %s)",
                len(sp_df),
                str(research_score_dates.min()) if len(research_score_dates) else "n/a",
                str(research_score_dates.max()) if len(research_score_dates) else "n/a",
            )
        else:
            log.info("score_performance table empty — research calibrator will use priors")
    except Exception as e:
        log.warning("Could not load research calibrator data: %s — using priors", e)

    # ── Step 5b: Production research calibrator (fit on full data) ───────────
    # Two distinct calibrators are needed:
    #   1. ``prod_calibrator`` — fit on the full score_performance history,
    #      uploaded to S3 at the end of training. This is what inference
    #      reads at runtime via research_calibrator.json.
    #   2. ``fold_calibrator`` — fit per fold inside the walk-forward loop
    #      below using only score_performance entries dated <= train_end_date.
    #      Used to compute research_calibrator_prob for that fold's test rows
    #      so the meta-model never sees forward-looking calibration during
    #      training (the leakage that the original 2026-04-28 fix accepted
    #      and PR #56 closes).
    prod_calibrator = ResearchCalibrator()
    if len(research_scores) >= 10:
        prod_calibrator.fit(research_scores, research_beat_spy)
    else:
        log.info(
            "Research calibrator: insufficient data (%d rows) — using neutral "
            "priors. Per-row research_calibrator_prob will fall back to "
            "score_norm in row construction.",
            len(research_scores),
        )

    # Subsample IC gate — time-based holdout: fit a temporary calibrator on
    # the first ~80% of score_performance by date, evaluate Pearson IC on
    # the last 20%, compare to score/100 passthrough baseline. Production
    # ``prod_calibrator`` (fit on full data above) is unaffected — this is
    # purely a promotion-decision input. See model/subsample_validator.py.
    from model.subsample_validator import validate_research_calibrator
    research_subsample_result = validate_research_calibrator(
        scores=research_scores,
        beat_spy=research_beat_spy,
        score_dates=research_score_dates,
    )
    research_subsample_result.log()

    # ── Step 5c: Load signals.json history for per-row research feature lookup ─
    # Each weekly snapshot at signals/{YYYY-MM-DD}/signals.json carries the
    # research universe (per-ticker score / conviction / sector) and the
    # top-level sector_modifiers map. Walk-forward row construction below
    # joins (test_date, ticker) → most-recent-prior snapshot to pull real
    # values for the four meta-model research features.
    try:
        import boto3 as _boto3_sig
        _s3_signals = _boto3_sig.client("s3")
        signals_history = _load_signals_history(_s3_signals, bucket)
        signals_lookup = _build_signals_lookup_by_test_date(
            signals_history, [str(d) for d in sorted(set(all_dates))]
        )
    except Exception as exc:
        # Hard-fail per feedback_no_silent_fails — a training run that
        # silently falls back to constants is the bug we are fixing.
        raise RuntimeError(
            f"Failed to load signals history from S3 for meta-trainer: {exc}. "
            "Without research history the meta-model collapses to expected_move + "
            "macro features only (the 2026-04-28 incident). Refusing to proceed."
        ) from exc

    # ── Step 6: Walk-forward validation ──────────────────────────────────────
    log.info("Walk-forward validation...")
    unique_dates = sorted(set(all_dates))
    n_unique = len(unique_dates)
    test_window = cfg.WF_TEST_WINDOW_DAYS
    min_train = cfg.WF_MIN_TRAIN_DAYS
    purge_days = cfg.WF_PURGE_DAYS

    # Build fold boundaries
    folds = []
    fold_start_idx = min_train
    while fold_start_idx < n_unique:
        remaining = n_unique - fold_start_idx
        if remaining < test_window // 2:
            break
        test_start_date = unique_dates[fold_start_idx]
        test_end_idx = min(fold_start_idx + test_window - 1, n_unique - 1)
        test_end_date = unique_dates[test_end_idx]
        train_end_idx = fold_start_idx - purge_days
        if train_end_idx < min_train // 2:
            fold_start_idx += test_window
            continue
        train_end_date = unique_dates[train_end_idx]

        train_mask = np.array([d <= train_end_date for d in all_dates])
        test_mask = np.array([test_start_date <= d <= test_end_date for d in all_dates])
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) < 1000 or len(test_idx) < 100:
            fold_start_idx += test_window
            continue
        folds.append({"train_idx": train_idx, "test_idx": test_idx,
                       "train_end": str(train_end_date), "test_start": str(test_start_date),
                       "test_end": str(test_end_date)})
        fold_start_idx += test_window

    log.info("Walk-forward: %d folds", len(folds))

    tuned_params = getattr(cfg, "GBM_TUNED_PARAMS", None)
    wf_n_est = getattr(cfg, "WF_N_ESTIMATORS", None) or cfg.GBM_N_ESTIMATORS
    wf_es = getattr(cfg, "WF_EARLY_STOPPING", None) or cfg.GBM_EARLY_STOPPING_ROUNDS

    # Momentum base params: YAML defaults from config.MOMENTUM_GBM_* with
    # optional S3 override from config/predictor_momentum_params.json (written
    # by the backtester's hyperparam sweep). Volatility base keeps the shared
    # GBM params — only momentum has the weak-signal / overfitting issue.
    mom_tuned_params = dict(cfg.MOMENTUM_GBM_TUNED_PARAMS)
    mom_n_est = cfg.MOMENTUM_GBM_N_ESTIMATORS
    mom_es = cfg.MOMENTUM_GBM_EARLY_STOPPING_ROUNDS
    _s3_override = _load_momentum_params_from_s3(bucket)
    if _s3_override:
        if "tuned_params" in _s3_override:
            mom_tuned_params.update(_s3_override["tuned_params"])
        mom_n_est = _s3_override.get("n_estimators", mom_n_est)
        mom_es = _s3_override.get("early_stopping_rounds", mom_es)
        log.info("Momentum params overridden from S3: n_est=%d es=%d keys=%s",
                 mom_n_est, mom_es, list(_s3_override.get("tuned_params", {}).keys()))

    # Collect OOS predictions for meta-model training
    oos_meta_rows = []  # list of dicts with meta-features + actual outcome
    fold_results = []
    mom_fold_ics = []
    # Counters for the research-signal join (added 2026-04-28 alongside
    # the real-signals lookup). Tracked here, logged after the fold loop.
    n_rows_dropped_no_signals_snapshot = 0  # test_date predates first signals
    n_rows_dropped_ticker_missing = 0       # ticker absent from snapshot's universe
    n_rows_with_real_signals = 0            # rows actually retained for L2 training
    vol_fold_ics = []

    for i, fold in enumerate(folds):
        fold_start = time.time()
        tr = fold["train_idx"]
        te = fold["test_idx"]

        # Per-fold research calibrator (added 2026-04-28, PR #56). Fit only
        # on score_performance entries dated <= train_end_date so the
        # walk-forward construction never sees forward-looking calibration
        # for the test fold's rows. Falls through to ``prod_calibrator`` if
        # we don't have enough train-fold history (e.g. the very first fold
        # with score_performance entries) — that path keeps the leakage we
        # accepted in the original 2026-04-28 fix and tags it explicitly so
        # operators see when it triggers.
        fold_calibrator = None
        if len(research_score_dates) > 0:
            train_end_np = np.datetime64(fold["train_end"], "D")
            train_mask = research_score_dates <= train_end_np
            n_train_cal = int(train_mask.sum())
            if n_train_cal >= 10:
                fold_calibrator = ResearchCalibrator()
                fold_calibrator.fit(
                    research_scores[train_mask],
                    research_beat_spy[train_mask],
                )
                log.info(
                    "  Fold %d/%d: research calibrator fit on %d train-fold rows "
                    "(score_date <= %s)",
                    i + 1, len(folds), n_train_cal, fold["train_end"],
                )
            else:
                log.warning(
                    "  Fold %d/%d: only %d score_performance rows dated <= %s — "
                    "falling back to prod_calibrator (leakage). Per-row "
                    "research_calibrator_prob may use slightly forward-looking "
                    "calibration for this fold.",
                    i + 1, len(folds), n_train_cal, fold["train_end"],
                )
                fold_calibrator = prod_calibrator
        else:
            fold_calibrator = prod_calibrator

        # Train momentum model (low-capacity — see mom_tuned_params above)
        n_sub = int(len(tr) * 0.85)
        mom_scorer = GBMScorer(params=mom_tuned_params, n_estimators=mom_n_est,
                               early_stopping_rounds=mom_es)
        mom_scorer.fit(X_mom[tr[:n_sub]], y_fwd[tr[:n_sub]],
                       X_mom[tr[n_sub:]], y_fwd[tr[n_sub:]],
                       feature_names=cfg.MOMENTUM_FEATURES)
        mom_preds = mom_scorer.predict(X_mom[te])
        mom_ic = float(np.corrcoef(mom_preds, y_fwd[te])[0, 1]) if np.std(mom_preds) > 1e-10 else 0.0
        mom_fold_ics.append(mom_ic)

        # Train volatility model (predicts absolute return magnitude)
        abs_fwd = np.abs(y_fwd)
        vol_scorer = GBMScorer(params=tuned_params, n_estimators=wf_n_est,
                               early_stopping_rounds=wf_es)
        vol_scorer.fit(X_vol[tr[:n_sub]], abs_fwd[tr[:n_sub]],
                       X_vol[tr[n_sub:]], abs_fwd[tr[n_sub:]],
                       feature_names=cfg.VOLATILITY_FEATURES)
        vol_preds = vol_scorer.predict(X_vol[te])
        vol_ic = float(np.corrcoef(vol_preds, abs_fwd[te])[0, 1]) if np.std(vol_preds) > 1e-10 else 0.0
        vol_fold_ics.append(vol_ic)

        # Build meta-model OOS rows for this fold's test set. Raw macro
        # features are extracted per test_date and broadcast to every ticker
        # on that date (market-wide values, not per-ticker). RegimePredictor
        # here is used only as a feature-engineering utility via the static
        # regime_features_df built once at Step 4 — the classifier model itself
        # is no longer trained, uploaded, or consumed by the meta-model (see
        # note at top of meta_model.py and the roadmap entry for the Tier 1
        # regime rewrite).
        test_dates_unique = sorted(set(all_dates[j] for j in te))
        for test_date in test_dates_unique:
            date_mask = np.array([all_dates[j] == test_date for j in te])
            date_indices = te[date_mask]

            macro_row: dict[str, float] = {}
            if test_date in regime_features_df.index:
                for src_name, meta_name in MACRO_FEATURE_META_MAP.items():
                    macro_row[meta_name] = float(regime_features_df.at[test_date, src_name])
            else:
                for meta_name in MACRO_FEATURE_META_MAP.values():
                    macro_row[meta_name] = 0.0

            # Lookup the most-recent-prior weekly signals snapshot for this
            # test_date (built once before the fold loop in Step 5c).
            signals_payload = signals_lookup.get(str(test_date))

            for idx in date_indices:
                local_idx = np.where(te == idx)[0][0]
                ticker = all_tickers[idx]

                # Extract real research features from signals.json. Drops
                # the row when no snapshot exists for the test_date or the
                # ticker is missing from the snapshot's universe — per
                # `feedback_no_silent_fails` we refuse to silently fill
                # with constants (the pre-2026-04-28 bug behavior). L1
                # models above still train on the full (date, ticker)
                # set; only L2 narrows to rows where research signals
                # were actually present in production.
                if signals_payload is None:
                    n_rows_dropped_no_signals_snapshot += 1
                    continue
                research_features = _extract_research_features(
                    signals_payload, ticker, fold_calibrator
                )
                if research_features is None:
                    n_rows_dropped_ticker_missing += 1
                    continue

                row = {
                    "momentum_score": float(mom_preds[local_idx]),
                    "expected_move": float(vol_preds[local_idx]),
                    **research_features,
                    "actual_fwd": float(y_fwd[idx]),
                    # Track 1-of-2 of the audit Phase 1 horizon battery (2026-05-07):
                    # carry the row's date so subsequent IC computations can
                    # subsample to non-overlapping windows. Cheap (one column),
                    # additive (no behavior change to training itself).
                    "date": all_dates[idx],
                    **macro_row,  # raw macro features
                }
                # Multi-horizon labels (diagnostic only)
                for hi, h in enumerate(_DIAGNOSTIC_HORIZONS):
                    row[f"actual_fwd_{h}d"] = float(y_fwd_horizons[idx, hi])
                oos_meta_rows.append(row)
                n_rows_with_real_signals += 1

        elapsed = time.time() - fold_start
        fold_results.append({
            "fold": i + 1, "mom_ic": round(mom_ic, 6), "vol_ic": round(vol_ic, 6),
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "n_test": len(te), "elapsed_s": round(elapsed, 1),
        })
        log.info("  Fold %d/%d: mom_IC=%.4f  vol_IC=%.4f  (%.1fs)",
                 i + 1, len(folds), mom_ic, vol_ic, elapsed)

    # ── Step 6b: Research-signal join summary (added 2026-04-28) ────────────
    # Before-fix the meta-trainer hardcoded constants for the four research
    # features and Ridge zeroed their coefficients. Surface the join stats
    # so a future regression — empty signals history, sector-name drift in
    # research's output, etc — fails loud instead of silently rebuilding
    # the original collapse bug.
    n_rows_total = (
        n_rows_with_real_signals
        + n_rows_dropped_no_signals_snapshot
        + n_rows_dropped_ticker_missing
    )
    log.info(
        "Research-signal join: kept=%d  dropped_no_snapshot=%d  "
        "dropped_ticker_missing=%d  total_walked=%d",
        n_rows_with_real_signals,
        n_rows_dropped_no_signals_snapshot,
        n_rows_dropped_ticker_missing,
        n_rows_total,
    )
    if n_rows_with_real_signals == 0:
        raise RuntimeError(
            "Meta-trainer: 0 rows survived the research-signal join. "
            "Either no test_date in the walk-forward folds maps to a "
            "signals.json snapshot, or every signals snapshot has an "
            "empty universe / unmappable sector. Investigate before "
            "retraining; do not silently fall back to placeholder "
            "constants (pre-2026-04-28 bug)."
        )

    # ── Step 7: Train meta-model on pooled OOS ───────────────────────────────
    log.info("Training meta-model on %d OOS rows...", len(oos_meta_rows))
    meta_X = np.array([[r[f] for f in META_FEATURES] for r in oos_meta_rows])
    meta_y = np.array([r["actual_fwd"] for r in oos_meta_rows])
    meta_model = MetaModel(alpha=1.0)
    meta_model.fit(meta_X, meta_y, feature_names=META_FEATURES)

    # ── Step 7.1: Multi-horizon forward IC diagnostic (ROADMAP Predictor P2) ─
    # Evaluate how well the 5d-trained meta-model ranks tickers at multiple
    # forward horizons. 2026-04-15 first measurement (5d vs 21d only) showed
    # 21d Spearman IC 2.2× 5d IC — suggested 5d was the wrong horizon.
    # Extended to 5d / 10d / 15d / 21d / 40d so the data (not intuition)
    # reveals where the signal peaks. Autonomous: system shouldn't be
    # tuned to match current trader behavior; label horizon should track
    # strongest signal. Pure sidecar — model still trained/promoted on 5d.
    from scipy.stats import spearmanr
    meta_preds_oos_insample = meta_model.predict(meta_X).ravel()
    horizon_ics: dict[str, dict] = {}
    # Track 1-of-2 of the audit Phase 1 horizon battery (2026-05-07): in
    # addition to the legacy overlapping IC (each row's forward window
    # shares 20 of 21 days with its neighbor at h=21d), compute the IC on
    # a non-overlapping subsample so we can see how much of the
    # cross-horizon ratio is real signal vs autocorrelation inflation.
    # Both numbers persist in the manifest; downstream analysis decides
    # which to trust for the canonical horizon decision.
    row_dates = [r.get("date") for r in oos_meta_rows]
    for h in _DIAGNOSTIC_HORIZONS:
        y_h = np.array([r[f"actual_fwd_{h}d"] for r in oos_meta_rows])
        mask = np.isfinite(y_h)
        if mask.sum() >= 100:
            sp = spearmanr(meta_preds_oos_insample[mask], y_h[mask])
            ic = float(sp.correlation) if np.isfinite(sp.correlation) else 0.0
        else:
            ic = float("nan")

        # Bootstrap 95% CI on overlapping IC. Track 2-of-N of the audit
        # Phase 1 horizon battery (2026-05-07): 1000-iter bootstrap
        # resampling unique dates with replacement (not rows — see helper
        # docstring for why). Tells the operator how much of the cross-
        # horizon IC ratio is explained by sampling noise vs real signal.
        ic_ci_lo, ic_ci_hi = float("nan"), float("nan")
        if mask.sum() >= 100 and all(d is not None for d in row_dates):
            ic_ci_lo, ic_ci_hi = _bootstrap_ic_ci_by_date(
                meta_preds_oos_insample[mask],
                y_h[mask],
                [d for d, m in zip(row_dates, mask) if m],
                n_iter=1000,
                ci=0.95,
                seed=42 + h,  # unique seed per horizon for IID bootstrap streams
            )

        # Non-overlapping subsample IC: keep one row per non-overlapping
        # h-day window, then re-mask for finite labels. Min sample size is
        # lower (10) because non-overlapping naturally shrinks N at long
        # horizons — a 90d horizon over 24 months yields only ~8 windows.
        # NaN means "not enough independent samples to estimate IC" — the
        # downstream consumer reads this as "consult the overlapping number
        # at this horizon, or extend the training window."
        nonoverlap_ic: float = float("nan")
        nonoverlap_n: int = 0
        nonoverlap_ci_lo, nonoverlap_ci_hi = float("nan"), float("nan")
        if all(d is not None for d in row_dates):
            nonoverlap_mask = _nonoverlapping_date_mask(row_dates, h)
            combined_mask = np.array(nonoverlap_mask) & mask
            if combined_mask.sum() >= 10:
                sp_no = spearmanr(
                    meta_preds_oos_insample[combined_mask], y_h[combined_mask]
                )
                nonoverlap_ic = (
                    float(sp_no.correlation)
                    if np.isfinite(sp_no.correlation) else float("nan")
                )
                nonoverlap_n = int(combined_mask.sum())
                # Bootstrap CI on the non-overlap subsample. CI naturally
                # widens at long horizons (8 dates at h=90d → much wider
                # than at h=5d's ~144 dates). The width is the diagnostic.
                nonoverlap_ci_lo, nonoverlap_ci_hi = _bootstrap_ic_ci_by_date(
                    meta_preds_oos_insample[combined_mask],
                    y_h[combined_mask],
                    [d for d, m in zip(row_dates, combined_mask) if m],
                    n_iter=1000,
                    ci=0.95,
                    seed=42 + h + 1000,  # offset from overlap seed
                )

        horizon_ics[f"{h}d"] = {
            "spearman": ic,
            "n": int(mask.sum()),
            "spearman_ci_lo": ic_ci_lo,
            "spearman_ci_hi": ic_ci_hi,
            "spearman_nonoverlap": nonoverlap_ic,
            "n_nonoverlap": nonoverlap_n,
            "spearman_nonoverlap_ci_lo": nonoverlap_ci_lo,
            "spearman_nonoverlap_ci_hi": nonoverlap_ci_hi,
        }

    # Backwards-compat: keep spearman_5d / spearman_21d scalars for the
    # existing result-dict field consumers (email, dashboard, etc.).
    spearman_5d = horizon_ics["5d"]["spearman"]
    spearman_21d = horizon_ics["21d"]["spearman"]
    mask_5d_count = horizon_ics["5d"]["n"]
    mask_21d_count = horizon_ics["21d"]["n"]

    def _fmt_ci(lo: float, hi: float) -> str:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "[—,—]"
        return f"[{lo:+.4f},{hi:+.4f}]"

    _ic_line = "  ".join(
        f"{k}={v['spearman']:+.4f}{_fmt_ci(v['spearman_ci_lo'], v['spearman_ci_hi'])} (n={v['n']})"
        for k, v in horizon_ics.items()
    )
    log.info("Horizon IC curve (overlapping, 95%% CI): %s", _ic_line)
    _no_line = "  ".join(
        f"{k}={v['spearman_nonoverlap']:+.4f}{_fmt_ci(v['spearman_nonoverlap_ci_lo'], v['spearman_nonoverlap_ci_hi'])} (n={v['n_nonoverlap']})"
        for k, v in horizon_ics.items()
    )
    log.info("Horizon IC curve (non-overlapping, 95%% CI): %s", _no_line)
    # Flag peak horizon for quick visual in logs/email.
    _peak = max(horizon_ics.items(),
                key=lambda kv: abs(kv[1]["spearman"]) if np.isfinite(kv[1]["spearman"]) else -1)
    log.info("Peak IC horizon: %s (Spearman=%+.4f)", _peak[0], _peak[1]["spearman"])

    # ── Step 7b: Fit isotonic calibrator on meta OOS predictions ─────────────
    # ROADMAP P1: collapse FLAT, use calibrated P(UP) as confidence.
    #
    # Input to isotonic: continuous meta-model output on the pooled OOS
    # rows. These predictions are OOS with respect to the Layer-1 base
    # models (momentum/volatility/regime were held out by fold) but are
    # in-sample for the meta ridge itself, which was just fit on meta_X.
    # Pure nested CV would be cleaner; the pragmatic tradeoff is that
    # ridge has low capacity (8 coefficients) and overfits minimally, so
    # in-sample meta predictions are a reasonable calibration substrate.
    #
    # Output: P(actual_fwd > 0 | meta prediction). Binary target from
    # sign(meta_y). This is the canonical isotonic use case — no sigmoid
    # wrapper needed; isotonic produces a monotonic calibration curve
    # directly from continuous input to calibrated probability.
    #
    # Hard-fail per feedback_hard_fail_until_stable: a broken calibrator
    # silently degrades inference quality. Catch it here rather than
    # letting the inference path log a WARNING and fall back to heuristic.
    from model.calibrator import PlattCalibrator
    oos_meta_preds = meta_model.predict(meta_X).ravel()
    oos_up_labels = (meta_y > 0).astype(np.int32)
    log.info(
        "Fitting isotonic calibrator: n=%d  up_rate=%.3f  pred_std=%.6f",
        len(oos_meta_preds), float(oos_up_labels.mean()), float(np.std(oos_meta_preds)),
    )
    calibrator = PlattCalibrator(method="isotonic")
    calibrator.fit(oos_meta_preds, oos_up_labels, label_clip=float(cfg.LABEL_CLIP))
    if not calibrator.is_fitted:
        raise RuntimeError(
            f"Isotonic calibrator did not fit (n={len(oos_meta_preds)} samples, "
            f"min required 100). Training must produce a calibrator — see "
            f"model/calibrator.py for the sample threshold."
        )
    log.info(
        "Isotonic calibrator: ECE_before=%.4f  ECE_after=%.4f  (%.1f%% reduction)",
        calibrator._ece_before, calibrator._ece_after,
        (1 - calibrator._ece_after / max(calibrator._ece_before, 1e-8)) * 100,
    )

    # Walk-forward summary
    mom_median_ic = float(np.median(mom_fold_ics)) if mom_fold_ics else 0.0
    vol_median_ic = float(np.median(vol_fold_ics)) if vol_fold_ics else 0.0
    log.info("WF summary: momentum median_IC=%.4f  volatility median_IC=%.4f",
             mom_median_ic, vol_median_ic)

    # ── Step 8: Train production models on full data ─────────────────────────
    log.info("Training production models on full dataset...")
    n_train = int(N * cfg.TRAIN_FRAC)
    n_val_raw = int(N * cfg.VAL_FRAC)
    val_end = min(n_train + n_val_raw, N)

    # Momentum production model (low-capacity — see mom_tuned_params above)
    prod_mom = GBMScorer(params=mom_tuned_params, n_estimators=mom_n_est,
                         early_stopping_rounds=mom_es)
    prod_mom.fit(X_mom[:n_train], y_fwd[:n_train],
                 X_mom[n_train:val_end], y_fwd[n_train:val_end],
                 feature_names=cfg.MOMENTUM_FEATURES)
    mom_test_preds = prod_mom.predict(X_mom[val_end:])
    mom_test_ic = float(np.corrcoef(mom_test_preds, y_fwd[val_end:])[0, 1]) if len(mom_test_preds) > 1 else 0.0
    log.info("Momentum production: test_IC=%.4f  best_iter=%d", mom_test_ic, prod_mom._best_iteration)

    # Volatility production model
    prod_vol = GBMScorer(params=tuned_params, n_estimators=cfg.GBM_N_ESTIMATORS,
                         early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS)
    prod_vol.fit(X_vol[:n_train], np.abs(y_fwd[:n_train]),
                 X_vol[n_train:val_end], np.abs(y_fwd[n_train:val_end]),
                 feature_names=cfg.VOLATILITY_FEATURES)
    vol_test_preds = prod_vol.predict(X_vol[val_end:])
    vol_test_ic = float(np.corrcoef(vol_test_preds, np.abs(y_fwd[val_end:]))[0, 1]) if len(vol_test_preds) > 1 else 0.0
    log.info("Volatility production: test_IC=%.4f  best_iter=%d", vol_test_ic, prod_vol._best_iteration)

    # ── Step 8b: Short-history subsample IC gate ─────────────────────────────
    # Per ROADMAP P1 "NaN-feature handling audit + short-history subsample
    # validation" + feedback_component_baseline_validation: each L1 GBM
    # must beat a named simple-fallback baseline on the short-history
    # subsample (rows where any L1 input was NaN before rank-norm),
    # otherwise the GBM isn't earning its complexity on the slice that
    # mimics inference-time short-history tickers (SNDK-class).
    #
    # Hard-fail block: if any component fails its subsample gate, the
    # whole deploy is blocked. The meta-model is structurally trained on
    # the L1 outputs; promoting a stack with a regressed component would
    # ship a model with a known weakness on short-history tickers, which
    # the data layer is now actively shipping (PR #78).
    from model.subsample_validator import (
        momentum_baseline_predict, volatility_baseline_predict,
        validate_component,
    )
    mom_subsample_mask_test = mom_nan_mask[val_end:]
    vol_subsample_mask_test = vol_nan_mask[val_end:]
    mom_baseline_preds = momentum_baseline_predict(
        X_mom_raw[val_end:], cfg.MOMENTUM_FEATURES,
    )
    vol_baseline_preds = volatility_baseline_predict(
        X_vol_raw[val_end:], cfg.VOLATILITY_FEATURES,
    )
    mom_subsample_result = validate_component(
        component_name="momentum",
        component_preds=mom_test_preds,
        baseline_preds=mom_baseline_preds,
        y_true=y_fwd[val_end:],
        subsample_mask=mom_subsample_mask_test,
    )
    vol_subsample_result = validate_component(
        component_name="volatility",
        component_preds=vol_test_preds,
        baseline_preds=vol_baseline_preds,
        y_true=np.abs(y_fwd[val_end:]),
        subsample_mask=vol_subsample_mask_test,
    )
    mom_subsample_result.log()
    vol_subsample_result.log()
    subsample_gate_passed = (
        mom_subsample_result.passed
        and vol_subsample_result.passed
        and research_subsample_result.passed
    )

    # Regime classifier removed from the critical path 2026-04-16. Its
    # Tier 0 implementation (multinomial logistic on 6 macro features) could
    # not clear an honest baseline — walk-forward OOS accuracy of 39.5% sat
    # below always-predict-majority (~49%), with bear recall of 0.23 under
    # balanced class weights. Raw macro features now feed the meta ridge
    # directly via MACRO_FEATURE_META_MAP above. Tier 1 regime rewrite
    # (LightGBM + broader features + triple-barrier labeling + proper purge
    # ≥ label horizon) is tracked in the roadmap; when it ships, it will
    # enter the stack through a dedicated promotion gate referencing named
    # baselines (majority-class, random, persistence) rather than absolute
    # thresholds.

    # Research calibrator (`prod_calibrator`) was fitted earlier in Step 5b
    # so it could be consumed during walk-forward row construction. By this
    # point it is already in its final form; nothing to do here.

    # ── Step 9: Upload to S3 ─────────────────────────────────────────────────
    # Gate promotion on the meta-model composite IC rather than requiring
    # each base model's walk-forward median to be strictly positive. The
    # meta blend's validation IC already reflects how momentum + volatility
    # combine out-of-sample; double-gating on per-base-model walk-forward
    # blocks legitimate promotions when one base is weak but the blend is
    # strong.
    #
    # 2026-04-11 regression the new gate fixes: meta-model IC was 0.0525
    # (well above the 0.02 gate), volatility walk-forward 0.33 median with
    # 12/12 positive folds, but momentum walk-forward was -0.0017. The old
    # strict-positive gate blocked promotion, leaving gbm_latest.txt at a
    # 2026-03-28 snapshot for 16+ days while two weekly cycles produced
    # passing candidates. Silent alpha cap on every daily inference.
    #
    # Threshold matches the v2 single-model path (cfg.WF_MEDIAN_IC_GATE,
    # default 0.02 per config.py, configurable via predictor.yaml).
    _meta_ic_gate = cfg.WF_MEDIAN_IC_GATE
    meta_ic_passed = meta_model._val_ic >= _meta_ic_gate
    # Composite gate: meta-IC AND per-component subsample-IC. Both must
    # pass for promotion. The short-history subsample gate is what was
    # missing pre-2026-04-29 — meta_ic≥0.02 alone proved 2026-04-28 that
    # the ensemble could "work" while the components had silent NaN
    # poisoning the data layer was actively shipping.
    promoted = meta_ic_passed and subsample_gate_passed
    log.info(
        "Promotion gate: meta_IC=%.4f %s %.4f (meta=%s) AND component "
        "subsample (mom=%s, vol=%s, research=%s) → %s "
        "(walk-forward for reference: mom_median=%.4f, vol_median=%.4f)",
        meta_model._val_ic, ">=" if meta_ic_passed else "<", _meta_ic_gate,
        "PASS" if meta_ic_passed else "BLOCK",
        "PASS" if mom_subsample_result.passed else "BLOCK",
        "PASS" if vol_subsample_result.passed else "BLOCK",
        "PASS" if research_subsample_result.passed else "BLOCK",
        "PROMOTE" if promoted else "BLOCK",
        mom_median_ic, vol_median_ic,
    )
    elapsed_s = (datetime.now(timezone.utc) - start_ts).total_seconds()

    if not dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            import boto3 as _b3
            s3_up = _b3.client("s3")
            prefix = cfg.META_WEIGHTS_PREFIX  # "predictor/weights/meta/"

            # Save all models
            # isotonic_calibrator.pkl — binary P(UP) head on meta output.
            # Filename is contractual with the backtester's retrain_alert
            # grace-period check (reads .meta.json sidecar). Do not rename
            # without updating analysis/production_health.py.
            # Regime predictor removed from upload set 2026-04-16 — see note
            # in Step 8 above. Existing s3://.../regime_predictor.pkl from the
            # prior training cycle is left in place; inference no longer loads
            # it (see inference/stages/load_model.py). Cleanup of the stale
            # S3 artifact is tracked in the Tier 1 regime roadmap entry.
            models = {
                "momentum": (prod_mom, "momentum_model.txt"),
                "volatility": (prod_vol, "volatility_model.txt"),
                "research_calibrator": (prod_calibrator, "research_calibrator.json"),
                "meta_model": (meta_model, "meta_model.pkl"),
                "isotonic_calibrator": (calibrator, "isotonic_calibrator.pkl"),
            }
            # Dated archive is always written (records what training produced
            # regardless of gate decision). Live path is only overwritten when
            # the promotion gate passes — otherwise live weights stay at the
            # last-promoted snapshot. Pre-fix, both writes were unconditional,
            # so a gate-failed run silently swapped in unblessed weights; the
            # `promoted: false` flag in manifest became informational only.
            # Caught 2026-05-04 after the 2026-05-02 training (meta_IC=0.090
            # passed, momentum subsample failed → promoted=False) overwrote
            # the 2026-04-28 collapse-fix model (val_IC=0.132). Monday daily
            # inference produced a coarse 4-bucket bearish-collapsed
            # distribution; the issue was only visible after rolling live
            # weights back to the 2026-04-28 archive snapshot.
            for name, (model, filename) in models.items():
                local_path = Path(tmp) / filename
                model.save(local_path)

                # Dated backup — always written
                dated_key = f"{prefix}archive/{date_str}/{filename}"
                s3_up.upload_file(str(local_path), bucket, dated_key)

                # Live promotion — gated on composite promotion gate
                if promoted:
                    s3_key = f"{prefix}{filename}"
                    s3_up.upload_file(str(local_path), bucket, s3_key)
                    meta_path = Path(str(local_path) + ".meta.json")
                    if meta_path.exists():
                        s3_up.upload_file(str(meta_path), bucket, f"{s3_key}.meta.json")
                    log.info("Uploaded %s → s3://%s/%s (PROMOTED)", name, bucket, s3_key)
                else:
                    log.info(
                        "Skipped live promotion of %s — gate blocked. "
                        "Archive only at s3://%s/%s",
                        name, bucket, dated_key,
                    )

            # Write manifest
            manifest = {
                "date": date_str,
                "version": "v3.0-meta",
                "promoted": promoted,
                "models": {
                    "momentum": {"key": f"{prefix}momentum_model.txt", "test_ic": round(mom_test_ic, 6)},
                    "volatility": {"key": f"{prefix}volatility_model.txt", "test_ic": round(vol_test_ic, 6)},
                    # regime: removed from manifest 2026-04-16 (Tier 0 model
                    # retired; see Step 8 note and roadmap).
                    "research_calibrator": {"key": f"{prefix}research_calibrator.json",
                                            "n_samples": prod_calibrator._n_samples},
                    "meta_model": {
                        "key": f"{prefix}meta_model.pkl",
                        "ic": round(meta_model._val_ic, 6),
                        "importance": meta_model._importance,
                    },
                    "isotonic_calibrator": {
                        "key": f"{prefix}isotonic_calibrator.pkl",
                        "ece_before": round(calibrator._ece_before or 0.0, 6),
                        "ece_after": round(calibrator._ece_after or 0.0, 6),
                        "n_samples": calibrator._n_samples,
                    },
                },
                "walk_forward": {
                    "momentum_median_ic": round(mom_median_ic, 6),
                    "volatility_median_ic": round(vol_median_ic, 6),
                    "n_folds": len(folds),
                },
                "meta_coefficients": meta_model._coefficients,
            }
            s3_up.put_object(
                Bucket=bucket, Key=cfg.META_MANIFEST_KEY,
                Body=json.dumps(manifest, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("Manifest written to s3://%s/%s", bucket, cfg.META_MANIFEST_KEY)

    log.info(
        "Meta-training complete: mom_IC=%.4f  vol_IC=%.4f  meta_IC=%.4f  "
        "promoted=%s  elapsed=%.0fs",
        mom_test_ic, vol_test_ic, meta_model._val_ic, promoted, elapsed_s,
    )

    return {
        "model_version": "v3.0-meta",
        "promoted": promoted,
        "promoted_mode": "meta" if promoted else None,
        "elapsed_s": round(elapsed_s, 1),
        "n_train": n_train,
        "n_val": val_end - n_train,
        "n_test": N - val_end,
        "momentum_test_ic": round(mom_test_ic, 6),
        "volatility_test_ic": round(vol_test_ic, 6),
        # regime_* fields removed from result dict 2026-04-16 (Tier 0 model
        # retired). Email formatter and downstream consumers must not expect
        # regime_accuracy / regime_oos_* keys; they will be restored when the
        # Tier 1 regime model ships with its own promotion gate + metrics.
        "research_calibrator_n": prod_calibrator._n_samples,
        "research_calibrator_metrics": prod_calibrator.metrics(),
        "meta_model_ic": round(meta_model._val_ic, 6),
        "meta_coefficients": meta_model._coefficients,
        "meta_importance": meta_model._importance,
        "horizon_diagnostic": {
            # Backwards-compat scalars (existing consumers):
            "spearman_5d": round(spearman_5d, 6),
            "spearman_21d": round(spearman_21d, 6) if np.isfinite(spearman_21d) else None,
            "n_5d": mask_5d_count,
            "n_21d": mask_21d_count,
            # v3.2: full curve across _DIAGNOSTIC_HORIZONS.
            # v3.3 (audit Phase 1 Track B 1/N, 2026-05-07): added
            # non-overlapping subsample fields. The legacy `spearman`/`n`
            # are computed on all rows (overlapping forward windows).
            # `spearman_nonoverlap` / `n_nonoverlap` greedily subsample
            # one row per h-day window so the IC reflects independent
            # observations. Cross-horizon ratios computed on the
            # non-overlapping fields are honest; ratios on the legacy
            # fields are autocorrelation-inflated.
            # v3.4 (Track B 2/N, 2026-05-07): added 95% bootstrap CI
            # fields on both overlap and non-overlap. CI is computed by
            # 1000-iter resampling of unique dates with replacement (not
            # rows) so the CI respects within-date cross-section but
            # properly accounts for between-date uncertainty.
            "curve": {
                k: {
                    "spearman": round(v["spearman"], 6) if np.isfinite(v["spearman"]) else None,
                    "n": v["n"],
                    "spearman_ci_lo": (
                        round(v["spearman_ci_lo"], 6)
                        if np.isfinite(v["spearman_ci_lo"]) else None
                    ),
                    "spearman_ci_hi": (
                        round(v["spearman_ci_hi"], 6)
                        if np.isfinite(v["spearman_ci_hi"]) else None
                    ),
                    "spearman_nonoverlap": (
                        round(v["spearman_nonoverlap"], 6)
                        if np.isfinite(v["spearman_nonoverlap"]) else None
                    ),
                    "n_nonoverlap": v["n_nonoverlap"],
                    "spearman_nonoverlap_ci_lo": (
                        round(v["spearman_nonoverlap_ci_lo"], 6)
                        if np.isfinite(v["spearman_nonoverlap_ci_lo"]) else None
                    ),
                    "spearman_nonoverlap_ci_hi": (
                        round(v["spearman_nonoverlap_ci_hi"], 6)
                        if np.isfinite(v["spearman_nonoverlap_ci_hi"]) else None
                    ),
                }
                for k, v in horizon_ics.items()
            },
            "peak_horizon": _peak[0],
            "peak_spearman": round(_peak[1]["spearman"], 6) if np.isfinite(_peak[1]["spearman"]) else None,
        },
        "walk_forward": {
            "momentum_median_ic": round(mom_median_ic, 6),
            "volatility_median_ic": round(vol_median_ic, 6),
            "n_folds": len(folds),
            "folds": fold_results,
            "passes_wf": mom_median_ic > 0 and vol_median_ic > 0,
        },
        "short_history_subsample": {
            "momentum": {
                "n": mom_subsample_result.n,
                "component_ic": round(mom_subsample_result.component_ic, 6),
                "baseline_ic": round(mom_subsample_result.baseline_ic, 6),
                "passed": mom_subsample_result.passed,
                "skip_reason": mom_subsample_result.skip_reason,
            },
            "volatility": {
                "n": vol_subsample_result.n,
                "component_ic": round(vol_subsample_result.component_ic, 6),
                "baseline_ic": round(vol_subsample_result.baseline_ic, 6),
                "passed": vol_subsample_result.passed,
                "skip_reason": vol_subsample_result.skip_reason,
            },
            "research_calibrator": {
                "n": research_subsample_result.n,
                "component_ic": round(research_subsample_result.component_ic, 6),
                "baseline_ic": round(research_subsample_result.baseline_ic, 6),
                "passed": research_subsample_result.passed,
                "skip_reason": research_subsample_result.skip_reason,
            },
            "gate_passed": subsample_gate_passed,
        },
        # Compat fields for email/summary (reuse existing reporting)
        "test_ic": round(meta_model._val_ic, 6),
        "mse_ic": round(mom_test_ic, 6),
        "val_ic": round(meta_model._val_ic, 6),
        "passes_ic_gate": promoted,
        "ic_ir": 0.0,
        "feature_importance_top10": [],
        "feature_ics": {},
        "noise_candidates": [],
        "calibration": calibrator.metrics(),
    }
