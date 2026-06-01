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


def _log_rss(label: str) -> int:
    """Log current process RSS in MB and return the integer MB value.

    Used at major training-stage boundaries to surface memory pressure
    cycle-over-cycle. The 2026-05-24 redrive's predictor SIGKILL (likely
    OOM during the third sequential volatility GBM fit) was diagnostically
    invisible because the trainer had no per-stage RSS instrumentation;
    closing that gap is the institutional pre-requisite to refactoring
    memory usage with measured before/after numbers per
    [[feedback_sota_institutional_default_no_shortcuts]].

    Returns the RSS in MB so callers can track running max.

    Soft-fails to a no-op if ``psutil`` import or syscall raises — never
    blocks training.
    """
    try:
        import psutil  # type: ignore[import]
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        log.info("RSS %s: %.0f MB", label, rss_mb)
        return int(rss_mb)
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("RSS profiling unavailable (%s)", exc)
        return 0


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


def _classify_regime(macro_features: dict) -> str:
    """Classify a row's market regime as ``bull`` / ``neutral`` / ``bear``.

    Used by the Track B per-regime IC breakdown (3-of-N, audit Phase 1).
    Inputs are the 6 raw macro features already attached to every
    ``oos_meta_row`` — ``macro_spy_20d_return``, ``macro_spy_20d_vol``,
    ``macro_vix_level``, ``macro_vix_term_slope``, ``macro_yield_curve_slope``,
    ``macro_market_breadth``.

    Classification rule (heuristic, single primary axis):

    - ``bull``    if ``macro_spy_20d_return > +0.03`` (3% over 20 trading days)
    - ``bear``    if ``macro_spy_20d_return < -0.03``
    - ``neutral`` otherwise

    This is a single-feature heuristic chosen for transparency. The retired
    Tier-0 regime classifier (`model/regime_predictor.py`) used a richer
    multi-feature model that failed honest baseline validation (39.5% OOS
    accuracy, below majority-class baseline of 49% per memory). For this
    diagnostic, the goal isn't a great regime model — it's *some* regime
    label that lets us slice IC by market state and see whether the model's
    cross-horizon ratio is regime-dependent. A simple SPY 20d-return
    threshold is defensible and reproducible.

    Future ROADMAP P1: Tier-1 regime classifier with triple-barrier
    labeling and walk-forward purge will replace this once it clears its
    own promotion gate. Per audit §6.2 + §6.3.
    """
    spy_20d = macro_features.get("macro_spy_20d_return")
    if spy_20d is None or not isinstance(spy_20d, (int, float)):
        return "neutral"
    import math
    if math.isnan(float(spy_20d)):
        return "neutral"
    if spy_20d > 0.03:
        return "bull"
    if spy_20d < -0.03:
        return "bear"
    return "neutral"


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


def _emit_research_join_coverage_metrics(
    kept: int,
    dropped_no_snapshot: int,
    dropped_ticker_missing: int,
) -> None:
    """Emit CloudWatch gauges for the research-signal join's kept/dropped
    row counts so the drop rate doesn't silently shift.

    Per the ROADMAP L1465 investigation (closed 2026-05-12): dropping
    rows where the ticker isn't in the snapshot's ``universe`` is
    DELIBERATE — research only ranks the tracked-population universe
    (~25 tickers), so non-population rows have no real research signal.
    Filling with neutral defaults would reintroduce the pre-2026-04-28
    meta-model collapse (Ridge zeros coefficients on constant-fed
    features). See ``feedback_no_silent_fails`` in the memory store.

    The gauges exist so a future regression — research universe
    expansion/contraction, snapshot schema drift, lookup misalignment
    — surfaces as a drop-rate shift instead of silently shrinking the
    L2 fit corpus.

    Metrics (namespace ``AlphaEngine/Predictor``):

    - ``research_join_kept_count``: L2 training corpus size
    - ``research_join_dropped_no_snapshot_count``: cold-start (test_date
      predates first signals.json) — corpus-depth, not coverage
    - ``research_join_dropped_ticker_missing_count``: deliberate drops
    - ``research_join_kept_ratio``: kept / (kept + dropped_ticker_missing),
      only meaningful within "we had a snapshot for the date"

    Best-effort: CloudWatch errors WARN but never fail training. Counts
    are already in the inline log so a failed emit leaves a paper trail.
    """
    denom = kept + dropped_ticker_missing
    kept_ratio = (kept / denom) if denom > 0 else 0.0
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Predictor",
            MetricData=[
                {"MetricName": "research_join_kept_count",
                 "Value": float(kept), "Unit": "Count"},
                {"MetricName": "research_join_dropped_no_snapshot_count",
                 "Value": float(dropped_no_snapshot), "Unit": "Count"},
                {"MetricName": "research_join_dropped_ticker_missing_count",
                 "Value": float(dropped_ticker_missing), "Unit": "Count"},
                {"MetricName": "research_join_kept_ratio",
                 "Value": float(kept_ratio), "Unit": "None"},
            ],
        )
    except Exception as exc:  # noqa: BLE001 — observability, never block training
        log.warning(
            "CloudWatch research_join_* metric emit failed: %s. "
            "Counts still surface via the inline 'Research-signal join: ...' "
            "log; investigate the EC2 training-role IAM grant if this fires "
            "every Saturday.",
            exc,
        )


def _compute_overfit_signal(
    *,
    train_ic: float | None,
    val_ic: float,
    warn_threshold: float = 3.0,
) -> tuple[float | None, bool]:
    """Return (train_val_ic_ratio, overfit_warn_flag).

    ROADMAP L1816 (2026-05-19): the ResearchGBM is sensitive to
    overfitting at the 21d horizon because the labeled corpus shrinks
    vs the pre-cutover 5d horizon. Formalize the ``train_ic / val_ic``
    ratio as a first-class signal so the manifest + CW alarm can
    monitor regression after the params-reduction shipped in this PR.

    Ratio convention:

    - Computed as ``train_ic / abs(val_ic)`` so a large positive ratio
      always means "training fit dominates validation fit" regardless
      of validation sign (negative val_ic is itself a different
      pathology, but still triggers the warn since |val_ic| → 0
      blows up the ratio).
    - Returns ``None`` when ``train_ic`` is missing OR when val_ic's
      magnitude is below 1e-3 (the ratio's numerator/denominator
      noise floor) — surfacing "we couldn't measure" as a separate
      state from "we measured and it's healthy".
    - Warn flag fires on ratio > ``warn_threshold`` (default 3.0,
      halfway between the >2× watch level docs cite and the 5.05×
      we observed on the 21d retrain).
    """
    if train_ic is None:
        return None, False
    if abs(val_ic) < 1e-3:
        return None, False
    ratio = train_ic / abs(val_ic)
    warn = ratio > warn_threshold
    return ratio, warn


def _emit_research_gbm_overfit_metrics(
    *,
    train_ic: float | None,
    val_ic: float,
    ratio: float | None,
    warn: bool,
) -> None:
    """Emit ResearchGBM train/val IC + overfit gauges to CloudWatch.

    Mirrors ``_emit_research_join_coverage_metrics`` — best-effort, never
    blocks training. Counts are already in the inline log so a failed
    emit leaves a paper trail (same EC2 training-role IAM grant under
    ``alpha-engine-cloudwatch-metrics``).

    Gauges (namespace ``AlphaEngine/Predictor``):

    - ``research_gbm_train_ic`` — train-split Pearson IC.
    - ``research_gbm_val_ic`` — early-stop holdout Pearson IC.
    - ``research_gbm_train_val_ic_ratio`` — ratio for the 2-cycle
      persistence alarm (alarm on > 3.0 for >= 2 consecutive Saturday
      datapoints).
    - ``research_gbm_overfit_warn`` — 0/1 flag, same persistence rule
      via a Count gauge.

    Returns early without emitting when ``ratio`` is None (insufficient
    measurement — surfaced via the absence of datapoints rather than a
    spurious zero).
    """
    if ratio is None:
        return
    metric_data: list[dict] = []
    if train_ic is not None:
        metric_data.append(
            {"MetricName": "research_gbm_train_ic",
             "Value": float(train_ic), "Unit": "None"}
        )
    metric_data.extend([
        {"MetricName": "research_gbm_val_ic",
         "Value": float(val_ic), "Unit": "None"},
        {"MetricName": "research_gbm_train_val_ic_ratio",
         "Value": float(ratio), "Unit": "None"},
        {"MetricName": "research_gbm_overfit_warn",
         "Value": 1.0 if warn else 0.0, "Unit": "Count"},
    ])
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Predictor",
            MetricData=metric_data,
        )
    except Exception as exc:  # noqa: BLE001 — observability, never block training
        log.warning(
            "CloudWatch research_gbm_* metric emit failed: %s. "
            "train_ic=%s val_ic=%.4f ratio=%.4f warn=%s still surface via "
            "the inline ResearchGBMScorer + overfit-signal log; investigate "
            "the EC2 training-role IAM grant if this fires every Saturday.",
            exc,
            (f"{train_ic:.4f}" if train_ic is not None else "None"),
            val_ic,
            ratio,
            warn,
        )


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
from model import momentum_scorer
from model import residual_momentum_scorer
from model.residual_momentum_scorer import RESIDUAL_MOMENTUM_FEATURES
from data.residual_momentum_features import compute_residual_momentum_features


def build_train_meta_features(resid_mom_enabled: bool) -> list[str]:
    """W2 (L4469) observe gate: the L2 training-time meta-feature list.

    IDENTICAL to ``META_FEATURES`` unless the residual-momentum L1 is enabled,
    in which case ``residual_momentum_score`` is appended. NEVER mutates the
    ``META_FEATURES`` module constant — inference imports it, and the persisted
    ``meta_model._feature_names`` must stay byte-identical when the gate is
    closed so the live prediction path is unchanged. Promotion = flip the flag.
    """
    from model.meta_model import META_FEATURES
    return list(META_FEATURES) + (
        ["residual_momentum_score"] if resid_mom_enabled else []
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


def _build_macro_array(
    all_dates: list,
    regime_features_df: pd.DataFrame,
    macro_features: list[str],
) -> np.ndarray:
    """Build (N, n_macros) array broadcasting per-date macro values to every row.

    Used by Stage 1b for parallel-observation macro-augmented volatility
    GBM training. Each per-(date, ticker) row gets the same macro values
    on a given date — macros are constant cross-sectionally. Rows whose
    date isn't in ``regime_features_df.index`` get NaN; LightGBM handles
    NaN natively.

    Args:
        all_dates: list of pd.Timestamp, length N (per-(date, ticker) rows).
        regime_features_df: date-indexed DataFrame (from
            ``RegimePredictor.build_features``). Columns must include
            every name in ``macro_features``.
        macro_features: list of column names to extract, in canonical
            order. The output array preserves this column order.

    Returns:
        (N, n_macros) float64 array.
    """
    n = len(all_dates)
    n_macros = len(macro_features)
    X_macro = np.full((n, n_macros), np.nan, dtype=np.float64)

    if regime_features_df.empty:
        return X_macro

    macro_subset = regime_features_df[macro_features]
    for i, d in enumerate(all_dates):
        if d in macro_subset.index:
            X_macro[i] = macro_subset.loc[d].to_numpy(dtype=np.float64)

    return X_macro


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
    from model.research_gbm import RESEARCH_GBM_FEATURES

    start_ts = datetime.now(timezone.utc)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    log.info("Meta-training: loading data from %s", data_dir)
    data_path = Path(data_dir)

    from data.dataset import _load_ticker_parquet, cross_sectional_rank_normalize
    from data.label_generator import (
        compute_labels,
        compute_triple_barrier_alpha_labels,
        compute_triple_barrier_touch_labels,
    )
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
    # Stage 2c-full (regime-conditioning rebuild): FRED-only macros
    # backfilled to predictor/price_cache via collectors/fred_history.py.
    # _load_close returns None when the parquet is absent so the predictor
    # still trains during data-availability transitions; build_features()
    # falls back to neutral defaults for those macro features.
    two_series = _load_close("TWO.parquet")
    hyoas_series = _load_close("HYOAS.parquet")
    baa10y_series = _load_close("BAA10Y.parquet")
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
    # Stage 2b: per-ticker risk features for the parallel
    # ``prod_vol_risk_aug`` variant. Tolerant of missing columns
    # (NaN-fill rather than reject) so older parquets without the
    # alpha-engine-data #202 risk features still flow through plain
    # vol GBM training. LightGBM handles NaN natively.
    risk_chunks: list[np.ndarray] = []
    fwd_chunks: list[np.ndarray] = []
    fwd_horizons_chunks: list[np.ndarray] = []
    # Stage 3 PR 2 (regime-conditioning rebuild): parallel triple-barrier
    # alpha labels alongside the canonical fixed-21d-horizon target. NaN
    # at front-of-history (vol-window warm-up) and tail (past data end)
    # — these align naturally with the canonical fwd_chunks tail-drop.
    fwd_tb_chunks: list[np.ndarray] = []
    # Task B: parallel triple-barrier touch-order meta-labels (1=up barrier
    # first, 0=down first, NaN=timeout) on the SAME vol-scaled barriers as the
    # alpha label above. Supervision target for the observe-only meta-label
    # classifier whose calibrated P(up before down) feeds executor sizing.
    fwd_tb_touch_chunks: list[np.ndarray] = []
    # W2 (L4469): per-ticker residual/idiosyncratic-momentum features, computed
    # predictor-locally from the raw close + benchmark series (NOT from the
    # ArcticDB feature store — net-new single-source research features). Aligned
    # to the same tail-dropped labeled.index as mom_chunks. OBSERVE-mode: gated
    # out of META_FEATURES unless cfg.RESIDUAL_MOMENTUM_ENABLED.
    resid_mom_chunks: list[np.ndarray] = []

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
            # Stage 3 PR 2: parallel triple-barrier alpha label, vol-scaled
            # barriers (LdP Ch. 3.4). Sector-neutral residual log-returns;
            # σ_t = EWMA std of residual log-return; barrier = k × σ_t.
            # Front-of-history rows where σ_t hasn't warmed up get NaN
            # labels and are filtered downstream at the parallel-Ridge fit.
            labeled = compute_triple_barrier_alpha_labels(
                labeled,
                benchmark_returns=sector_etf_s if sector_etf_s is not None else spy_series,
                forward_window=cfg.TRIPLE_BARRIER_FORWARD_WINDOW,
                vol_window=cfg.TRIPLE_BARRIER_VOL_WINDOW,
                vol_multiplier=cfg.TRIPLE_BARRIER_VOL_MULTIPLIER,
                min_periods=cfg.TRIPLE_BARRIER_MIN_PERIODS,
            )
            # Task B: parallel touch-order meta-label on the same barriers.
            labeled = compute_triple_barrier_touch_labels(
                labeled,
                benchmark_returns=sector_etf_s if sector_etf_s is not None else spy_series,
                forward_window=cfg.TRIPLE_BARRIER_FORWARD_WINDOW,
                vol_window=cfg.TRIPLE_BARRIER_VOL_WINDOW,
                vol_multiplier=cfg.TRIPLE_BARRIER_VOL_MULTIPLIER,
                min_periods=cfg.TRIPLE_BARRIER_MIN_PERIODS,
            )
        except Exception as exc:
            reject_label_error.append((ticker, f"{type(exc).__name__}: {exc}"))
            continue
        if labeled.empty:
            reject_empty_labeled.append(ticker)
            continue

        # ROADMAP L1581 diagnostic — optional pre-cutoff drop. Applied to
        # ``labeled`` only; ``raw_df`` (and therefore the multi-horizon
        # shift(-N) sidecars below) keeps full history so forward labels
        # near the filter boundary aren't distorted by an artificial
        # front-of-window. Default ``TRAIN_START_DATE`` is None which is
        # a no-op.
        if cfg.TRAIN_START_DATE is not None:
            _cutoff_ts = pd.Timestamp(cfg.TRAIN_START_DATE)
            labeled = labeled.loc[labeled.index >= _cutoff_ts]
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
        # W2 (L4469): residual-momentum features. Computed on the FULL raw_df
        # close index (backward-only, point-in-time) then reindexed to the
        # tail-dropped labeled.index so it aligns row-for-row with mom/vol.
        # Benchmark = sector ETF if present else SPY (matches the label
        # convention). Always emitted (the column is harmless extra data); the
        # observe gate controls only META_FEATURES inclusion downstream.
        resid_mom_df = compute_residual_momentum_features(
            raw_df["Close"].astype(float),
            benchmark_close=(sector_etf_s if sector_etf_s is not None else spy_series),
            spy_close=spy_series,
            beta_window=cfg.RESID_MOM_BETA_WINDOW,
            mom_window=cfg.RESID_MOM_WINDOW,
            skip_window=cfg.RESID_MOM_SKIP_DAYS,
            vol_window=cfg.RESID_MOM_VOL_WINDOW,
            mom_change_window=cfg.RESID_MOM_CHANGE_WINDOW,
        )
        resid_mom_chunks.append(
            resid_mom_df.reindex(labeled.index)[RESIDUAL_MOMENTUM_FEATURES]
            .to_numpy(dtype=np.float32)
        )
        # Stage 2b: per-ticker risk features. ``reindex`` returns NaN
        # for any column missing from the parquet — older feature_engineer
        # outputs (pre-#202) lack these columns, but LightGBM handles
        # NaN natively so the parallel variant trains on whatever subset
        # is populated.
        risk_chunks.append(
            labeled.reindex(columns=cfg.RISK_AUG_FEATURES)
                   .to_numpy(dtype=np.float32)
        )
        fwd_chunks.append(
            labeled["forward_return_5d"].to_numpy(dtype=np.float32)
        )
        # Stage 3 PR 2: triple-barrier label per row, aligned to the same
        # tail-dropped index as fwd_chunks. Column may be absent if the
        # PR 1 generator returned early (empty input); fall back to NaN.
        fwd_tb_chunks.append(
            labeled.get(
                "triple_barrier_alpha_21d",
                pd.Series(float("nan"), index=labeled.index),
            ).to_numpy(dtype=np.float32)
        )
        # Task B: touch-order meta-label, same tail-dropped index. Absent if
        # the generator returned early (empty input) → NaN fallback.
        fwd_tb_touch_chunks.append(
            labeled.get(
                "triple_barrier_touch_21d",
                pd.Series(float("nan"), index=labeled.index),
            ).to_numpy(dtype=np.float32)
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
        del labeled, raw_df, close_for_horizon, bench_for_horizon, resid_mom_df

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
    X_risk_unsorted = np.concatenate(risk_chunks, axis=0)
    y_fwd_unsorted = np.concatenate(fwd_chunks)
    y_fwd_horizons_unsorted = np.concatenate(fwd_horizons_chunks, axis=0)
    y_fwd_tb_unsorted = np.concatenate(fwd_tb_chunks)
    y_fwd_tb_touch_unsorted = np.concatenate(fwd_tb_touch_chunks)
    X_resid_mom_unsorted = np.concatenate(resid_mom_chunks, axis=0)  # W2

    # Free the chunk lists — referenced data has been copied into the
    # contiguous arrays above. Without this `del` the chunks linger
    # through walk-forward and double the RSS we just reduced.
    del date_chunks, ticker_chunks, mom_chunks, vol_chunks, risk_chunks
    del fwd_chunks, fwd_horizons_chunks, fwd_tb_chunks, fwd_tb_touch_chunks
    del resid_mom_chunks

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
    X_risk = X_risk_unsorted[sort_idx]
    y_fwd = y_fwd_unsorted[sort_idx]
    y_fwd_horizons = y_fwd_horizons_unsorted[sort_idx]
    y_fwd_tb = y_fwd_tb_unsorted[sort_idx]
    # Task B touch-order labels are 0.0/1.0/NaN — NOT clipped (clip below
    # applies only to the continuous alpha labels).
    y_fwd_tb_touch = y_fwd_tb_touch_unsorted[sort_idx]
    X_resid_mom = X_resid_mom_unsorted[sort_idx]  # W2

    del date_unsorted, ticker_unsorted
    del X_mom_unsorted, X_vol_unsorted, X_risk_unsorted
    del y_fwd_unsorted, y_fwd_horizons_unsorted, y_fwd_tb_unsorted
    del y_fwd_tb_touch_unsorted, X_resid_mom_unsorted

    # Winsorize
    if cfg.LABEL_CLIP:
        y_fwd = np.clip(y_fwd, -cfg.LABEL_CLIP, cfg.LABEL_CLIP)
        # Stage 3 PR 2: triple-barrier labels are already capped at the
        # vol-scaled barrier per row (LdP path-walking), so winsorization
        # is largely a no-op. Apply for symmetry with y_fwd in case the
        # barrier ever yields a value beyond LABEL_CLIP at extreme vol;
        # NaN values are preserved by np.clip.
        y_fwd_tb = np.clip(y_fwd_tb, -cfg.LABEL_CLIP, cfg.LABEL_CLIP)

    # Capture NaN masks BEFORE rank-normalization erases them (NaN inputs
    # become rank 1.0 after argsort → numpy puts NaN at the end). Used by
    # the short-history subsample IC gate at promotion time — see
    # ``model.subsample_validator`` and the gate block below the
    # production-train step.
    X_mom_raw = X_mom.copy()
    X_vol_raw = X_vol.copy()
    mom_nan_mask = np.any(np.isnan(X_mom_raw), axis=1)
    vol_nan_mask = np.any(np.isnan(X_vol_raw), axis=1)
    # W2: the deterministic residual-momentum scorer reads the RAW matrix (like
    # momentum); keep the nan-mask for a future short-history subsample gate.
    X_resid_mom_raw = X_resid_mom.copy()
    resid_mom_nan_mask = np.any(np.isnan(X_resid_mom_raw), axis=1)

    # Cross-sectional rank normalize (per-date, per-feature)
    X_mom = cross_sectional_rank_normalize(X_mom, all_dates)
    X_vol = cross_sectional_rank_normalize(X_vol, all_dates)
    # W2: rank-normed residual-mom kept available for a future direct-to-
    # META_FEATURES promotion; the scorer itself uses X_resid_mom_raw.
    X_resid_mom = cross_sectional_rank_normalize(X_resid_mom, all_dates)
    # Stage 2b: rank-norm risk features alongside vol. Per-ticker
    # cross-sectional variation lets the trees split on relative rank
    # (e.g., "this ticker is in the top 10% of beta cross-section today").
    X_risk = cross_sectional_rank_normalize(X_risk, all_dates)

    N = len(y_fwd)
    log.info("Arrays built: %d samples, mom=%d features, vol=%d features",
             N, X_mom.shape[1], X_vol.shape[1])

    # ── Step 3b: Canonical alpha labels (audit Track A PR 2/6, observe-only) ─
    # Computes log-domain risk-matched alpha vs vol-cohort EW basket per
    # (date, ticker) from all_close_prices. Per audit §7.3-§7.4 this is
    # the canonical replacement for the legacy
    # ``stock_5d_return - sector_etf_5d_return`` arithmetic label that
    # gets attached to oos_meta_rows below in Step 6.
    #
    # Observe-only in this PR — labels are attached to each row but the
    # meta-Ridge still trains on the legacy ``actual_fwd`` label.
    # Track A PR 3 adds a parallel canonical-label Ridge fit; PR 5 cuts
    # over to canonical labels as production training target.
    #
    # Computes once on the full universe (not per-fold) since the
    # cohort partitioning is cross-sectional per-date and doesn't depend
    # on walk-forward fold boundaries — the cohort assignment only uses
    # *trailing* vol, no forward leakage.
    from model.canonical_alpha_labels import compute_canonical_alpha

    canonical_alpha_by_ticker: dict[str, "pd.Series"] = {}
    try:
        canonical_alpha_by_ticker = compute_canonical_alpha(
            all_close_prices,
            horizon=cfg.FORWARD_DAYS,
            vol_lookback_days=20,
            n_cohorts=10,
        )
        n_finite = sum(int(s.notna().sum()) for s in canonical_alpha_by_ticker.values())
        log.info(
            "Canonical alpha labels: %d tickers, %d finite (date, ticker) cells "
            "— observe-only Track A PR 2",
            len(canonical_alpha_by_ticker), n_finite,
        )
    except Exception as e:
        log.warning(
            "compute_canonical_alpha failed (non-blocking, observe-only): %s — "
            "rows will get None for actual_fwd_canonical",
            e,
        )

    # ── Step 4: Build regime features + labels ───────────────────────────────
    regime_predictor = RegimePredictor()
    regime_features_df = regime_predictor.build_features(
        spy_series, vix_series, vix3m_series, tnx_series, irx_series,
        all_close_prices,
        two_series=two_series,
        hyoas_series=hyoas_series,
        baa10y_series=baa10y_series,
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

    # ── Step 4b: Build macro feature array for Stage 1b parallel observation ─
    # Stage 1b of the regime-conditioning rebuild (plan doc:
    # alpha-engine-docs/private/regime-conditioning-260510.md). Builds a
    # (N, 6) macro array broadcasting per-date values to every per-(date,
    # ticker) row in all_dates, then applies time-series z-score over a
    # rolling 252-day window.
    #
    # Per-feature normalization is the institutional SOTA pattern: cross-
    # sectional rank-norm for ticker features (existing X_vol, X_mom),
    # time-series z-score for macros (this block). Macros are constant
    # across tickers on a given date — rank-norm degenerates them to 0.5
    # for every row, losing all signal. Z-score over rolling window
    # preserves the regime signal (high VIX vs low VIX) and lets trees
    # discover macro-conditioned interactions through sequential splits.
    #
    # Used downstream by the parallel macro-augmented volatility GBM
    # (Step 7e). When regime_features_df doesn't have a date, that row
    # gets NaN for every macro — LightGBM handles NaN natively.
    from data.dataset import time_series_zscore_normalize

    X_macro_raw = _build_macro_array(
        all_dates, regime_features_df, list(cfg.MACRO_NORM_FEATURES),
    )
    X_macro_zscored = time_series_zscore_normalize(
        X_macro_raw, all_dates, window=cfg.MACRO_NORM_WINDOW,
    )
    log.info(
        "Macro feature array (Stage 1b): %d rows × %d cols, "
        "rolling z-score window=%d, finite_pct=%.2f",
        X_macro_zscored.shape[0], X_macro_zscored.shape[1],
        cfg.MACRO_NORM_WINDOW,
        100.0 * float(np.isfinite(X_macro_zscored).all(axis=1).mean()),
    )

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

    # Bucket-lookup ResearchCalibrator subsample gate retired 2026-05-09
    # (Phase 3 PR 4/5 cutover). The bucket-lookup is no longer the L1
    # component sourcing research_calibrator_prob — the ResearchGBMScorer
    # fit at Step 6c is, and its val_ic (held-out 20%) provides the same
    # "earns-its-complexity" signal this subsample gate did. The bucket-
    # lookup remains as a degraded fallback for cycles where the GBM fit
    # is skipped (n_finite < 100); a fallback path doesn't need its own
    # promotion gate. ``validate_research_calibrator`` is still defined
    # in model/subsample_validator.py for legacy tests / future re-use.

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

    # Momentum L1 hyperparams (cfg.MOMENTUM_GBM_*, S3 override
    # config/predictor_momentum_params.json) are no longer consumed —
    # the L1 component is now the deterministic baseline at
    # model/momentum_scorer.py. Config keys are left in place for
    # backward compatibility with downstream readers (backtester
    # hyperparam sweep) but their values are inert until / unless a
    # future momentum L1 model returns through a named-baseline
    # promotion gate.

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
    resid_mom_fold_ics = []  # W2: per-fold raw IC of the residual-momentum L1

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

        # Momentum L1 component is now the deterministic weighted-blend
        # baseline (was a low-capacity LightGBM). Across 16 weeks of
        # walk-forward validation the GBM consistently failed to beat
        # the baseline (GBM val_IC ~0.01, baseline val_IC ~0.31), so
        # the GBM was a calibration tax on the meta-Ridge stack.
        # Replaced 2026-05-09. See model/momentum_scorer.py for the
        # single source of truth and PR alpha-engine-predictor (post-#111)
        # for the rationale.
        n_sub = int(len(tr) * 0.85)  # retained for volatility split below
        mom_preds = momentum_scorer.predict_array(
            X_mom_raw[te], cfg.MOMENTUM_FEATURES,
        )
        mom_ic = float(np.corrcoef(mom_preds, y_fwd[te])[0, 1]) if np.std(mom_preds) > 1e-10 else 0.0
        mom_fold_ics.append(mom_ic)

        # W2 (L4469): residual-momentum L1 — deterministic, reads the raw
        # matrix like momentum. Per-fold raw IC tracked for the manifest; the
        # standalone leak-free read (purged + embargoed cross-sectional rank IC)
        # is computed after the loop. The score column is always written to the
        # OOS row dict; the observe gate controls only META_FEATURES inclusion.
        resid_mom_preds = residual_momentum_scorer.predict_array(
            X_resid_mom_raw[te], RESIDUAL_MOMENTUM_FEATURES,
        )
        resid_mom_ic = (
            float(np.corrcoef(resid_mom_preds, y_fwd[te])[0, 1])
            if np.std(resid_mom_preds) > 1e-10 else 0.0
        )
        resid_mom_fold_ics.append(resid_mom_ic)

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
                # Stage D: regime-derived features (currently just
                # intensity_z). Same date-indexed lookup pattern; missing
                # columns default to 0.0 so older code paths that
                # haven't repopulated regime_features_df via
                # RegimePredictor.build_features (which now adds
                # intensity_z) degrade gracefully.
                from model.meta_model import REGIME_DERIVED_FEATURE_META_MAP
                for src_name, meta_name in REGIME_DERIVED_FEATURE_META_MAP.items():
                    if src_name in regime_features_df.columns:
                        macro_row[meta_name] = float(regime_features_df.at[test_date, src_name])
                    else:
                        macro_row[meta_name] = 0.0
            else:
                for meta_name in MACRO_FEATURE_META_MAP.values():
                    macro_row[meta_name] = 0.0
                from model.meta_model import REGIME_DERIVED_FEATURE_META_MAP
                for meta_name in REGIME_DERIVED_FEATURE_META_MAP.values():
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
                    # W2: always written; only enters the L2 matrix when the
                    # observe gate (cfg.RESIDUAL_MOMENTUM_ENABLED) selects it
                    # into TRAIN_META_FEATURES below.
                    "residual_momentum_score": float(resid_mom_preds[local_idx]),
                    **research_features,
                    "actual_fwd": float(y_fwd[idx]),
                    # Stage 3 PR 2: parallel triple-barrier alpha label
                    # (vol-scaled barriers per LdP Ch. 3.4). NaN at front-of-
                    # history rows where σ_t didn't warm up; the parallel
                    # Ridge fit at Step 7b filters NaN rows separately from
                    # the canonical fit. Carrying ticker so Step 7b can
                    # compute per-ticker LdP Ch. 4.4 sample weights.
                    "actual_fwd_triple_barrier": float(y_fwd_tb[idx]),
                    # Task B: touch-order meta-label (1=up first, 0=down first,
                    # NaN=timeout under "nan" policy). Step 7c filters NaN rows
                    # separately for the meta-label classifier fit.
                    "actual_fwd_barrier_touch": float(y_fwd_tb_touch[idx]),
                    "ticker": ticker,
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
                # Audit Track A PR 2/6 (2026-05-07): canonical alpha label
                # (log-domain risk-matched vs vol-cohort EW basket).
                # Observe-only — meta-Ridge still trains on actual_fwd.
                # NaN for rows where the ticker was unclassifiable into a
                # cohort or the cohort had <2 members; the manifest gets
                # the finite-row count as a coverage metric.
                row_ts = all_dates[idx]
                _canonical_series = canonical_alpha_by_ticker.get(ticker)
                if _canonical_series is not None:
                    _canonical_value = _canonical_series.get(row_ts)
                    if _canonical_value is not None and np.isfinite(_canonical_value):
                        row["actual_fwd_canonical"] = float(_canonical_value)
                    else:
                        row["actual_fwd_canonical"] = float("nan")
                else:
                    row["actual_fwd_canonical"] = float("nan")
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
    # Closes ROADMAP L1465: CW gauges so the drop rate doesn't silently
    # shift on a future research universe / snapshot-schema change.
    _emit_research_join_coverage_metrics(
        kept=n_rows_with_real_signals,
        dropped_no_snapshot=n_rows_dropped_no_signals_snapshot,
        dropped_ticker_missing=n_rows_dropped_ticker_missing,
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

    # ── Step 6c: ResearchGBMScorer fit + research_calibrator_prob cutover ─────
    # Audit Phase 3 PR 4/5 (2026-05-09): the GBM is now the canonical source
    # for the `research_calibrator_prob` META_FEATURE the meta-Ridge sees.
    # The bucket-lookup ResearchCalibrator (prod_calibrator from Step 5b)
    # remains as the per-fold WF source for `research_calibrator_prob` in
    # `_extract_research_features` AND as the inference-time fallback when
    # the GBM isn't loaded (n_finite < 100 cycles).
    #
    # Training shape: same OOS rows the meta-Ridge consumes (each row has
    # all 9 RESEARCH_GBM_FEATURES + multi-horizon labels from Track B).
    # Label is binary beat-SPY-at-10d (`actual_fwd_10d > 0`) — same target
    # the bucket-lookup is graded on. WF isolation: rows are date-sorted
    # and we split 80/20 temporally for early-stopping validation.
    #
    # Caveat (PR 4 cutover, 2026-05-09): at ~24 weeks of labeled history
    # (~797 OOS rows) per-fold WF GBM training is infeasible — early folds
    # would have <50 rows of feature-matrix to train a 9-feature LGB.
    # Today's cutover accepts in-sample predictions on training rows for
    # Ridge fit purposes; the GBM's val_ic measures the OOS gap. Manifest
    # emits both train_ic and val_ic so the gap can be monitored. Per-fold
    # WF tightening tracked for ~30-week milestone (2026-06-20-ish).
    from model.research_gbm import ResearchGBMScorer, RESEARCH_GBM_FEATURES

    prod_research_gbm: ResearchGBMScorer | None = None
    research_gbm_metrics: dict | None = None
    research_gbm_train_ic: float | None = None
    research_gbm_train_val_ic_ratio: float | None = None
    research_gbm_overfit_warn: bool = False
    try:
        finite_mask = np.array([
            np.isfinite(r.get("actual_fwd_10d", float("nan")))
            for r in oos_meta_rows
        ])
        n_finite = int(finite_mask.sum())
        if n_finite >= 100:
            X_research = np.array([
                [r.get(f, 0.0) or 0.0 for f in RESEARCH_GBM_FEATURES]
                for r, m in zip(oos_meta_rows, finite_mask) if m
            ], dtype=np.float32)
            y_research = np.array([
                1.0 if r.get("actual_fwd_10d", 0.0) > 0 else 0.0
                for r, m in zip(oos_meta_rows, finite_mask) if m
            ], dtype=np.float32)
            split_idx = int(n_finite * 0.8)
            prod_research_gbm = ResearchGBMScorer(
                n_estimators=500, early_stopping_rounds=30,
            )
            prod_research_gbm.fit(
                X_research[:split_idx], y_research[:split_idx],
                X_research[split_idx:], y_research[split_idx:],
            )
            research_gbm_metrics = prod_research_gbm.metrics()

            # Train_ic — Pearson IC on the same training split the GBM saw.
            # train_ic >> val_ic (e.g. >2×) signals overfitting; emit both
            # in the manifest so monitoring can flag a degenerating gap.
            train_preds = prod_research_gbm.predict(X_research[:split_idx])
            train_y = y_research[:split_idx]
            if np.std(train_preds) > 1e-10 and np.std(train_y) > 1e-10:
                research_gbm_train_ic = float(np.corrcoef(train_preds, train_y)[0, 1])
            log.info(
                "ResearchGBMScorer fit on %d rows (train_ic=%.4f, val_ic=%.4f, "
                "best_iter=%d) — CANONICAL source for research_calibrator_prob "
                "(Phase 3 PR 4/5 cutover)",
                n_finite,
                research_gbm_train_ic if research_gbm_train_ic is not None else float("nan"),
                prod_research_gbm._val_ic,
                prod_research_gbm._best_iteration,
            )

            # 2026-05-19 (ROADMAP L1816): formalize the train/val IC ratio
            # as a first-class manifest field + warn flag + CW gauge.
            #
            # Ratio definition: train_ic / val_ic, sign-agnostic via abs
            # in the denominator. Threshold 3.0 is the half-way between
            # the "healthy" 2× watch-level the existing docs cite and the
            # 5.05× we observed on the 21d retrain, so a fresh > 3 cycle
            # would be unambiguously degraded — operator alerts only on
            # confirmed regression, not on noisy borderline ratios.
            (
                research_gbm_train_val_ic_ratio,
                research_gbm_overfit_warn,
            ) = _compute_overfit_signal(
                train_ic=research_gbm_train_ic,
                val_ic=float(prod_research_gbm._val_ic),
                warn_threshold=3.0,
            )
            if research_gbm_overfit_warn:
                log.warning(
                    "ResearchGBMScorer overfit signal — train_ic=%.4f / "
                    "val_ic=%.4f → ratio=%.2f (> 3.0). Manifest emits the "
                    "ratio + warn flag; CloudWatch carries the gauge for "
                    "the 2-cycle persistence alarm. The Ridge's regularization "
                    "still absorbs the overfit research scorer today so this "
                    "is observability, not blocking.",
                    research_gbm_train_ic or float("nan"),
                    prod_research_gbm._val_ic,
                    research_gbm_train_val_ic_ratio or float("nan"),
                )
            _emit_research_gbm_overfit_metrics(
                train_ic=research_gbm_train_ic,
                val_ic=float(prod_research_gbm._val_ic),
                ratio=research_gbm_train_val_ic_ratio,
                warn=research_gbm_overfit_warn,
            )

            # Recompute research_calibrator_prob in oos_meta_rows using the
            # GBM's predictions (in-sample for the rows the GBM trained on,
            # OOS for any non-finite-label rows that weren't part of the fit
            # set). The meta-Ridge below now sees GBM-sourced probs — same
            # contract inference will see.
            n_recomputed = 0
            for row in oos_meta_rows:
                row["research_calibrator_prob"] = float(
                    prod_research_gbm.predict_from_dict({
                        f: float(row.get(f, 0.0) or 0.0)
                        for f in RESEARCH_GBM_FEATURES
                    })
                )
                n_recomputed += 1
            log.info(
                "research_calibrator_prob recomputed via GBM for %d oos_meta_rows "
                "(in-sample for ~80%%, OOS-fitted for ~20%% per the GBM's "
                "temporal early-stop split)",
                n_recomputed,
            )
        else:
            log.info(
                "ResearchGBMScorer: only %d finite-label rows (need >=100) — "
                "skipped this cycle. Bucket-lookup ResearchCalibrator-sourced "
                "research_calibrator_prob remains in oos_meta_rows.",
                n_finite,
            )
    except Exception as e:
        # Non-blocking: failure here logs a warning but doesn't fail the
        # whole training run. The bucket-lookup-sourced research_calibrator_prob
        # already populated by the WF loop survives.
        log.warning(
            "ResearchGBMScorer fit failed (non-blocking, falling back to "
            "bucket-lookup-sourced research_calibrator_prob): %s", e,
        )

    # ── Step 7: Train meta-model on pooled OOS ───────────────────────────────
    # Audit Track A PR 5/6 cutover (2026-05-09): the meta-Ridge now trains on
    # canonical alpha labels (`actual_fwd_canonical` — log-domain risk-matched
    # vs vol-cohort EW basket), retiring the legacy `actual_fwd` arithmetic
    # target. Today's smoke (legacy still canonical) showed canonical Ridge
    # at val_ic≈0.19 on canonical labels vs legacy Ridge at val_ic≈0.07 on
    # legacy labels — 2.6× stronger meta-Ridge under the canonical target.
    #
    # Filter to rows with finite canonical labels (NaN for tickers that were
    # unclassifiable into a vol cohort, or cohorts with <2 members). The
    # full meta_X is preserved as `meta_X_all` for downstream diagnostics
    # that span all rows; the Ridge fit + isotonic + per-regime stack all
    # consume the canonical-finite subset.
    # W2 (L4469) OBSERVE gate: the residual-momentum column enters the L2 stack
    # ONLY when cfg.RESIDUAL_MOMENTUM_ENABLED. When False, TRAIN_META_FEATURES is
    # IDENTICAL to META_FEATURES → meta_X columns, the fitted/persisted
    # meta_model._feature_names, and the inference path are all byte-identical to
    # the pre-W2 trainer. We never mutate the META_FEATURES module constant
    # (inference imports it). Promotion = flip the flag after 2-3 observe firings.
    _resid_mom_enabled = bool(getattr(cfg, "RESIDUAL_MOMENTUM_ENABLED", False))
    TRAIN_META_FEATURES = build_train_meta_features(_resid_mom_enabled)
    log.info(
        "Training meta-model on %d OOS rows... (residual_momentum L1 in stack: %s)",
        len(oos_meta_rows), _resid_mom_enabled,
    )
    meta_X_all = np.array([[r[f] for f in TRAIN_META_FEATURES] for r in oos_meta_rows])
    meta_y_all = np.array([r["actual_fwd"] for r in oos_meta_rows])  # legacy, kept for diagnostics
    canonical_y_full = np.array([
        r.get("actual_fwd_canonical", float("nan")) for r in oos_meta_rows
    ])
    canonical_finite_mask = np.isfinite(canonical_y_full)
    n_canonical = int(canonical_finite_mask.sum())
    if n_canonical < 100:
        raise RuntimeError(
            f"Track A cutover: only {n_canonical} OOS rows have finite "
            "actual_fwd_canonical labels (need ≥100). Check Step 3b's "
            "canonical-alpha computation — vol-cohort assignment may have "
            "failed for the majority of tickers, or the canonical-alpha "
            "module raised non-fatally and skipped the population."
        )
    meta_X = meta_X_all[canonical_finite_mask]
    meta_y = canonical_y_full[canonical_finite_mask]
    meta_model = MetaModel(alpha=1.0)
    meta_model.fit(meta_X, meta_y, feature_names=TRAIN_META_FEATURES)
    log.info(
        "Meta-Ridge fit on canonical labels: n_canonical=%d / n_total=%d "
        "(val_ic_canonical=%.4f)",
        n_canonical, len(oos_meta_rows), meta_model._val_ic,
    )

    # ── W1.1a (ROADMAP L4469): leak-free walk-forward OOS meta IC (OBSERVE) ──
    # The val_ic just logged (and the manifest's `meta_model_oos_ic`) are
    # IN-SAMPLE at the META level: the L2 (BayesianRidge) is fit on the full
    # stack of OOS-L1 rows, then its IC is read back on that SAME stack
    # (`meta_preds_oos_insample` below). The L1 predictions are OOS but the L2
    # aggregation is not, so the number is inflated (~0.50 on 2026-05-30) vs
    # the Gu-Kelly-Xiu realistic ceiling (IC ~0.03-0.07). This runs a SECOND-
    # level purged(+embargoed) expanding walk-forward over the meta rows and
    # reports the CROSS-SECTIONAL rank IC (per-date Spearman averaged — the
    # Fama-MacBeth / GKX standard) with a date-blocked bootstrap CI. OBSERVE
    # ONLY: it is reported in the manifest, NOT used to gate promotion. L4469
    # W1.4 flips the gate from the in-sample _val_ic to this number after 2-3
    # Saturday observe firings. See training/leakfree_meta_ic.py.
    leakfree_meta_ic = {"status": "not_run"}
    cpcv_meta_ic = {"status": "not_run"}
    promotion_stats = {"status": "not_run"}
    try:
        from training.leakfree_meta_ic import (
            cpcv_meta_oos_ic,
            leakfree_meta_oos_ic,
        )
        _meta_dates = [
            r.get("date")
            for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
        ]

        def _meta_fit_predict(_Xtr, _ytr, _Xte):
            _m = MetaModel(alpha=1.0)
            _m.fit(_Xtr, _ytr, feature_names=TRAIN_META_FEATURES)
            return _m.predict(_Xte).ravel()

        leakfree_meta_ic = leakfree_meta_oos_ic(
            meta_X, meta_y, _meta_dates,
            fit_predict_fn=_meta_fit_predict,
            forward_days=cfg.FORWARD_DAYS,
            embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
            bootstrap_fn=lambda p, y, d: _bootstrap_ic_ci_by_date(
                p, y, d, n_iter=1000, ci=0.95, seed=4469),
        )
        log.info(
            "W1.1a leak-free meta OOS IC (OBSERVE, NOT gated): xsec=%s "
            "pooled=%s [%s, %s] over %s folds / %s rows / %s dates — vs "
            "IN-SAMPLE _val_ic=%.4f. Expect xsec << _val_ic (leakage); the "
            "drop is the honest number, not a regression. L4469 W1.4 flips "
            "the gate after observe firings.",
            leakfree_meta_ic.get("xsec_ic"), leakfree_meta_ic.get("pooled_ic"),
            leakfree_meta_ic.get("pooled_ci_lo"),
            leakfree_meta_ic.get("pooled_ci_hi"),
            leakfree_meta_ic.get("n_folds"), leakfree_meta_ic.get("n"),
            leakfree_meta_ic.get("n_dates"), meta_model._val_ic,
        )

        # W1.2 (L4469, OBSERVE): combinatorial purged CV — a DISTRIBUTION of
        # leak-free cross-sectional OOS ICs (the single-path WF above tests one
        # easily-overfit path). The distribution is the input to W1.3's
        # Deflated-Sharpe / PBO promotion gate. OBSERVE ONLY.
        cpcv_meta_ic = cpcv_meta_oos_ic(
            meta_X, meta_y, _meta_dates,
            fit_predict_fn=_meta_fit_predict,
            forward_days=cfg.FORWARD_DAYS,
            embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
            n_groups=getattr(cfg, "WF_CPCV_N_GROUPS", 6),
            k_test=getattr(cfg, "WF_CPCV_K_TEST", 2),
        )
        log.info(
            "W1.2 CPCV meta OOS IC (OBSERVE, NOT gated): mean=%s std=%s "
            "[p05=%s p50=%s p95=%s] frac_pos=%s over %s combos / %s paths "
            "(N=%s,k=%s). Distribution feeds the W1.3 DSR/PBO gate.",
            cpcv_meta_ic.get("mean_ic"), cpcv_meta_ic.get("std_ic"),
            cpcv_meta_ic.get("p05_ic"), cpcv_meta_ic.get("p50_ic"),
            cpcv_meta_ic.get("p95_ic"), cpcv_meta_ic.get("frac_positive"),
            cpcv_meta_ic.get("n_combos"), cpcv_meta_ic.get("n_backtest_paths"),
            cpcv_meta_ic.get("n_groups"), cpcv_meta_ic.get("k_test"),
        )

        # W1.3 (L4469, OBSERVE): TWO lenses on the leak-free per-date IC series.
        # (1) DOWNSIDE-AWARE PERFORMANCE — Sortino-of-IC + CVaR-of-IC, matching
        #     the house skilled-risk basket (Sortino + CVaR + maxDD; Sharpe is
        #     RETIRED to legacy, anchor-gates-on-skilled-risk-not-sharpe). This
        #     is the headline "is the skill good + downside-robust" lens.
        # (2) OVERFIT-SIGNIFICANCE — PSR/Deflated-Sharpe (Bailey-LdP): is the IC
        #     genuinely non-zero vs multiple-testing false discovery? PSR is
        #     itself in the basket. IC-IR here is symmetric BY DESIGN (a
        #     significance test, not the performance metric).
        # NOT gated. W1.4 composes BOTH (real AND downside-robust). PBO/CSCV
        # deferred to W1.3b (needs a config grid). See training/deflated_sharpe.
        from training.deflated_sharpe import (
            deflated_sharpe_ratio,
            downside_ic_stats,
        )
        _ic_series = (
            leakfree_meta_ic.get("ic_series", [])
            if isinstance(leakfree_meta_ic, dict) else []
        )
        # n_trials: the CPCV combination count is a conservative within-run
        # trial proxy until cumulative cross-week trial-count tracking lands
        # (W1.3b). max(.,WF_DSR_N_TRIALS) so a degenerate CPCV doesn't zero the
        # deflation.
        _n_trials = max(
            int(cpcv_meta_ic.get("n_combos", 1) or 1),
            int(getattr(cfg, "WF_DSR_N_TRIALS", 10)),
        )
        _downside = downside_ic_stats(
            _ic_series,
            sortino_threshold=float(getattr(cfg, "WF_SORTINO_IC_THRESHOLD", 0.0)),
        )
        _overfit = deflated_sharpe_ratio(
            _ic_series, n_trials=_n_trials,
            threshold=float(getattr(cfg, "WF_DSR_THRESHOLD", 0.95)),
        )
        # Headline = the downside-aware performance lens; DSR/PSR is the
        # significance leg. Structured so W1.4 can gate on both legs.
        promotion_stats = {
            "status": _overfit.get("status"),
            "downside": _downside,
            "overfit": _overfit,
        }
        log.info(
            "W1.3 promotion battery (OBSERVE, NOT gated) — DOWNSIDE "
            "(skilled-risk basket): Sortino-of-IC=%s CVaR%s%%-of-IC=%s "
            "frac_neg_IC=%s worst=%s | OVERFIT (significance): PSR(0)=%s DSR=%s "
            "(n_trials=%s) | passes_downside=%s passes_overfit=%s. W1.4 gates "
            "on BOTH; Sortino/CVaR are the house metrics, IC-IR is symmetric "
            "significance only.",
            _downside.get("sortino_of_ic"), _downside.get("cvar_pct"),
            _downside.get("cvar_of_ic"), _downside.get("frac_negative_ic"),
            _downside.get("worst_ic"), _overfit.get("psr_vs_zero"),
            _overfit.get("dsr"), _overfit.get("n_trials"),
            _downside.get("passes_downside_gate"),
            _overfit.get("passes_overfit_gate"),
        )
    except Exception as _e:  # observe-only diagnostic must never fail training
        log.warning(
            "W1.1a/W1.2/W1.3 leak-free meta OOS IC failed (OBSERVE, "
            "non-fatal): %s", _e)
        if leakfree_meta_ic.get("status") == "not_run":
            leakfree_meta_ic = {"status": "error", "error": str(_e)}
        if cpcv_meta_ic.get("status") == "not_run":
            cpcv_meta_ic = {"status": "error", "error": str(_e)}
        if promotion_stats.get("status") == "not_run":
            promotion_stats = {"status": "error", "error": str(_e)}

    # ── W2 (L4469): standalone leak-free read of the residual-momentum L1 ────
    # The scorer is DETERMINISTIC (fit-free), so its honest skill is the
    # CROSS-SECTIONAL rank IC (Fama-MacBeth, per-date Spearman) of its
    # predictions vs the canonical label — no train/test split to leak through.
    # Same machinery + downside/DSR legs as the W1 meta read, over the SAME
    # canonical-finite rows for parity. This is the standalone "isolated
    # promotion gate" number W2's acceptance criterion needs. OBSERVE ONLY:
    # emitted to the manifest, enters NO promotion gate, and runs REGARDLESS of
    # cfg.RESIDUAL_MOMENTUM_ENABLED.
    residual_momentum_leakfree = {"status": "not_run"}
    try:
        from training.leakfree_meta_ic import cross_sectional_ic_series
        from training.deflated_sharpe import (
            deflated_sharpe_ratio as _rm_dsr,
            downside_ic_stats as _rm_dstats,
        )
        _rm_preds = residual_momentum_scorer.predict_array(
            X_resid_mom_raw[canonical_finite_mask], RESIDUAL_MOMENTUM_FEATURES,
        )
        _rm_dates = [
            r.get("date")
            for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
        ]
        _rm_ic_series = cross_sectional_ic_series(_rm_preds, meta_y, _rm_dates)
        _rm_xsec = float(np.mean(_rm_ic_series)) if _rm_ic_series else float("nan")
        _rm_downside = _rm_dstats(
            _rm_ic_series,
            sortino_threshold=float(getattr(cfg, "WF_SORTINO_IC_THRESHOLD", 0.0)),
        )
        _rm_overfit = _rm_dsr(
            _rm_ic_series,
            n_trials=int(getattr(cfg, "WF_DSR_N_TRIALS", 10)),
            threshold=float(getattr(cfg, "WF_DSR_THRESHOLD", 0.95)),
        )
        residual_momentum_leakfree = {
            "status": "ok" if _rm_ic_series else "insufficient_dates",
            "xsec_ic": round(_rm_xsec, 6) if _rm_ic_series else None,
            "n": int(len(_rm_preds)),
            "n_dates": len(_rm_ic_series),
            "ic_series": [round(float(x), 6) for x in _rm_ic_series],
            "downside": _rm_downside,
            "overfit": _rm_overfit,
            "enabled_in_stack": _resid_mom_enabled,
            "forward_days": cfg.FORWARD_DAYS,
            "embargo_days": int(getattr(cfg, "WF_EMBARGO_DAYS", 0)),
        }
        log.info(
            "W2 residual-momentum L1 leak-free read (OBSERVE, NOT gated): "
            "xsec_ic=%s over %s dates / %s rows | Sortino-of-IC=%s DSR=%s | "
            "in_stack=%s. vs dead-momentum baseline ~-0.001 and the realistic "
            "0.03-0.07 live-IC range.",
            residual_momentum_leakfree.get("xsec_ic"),
            residual_momentum_leakfree.get("n_dates"),
            residual_momentum_leakfree.get("n"),
            _rm_downside.get("sortino_of_ic"), _rm_overfit.get("dsr"),
            _resid_mom_enabled,
        )
    except Exception as _e:  # observe-only diagnostic must never fail training
        log.warning(
            "W2 residual-momentum leak-free read failed (OBSERVE, non-fatal): %s",
            _e)
        residual_momentum_leakfree = {"status": "error", "error": str(_e)}

    # ── W3.2 (L4469): leak-free per-HORIZON IC curve (OBSERVE) ───────────────
    # The operator's 5/21/60/90d question — where does the HONEST (leak-free)
    # cross-sectional IC peak? The existing `horizon_ics` block uses a POOLED
    # overlapping Spearman + a non-overlapping subsample; this runs the SAME W1
    # purged+embargoed expanding walk-forward (refit the L2 per fold on EACH
    # horizon's realized label, purge = h days), reporting the leak-free
    # cross-sectional rank IC + downside/DSR legs per horizon. Same meta_X /
    # canonical-finite row population across horizons so the curve is
    # apples-to-apples. OBSERVE ONLY: emitted to the manifest, gates NOTHING —
    # the canonical 21d target is unchanged. Gross IC alone does NOT settle the
    # horizon; net-of-cost judgment is W3.4 (turnover-adjusted alpha, deferred).
    horizon_ics_leakfree: dict[str, dict] = {}
    try:
        from training.deflated_sharpe import (
            deflated_sharpe_ratio as _h_dsr,
            downside_ic_stats as _h_dstats,
        )
        from training.leakfree_meta_ic import leakfree_horizon_ic_curve

        def _h_fit_predict(_Xtr, _ytr, _Xte):
            _m = MetaModel(alpha=1.0)
            _m.fit(_Xtr, _ytr, feature_names=TRAIN_META_FEATURES)
            return _m.predict(_Xte).ravel()

        _h_dates = [
            r.get("date")
            for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
        ]
        _h_labels = {
            _h: np.array([
                r.get(f"actual_fwd_{_h}d", float("nan"))
                for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
            ])
            for _h in _DIAGNOSTIC_HORIZONS
        }
        _raw_curve = leakfree_horizon_ic_curve(
            meta_X, _h_labels, _h_dates,
            fit_predict_fn=_h_fit_predict,
            embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
        )
        for _key, _lf in _raw_curve.items():
            _entry = {
                "status": _lf.get("status"),
                "xsec_ic": _lf.get("xsec_ic"),
                "pooled_ic": _lf.get("pooled_ic"),
                "n": _lf.get("n"),
                "n_dates": _lf.get("n_dates"),
                "n_folds": _lf.get("n_folds"),
                "forward_days": _lf.get("forward_days"),
            }
            _ic_series_h = _lf.get("ic_series", []) if isinstance(_lf, dict) else []
            if _ic_series_h:
                _entry["downside"] = _h_dstats(
                    _ic_series_h,
                    sortino_threshold=float(getattr(cfg, "WF_SORTINO_IC_THRESHOLD", 0.0)),
                )
                _entry["overfit"] = _h_dsr(
                    _ic_series_h,
                    n_trials=int(getattr(cfg, "WF_DSR_N_TRIALS", 10)),
                    threshold=float(getattr(cfg, "WF_DSR_THRESHOLD", 0.95)),
                )
            horizon_ics_leakfree[_key] = _entry
        _curve = {
            k: v["xsec_ic"] for k, v in horizon_ics_leakfree.items()
            if isinstance(v.get("xsec_ic"), (int, float)) and np.isfinite(v["xsec_ic"])
        }
        _peak = max(_curve, key=_curve.get) if _curve else None
        log.info(
            "W3.2 leak-free horizon IC curve (OBSERVE, NOT gated): %s | peak=%s "
            "vs canonical target=21d. Settle the horizon net-of-cost (W3.4) "
            "before any cutover — gross IC alone does not decide it.",
            {k: round(float(v), 4) for k, v in _curve.items()}, _peak,
        )
    except Exception as _e:  # observe-only diagnostic must never fail training
        log.warning("W3.2 leak-free horizon IC failed (OBSERVE, non-fatal): %s", _e)
        horizon_ics_leakfree = {"status": "error", "error": str(_e)}

    # ── W4 watch + L4469 P2 (OBSERVE): is the leak-free meta IC real ALPHA, ──
    # and does pre-2020 data drag it? Three cheap leak-free reads, all reusing
    # the W1 machinery, all gating NOTHING:
    #   (A)  per-input standalone cross-sectional alpha-IC — surfaces whether
    #        `expected_move` (the meta's dominant coef, a VOL L1) actually
    #        predicts cross-sectional ALPHA or is in-sample dominance only.
    #   (A2) leave-one-out leak-free meta IC dropping `expected_move` — if the
    #        leak-free meta IC barely moves, expected_move isn't carrying alpha.
    #   (B)  leak-free meta IC on post-2020-03-31 rows vs full history — settles
    #        the standing "pre-2020 may be noise" P2 in ONE run (no extra cycle).
    # Self-contained (own dates + fit_predict) so a failure in the W1/W3 blocks
    # above can't take it down. The canonical 21d target is unchanged.
    meta_l1_standalone_alpha_ic: dict = {"status": "not_run"}
    meta_oos_ic_leakfree_no_expected_move: dict = {"status": "not_run"}
    meta_oos_ic_leakfree_post2020: dict = {"status": "not_run"}
    try:
        from training.leakfree_meta_ic import (
            leakfree_meta_oos_ic as _d_lf,
            per_feature_standalone_ic as _d_pfsi,
        )

        _d_dates = [
            r.get("date")
            for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
        ]

        def _d_fit_predict(_Xtr, _ytr, _Xte, _names=TRAIN_META_FEATURES):
            _m = MetaModel(alpha=1.0)
            _m.fit(_Xtr, _ytr, feature_names=_names)
            return _m.predict(_Xte).ravel()

        # (A) per-input standalone alpha-IC
        meta_l1_standalone_alpha_ic = _d_pfsi(
            meta_X, meta_y, _d_dates, TRAIN_META_FEATURES,
        )

        # (A2) leave-one-out: drop expected_move from the L2
        if "expected_move" in TRAIN_META_FEATURES:
            _keep = [f for f in TRAIN_META_FEATURES if f != "expected_move"]
            _keep_idx = [
                i for i, f in enumerate(TRAIN_META_FEATURES) if f != "expected_move"
            ]
            meta_oos_ic_leakfree_no_expected_move = _d_lf(
                meta_X[:, _keep_idx], meta_y, _d_dates,
                fit_predict_fn=lambda a, b, c, _n=_keep: _d_fit_predict(a, b, c, _names=_n),
                forward_days=cfg.FORWARD_DAYS,
                embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
            )

        # (B) pre-2020 clamp A/B (lexical ISO-date compare; ≥200 post rows).
        _post = np.array([
            (d is not None and str(d)[:10] >= "2020-03-31") for d in _d_dates
        ])
        if int(_post.sum()) >= 200:
            meta_oos_ic_leakfree_post2020 = _d_lf(
                meta_X[_post], meta_y[_post],
                [d for d, m in zip(_d_dates, _post) if m],
                fit_predict_fn=_d_fit_predict,
                forward_days=cfg.FORWARD_DAYS,
                embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
            )

        _em = meta_l1_standalone_alpha_ic.get("expected_move", {}) if isinstance(
            meta_l1_standalone_alpha_ic, dict) else {}
        log.info(
            "W4/P2 observe (NOT gated): expected_move standalone alpha-IC=%s | "
            "meta leak-free full xsec=%s / drop-expected_move=%s / post-2020=%s. "
            "If full≈drop-EM and EM-standalone≈0, the headline IC is NOT EM-alpha.",
            _em.get("xsec_ic"),
            leakfree_meta_ic.get("xsec_ic") if isinstance(leakfree_meta_ic, dict) else None,
            meta_oos_ic_leakfree_no_expected_move.get("xsec_ic"),
            meta_oos_ic_leakfree_post2020.get("xsec_ic"),
        )
    except Exception as _e:  # observe-only diagnostic must never fail training
        log.warning("W4/P2 observe diagnostics failed (OBSERVE, non-fatal): %s", _e)
        meta_l1_standalone_alpha_ic = {"status": "error", "error": str(_e)}

    # ── W4.1 (L4469, OBSERVE): nonlinear-L2 SHADOW blender ───────────────────
    # W1 established the live meta is a LINEAR BayesianRidge. Gu-Kelly-Xiu: a
    # nonlinear blender materially beats OLS on the SAME factor inputs (NN
    # Sharpe 1.35 vs OLS 0.61). This measures — SHADOW, live L2 UNCHANGED —
    # whether a LightGBM meta recovers more leak-free IC from the identical L1
    # outputs, via the SAME purged+embargoed expanding WF + meta_X / folds as
    # the W1 linear read (so `meta_oos_ic_leakfree_nonlinear.xsec_ic` vs
    # `meta_model_oos_ic_leakfree.xsec_ic` is apples-to-apples). Tells us whether
    # the next move is "better L1s" or "nonlinear meta." OBSERVE; gates nothing.
    meta_oos_ic_leakfree_nonlinear: dict = {"status": "not_run"}
    try:
        from training.leakfree_meta_ic import (
            gbm_blender_fit_predict,
            leakfree_meta_oos_ic as _nl_lf,
        )

        _nl_dates = [
            r.get("date")
            for r, _m in zip(oos_meta_rows, canonical_finite_mask) if _m
        ]
        meta_oos_ic_leakfree_nonlinear = _nl_lf(
            meta_X, meta_y, _nl_dates,
            fit_predict_fn=gbm_blender_fit_predict(),
            forward_days=cfg.FORWARD_DAYS,
            embargo_days=getattr(cfg, "WF_EMBARGO_DAYS", 0),
        )
        log.info(
            "W4.1 nonlinear-L2 shadow leak-free meta IC (OBSERVE, NOT gated): "
            "nonlinear xsec=%s vs linear xsec=%s. Does a nonlinear blender "
            "recover more leak-free IC from the SAME L1s (GKX NN>>OLS)? Live L2 "
            "(BayesianRidge) UNCHANGED — this only informs the next architecture move.",
            meta_oos_ic_leakfree_nonlinear.get("xsec_ic"),
            leakfree_meta_ic.get("xsec_ic") if isinstance(leakfree_meta_ic, dict) else None,
        )
    except Exception as _e:  # observe-only diagnostic must never fail training
        log.warning("W4.1 nonlinear-L2 shadow failed (OBSERVE, non-fatal): %s", _e)
        meta_oos_ic_leakfree_nonlinear = {"status": "error", "error": str(_e)}

    # Parity diagnostic: canonical Ridge predictions vs legacy labels
    # (cross-target). Useful for monitoring whether the cutover degrades
    # the historical reporting metric. Same 4-cell-grid spirit as Track A
    # PR 3's observe-only diagnostic; post-cutover we only retain the
    # canonical→legacy cross-IC because the legacy-Ridge half is gone.
    legacy_y_subset = meta_y_all[canonical_finite_mask]
    mask_finite_legacy = np.isfinite(legacy_y_subset)
    if mask_finite_legacy.sum() >= 50:
        canonical_preds_for_diag = meta_model.predict(meta_X).ravel()
        ic_canonical_ridge_vs_legacy = float(np.corrcoef(
            canonical_preds_for_diag[mask_finite_legacy],
            legacy_y_subset[mask_finite_legacy],
        )[0, 1])
    else:
        ic_canonical_ridge_vs_legacy = float("nan")
    log.info(
        "Canonical-vs-legacy diagnostic: ic_canonical_ridge_vs_legacy=%s "
        "(post-cutover monitoring; canonical is canonical)",
        f"{ic_canonical_ridge_vs_legacy:+.4f}"
        if np.isfinite(ic_canonical_ridge_vs_legacy) else "n/a",
    )

    # ── Step 7b: Parallel triple-barrier L2 Ridge (Stage 3 PR 2, observe-only) ─
    # Audit Phase 4 / regime-conditioning rebuild Stage 3: fit a parallel
    # Ridge on the same META_FEATURES but supervised on the triple-barrier
    # alpha label (vol-scaled barriers per LdP Ch. 3.4). Sample weights
    # follow LdP Ch. 4.4: average-uniqueness weights computed per-ticker
    # so heavily-overlapping consecutive-day labels for the same ticker
    # don't double-count their gradient contribution. Inference (PR 3)
    # adds the parallel field; cutover (PR 5) flips `enforce_cutover`
    # after ≥3 Saturday SF firings of variant_cutover_gate show ≥15%
    # relative IC lift over the canonical fixed-21d Ridge.
    #
    # The parallel fit operates on the FULL `oos_meta_rows` set (before
    # the canonical-finite filter below) since triple-barrier finiteness
    # is a different mask than canonical finiteness — front-of-history
    # vol-warmup vs cohort assignability.
    from labeling.sample_weights import average_uniqueness_weights
    meta_model_tb: MetaModel | None = None
    tb_y_full = np.array([
        r.get("actual_fwd_triple_barrier", float("nan")) for r in oos_meta_rows
    ])
    tb_finite_mask = np.isfinite(tb_y_full)
    n_tb_finite = int(tb_finite_mask.sum())
    if n_tb_finite >= 100:
        tb_X = meta_X_all[tb_finite_mask]
        tb_y = tb_y_full[tb_finite_mask]
        # LdP Ch. 4.4 average-uniqueness weights — per-ticker so different
        # tickers don't pollute each other's concurrency. Window-start is
        # the row's date in trading-day-position space; window length is
        # the triple-barrier forward window.
        tb_dates = np.array([
            r["date"] for r in oos_meta_rows if r is not None
        ])[tb_finite_mask]
        tb_tickers = np.array([
            r.get("ticker", "_unknown") for r in oos_meta_rows
        ])[tb_finite_mask]
        # Convert pd.Timestamp dates to integer day positions (relative
        # to the earliest date) for the uniqueness algorithm.
        date_min_ts = pd.Timestamp(min(tb_dates))
        date_positions = np.array([
            int((pd.Timestamp(d) - date_min_ts).days) for d in tb_dates
        ], dtype=np.int64)
        tb_sample_weights = average_uniqueness_weights(
            window_starts=date_positions,
            window_length=cfg.TRIPLE_BARRIER_FORWARD_WINDOW,
            group_ids=tb_tickers,
        )
        meta_model_tb = MetaModel(alpha=1.0)
        meta_model_tb.fit(
            tb_X, tb_y,
            feature_names=META_FEATURES,
            sample_weight=tb_sample_weights,
        )
        log.info(
            "Triple-barrier Meta-Ridge fit (Stage 3 PR 2 observe-only): "
            "n_tb_finite=%d / n_total=%d (val_ic_tb=%.4f)",
            n_tb_finite, len(oos_meta_rows), meta_model_tb._val_ic,
        )
        # Cross-target diagnostic: IC of TB Ridge predictions against the
        # canonical label, parallel to the canonical-vs-legacy diagnostic
        # above. Tells us whether the TB Ridge ranks tickers similarly
        # to the canonical Ridge on the canonical-grading axis.
        canonical_y_for_tb = canonical_y_full[tb_finite_mask]
        canonical_finite_for_tb = np.isfinite(canonical_y_for_tb)
        if canonical_finite_for_tb.sum() >= 50:
            tb_preds_for_diag = meta_model_tb.predict(tb_X).ravel()
            ic_tb_ridge_vs_canonical = float(np.corrcoef(
                tb_preds_for_diag[canonical_finite_for_tb],
                canonical_y_for_tb[canonical_finite_for_tb],
            )[0, 1])
            log.info(
                "Triple-barrier-vs-canonical diagnostic: ic_tb_ridge_vs_canonical=%+.4f "
                "(observe-only — variant_cutover_gate uses prediction-pairs vs "
                "actual realized alpha, not this in-sample cross-IC)",
                ic_tb_ridge_vs_canonical,
            )
    else:
        log.info(
            "Triple-barrier Meta-Ridge: only %d finite-label rows (need >=100) — "
            "skipped this cycle. Canonical Ridge unaffected.",
            n_tb_finite,
        )

    # ── Step 7c: Meta-label classifier (Task B, observe-only) ───────────────
    # López de Prado meta-labeling: a calibrated classifier on the SAME
    # META_FEATURES, supervised on the triple-barrier touch-order label
    # (1=up/profit barrier first, 0=down/stop first; timeouts are NaN and
    # excluded). Its calibrated P(up before down) is emitted by inference as
    # the observe-only ``barrier_win_prob`` field; the executor sizing
    # consumer (Task B2) stays dormant until a cutover gate clears. Mirrors
    # the Step 7b observe-only lifecycle: fit on the FULL oos_meta_rows set
    # (touch-finiteness is its own mask), persist if fit succeeds, never
    # gate the canonical path. Behind ``META_LABEL_CLASSIFIER_ENABLED``
    # (default on; observe-only so safe) for an operator kill-switch.
    meta_label_clf = None
    if getattr(cfg, "META_LABEL_CLASSIFIER_ENABLED", True):
        from model.meta_label_classifier import MetaLabelClassifier
        touch_y_full = np.array([
            r.get("actual_fwd_barrier_touch", float("nan")) for r in oos_meta_rows
        ])
        touch_finite_mask = np.isfinite(touch_y_full)
        n_touch_finite = int(touch_finite_mask.sum())
        if n_touch_finite >= 100:
            touch_X = meta_X_all[touch_finite_mask]
            touch_y = touch_y_full[touch_finite_mask]
            # LdP Ch. 4.4 average-uniqueness weights, per-ticker (same as 7b).
            touch_dates = np.array([
                r["date"] for r in oos_meta_rows if r is not None
            ])[touch_finite_mask]
            touch_tickers = np.array([
                r.get("ticker", "_unknown") for r in oos_meta_rows
            ])[touch_finite_mask]
            t_date_min = pd.Timestamp(min(touch_dates))
            t_positions = np.array([
                int((pd.Timestamp(d) - t_date_min).days) for d in touch_dates
            ], dtype=np.int64)
            touch_weights = average_uniqueness_weights(
                window_starts=t_positions,
                window_length=cfg.TRIPLE_BARRIER_FORWARD_WINDOW,
                group_ids=touch_tickers,
            )
            meta_label_clf = MetaLabelClassifier()
            meta_label_clf.fit(
                touch_X, touch_y,
                feature_names=META_FEATURES,
                sample_weight=touch_weights,
            )
            log.info(
                "Meta-label classifier fit (Task B observe-only): "
                "n_touch_finite=%d / n_total=%d (AUC=%.4f, base_rate=%.3f)",
                n_touch_finite, len(oos_meta_rows),
                meta_label_clf._auc, meta_label_clf._base_rate,
            )
            if not meta_label_clf.is_fitted:
                # fit() skipped internally (e.g. single-class subsample) —
                # don't persist an unfitted shell.
                meta_label_clf = None
        else:
            log.info(
                "Meta-label classifier: only %d touch-finite rows (need >=100) "
                "— skipped this cycle. Canonical path unaffected.",
                n_touch_finite,
            )

    # Filter oos_meta_rows to canonical-finite rows so downstream consumers
    # (horizon battery, row_regimes, isotonic calibrator) align naturally
    # with meta_X / meta_y / Ridge predictions.
    # Pre-cutover the Ridge fit on the full row set; post-cutover the fit
    # is on the canonical-finite subset and every downstream operation
    # against Ridge outputs must use the same subset.
    oos_meta_rows = [
        r for r, ok in zip(oos_meta_rows, canonical_finite_mask) if ok
    ]

    # Step 7e (parallel canonical-label Ridge, audit Track A PR 3) retired
    # 2026-05-09 (Track A PR 5/6 cutover). The single Ridge above NOW
    # trains on canonical labels; there is no parallel "legacy Ridge" to
    # compare against. Post-cutover monitoring uses
    # ``ic_canonical_ridge_vs_legacy`` emitted from the diagnostic above
    # (one cross-target IC; the 4-cell grid collapsed to 1 cell since
    # canonical-vs-canonical IS the production val_ic).

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
    # Track 3-of-N of the audit Phase 1 horizon battery: classify each row's
    # market regime so per-horizon ICs can be sliced by regime. Heuristic
    # is bull/neutral/bear via macro_spy_20d_return thresholds at ±3% (see
    # _classify_regime docstring for rationale + future-replacement plan).
    row_regimes = [_classify_regime(r) for r in oos_meta_rows]
    regime_counts = {
        "bull": row_regimes.count("bull"),
        "neutral": row_regimes.count("neutral"),
        "bear": row_regimes.count("bear"),
    }
    log.info(
        "Per-row regime distribution: bull=%d  neutral=%d  bear=%d  total=%d",
        regime_counts["bull"], regime_counts["neutral"], regime_counts["bear"],
        len(row_regimes),
    )
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

        # Per-regime IC slice (Track B 3-of-N). Compute IC within each of
        # bull/neutral/bear regime buckets — operator can see whether the
        # cross-horizon IC ratio is regime-dependent (e.g. 21d signal is
        # 3× 5d in bull, 1.2× in bear) or stable (regime-invariant). Min
        # sample size is 30 per regime — below that the IC is too noisy
        # to interpret. NaN signals "insufficient sample in this regime";
        # downstream consumers should treat the regime as unmeasured at
        # this horizon rather than confusing low-sample noise with signal.
        regime_ics: dict = {}
        for regime in ("bull", "neutral", "bear"):
            r_mask = np.array([rr == regime for rr in row_regimes]) & mask
            if r_mask.sum() >= 30:
                sp_r = spearmanr(meta_preds_oos_insample[r_mask], y_h[r_mask])
                ic_r = float(sp_r.correlation) if np.isfinite(sp_r.correlation) else float("nan")
            else:
                ic_r = float("nan")
            regime_ics[regime] = {
                "spearman": ic_r,
                "n": int(r_mask.sum()),
            }

        horizon_ics[f"{h}d"] = {
            "spearman": ic,
            "n": int(mask.sum()),
            "spearman_ci_lo": ic_ci_lo,
            "spearman_ci_hi": ic_ci_hi,
            "spearman_nonoverlap": nonoverlap_ic,
            "n_nonoverlap": nonoverlap_n,
            "spearman_nonoverlap_ci_lo": nonoverlap_ci_lo,
            "spearman_nonoverlap_ci_hi": nonoverlap_ci_hi,
            "by_regime": regime_ics,
        }

    # Audit Track A PR 2/6 (2026-05-07): canonical-label IC diagnostic.
    # Computes Pearson correlation of meta_preds_oos_insample (single-Ridge
    # Track A PR 5/6 cutover (2026-05-09): the legacy-Ridge-vs-canonical
    # IC diagnostic that lived here became vacuous post-cutover (the
    # single Ridge IS canonical-trained; "single_ridge_ic_vs_canonical"
    # is the same as meta_model._val_ic). The legacy parity comparison
    # (``ic_canonical_ridge_vs_legacy``) now ships out of Step 7 above
    # in `track_a_canonical_diagnostic` for cross-cycle trend monitoring.
    canonical_ic_metrics: dict = {
        "computed": True,
        "n_finite_canonical_labels": n_canonical,
        "n_total_rows": n_canonical,  # post-cutover oos_meta_rows is canonical-filtered
        "single_ridge_ic_vs_canonical": round(meta_model._val_ic, 6),
        "single_ridge_ic_vs_legacy": (
            round(ic_canonical_ridge_vs_legacy, 6)
            if np.isfinite(ic_canonical_ridge_vs_legacy) else None
        ),
        "post_cutover_note": (
            "Post Track A PR 5/6 cutover: meta_model trains on canonical "
            "labels; single_ridge_ic_vs_canonical IS meta_model._val_ic. "
            "single_ridge_ic_vs_legacy is the cross-target diagnostic for "
            "cross-cycle trend monitoring."
        ),
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
    # Per-regime breakdown — one log line per regime so operators can
    # eyeball whether the cross-horizon ratio is regime-dependent.
    for regime in ("bull", "neutral", "bear"):
        _r_line = "  ".join(
            f"{k}={v['by_regime'][regime]['spearman']:+.4f} (n={v['by_regime'][regime]['n']})"
            if np.isfinite(v["by_regime"][regime]["spearman"])
            else f"{k}=— (n={v['by_regime'][regime]['n']})"
            for k, v in horizon_ics.items()
        )
        log.info("Horizon IC curve (%s regime): %s", regime, _r_line)
    # Flag peak horizon for quick visual in logs/email.
    _peak = max(horizon_ics.items(),
                key=lambda kv: abs(kv[1]["spearman"]) if np.isfinite(kv[1]["spearman"]) else -1)
    log.info("Peak IC horizon: %s (Spearman=%+.4f)", _peak[0], _peak[1]["spearman"])

    # ── Step 7b: Fit direction calibrator on meta OOS predictions ────────────
    # ROADMAP P1: collapse FLAT, use calibrated P(UP) as confidence.
    #
    # Input: continuous meta-model output on the pooled OOS rows. These
    # predictions are OOS with respect to the Layer-1 base models
    # (momentum/volatility/regime were held out by fold) but are
    # in-sample for the meta ridge itself, which was just fit on meta_X.
    # Pure nested CV would be cleaner; the pragmatic tradeoff is that
    # ridge has low capacity (8 coefficients) and overfits minimally, so
    # in-sample meta predictions are a reasonable calibration substrate.
    #
    # Output: P(actual_fwd > 0 | meta prediction). Binary target from
    # sign(meta_y), continuous input → calibrated probability (no sigmoid
    # wrapper needed for either method).
    #
    # Method is config-driven via cfg.CALIBRATION_METHOD (predictor.yaml
    # ``calibration.method``) — "platt" (L2-regularised logistic, the
    # tail-robust default) or "isotonic". ROADMAP L1873: the 2026-05-09
    # 75%+ DOWN-veto inversion was isotonic-tail overconfidence; "platt"
    # is the prescribed tail-robust fix. This call site previously
    # hardcoded "isotonic", silently ignoring the config (the knob was
    # dead — parsed at config.py but never read), so the committed
    # ``platt`` default never took effect. Honour the config here.
    #
    # Hard-fail per feedback_hard_fail_until_stable: a broken calibrator
    # silently degrades inference quality. Catch it here rather than
    # letting the inference path log a WARNING and fall back to heuristic.
    from model.calibrator import PlattCalibrator
    cal_method = cfg.CALIBRATION_METHOD
    oos_meta_preds = meta_model.predict(meta_X).ravel()
    oos_up_labels = (meta_y > 0).astype(np.int32)
    log.info(
        "Fitting %s direction calibrator: n=%d  up_rate=%.3f  pred_std=%.6f",
        cal_method, len(oos_meta_preds), float(oos_up_labels.mean()),
        float(np.std(oos_meta_preds)),
    )
    calibrator = PlattCalibrator(method=cal_method)
    calibrator.fit(
        oos_meta_preds, oos_up_labels,
        label_clip=float(cfg.LABEL_CLIP),
        class_weight=cfg.CALIBRATION_CLASS_WEIGHT,
        C=cfg.CALIBRATION_C,
    )
    if not calibrator.is_fitted:
        raise RuntimeError(
            f"{cal_method} direction calibrator did not fit "
            f"(n={len(oos_meta_preds)} samples, min required 100). Training "
            f"must produce a calibrator — see model/calibrator.py for the "
            f"sample threshold."
        )
    log.info(
        "%s direction calibrator: ECE_before=%.4f  ECE_after=%.4f  (%.1f%% reduction)",
        cal_method, calibrator._ece_before, calibrator._ece_after,
        (1 - calibrator._ece_after / max(calibrator._ece_before, 1e-8)) * 100,
    )

    # ── Step 7b-shadow: Fit shadow calibrator + run gates (observability) ────
    # Added 2026-05-24 (calibrator-shadow arc) in response to the 2026-05-23
    # Platt-collapse incident. When ``cfg.CALIBRATION_SHADOW_METHOD`` is set,
    # fit a second calibrator on the SAME rows / labels with potentially
    # different method + class_weight + C, run it through both promotion-gate
    # checks, and persist the result in the manifest. Pure observability:
    # the shadow calibrator does NOT feed inference, does NOT influence the
    # ``promoted`` boolean, and is NOT saved to the live calibrator S3 key.
    # Lets us iterate Platt's class-balance / regularisation knobs across
    # Saturday cycles while isotonic remains live, then flip live by swapping
    # ``calibration.method`` in predictor.yaml once shadow gates pass for ≥1
    # cycle. Skipped entirely when shadow_method is None (no extra work, no
    # extra log noise on operators who haven't opted in).
    shadow_calibrator = None
    shadow_method = cfg.CALIBRATION_SHADOW_METHOD
    if shadow_method:
        try:
            shadow_calibrator = PlattCalibrator(method=shadow_method)
            shadow_calibrator.fit(
                oos_meta_preds, oos_up_labels,
                label_clip=float(cfg.LABEL_CLIP),
                class_weight=cfg.CALIBRATION_SHADOW_CLASS_WEIGHT,
                C=cfg.CALIBRATION_SHADOW_C,
            )
            if shadow_calibrator.is_fitted:
                log.info(
                    "Shadow %s calibrator (class_weight=%r, C=%.2f): "
                    "ECE_before=%.4f  ECE_after=%.4f  (%.1f%% reduction)",
                    shadow_method, cfg.CALIBRATION_SHADOW_CLASS_WEIGHT,
                    cfg.CALIBRATION_SHADOW_C,
                    shadow_calibrator._ece_before, shadow_calibrator._ece_after,
                    (
                        1 - shadow_calibrator._ece_after
                        / max(shadow_calibrator._ece_before, 1e-8)
                    ) * 100,
                )
            else:
                log.warning(
                    "Shadow %s calibrator did not fit "
                    "(n=%d, min required 100) — shadow gate result will be n/a",
                    shadow_method, len(oos_meta_preds),
                )
                shadow_calibrator = None
        except Exception as e:
            log.warning(
                "Shadow calibrator fit failed (non-blocking — live calibrator "
                "unaffected): %s",
                e,
            )
            shadow_calibrator = None

    # Walk-forward summary
    mom_median_ic = float(np.median(mom_fold_ics)) if mom_fold_ics else 0.0
    vol_median_ic = float(np.median(vol_fold_ics)) if vol_fold_ics else 0.0
    resid_mom_median_ic = float(np.median(resid_mom_fold_ics)) if resid_mom_fold_ics else 0.0  # W2
    log.info("WF summary: momentum median_IC=%.4f  volatility median_IC=%.4f",
             mom_median_ic, vol_median_ic)

    # ── Step 8: Train production models on full data ─────────────────────────
    log.info("Training production models on full dataset...")
    n_train = int(N * cfg.TRAIN_FRAC)
    n_val_raw = int(N * cfg.VAL_FRAC)
    val_end = min(n_train + n_val_raw, N)

    # Momentum L1 component is the deterministic weighted-blend baseline.
    # No fit step — the formula is fixed (see model/momentum_scorer.py).
    # test_IC is computed against the held-out slice for manifest reporting
    # and parity with the volatility component below.
    mom_test_preds = momentum_scorer.predict_array(
        X_mom_raw[val_end:], cfg.MOMENTUM_FEATURES,
    )
    mom_test_ic = float(np.corrcoef(mom_test_preds, y_fwd[val_end:])[0, 1]) if len(mom_test_preds) > 1 else 0.0
    log.info("Momentum production (deterministic baseline): test_IC=%.4f", mom_test_ic)

    # Volatility production model
    peak_rss_mb = _log_rss("before prod_vol (plain) fit")
    prod_vol = GBMScorer(params=tuned_params, n_estimators=cfg.GBM_N_ESTIMATORS,
                         early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS)
    prod_vol.fit(X_vol[:n_train], np.abs(y_fwd[:n_train]),
                 X_vol[n_train:val_end], np.abs(y_fwd[n_train:val_end]),
                 feature_names=cfg.VOLATILITY_FEATURES)
    vol_test_preds = prod_vol.predict(X_vol[val_end:])
    vol_test_ic = float(np.corrcoef(vol_test_preds, np.abs(y_fwd[val_end:]))[0, 1]) if len(vol_test_preds) > 1 else 0.0
    log.info("Volatility production: test_IC=%.4f  best_iter=%d", vol_test_ic, prod_vol._best_iteration)
    peak_rss_mb = max(peak_rss_mb, _log_rss("after prod_vol (plain) fit"))

    # ── Step 8a: Parallel macro-augmented volatility GBM (Stage 1b observe-only) ──
    # Per regime-conditioning rebuild plan: train a parallel volatility
    # GBM that consumes the rank-normed per-ticker volatility features
    # PLUS the time-series-z-scored macros (from Step 4b). Trees discover
    # macro-conditioned interactions through sequential splits — the
    # institutional SOTA pattern (per-feature normalization).
    #
    # Ships ALONGSIDE the plain volatility GBM. Inference doesn't read it
    # yet (Stage 1c wires inference parallel observation; Stage 1d cuts
    # over after the variant_cutover_gate fires). This PR is training-side
    # parallel observation only — same pattern as audit Phase 3 PR 2/5
    # (ResearchGBMScorer parallel-observe before cutover).
    #
    # The augmented variant feeds 12 features (6 vol + 6 macro) vs the
    # plain variant's 6 vol features. Subsample IC gate (Step 8b) runs
    # against the plain variant only — adding the aug variant to the gate
    # would require coupling the gate to the cutover decision, which we
    # explicitly defer.
    # Float32 augmented feature matrices — LightGBM trains natively on
    # float32 and the augmented variants are the largest in-RAM matrices
    # in this stage. Halves memory vs the default float64.
    X_vol_aug = np.concatenate([X_vol, X_macro_zscored], axis=1).astype(np.float32, copy=False)
    VOL_AUG_FEATURES = list(cfg.VOLATILITY_FEATURES) + list(cfg.MACRO_NORM_FEATURES)
    prod_vol_macro_aug: GBMScorer | None = None
    vol_macro_aug_test_ic: float | None = None
    try:
        peak_rss_mb = max(peak_rss_mb, _log_rss("before prod_vol_macro_aug fit"))
        prod_vol_macro_aug = GBMScorer(
            params=tuned_params, n_estimators=cfg.GBM_N_ESTIMATORS,
            early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
        )
        prod_vol_macro_aug.fit(
            X_vol_aug[:n_train], np.abs(y_fwd[:n_train]),
            X_vol_aug[n_train:val_end], np.abs(y_fwd[n_train:val_end]),
            feature_names=VOL_AUG_FEATURES,
        )
        vol_aug_test_preds = prod_vol_macro_aug.predict(X_vol_aug[val_end:])
        vol_macro_aug_test_ic = (
            float(np.corrcoef(vol_aug_test_preds, np.abs(y_fwd[val_end:]))[0, 1])
            if len(vol_aug_test_preds) > 1 else 0.0
        )
        lift = vol_macro_aug_test_ic - vol_test_ic
        relative_lift = (
            vol_macro_aug_test_ic / vol_test_ic
            if vol_test_ic > 1e-6 else float("nan")
        )
        log.info(
            "Volatility-with-macros (Stage 1b parallel observe-only): "
            "test_IC=%.4f vs plain test_IC=%.4f (abs_lift=%+.4f, "
            "rel_lift=%.3fx)  best_iter=%d  n_features=%d",
            vol_macro_aug_test_ic, vol_test_ic, lift, relative_lift,
            prod_vol_macro_aug._best_iteration,
            prod_vol_macro_aug._booster.num_feature(),
        )
    except Exception as e:
        log.warning(
            "Volatility-with-macros parallel fit failed (non-blocking, "
            "observe-only): %s", e,
        )
        prod_vol_macro_aug = None
        vol_macro_aug_test_ic = None
    # Release the augmented feature matrix before the next variant's
    # concat allocates another large matrix. Together with the
    # Booster's `free_dataset()` (called inside GBMScorer.fit() post-
    # 2026-05-24), this caps peak RSS during the 3-variant volatility
    # training instead of letting it accumulate to OOM.
    import gc
    peak_rss_mb = max(peak_rss_mb, _log_rss("before X_vol_aug del"))
    del X_vol_aug
    gc.collect()
    peak_rss_mb = max(peak_rss_mb, _log_rss("after X_vol_aug del + gc"))

    # ── Step 8a-2: Parallel risk-augmented volatility GBM (Stage 2b observe-only) ──
    # Per regime-conditioning rebuild plan: train a parallel volatility
    # GBM that consumes the existing rank-normed vol features PLUS the
    # rank-normed per-ticker risk features (beta_60d, idio_vol_60d,
    # vol_of_vol_30d, max_drawdown_60d, realized_vol_63d). Mirrors the
    # Stage 1b macro-aug variant but for risk dimensions LightGBM trees
    # cannot split on through the existing 6 vol features.
    #
    # Ships ALONGSIDE the plain volatility GBM. Inference loads it
    # observe-only (Stage 2b inference path adds expected_move_risk_aug
    # parallel field). Cutover (Stage 2d) happens after the
    # variant_cutover_gate fires (≥15% relative IC lift over plain).
    X_vol_risk_aug = np.concatenate([X_vol, X_risk], axis=1).astype(np.float32, copy=False)
    VOL_RISK_AUG_FEATURES = list(cfg.VOLATILITY_FEATURES) + list(cfg.RISK_AUG_FEATURES)
    prod_vol_risk_aug: GBMScorer | None = None
    vol_risk_aug_test_ic: float | None = None
    try:
        peak_rss_mb = max(peak_rss_mb, _log_rss("before prod_vol_risk_aug fit"))
        prod_vol_risk_aug = GBMScorer(
            params=tuned_params, n_estimators=cfg.GBM_N_ESTIMATORS,
            early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
        )
        prod_vol_risk_aug.fit(
            X_vol_risk_aug[:n_train], np.abs(y_fwd[:n_train]),
            X_vol_risk_aug[n_train:val_end], np.abs(y_fwd[n_train:val_end]),
            feature_names=VOL_RISK_AUG_FEATURES,
        )
        vol_risk_aug_test_preds = prod_vol_risk_aug.predict(X_vol_risk_aug[val_end:])
        vol_risk_aug_test_ic = (
            float(np.corrcoef(vol_risk_aug_test_preds, np.abs(y_fwd[val_end:]))[0, 1])
            if len(vol_risk_aug_test_preds) > 1 else 0.0
        )
        lift = vol_risk_aug_test_ic - vol_test_ic
        relative_lift = (
            vol_risk_aug_test_ic / vol_test_ic
            if vol_test_ic > 1e-6 else float("nan")
        )
        log.info(
            "Volatility-with-risk (Stage 2b parallel observe-only): "
            "test_IC=%.4f vs plain test_IC=%.4f (abs_lift=%+.4f, "
            "rel_lift=%.3fx)  best_iter=%d  n_features=%d",
            vol_risk_aug_test_ic, vol_test_ic, lift, relative_lift,
            prod_vol_risk_aug._best_iteration,
            prod_vol_risk_aug._booster.num_feature(),
        )
    except Exception as e:
        log.warning(
            "Volatility-with-risk parallel fit failed (non-blocking, "
            "observe-only): %s", e,
        )
        prod_vol_risk_aug = None
        vol_risk_aug_test_ic = None
    # Same eager-release pattern as the macro-aug variant — drop the
    # large augmented feature matrix before downstream stages allocate.
    peak_rss_mb = max(peak_rss_mb, _log_rss("before X_vol_risk_aug del"))
    del X_vol_risk_aug
    gc.collect()
    peak_rss_mb = max(peak_rss_mb, _log_rss("after X_vol_risk_aug del + gc"))

    # ── Step 8b: Short-history subsample IC gate ─────────────────────────────
    # Per ROADMAP P1 "NaN-feature handling audit + short-history subsample
    # validation" + feedback_component_baseline_validation: each L1 GBM
    # must beat a named simple-fallback baseline on the short-history
    # subsample (rows where any L1 input was NaN before rank-norm),
    # otherwise the GBM isn't earning its complexity on the slice that
    # mimics inference-time short-history tickers (SNDK-class).
    #
    # Momentum is no longer in this gate: its component IS the named
    # baseline (deterministic weighted blend, model/momentum_scorer.py),
    # so a "component vs baseline" comparison is vacuous. Volatility +
    # research calibrator retain the gate.
    from model.subsample_validator import (
        volatility_baseline_predict, validate_component,
    )
    vol_subsample_mask_test = vol_nan_mask[val_end:]
    vol_baseline_preds = volatility_baseline_predict(
        X_vol_raw[val_end:], cfg.VOLATILITY_FEATURES,
    )
    vol_subsample_result = validate_component(
        component_name="volatility",
        component_preds=vol_test_preds,
        baseline_preds=vol_baseline_preds,
        y_true=np.abs(y_fwd[val_end:]),
        subsample_mask=vol_subsample_mask_test,
    )
    vol_subsample_result.log()
    subsample_gate_passed = vol_subsample_result.passed

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

    # Audit Phase 2a-PROMOTE (2026-05-07): output-distribution gate runs
    # the candidate calibrator on a synthetic alpha sweep and checks
    # four distribution-shape invariants (uniqueness / saturation /
    # stdev / direction skew). Catches the calibrator-plateau /
    # saturation / direction-skew failure class that bit production
    # 2026-04-28, 2026-05-04, 2026-05-07 — promotion-gate variants of
    # those incidents shipped to live S3 paths because no gate looked at
    # the calibrator's output the way the executor consumes it. Off
    # behind a feature flag for one Saturday-SF observation cycle, then
    # flips to blocking via predictor_params.json.
    from model.output_distribution_gate import (
        validate_calibrator_distribution,
        validate_stratified_per_regime,
    )
    output_dist_result = validate_calibrator_distribution(calibrator)
    output_dist_blocking = bool(
        cfg.OUTPUT_DISTRIBUTION_GATE_BLOCKING
        if hasattr(cfg, "OUTPUT_DISTRIBUTION_GATE_BLOCKING") else False
    )
    output_dist_gate_passed = output_dist_result.passed if output_dist_blocking else True

    # Audit Phase 2a-PROMOTE Option C (2026-05-07): stratified per-regime
    # variant. The synthetic-sweep gate above catches calibrator-internal
    # failures; this catches regime-conditional degeneracy where the
    # model's output collapses in only one regime (bull/neutral/bear).
    # Both can pass while a regime-conditional failure ships. Reuses the
    # same OOS rows + meta-model predictions the existing horizon-IC
    # diagnostic computed earlier in Step 7.1, plus runs them through
    # the calibrator and slices by regime label (already attached to
    # each oos_meta_row via the Track B 3/N PR).
    stratified_result = None
    try:
        # Calibrate the meta predictions to p_up + direction (one entry
        # per OOS row) then bucket by regime for the per-regime check.
        stratified_predictions: list[dict] = []
        stratified_regimes: list[str] = []
        for i, raw_alpha in enumerate(meta_preds_oos_insample):
            cal_result = calibrator.calibrate_prediction(float(raw_alpha))
            stratified_predictions.append({
                "p_up": cal_result["p_up"],
                "predicted_direction": cal_result["predicted_direction"],
            })
            stratified_regimes.append(row_regimes[i])
        stratified_result = validate_stratified_per_regime(
            stratified_predictions, stratified_regimes,
        )
    except Exception as e:
        log.warning(
            "Stratified per-regime gate computation failed (non-blocking): %s", e,
        )

    # Per-regime label up-rate — the empirical fraction of OOS rows
    # whose REALIZED canonical alpha is positive, sliced by regime.
    # Surfaced 2026-05-24 to answer "is the calibrator's bull-regime
    # direction-skew of 88% a defect (over-predicting UP) or a correct
    # match to reality (bull regimes have ~88% empirical up-rate)?"
    # Without this slice, the stratified-per-regime gate's 85% threshold
    # cannot be evaluated as too-strict-or-too-loose; with it, future
    # cycles can decide whether the gate threshold should be regime-aware.
    # The 5/24 cycle's bull-regime gate failure (n_up=227/256 calibrator
    # predictions) is the surfacing case. Independent of the calibrator —
    # this is a property of the labels + regime classifier alone.
    labels_up_rate_by_regime: dict[str, dict] = {}
    try:
        meta_y_positive = (meta_y > 0).astype(int)
        for regime in ("bull", "neutral", "bear"):
            r_mask = np.array([rr == regime for rr in row_regimes])
            n_regime = int(r_mask.sum())
            if n_regime > 0:
                n_up = int(meta_y_positive[r_mask].sum())
                labels_up_rate_by_regime[regime] = {
                    "n": n_regime,
                    "n_up": n_up,
                    "n_down": n_regime - n_up,
                    "labels_up_rate": round(n_up / n_regime, 4),
                }
            else:
                labels_up_rate_by_regime[regime] = {
                    "n": 0, "n_up": 0, "n_down": 0,
                    "labels_up_rate": None,
                }
        log.info(
            "Per-regime label up-rate (empirical realized): "
            "bull=%.3f (n=%d) neutral=%.3f (n=%d) bear=%.3f (n=%d)",
            labels_up_rate_by_regime["bull"].get("labels_up_rate") or 0,
            labels_up_rate_by_regime["bull"]["n"],
            labels_up_rate_by_regime["neutral"].get("labels_up_rate") or 0,
            labels_up_rate_by_regime["neutral"]["n"],
            labels_up_rate_by_regime["bear"].get("labels_up_rate") or 0,
            labels_up_rate_by_regime["bear"]["n"],
        )
    except Exception as e:
        log.warning(
            "Per-regime label up-rate computation failed (non-blocking): %s", e,
        )
    stratified_gate_passed = (
        stratified_result.passed if (stratified_result is not None and output_dist_blocking)
        else True
    )

    # Shadow calibrator gates — pure observability, results persist in the
    # manifest but do NOT influence ``promoted``. Lets operators compare
    # shadow-vs-live across Saturdays before promoting shadow → live by
    # flipping ``calibration.method`` in predictor.yaml.
    shadow_output_dist_result = None
    shadow_stratified_result = None
    if shadow_calibrator is not None:
        try:
            shadow_output_dist_result = validate_calibrator_distribution(shadow_calibrator)
        except Exception as e:
            log.warning("Shadow output-distribution gate computation failed: %s", e)
        try:
            shadow_stratified_predictions: list[dict] = []
            for i, raw_alpha in enumerate(meta_preds_oos_insample):
                cal_result = shadow_calibrator.calibrate_prediction(float(raw_alpha))
                shadow_stratified_predictions.append({
                    "p_up": cal_result["p_up"],
                    "predicted_direction": cal_result["predicted_direction"],
                })
            shadow_stratified_result = validate_stratified_per_regime(
                shadow_stratified_predictions, row_regimes,
            )
        except Exception as e:
            log.warning("Shadow stratified-per-regime gate computation failed: %s", e)
    if shadow_calibrator is not None:
        _shadow_od_status = (
            "n/a" if shadow_output_dist_result is None
            else ("PASS" if shadow_output_dist_result.passed else "FAIL")
        )
        _shadow_sp_status = (
            "n/a" if shadow_stratified_result is None
            else ("PASS" if shadow_stratified_result.passed else "FAIL")
        )
        log.info(
            "Shadow %s calibrator gates (observability): output_dist=%s  "
            "stratified_per_regime=%s",
            shadow_method, _shadow_od_status, _shadow_sp_status,
        )

    promoted = (
        meta_ic_passed
        and subsample_gate_passed
        and output_dist_gate_passed
        and stratified_gate_passed
    )
    # Promotion-blocker reason string. Resolved from the SAME gate booleans
    # the live `promoted` formula above uses, so it can never disagree with
    # the actual outcome. Persisted into training_summary so downstream
    # diagnostics ("why didn't this run promote?") don't have to re-run
    # training to figure out which gate fired — the original heuristic in
    # train_handler._build_failure_reason scanned walk_forward for negative
    # medians and routinely blamed momentum even when the real blocker was
    # output_dist or stratified, causing the 2026-05-13 incident (ROADMAP
    # L1668) where the email said "walk-forward failed" but the actual
    # cause was output_dist_gate_passed=False under
    # output_distribution_gate_blocking=true.
    _blockers: list[str] = []
    if not meta_ic_passed:
        _blockers.append("meta_ic")
    if not subsample_gate_passed:
        _blockers.append("subsample")
    if not output_dist_gate_passed:
        _blockers.append("output_dist")
    if not stratified_gate_passed:
        _blockers.append("stratified")
    promoted_blocker_reason: str | None = (
        None if promoted else ("+".join(_blockers) if _blockers else "unknown")
    )
    _stratified_status = (
        "n/a" if stratified_result is None
        else ("PASS" if stratified_result.passed else "FAIL")
    )
    log.info(
        "Promotion gate: meta_IC=%.4f %s %.4f (meta=%s) AND component "
        "subsample (vol=%s; momentum=DETERMINISTIC, research=GBM val_ic=%s) "
        "AND output_dist=%s AND stratified_per_regime=%s%s → %s "
        "(walk-forward for reference: mom_median=%.4f, vol_median=%.4f)",
        meta_model._val_ic, ">=" if meta_ic_passed else "<", _meta_ic_gate,
        "PASS" if meta_ic_passed else "BLOCK",
        "PASS" if vol_subsample_result.passed else "BLOCK",
        (
            f"{prod_research_gbm._val_ic:+.4f}"
            if prod_research_gbm is not None else "BUCKET-LOOKUP-FALLBACK"
        ),
        "PASS" if output_dist_result.passed else "FAIL",
        _stratified_status,
        "" if output_dist_blocking else " (OBSERVE-ONLY: not blocking)",
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
            #
            # Momentum LightGBM removed from upload set 2026-05-09 — its L1
            # component is now the deterministic baseline at
            # model/momentum_scorer.py. Existing s3://.../momentum_model.txt
            # from prior training cycles is left in place; inference's
            # tolerant load_meta_models block stops loading it (see
            # inference/stages/load_model.py).
            models = {
                "volatility": (prod_vol, "volatility_model.txt"),
                "research_calibrator": (prod_calibrator, "research_calibrator.json"),
                "meta_model": (meta_model, "meta_model.pkl"),
                "isotonic_calibrator": (calibrator, "isotonic_calibrator.pkl"),
            }
            # Audit Phase 3 PR 2/5: ResearchGBMScorer ships ALONGSIDE the
            # bucket-lookup in observe-only mode. Both get persisted with
            # their .meta.json sidecars; inference doesn't read the LGB
            # yet (PR 3 wires that). When prod_research_gbm is None
            # (insufficient data this cycle, or fit failed), the LGB key
            # is omitted — bucket-lookup remains the canonical path.
            if prod_research_gbm is not None:
                models["research_gbm"] = (prod_research_gbm, "research_gbm.pkl")
            # Stage 1b (regime-conditioning rebuild): parallel macro-augmented
            # volatility GBM ships ALONGSIDE the plain volatility GBM in
            # observe-only mode. Persisted to dated archive + live prefix
            # alongside the plain variant. Inference doesn't load it yet
            # (Stage 1c wires inference parallel observation; Stage 1d cuts
            # over after the variant_cutover_gate clears). When
            # prod_vol_macro_aug is None (parallel fit failed), the key is
            # omitted — plain volatility GBM remains the canonical path.
            if prod_vol_macro_aug is not None:
                models["volatility_macro_aug"] = (
                    prod_vol_macro_aug, "volatility_macro_aug_model.txt",
                )
            # Stage 2b (regime-conditioning rebuild): parallel risk-augmented
            # volatility GBM ships ALONGSIDE the plain volatility GBM in
            # observe-only mode. Inference reads it via Stage 2b's parallel
            # field; cutover (Stage 2d) happens after the variant_cutover_gate
            # fires. When prod_vol_risk_aug is None (parallel fit failed or
            # risk features absent from all parquets), the key is omitted.
            if prod_vol_risk_aug is not None:
                models["volatility_risk_aug"] = (
                    prod_vol_risk_aug, "volatility_risk_aug_model.txt",
                )
            # Stage 3 PR 2 (regime-conditioning rebuild): parallel triple-barrier
            # L2 Ridge ships ALONGSIDE the canonical Ridge in observe-only mode.
            # Inference (PR 3) emits a parallel `meta_alpha_tb` field; cutover
            # (PR 5) flips `enforce_cutover` after ≥3 Sat-SF firings of
            # variant_cutover_gate show ≥15% relative IC lift. When
            # meta_model_tb is None (insufficient TB-finite labels this cycle),
            # the key is omitted — canonical Ridge remains the sole live writer.
            if meta_model_tb is not None:
                models["meta_model_tb"] = (meta_model_tb, "meta_model_tb.pkl")
            # Task B (observe-only): meta-label classifier ships ALONGSIDE the
            # canonical Ridge. Inference emits `barrier_win_prob`; the executor
            # sizing consumer stays dormant until the Task B2 cutover. When the
            # classifier is None (disabled, or <100 touch-finite rows this
            # cycle), the key is omitted and inference emits null.
            if meta_label_clf is not None:
                models["meta_label_classifier"] = (
                    meta_label_clf, "meta_label_classifier.pkl",
                )
            # Shadow calibrator (2026-05-24 calibrator-shadow arc): persist
            # alongside the live calibrator under a distinct key so the
            # backtester's grace-period check on isotonic_calibrator.pkl.meta.json
            # isn't disturbed. Inference does NOT load this artifact; it
            # exists only so operators can `pickle.load` it from S3 to
            # introspect coefficients / fit details between Saturday cycles
            # while tuning the shadow knobs.
            if shadow_calibrator is not None:
                models["isotonic_calibrator_shadow"] = (
                    shadow_calibrator, "isotonic_calibrator_shadow.pkl",
                )
            # Audit Track A PR 5/6 cutover (2026-05-09): canonical_meta_model
            # retired from the upload set. The single meta_model.pkl above
            # is now trained on canonical labels — no separate canonical
            # parallel. Existing s3://.../canonical_meta_model.pkl from
            # the observe-only era is left in place; inference's tolerant
            # loader stops reading it (see load_model.py).
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

            # Final RSS snapshot — captured AFTER all L1 fits + meta-Ridge
            # but before manifest write, so it reflects the high-water mark
            # of training. Surfaced in the manifest so cycle-over-cycle
            # drift (creeping memory pressure from new features / variants)
            # is observable BEFORE it OOMs the spot.
            peak_rss_mb = max(peak_rss_mb, _log_rss("end of training (pre-manifest-write)"))
            log.info("Peak training RSS: %d MB", peak_rss_mb)
            # Write manifest
            manifest = {
                "date": date_str,
                "version": "v3.0-meta",
                "peak_rss_mb": peak_rss_mb,
                # Per-regime empirical up-rate of REALIZED canonical alpha
                # (independent of calibrator) — observability for the
                # stratified-per-regime gate threshold calibration. See
                # the in-trainer block above for the full rationale; in
                # short: if the calibrator's per-regime direction_skew
                # matches this empirical up-rate, the calibrator is
                # tracking reality and the 85% gate threshold is too
                # strict for that regime. Surfacing this manifest field
                # is the prerequisite for any future regime-aware-
                # threshold redesign.
                "labels_up_rate_by_regime": labels_up_rate_by_regime,
                # Horizon-of-record + label-domain top-level metadata (added
                # 2026-05-09 post Track A PR 5/6 cutover) so downstream
                # consumers — dashboard, backtester analytics, evaluator —
                # can read the predictor's training horizon + target domain
                # without re-deriving from filenames or hardcoding. Per the
                # predictor-21d-migration plan: column names + display
                # strings should be horizon-agnostic, parameterized by this
                # field instead of literal "5d" / "21d" strings.
                "forward_days": int(cfg.FORWARD_DAYS),
                "label_domain": "canonical_log",  # vs "arithmetic_legacy" pre Track A PR 5/6
                # ROADMAP L1581 diagnostic — the effective pre-cutoff
                # clamp applied to the per-ticker labeled DataFrame.
                # ``null`` when full history was used. Persisted so a
                # clamp-vs-no-clamp comparison across Sat retrains is
                # auditable from the manifest alone.
                "train_start_date": (
                    str(cfg.TRAIN_START_DATE)
                    if cfg.TRAIN_START_DATE is not None else None
                ),
                "promoted": promoted,
                "models": {
                    "momentum": {
                        "kind": "deterministic_baseline",
                        "test_ic": round(mom_test_ic, 6),
                        "formula": "0.4*m5 + 0.3*m20 + 0.2*price_vs_ma50 + 0.1*(rsi-50)/100",
                    },
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
                    # Audit Phase 3 PR 4/5 (2026-05-09 cutover):
                    # ResearchGBMScorer is the canonical source for
                    # research_calibrator_prob. train_ic is emitted
                    # alongside val_ic so a widening train-vs-val gap
                    # (> ~2× ratio is the typical overfitting signal)
                    # surfaces in the dashboard. When None, the LGB
                    # wasn't fit this cycle (insufficient data or fit
                    # failed); bucket-lookup-sourced research_calibrator_prob
                    # is the fallback for that cycle.
                    **(
                        {"research_gbm": {
                            "key": f"{prefix}research_gbm.pkl",
                            "canonical_for": "research_calibrator_prob",
                            "train_ic": (
                                round(research_gbm_train_ic, 6)
                                if research_gbm_train_ic is not None else None
                            ),
                            # ROADMAP L1816 (2026-05-19): train/val IC ratio
                            # + overfit warn surface in the manifest so a
                            # cycle-over-cycle regression is auditable
                            # without re-reading the LightGBM booster.
                            # ``ratio is None`` means "couldn't measure"
                            # (train_ic absent or |val_ic|<1e-3); ratio
                            # present with warn=True means "regression
                            # observed this cycle"; ratio present with
                            # warn=False is the healthy state.
                            "train_val_ic_ratio": (
                                round(research_gbm_train_val_ic_ratio, 6)
                                if research_gbm_train_val_ic_ratio is not None
                                else None
                            ),
                            "overfit_warn": research_gbm_overfit_warn,
                            **(research_gbm_metrics or {}),
                        }}
                        if prod_research_gbm is not None else {}
                    ),
                    # Stage 1b (regime-conditioning rebuild): macro-augmented
                    # volatility GBM observe-only metrics. Carries test_IC
                    # for both variants + abs/relative lift so cumulative
                    # observation across Saturday SF firings can be fed to
                    # ``analysis.variant_cutover_gate`` for the cutover
                    # decision. Inference doesn't read this artifact yet
                    # (Stage 1c) — purely training-side parallel observation.
                    # When None, parallel fit failed this cycle (no impact
                    # on prod_vol training).
                    **(
                        {"volatility_macro_aug": {
                            "key": f"{prefix}volatility_macro_aug_model.txt",
                            "test_ic": (
                                round(vol_macro_aug_test_ic, 6)
                                if vol_macro_aug_test_ic is not None else None
                            ),
                            "test_ic_plain_vol": round(vol_test_ic, 6),
                            "abs_lift": (
                                round(vol_macro_aug_test_ic - vol_test_ic, 6)
                                if vol_macro_aug_test_ic is not None else None
                            ),
                            "relative_lift": (
                                round(vol_macro_aug_test_ic / vol_test_ic, 4)
                                if (
                                    vol_macro_aug_test_ic is not None
                                    and vol_test_ic > 1e-6
                                ) else None
                            ),
                            "val_ic": float(prod_vol_macro_aug._val_ic or 0.0),
                            "best_iteration": prod_vol_macro_aug._best_iteration,
                            "n_features": prod_vol_macro_aug._booster.num_feature(),
                            "macro_norm_window": cfg.MACRO_NORM_WINDOW,
                        }}
                        if prod_vol_macro_aug is not None else {}
                    ),
                    # Stage 2b (regime-conditioning rebuild): parallel
                    # risk-augmented volatility GBM observe-only metrics.
                    # Mirrors the volatility_macro_aug shape so the same
                    # variant_cutover_gate analyzer can compare it against
                    # plain vol baseline. Cutover is Stage 2d after the
                    # gate fires. When None, parallel fit failed or risk
                    # features were absent from all parquets.
                    **(
                        {"volatility_risk_aug": {
                            "key": f"{prefix}volatility_risk_aug_model.txt",
                            "test_ic": (
                                round(vol_risk_aug_test_ic, 6)
                                if vol_risk_aug_test_ic is not None else None
                            ),
                            "test_ic_plain_vol": round(vol_test_ic, 6),
                            "abs_lift": (
                                round(vol_risk_aug_test_ic - vol_test_ic, 6)
                                if vol_risk_aug_test_ic is not None else None
                            ),
                            "relative_lift": (
                                round(vol_risk_aug_test_ic / vol_test_ic, 4)
                                if (
                                    vol_risk_aug_test_ic is not None
                                    and vol_test_ic > 1e-6
                                ) else None
                            ),
                            "val_ic": float(prod_vol_risk_aug._val_ic or 0.0),
                            "best_iteration": prod_vol_risk_aug._best_iteration,
                            "n_features": prod_vol_risk_aug._booster.num_feature(),
                            "risk_features": list(cfg.RISK_AUG_FEATURES),
                        }}
                        if prod_vol_risk_aug is not None else {}
                    ),
                    # Track A PR 5/6 cutover (2026-05-09): canonical_meta_model
                    # observe-only entry retired. The meta_model entry above
                    # IS the canonical Ridge. Post-cutover monitoring is
                    # ic_canonical_ridge_vs_legacy at the diagnostics block
                    # below.
                },
                "track_a_canonical_diagnostic": {
                    "ic_canonical_ridge_vs_legacy": (
                        round(ic_canonical_ridge_vs_legacy, 6)
                        if np.isfinite(ic_canonical_ridge_vs_legacy) else None
                    ),
                    "n_canonical": n_canonical,
                    "n_total": len(oos_meta_rows),
                },
                "walk_forward": {
                    "momentum_median_ic": round(mom_median_ic, 6),
                    "volatility_median_ic": round(vol_median_ic, 6),
                    # W2 (L4469, OBSERVE): residual-momentum L1 raw fold-IC
                    # median + whether it was in the live stack this run.
                    "residual_momentum_median_ic": round(resid_mom_median_ic, 6),
                    "residual_momentum_enabled": _resid_mom_enabled,
                    "n_folds": len(folds),
                },
                # W1 (L4469, OBSERVE): the leak-free OOS meta-IC battery, also
                # carried in the run-result dict. Mirrored into the manifest so
                # it is the SINGLE authoritative training-state artifact (L4468
                # SSOT) — the dashboard/operator surface reads training state
                # from here, never from the inference-cadence latest.json copy.
                "meta_model_oos_ic_leakfree": leakfree_meta_ic,
                # W2 (L4469, OBSERVE): standalone leak-free read of the
                # residual-momentum L1 (the isolated promotion-gate number).
                "residual_momentum_leakfree_oos_ic": residual_momentum_leakfree,
                # W4 watch + L4469 P2 (OBSERVE): is the meta IC real alpha +
                # pre-2020 drag. NOT gated.
                "meta_l1_standalone_alpha_ic": meta_l1_standalone_alpha_ic,
                "meta_oos_ic_leakfree_no_expected_move": meta_oos_ic_leakfree_no_expected_move,
                "meta_oos_ic_leakfree_post2020": meta_oos_ic_leakfree_post2020,
                # W4.1 (OBSERVE): nonlinear-L2 shadow leak-free meta IC.
                "meta_oos_ic_leakfree_nonlinear": meta_oos_ic_leakfree_nonlinear,
                "meta_model_oos_ic_cpcv": cpcv_meta_ic,
                "meta_model_promotion_stats": promotion_stats,
                "meta_coefficients": meta_model._coefficients,
                # Audit Phase 2a (2026-05-07): output-distribution gate
                # result. Persisted regardless of pass/fail so a borderline
                # pass over multiple training cycles is itself a useful
                # trend signal. ``blocking`` reflects whether the gate
                # blocked promotion this run (only True when the gate
                # failed AND the feature flag was set to blocking mode).
                "output_distribution_gate": {
                    "passed": output_dist_result.passed,
                    "failed_check": output_dist_result.failed_check,
                    "reason": output_dist_result.reason,
                    "metrics": output_dist_result.metrics,
                    "blocking": output_dist_blocking,
                    "would_have_blocked_if_blocking": (
                        not output_dist_result.passed
                    ),
                    # Phase 2a-PROMOTE Option C (2026-05-07): stratified
                    # per-regime sub-gate. Persisted regardless of
                    # pass/fail — borderline-pass on one regime over
                    # multiple training cycles is a useful trend signal.
                    "stratified_per_regime": (
                        {
                            "passed": stratified_result.passed,
                            "failed_check": stratified_result.failed_check,
                            "reason": stratified_result.reason,
                            "metrics": stratified_result.metrics,
                            "would_have_blocked_if_blocking": (
                                not stratified_result.passed
                            ),
                        } if stratified_result is not None
                        else {"passed": None, "reason": "computation skipped"}
                    ),
                },
                # Shadow calibrator block — observability only. Mirrors the
                # output_distribution_gate structure above so dashboards /
                # diagnostics can render shadow-vs-live side-by-side without
                # special-casing. Empty dict when shadow is disabled or fit
                # failed (see Step 7b-shadow). NEVER feeds promotion logic.
                "shadow_calibrator": (
                    {
                        "method": shadow_calibrator.method,
                        "metrics": shadow_calibrator.metrics(),
                        "output_distribution_gate": (
                            {
                                "passed": shadow_output_dist_result.passed,
                                "failed_check": shadow_output_dist_result.failed_check,
                                "reason": shadow_output_dist_result.reason,
                                "metrics": shadow_output_dist_result.metrics,
                            } if shadow_output_dist_result is not None
                            else {"passed": None, "reason": "computation skipped"}
                        ),
                        "stratified_per_regime": (
                            {
                                "passed": shadow_stratified_result.passed,
                                "failed_check": shadow_stratified_result.failed_check,
                                "reason": shadow_stratified_result.reason,
                                "metrics": shadow_stratified_result.metrics,
                            } if shadow_stratified_result is not None
                            else {"passed": None, "reason": "computation skipped"}
                        ),
                    } if shadow_calibrator is not None
                    else {}
                ),
            }
            s3_up.put_object(
                Bucket=bucket, Key=cfg.META_MANIFEST_KEY,
                Body=json.dumps(manifest, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("Manifest written to s3://%s/%s", bucket, cfg.META_MANIFEST_KEY)

            # feature_list.json — inference-universe disclosure (ROADMAP P2,
            # added 2026-05-05; closed 2026-05-12). Single source of truth
            # for the dashboard's Feature Store inventory page to render the
            # production-vs-research delta (features in ArcticDB but not yet
            # wired into inference) without parsing LightGBM .txt files.
            # The momentum L1 is a deterministic baseline (see
            # ``model.momentum_scorer``); its consumed-feature set is the
            # 4-input formula, not the full ``cfg.MOMENTUM_FEATURES``
            # subscription list.
            feature_list = {
                "trained_at": date_str,
                "version": "v3.0-meta",
                # W2: reflect the ACTUAL L2 feature set the Ridge was fit on
                # (identical to META_FEATURES while the observe gate is closed).
                "l2_features": list(TRAIN_META_FEATURES),
                "l1_features": {
                    "momentum": ["momentum_5d", "momentum_20d",
                                 "price_vs_ma50", "rsi_14"],
                    "volatility": list(cfg.VOLATILITY_FEATURES),
                    "research_calibrator": list(RESEARCH_GBM_FEATURES),
                    # W2 (observe): residual-momentum L1 inputs (computed +
                    # measured regardless of the gate; in the L2 only when
                    # residual_momentum_enabled).
                    "residual_momentum": list(RESIDUAL_MOMENTUM_FEATURES),
                },
            }
            s3_up.put_object(
                Bucket=bucket, Key=cfg.META_FEATURE_LIST_KEY,
                Body=json.dumps(feature_list, indent=2).encode(),
                ContentType="application/json",
            )
            log.info(
                "Feature list written to s3://%s/%s",
                bucket, cfg.META_FEATURE_LIST_KEY,
            )

            # Track 4-of-N of audit Phase 1 horizon battery (2026-05-07):
            # persist OOS meta-rows to S3 so the standalone analysis module
            # (analysis/horizon_battery.py) can iterate offline — varying
            # regime thresholds, horizon sets, bootstrap params — without
            # waiting for the next Saturday training cycle. Parquet for
            # efficiency on the ~10-100k row × ~25 column shape.
            try:
                import io as _io
                import pandas as _pd
                # Ensure Timestamp columns serialize cleanly; dict comprehension
                # preserves ordering. Drop date Timestamps to ISO strings since
                # parquet's pyarrow backend handles those natively but keeping
                # them as strings avoids any timezone-roundtrip subtleties on
                # the analysis-side read.
                _df_rows = _pd.DataFrame(oos_meta_rows)
                if "date" in _df_rows.columns:
                    _df_rows["date"] = _pd.to_datetime(_df_rows["date"]).dt.strftime("%Y-%m-%d")
                _buf = _io.BytesIO()
                _df_rows.to_parquet(_buf, index=False)
                _buf.seek(0)
                _oos_key = f"predictor/diagnostics/oos_rows/{date_str}.parquet"
                s3_up.put_object(
                    Bucket=bucket, Key=_oos_key,
                    Body=_buf.getvalue(),
                    ContentType="application/octet-stream",
                )
                # Also write a 'latest' alias so the analysis module can
                # default-read the most recent run without knowing the date.
                _buf.seek(0)
                s3_up.put_object(
                    Bucket=bucket, Key="predictor/diagnostics/oos_rows/latest.parquet",
                    Body=_buf.getvalue(),
                    ContentType="application/octet-stream",
                )
                log.info(
                    "OOS rows persisted to s3://%s/%s (%d rows × %d cols)",
                    bucket, _oos_key, len(_df_rows), len(_df_rows.columns),
                )
            except Exception as e:
                # Non-blocking — analysis can fall back to re-running training
                # to regenerate the rows. Don't fail a training run because
                # the diagnostic-persist step failed.
                log.warning("OOS-row persistence to S3 failed (non-blocking): %s", e)

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
        # ResearchGBMScorer is the canonical source for
        # research_calibrator_prob META_FEATURE (Phase 3 PR 4/5 cutover
        # 2026-05-09). val_ic is the honest held-out 20%-OOS measurement;
        # train_ic is paired so a widening train_ic/val_ic gap
        # (> ~2× ratio is the typical overfitting signal) surfaces
        # without manifest spelunking. None when the GBM wasn't fit
        # (n_finite < 100 finite-label rows in oos_meta_rows).
        "research_gbm_val_ic": (
            round(prod_research_gbm._val_ic, 6)
            if prod_research_gbm is not None else None
        ),
        "research_gbm_train_ic": (
            round(research_gbm_train_ic, 6)
            if research_gbm_train_ic is not None else None
        ),
        # ── Meta-model IC fields ────────────────────────────────────────
        # `meta_model_ic` + `meta_model_in_sample_ic` both hold the L2
        # Ridge's IN-SAMPLE Pearson fit — `np.corrcoef(model.predict(X), y)`
        # over the same pooled-OOF rows the Ridge was just fit on (see
        # model/meta_model.py:149-151, no held-out split inside the
        # Ridge fit step). Expect ~3× shrinkage to true OOS.
        #
        # `meta_model_oos_ic` is the honest reference: walk-forward
        # Spearman at the active label horizon (cfg.FORWARD_DAYS),
        # sourced from horizon_diagnostic.curve.{H}d.spearman just below.
        # Downstream consumers (alpha-engine-backtester production_health
        # retrain-alert plumbing, dashboard panels) MUST read
        # meta_model_oos_ic for degradation comparisons. The
        # _in_sample alias is published so the deprecation is
        # explicit in the schema; the unsuffixed key is preserved for
        # backward compat per S3 contract safety (additive only).
        #
        # Surfaced 2026-05-11 from the false-positive retrain alert
        # (rolling Pearson -0.10 vs reference 0.4634 → -22% ic_ratio →
        # HIGH-severity degradation flag). Consumer-side fix:
        # alpha-engine-backtester PR #181.
        "meta_model_ic": round(meta_model._val_ic, 6),
        "meta_model_in_sample_ic": round(meta_model._val_ic, 6),
        "meta_model_oos_ic": (
            round(float(horizon_ics[f"{cfg.FORWARD_DAYS}d"]["spearman"]), 6)
            if (
                f"{cfg.FORWARD_DAYS}d" in horizon_ics
                and np.isfinite(horizon_ics[f"{cfg.FORWARD_DAYS}d"]["spearman"])
            )
            else None
        ),
        # W1.1a (L4469, OBSERVE): the genuinely leak-free meta IC. The
        # `meta_model_oos_ic` above is OOS at the L1 level but IN-SAMPLE at
        # the META level (the L2 is read back on its own fit pool) and is a
        # *pooled* overlapping Spearman; this field refits the L2 per fold in
        # a purged/embargoed walk-forward and reports the CROSS-SECTIONAL rank
        # IC (`xsec_ic`) + pooled CI. NOT gated yet — W1.4 flips the promotion
        # gate from the in-sample _val_ic to this after observe firings.
        # Additive per S3 contract safety.
        "meta_model_oos_ic_leakfree": leakfree_meta_ic,
        # W2 (L4469, OBSERVE): standalone leak-free cross-sectional rank IC of
        # the residual-momentum L1 — its isolated promotion-gate number, vs the
        # dead-momentum baseline (~-0.001). NOT gated. Additive per S3 contract.
        "residual_momentum_leakfree_oos_ic": residual_momentum_leakfree,
        # W4 watch + L4469 P2 (OBSERVE, NOT gated): is the headline leak-free
        # meta IC real ALPHA, and does pre-2020 data drag it? (A) per-input
        # standalone alpha-IC; (A2) leak-free meta IC w/o expected_move; (B)
        # leak-free meta IC on post-2020-03-31 rows. Additive per S3 contract.
        "meta_l1_standalone_alpha_ic": meta_l1_standalone_alpha_ic,
        "meta_oos_ic_leakfree_no_expected_move": meta_oos_ic_leakfree_no_expected_move,
        "meta_oos_ic_leakfree_post2020": meta_oos_ic_leakfree_post2020,
        # W4.1 (OBSERVE): nonlinear-L2 shadow leak-free meta IC — vs the linear
        # meta_model_oos_ic_leakfree on the SAME folds. Additive per S3 contract.
        "meta_oos_ic_leakfree_nonlinear": meta_oos_ic_leakfree_nonlinear,
        # W1.2 (L4469, OBSERVE): combinatorial purged CV distribution of
        # leak-free cross-sectional OOS ICs (mean/std/percentiles/frac_positive
        # over C(N,k) combinations). Feeds the W1.3 Deflated-Sharpe / PBO gate.
        # NOT gated. Additive per S3 contract safety.
        "meta_model_oos_ic_cpcv": cpcv_meta_ic,
        # W1.3 (L4469, OBSERVE): two-lens promotion battery on the leak-free IC
        # series. `downside` = the house skilled-risk lens (Sortino-of-IC +
        # CVaR-of-IC + frac_negative_ic — Sortino/CVaR/maxDD basket, Sharpe is
        # legacy); `overfit` = the significance lens (PSR/Deflated-Sharpe,
        # Bailey-LdP; IC-IR symmetric by design for significance). NOT gated —
        # W1.4 composes both (downside-robust AND real). Additive per S3
        # contract safety.
        "meta_model_promotion_stats": promotion_stats,
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
                    # v3.5 (Track B 3/N): per-regime IC slice. Operators
                    # eyeball whether cross-horizon IC ratios are regime-
                    # dependent — does 21d=2.2×5d hold across bull / neutral /
                    # bear, or is it concentrated in one regime?
                    "by_regime": {
                        regime: {
                            "spearman": (
                                round(v["by_regime"][regime]["spearman"], 6)
                                if np.isfinite(v["by_regime"][regime]["spearman"])
                                else None
                            ),
                            "n": v["by_regime"][regime]["n"],
                        }
                        for regime in ("bull", "neutral", "bear")
                    },
                }
                for k, v in horizon_ics.items()
            },
            # Top-level regime counts so operators can immediately see
            # the bull/neutral/bear distribution across the OOS set.
            "regime_distribution": regime_counts,
            "peak_horizon": _peak[0],
            "peak_spearman": round(_peak[1]["spearman"], 6) if np.isfinite(_peak[1]["spearman"]) else None,
            # W3.2 (L4469, OBSERVE): the LEAK-FREE per-horizon IC curve — the
            # purged+embargoed expanding-WF cross-sectional rank IC (L2 refit
            # per fold on each horizon's label) + downside/DSR legs. This is
            # the honest curve that settles the operator's 5/21/60/90d question;
            # the `curve` field above is pooled/non-overlapping (inflated). NOT
            # gated; the canonical 21d target is unchanged. Additive per S3
            # contract. Net-of-cost judgment is W3.4 (deferred).
            "curve_leakfree": horizon_ics_leakfree,
            # Audit Track A PR 2/6 (2026-05-07): canonical-label IC
            # diagnostic. Persisted regardless of pass/fail for cross-cycle
            # trend monitoring. The audit's PR 5 cutover decision will
            # compare ``single_ridge_ic_vs_canonical`` against the future
            # ``canonical_ridge_ic_vs_canonical`` (Track A PR 3) — the
            # canonical-label Ridge should dominate the legacy-label
            # Ridge against the canonical target.
            "canonical_label_ic": canonical_ic_metrics,
        },
        "walk_forward": {
            "momentum_median_ic": round(mom_median_ic, 6),
            "volatility_median_ic": round(vol_median_ic, 6),
            # W2 (L4469, OBSERVE): residual-momentum L1 raw fold-IC median +
            # whether it was in the live L2 stack this run. NOT in passes_wf.
            "residual_momentum_median_ic": round(resid_mom_median_ic, 6),
            "residual_momentum_enabled": _resid_mom_enabled,
            "n_folds": len(folds),
            "folds": fold_results,
            # Momentum dropped from the WF promotion gate 2026-05-13. Per
            # scripts/momentum_ic_study_21d.py + the post-retire spot run
            # delta (~/Development/alpha-engine-docs/private/
            # momentum_l1_retirement-260513.md), momentum's standalone IC
            # at the canonical 21d horizon is ~0 but its Ridge-stack
            # interaction value carries measurable OOS signal — removing
            # it dropped OOS Spearman from 0.166 → 0.126. The component
            # stays in META_FEATURES; only the gate criterion is updated.
            "passes_wf": vol_median_ic > 0,
        },
        # Promotion-gate diagnostic block. Mirrors the live `promoted`
        # formula one-to-one (meta_ic AND subsample AND output_dist AND
        # stratified). Each gate's resolved boolean is the value actually
        # AND'd into `promoted`; the raw `_result_passed` and `_blocking`
        # fields preserve the upstream observation + its blocking-mode
        # context so an OBSERVE-ONLY failure (result.passed=False,
        # blocking=False → gate_passed=True) is distinguishable from a
        # blocking failure (result.passed=False, blocking=True →
        # gate_passed=False). `promoted_blocker_reason` is the +-joined
        # list of gates that failed (None when promoted=True). See
        # ROADMAP L1668 for the diagnostic gap this closes.
        "promotion_gate_detail": {
            "meta_ic_passed": bool(meta_ic_passed),
            "meta_ic_threshold": round(float(_meta_ic_gate), 6),
            "subsample_gate_passed": bool(subsample_gate_passed),
            "output_dist_result_passed": bool(output_dist_result.passed),
            "output_dist_blocking": bool(output_dist_blocking),
            "output_dist_gate_passed": bool(output_dist_gate_passed),
            "stratified_result_passed": (
                bool(stratified_result.passed) if stratified_result is not None else None
            ),
            "stratified_gate_passed": bool(stratified_gate_passed),
            "promoted_blocker_reason": promoted_blocker_reason,
        },
        "short_history_subsample": {
            # Momentum subsample gate retired 2026-05-09 — its L1 component
            # is now the deterministic baseline so component-vs-baseline is
            # vacuous. Carrying explicit kind="deterministic_baseline" here
            # so dashboard / health-check consumers can detect the change
            # without inferring from missing keys.
            "momentum": {
                "kind": "deterministic_baseline",
                "skip_reason": "component is the named baseline (no GBM to validate)",
            },
            "volatility": {
                "n": vol_subsample_result.n,
                "component_ic": round(vol_subsample_result.component_ic, 6),
                "baseline_ic": round(vol_subsample_result.baseline_ic, 6),
                "passed": vol_subsample_result.passed,
                "skip_reason": vol_subsample_result.skip_reason,
            },
            # Bucket-lookup ResearchCalibrator subsample gate retired
            # 2026-05-09 (Phase 3 PR 4/5 cutover). The L1 source for
            # research_calibrator_prob is now ResearchGBMScorer; its
            # val_ic + train_ic are reported in `models.research_gbm`
            # below and the gate-derived "earns its complexity" signal
            # is the GBM's val_ic on its own held-out 20%.
            "research_calibrator": {
                "kind": "retired_bucket_lookup_was_canonical_pre_2026_05_09",
                "skip_reason": "L1 source is now ResearchGBMScorer; see models.research_gbm.val_ic",
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
        # Shadow calibrator metrics — observability only. Empty dict when
        # shadow is disabled. Email formatter (train_handler.py) renders a
        # second calibration row when present.
        "shadow_calibration": (
            {
                **shadow_calibrator.metrics(),
                "output_distribution_gate_passed": (
                    shadow_output_dist_result.passed
                    if shadow_output_dist_result is not None else None
                ),
                "output_distribution_gate_failed_check": (
                    shadow_output_dist_result.failed_check
                    if shadow_output_dist_result is not None else None
                ),
                "stratified_per_regime_passed": (
                    shadow_stratified_result.passed
                    if shadow_stratified_result is not None else None
                ),
                "stratified_per_regime_failed_check": (
                    shadow_stratified_result.failed_check
                    if shadow_stratified_result is not None else None
                ),
            } if shadow_calibrator is not None else {}
        ),
    }
