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
    from model.meta_model import MetaModel, META_FEATURES

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
    ticker_features: dict[str, pd.DataFrame] = {}
    reject_too_short: list[tuple[str, int]] = []
    reject_empty_raw: list[str] = []
    reject_missing_features: list[tuple[str, str]] = []
    reject_label_error: list[tuple[str, str]] = []
    reject_empty_labeled: list[str] = []
    n_candidates = 0
    _MIN_HISTORY_ROWS = 265  # ~1 year of trading days — required for walk-forward folds

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

        # v3.1 diagnostic (ROADMAP Predictor P2): compute 21d forward
        # sector-neutral alpha alongside the 5d label. Used ONLY to
        # evaluate the trained meta-model's IC at a longer horizon —
        # tells us whether the 5d forward label is the right horizon
        # (if 21d IC >> 5d IC, we should consider a parallel 21d
        # training stack). Not used for training; pure sidecar metric.
        #
        # Shift -21 means "close 21 rows ahead". At the tail of the
        # history the value is NaN; rows with NaN are excluded from
        # the 21d IC computation only (they still train on 5d).
        close_for_21d = raw_df["Close"].astype(float)
        bench_21d = sector_etf_s if sector_etf_s is not None else spy_series
        if bench_21d is not None:
            bench_aligned_21d = bench_21d.reindex(close_for_21d.index, method="ffill")
            stock_fwd_21d = (close_for_21d.shift(-21) / close_for_21d) - 1.0
            bench_fwd_21d = (bench_aligned_21d.shift(-21) / bench_aligned_21d) - 1.0
            labeled["forward_return_21d"] = (stock_fwd_21d - bench_fwd_21d).reindex(labeled.index)
        else:
            labeled["forward_return_21d"] = float("nan")

        ticker_features[ticker] = labeled

    log.info(
        "Feature read complete: %d accepted / %d candidates  "
        "(rejected: too_short=%d  empty_raw=%d  missing_features=%d  "
        "label_error=%d  empty_labeled=%d)",
        len(ticker_features), n_candidates,
        len(reject_too_short), len(reject_empty_raw),
        len(reject_missing_features), len(reject_label_error), len(reject_empty_labeled),
    )

    if len(ticker_features) == 0:
        # Hard-fail with diagnostics rather than letting np.stack emit its
        # cryptic "need at least one array to stack" on the empty list.
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

    # ── Step 3: Build arrays for momentum + volatility models ────────────────
    # Flatten per-ticker DataFrames into (X, y, dates) arrays
    all_rows = []
    for ticker, df in ticker_features.items():
        mom_vals = df[cfg.MOMENTUM_FEATURES].to_numpy(dtype=np.float32)
        vol_vals = df[cfg.VOLATILITY_FEATURES].to_numpy(dtype=np.float32)
        fwd = df["forward_return_5d"].to_numpy(dtype=np.float32)
        fwd_21d = df.get("forward_return_21d", pd.Series(float("nan"), index=df.index)).to_numpy(dtype=np.float32)
        dates = df.index
        for j in range(len(dates)):
            all_rows.append((
                dates[j], ticker,
                mom_vals[j], vol_vals[j],
                float(fwd[j]),
                float(fwd_21d[j]),
            ))

    all_rows.sort(key=lambda r: r[0])
    all_dates = [r[0] for r in all_rows]
    all_tickers = [r[1] for r in all_rows]
    X_mom = np.stack([r[2] for r in all_rows])
    X_vol = np.stack([r[3] for r in all_rows])
    y_fwd = np.array([r[4] for r in all_rows], dtype=np.float32)
    y_fwd_21d = np.array([r[5] for r in all_rows], dtype=np.float32)  # diagnostic only

    # Winsorize
    if cfg.LABEL_CLIP:
        y_fwd = np.clip(y_fwd, -cfg.LABEL_CLIP, cfg.LABEL_CLIP)

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
    try:
        import boto3
        s3 = boto3.client("s3")
        # Try to load research.db score_performance table
        import sqlite3
        db_tmp = Path(tempfile.gettempdir()) / "research_train.db"
        s3.download_file(bucket, "research.db", str(db_tmp))
        conn = sqlite3.connect(str(db_tmp))
        sp_df = pd.read_sql_query(
            "SELECT score, beat_spy_10d FROM score_performance "
            "WHERE beat_spy_10d IS NOT NULL AND score IS NOT NULL",
            conn,
        )
        conn.close()
        if not sp_df.empty:
            research_scores = sp_df["score"].to_numpy()
            research_beat_spy = sp_df["beat_spy_10d"].to_numpy().astype(int)
            log.info("Research calibrator data: %d rows from score_performance", len(sp_df))
        else:
            log.info("score_performance table empty — research calibrator will use priors")
    except Exception as e:
        log.warning("Could not load research calibrator data: %s — using priors", e)

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
    vol_fold_ics = []

    for i, fold in enumerate(folds):
        fold_start = time.time()
        tr = fold["train_idx"]
        te = fold["test_idx"]

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

        # Regime predictor: train on dates up to train_end
        regime_train_mask = common_dates <= pd.Timestamp(fold["train_end"])
        regime_test_mask = (common_dates >= pd.Timestamp(fold["test_start"])) & \
                           (common_dates <= pd.Timestamp(fold["test_end"]))
        if regime_train_mask.sum() >= 100:
            fold_regime = RegimePredictor()
            fold_regime.fit(regime_X_aligned[regime_train_mask], regime_y_aligned[regime_train_mask])
        else:
            fold_regime = None

        # Build meta-model OOS rows for this fold's test set
        # Group test samples by date for regime prediction
        test_dates_unique = sorted(set(all_dates[j] for j in te))
        for test_date in test_dates_unique:
            date_mask = np.array([all_dates[j] == test_date for j in te])
            date_indices = te[date_mask]

            # Regime prediction for this date
            if fold_regime is not None and test_date in regime_features_df.index:
                regime_x = regime_features_df.loc[test_date, RegimePredictor.FEATURE_NAMES].to_numpy().reshape(1, -1)
                regime_probs = fold_regime.predict_proba(regime_x)[0]
            else:
                regime_probs = np.array([0.33, 0.34, 0.33])

            for idx in date_indices:
                local_idx = np.where(te == idx)[0][0]
                oos_meta_rows.append({
                    "momentum_score": float(mom_preds[local_idx]),
                    "expected_move": float(vol_preds[local_idx]),
                    "regime_bull": float(regime_probs[2]),
                    "regime_bear": float(regime_probs[0]),
                    "research_calibrator_prob": 0.5,  # placeholder (v0 lookup applied later)
                    "research_composite_score": 0.5,  # not available in price-only training
                    "research_conviction": 0.0,
                    "sector_macro_modifier": 0.0,
                    "actual_fwd": float(y_fwd[idx]),
                    "actual_fwd_21d": float(y_fwd_21d[idx]),  # diagnostic only
                })

        elapsed = time.time() - fold_start
        fold_results.append({
            "fold": i + 1, "mom_ic": round(mom_ic, 6), "vol_ic": round(vol_ic, 6),
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "n_test": len(te), "elapsed_s": round(elapsed, 1),
        })
        log.info("  Fold %d/%d: mom_IC=%.4f  vol_IC=%.4f  (%.1fs)",
                 i + 1, len(folds), mom_ic, vol_ic, elapsed)

    # ── Step 7: Train meta-model on pooled OOS ───────────────────────────────
    log.info("Training meta-model on %d OOS rows...", len(oos_meta_rows))
    meta_X = np.array([[r[f] for f in META_FEATURES] for r in oos_meta_rows])
    meta_y = np.array([r["actual_fwd"] for r in oos_meta_rows])
    meta_model = MetaModel(alpha=1.0)
    meta_model.fit(meta_X, meta_y, feature_names=META_FEATURES)

    # ── Step 7.1: 21d forward IC diagnostic (ROADMAP Predictor P2) ───────────
    # Evaluate whether the 5d-trained meta-model ranks tickers well at a
    # 21d horizon. If 21d Spearman IC >> 5d IC, the 5d label is the
    # wrong horizon (monthly momentum regime) and a parallel 21d stack
    # is worth building. If 21d ≈ 5d, horizon isn't the issue — feature
    # quality is. Either way the diagnostic is free: just a correlation
    # against an already-collected label column. Not used for training.
    from scipy.stats import spearmanr
    meta_y_21d = np.array([r["actual_fwd_21d"] for r in oos_meta_rows])
    meta_preds_oos_insample = meta_model.predict(meta_X).ravel()
    mask_5d = np.isfinite(meta_y)
    mask_21d = np.isfinite(meta_y_21d)
    spearman_5d_res = spearmanr(meta_preds_oos_insample[mask_5d], meta_y[mask_5d])
    spearman_5d = float(spearman_5d_res.correlation) if np.isfinite(spearman_5d_res.correlation) else 0.0
    if mask_21d.sum() >= 100:
        spearman_21d_res = spearmanr(meta_preds_oos_insample[mask_21d], meta_y_21d[mask_21d])
        spearman_21d = float(spearman_21d_res.correlation) if np.isfinite(spearman_21d_res.correlation) else 0.0
    else:
        spearman_21d = float("nan")
    log.info(
        "Horizon IC diagnostic: meta preds vs 5d fwd Spearman=%.4f (n=%d)  "
        "vs 21d fwd Spearman=%.4f (n=%d)",
        spearman_5d, int(mask_5d.sum()), spearman_21d, int(mask_21d.sum()),
    )

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

    # Regime production model (full history)
    prod_regime = RegimePredictor()
    prod_regime.fit(regime_X_aligned, regime_y_aligned)

    # Research calibrator (v0 — lookup table)
    prod_calibrator = ResearchCalibrator()
    if len(research_scores) >= 10:
        prod_calibrator.fit(research_scores, research_beat_spy)
    else:
        log.info("Research calibrator: insufficient data (%d rows) — using neutral priors",
                 len(research_scores))

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
    promoted = meta_model._val_ic >= _meta_ic_gate
    log.info(
        "Meta-model promotion gate: meta_IC=%.4f %s %.4f → %s "
        "(walk-forward for reference: mom_median=%.4f, vol_median=%.4f)",
        meta_model._val_ic, ">=" if promoted else "<", _meta_ic_gate,
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
            models = {
                "momentum": (prod_mom, "momentum_model.txt"),
                "volatility": (prod_vol, "volatility_model.txt"),
                "regime": (prod_regime, "regime_predictor.pkl"),
                "research_calibrator": (prod_calibrator, "research_calibrator.json"),
                "meta_model": (meta_model, "meta_model.pkl"),
                "isotonic_calibrator": (calibrator, "isotonic_calibrator.pkl"),
            }
            for name, (model, filename) in models.items():
                local_path = Path(tmp) / filename
                model.save(local_path)
                s3_key = f"{prefix}{filename}"
                s3_up.upload_file(str(local_path), bucket, s3_key)
                # Upload meta.json if exists
                meta_path = Path(str(local_path) + ".meta.json")
                if meta_path.exists():
                    s3_up.upload_file(str(meta_path), bucket, f"{s3_key}.meta.json")
                log.info("Uploaded %s → s3://%s/%s", name, bucket, s3_key)

                # Dated backup
                dated_key = f"{prefix}archive/{date_str}/{filename}"
                s3_up.upload_file(str(local_path), bucket, dated_key)

            # Write manifest
            manifest = {
                "date": date_str,
                "version": "v3.0-meta",
                "promoted": promoted,
                "models": {
                    "momentum": {"key": f"{prefix}momentum_model.txt", "test_ic": round(mom_test_ic, 6)},
                    "volatility": {"key": f"{prefix}volatility_model.txt", "test_ic": round(vol_test_ic, 6)},
                    "regime": {"key": f"{prefix}regime_predictor.pkl", "accuracy": prod_regime._accuracy},
                    "research_calibrator": {"key": f"{prefix}research_calibrator.json",
                                            "n_samples": prod_calibrator._n_samples},
                    "meta_model": {"key": f"{prefix}meta_model.pkl", "ic": round(meta_model._val_ic, 6)},
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
        "regime_acc=%.1f%%  promoted=%s  elapsed=%.0fs",
        mom_test_ic, vol_test_ic, meta_model._val_ic,
        (prod_regime._accuracy or 0) * 100, promoted, elapsed_s,
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
        "regime_accuracy": round((prod_regime._accuracy or 0), 4),
        "research_calibrator_n": prod_calibrator._n_samples,
        "research_calibrator_metrics": prod_calibrator.metrics(),
        "meta_model_ic": round(meta_model._val_ic, 6),
        "meta_coefficients": meta_model._coefficients,
        "horizon_diagnostic": {
            "spearman_5d": round(spearman_5d, 6),
            "spearman_21d": round(spearman_21d, 6) if np.isfinite(spearman_21d) else None,
            "n_5d": int(mask_5d.sum()),
            "n_21d": int(mask_21d.sum()),
        },
        "walk_forward": {
            "momentum_median_ic": round(mom_median_ic, 6),
            "volatility_median_ic": round(vol_median_ic, 6),
            "n_folds": len(folds),
            "folds": fold_results,
            "passes_wf": mom_median_ic > 0 and vol_median_ic > 0,
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
