"""Stage: run_inference — Compute features and run GBM or MLP inference."""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import arcticdb as adb  # Hard dep: PR #5 removed the try/except ImportError
                        # fallback that masked a missing Lambda layer for a
                        # week before detection. If arcticdb isn't in the
                        # deploy image, the Lambda must fail at cold start.
import numpy as np
import pandas as pd

import config as cfg
from inference.pipeline import PipelineContext, PipelineAbort
from model.momentum_scorer import predict_dict as _momentum_scorer_predict_dict

log = logging.getLogger(__name__)


def _safe_get_numeric(latest: pd.Series, key: str, default: float) -> float:
    """Read a numeric feature from a precomputed row, returning ``default``
    if the value is missing or NaN.

    NaN is structurally possible for short-history tickers — the data layer
    ships partial-NaN feature rows per the 2026-04-21 evening graceful-
    degrade policy. The momentum direct-fallback path's ``v or default``
    idiom was buggy because Python ``nan`` is truthy; this helper closes
    that NaN-survives path.
    """
    v = latest.get(key, default)
    if pd.isna(v):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _emit_nan_feature_tickers_metric(count: int) -> None:
    """Emit ``AlphaEngine/Predictor/nan_feature_tickers_count`` gauge.

    Best-effort: CloudWatch errors WARN but never fail inference. The
    counter itself is already in CloudWatch Logs (per-ticker WARN +
    inference-complete summary), so a metric-emission failure leaves
    a paper trail. Always emits — including value=0 — so alarm
    baselines are continuous rather than gappy.

    Parallel shape to executor's ``_emit_unscored_count_metric`` and
    ``_emit_admission_refused_metric``. Same namespace
    (``AlphaEngine/Predictor``) used by the existing executor-side
    coverage-gap gauge.
    """
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Predictor",
            MetricData=[{
                "MetricName": "nan_feature_tickers_count",
                "Value": float(count),
                "Unit": "Count",
            }],
        )
    except Exception as exc:  # noqa: BLE001 — observability, must never block inference
        log.warning(
            "CloudWatch nan_feature_tickers_count metric failed: %s. "
            "Counter still surfaces via the inference-complete summary log "
            "and per-ticker WARN; investigate the IAM grant if this fires "
            "every run.",
            exc,
        )


def _sanitize_meta_features(features: dict) -> tuple[dict, list[str]]:
    """Replace NaN values in a meta-features dict with 0.0 (neutral default
    matching ``predict_single``'s ``.get(f, 0.0)`` for missing keys).
    Returns the cleaned dict + list of feature names that were imputed.

    Ridge regression treats NaN as poison: any NaN input → NaN alpha →
    calibrator may crash or downstream consumers see NaN p_up. The data
    layer's graceful-degrade policy means NaN inputs ARE expected for
    short-history tickers; rather than quarantine those tickers (per
    feedback_no_unscoreable_labels), impute neutral and surface the
    coverage gap via per-ticker logging + a batch counter.
    """
    nan_keys = [
        k for k, v in features.items()
        if isinstance(v, float) and pd.isna(v)
    ]
    if not nan_keys:
        return features, []
    cleaned = {
        k: (0.0 if isinstance(v, float) and pd.isna(v) else v)
        for k, v in features.items()
    }
    return cleaned, nan_keys


def _load_precomputed_features_from_arcticdb(
    ctx: PipelineContext,
) -> dict[str, pd.Series]:
    """Read the latest feature row per ticker from ArcticDB's ``universe`` library.

    Raises RuntimeError on ArcticDB-wide failure (library unreachable,
    zero tickers readable). Individual missing/unreadable tickers are
    logged at WARNING and skipped — the meta-inference loop below filters
    them out. A ≥5% per-ticker error rate short-circuits with a raise.

    Replaces the prior three-tier read: S3 parquet feature store →
    inline compute_features. Those fallbacks masked a ``_run_gbm_inference``
    miswiring where ArcticDB was never actually consulted in production.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{ctx.bucket}?path_prefix=arcticdb&aws_auth=true"
    try:
        universe = adb.Arctic(uri).get_library("universe")
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB universe library unreachable at {uri}: {exc}"
        ) from exc

    precomputed: dict[str, pd.Series] = {}
    n_err = 0
    for ticker in ctx.tickers:
        try:
            df = universe.read(ticker).data
        except Exception as exc:
            log.warning("ArcticDB read failed for %s: %s", ticker, exc)
            n_err += 1
            continue
        if df.empty:
            log.warning("ArcticDB returned empty frame for %s", ticker)
            n_err += 1
            continue
        precomputed[ticker] = df.iloc[-1]

    err_rate = n_err / max(len(ctx.tickers), 1)
    if err_rate > 0.05:
        raise RuntimeError(
            f"ArcticDB read error rate {err_rate:.1%} exceeds 5% threshold "
            f"({n_err} failed of {len(ctx.tickers)}) — treating as pipeline failure"
        )
    log.info(
        "[data_source=arcticdb] Loaded %d/%d pre-computed tickers for %s",
        len(precomputed), len(ctx.tickers), ctx.date_str,
    )
    return precomputed


# ── Per-ticker MLP prediction (migrated from daily_predict.py) ───────────────

# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Run model inference across all tickers."""
    from inference.stages.write_output import write_predictions

    # Timeout gate
    if ctx.near_timeout():
        log.warning("Soft timeout before inference — writing partial predictions")
        write_predictions(ctx.predictions, ctx.date_str, ctx.bucket,
                          {"model_version": "timeout", "timed_out": True},
                          dry_run=ctx.dry_run, fd=ctx.fd)
        raise PipelineAbort("soft timeout before inference")

    ctx.predictions = []
    ctx.n_skipped = 0

    # v3 meta-model is the only supported inference path. The v2 single-GBM
    # branch (_run_gbm_inference) was deleted 2026-04-15 along with its
    # inline compute_features fallback — silent quality degradation on
    # meta-model load failure is worse than a loud cold-start abort.
    if ctx.inference_mode != "meta" or not ctx.meta_models:
        raise RuntimeError(
            f"v3 meta-model unavailable (inference_mode={ctx.inference_mode}, "
            f"meta_models={list(ctx.meta_models) if ctx.meta_models else 'empty'}). "
            "Inference cannot proceed — investigate the model-load step in "
            "load_model.py. The v2 single-GBM fallback was removed with the "
            "training/inference feature unification."
        )
    _run_meta_inference(ctx)

    log.info("Inference complete: %d predictions  %d skipped", len(ctx.predictions), ctx.n_skipped)

    # combined_rank is assigned per-prediction inside _run_meta_inference
    # (sorted by predicted_alpha desc, then 1..N). Re-sort here so the
    # outer caller sees the same best-first order.
    ctx.predictions.sort(key=lambda p: p.get("combined_rank") or 999)

    # Feature store writes removed — standalone compute.py handles this now.
    # See feature_store/compute.py and alpha-engine-data-feature-store-260402.md


def _run_meta_inference(ctx: PipelineContext) -> None:
    """Run Layer 1 specialized models → meta-model → predictions."""
    # Note: `data.feature_engineer.compute_features` is no longer imported here
    # after PR #5 removed the per-ticker inline compute fallback. Features now
    # come exclusively from ArcticDB via _load_precomputed_features_from_arcticdb.
    # Training still imports it — only inference was migrated.
    from model.meta_model import META_FEATURES, MACRO_FEATURE_META_MAP

    # Momentum L1 component is the deterministic baseline at
    # model/momentum_scorer.py — no scorer to look up. The
    # ctx.meta_models["momentum"] entry was retired 2026-05-09.
    vol_scorer = ctx.meta_models.get("volatility")
    # regime_model removed from ctx.meta_models lookup 2026-04-16 — Tier 0
    # classifier no longer loaded by load_model.py.
    # Audit Phase 3 PR 4/5 (2026-05-09): ResearchGBMScorer is now the
    # canonical source for the research_calibrator_prob META_FEATURE.
    # The bucket-lookup ResearchCalibrator rides alongside as a fallback
    # for cycles where the GBM wasn't fit (n_finite < 100 finite-label
    # rows in oos_meta_rows). When both are loaded, GBM wins.
    research_cal = ctx.meta_models.get("research_calibrator")
    research_gbm = ctx.meta_models.get("research_gbm")
    # Track A PR 5/6 cutover (2026-05-09): meta_model is now trained on
    # canonical alpha labels (log-domain risk-matched vs vol-cohort EW
    # basket); predicted_alpha IS the canonical alpha. No separate
    # canonical Ridge — the parallel observe-only canonical_meta_model
    # was retired with the cutover.
    meta_model = ctx.meta_models.get("meta")

    if vol_scorer is None and meta_model is None:
        # Per feedback_hard_fail_until_stable: this is a cold-start load
        # failure, not a transient condition. Silent fallback to the old v2
        # path (deleted 2026-04-15) would degrade quality without alerting.
        # Momentum L1 is now deterministic (no S3 weights), so absence of a
        # momentum scorer is no longer a load-failure signal.
        raise RuntimeError(
            "Volatility and meta-model scorers both failed to load. Check "
            "load_model.py diagnostics and verify predictor/weights/meta/ "
            "in S3 is populated and readable."
        )

    log.info("Momentum scoring mode: deterministic baseline (model/momentum_scorer.py)")

    # ── Step 1: Compute raw macro features (once, market-wide) ──────────────
    # RegimePredictor used here as a pure feature-engineering utility via
    # build_features(). The regime classifier itself was retired 2026-04-16
    # (Tier 0 model could not clear an honest baseline — see meta_model.py
    # note and roadmap). The 6 macro features below feed the meta ridge
    # directly; no classifier in the loop.
    macro_row_for_meta: dict[str, float] = {name: 0.0 for name in MACRO_FEATURE_META_MAP.values()}
    # Stage 1c (regime-conditioning rebuild): keep regime_features_df at
    # function scope so the macro-aug volatility GBM can apply
    # time-series z-score against the same history.
    regime_features_df: pd.DataFrame = pd.DataFrame()
    try:
        from model.regime_predictor import RegimePredictor as _RPFeatureBuilder
        spy_s = ctx.macro.get("SPY") if ctx.macro else None
        vix_s = ctx.macro.get("VIX") if ctx.macro else None
        vix3m_s = ctx.macro.get("VIX3M") if ctx.macro else None
        tnx_s = ctx.macro.get("TNX") if ctx.macro else None
        irx_s = ctx.macro.get("IRX") if ctx.macro else None
        # Stage 2c-full additions (regime-conditioning rebuild 2026-05-10)
        two_s = ctx.macro.get("TWO") if ctx.macro else None
        hyoas_s = ctx.macro.get("HYOAS") if ctx.macro else None
        baa10y_s = ctx.macro.get("BAA10Y") if ctx.macro else None

        if spy_s is not None and len(spy_s) >= 20:
            _close_prices = {}
            for _tk, _df in (ctx.price_data or {}).items():
                if _df is not None and not _df.empty and "Close" in _df.columns:
                    _close_prices[_tk] = _df["Close"].astype(float)
            regime_features_df = _RPFeatureBuilder().build_features(
                spy_s, vix_s, vix3m_s, tnx_s, irx_s, _close_prices,
                two_series=two_s,
                hyoas_series=hyoas_s,
                baa10y_series=baa10y_s,
            )
            if not regime_features_df.empty:
                latest_regime = regime_features_df.iloc[-1]
                for src_name, meta_name in MACRO_FEATURE_META_MAP.items():
                    macro_row_for_meta[meta_name] = float(latest_regime.get(src_name, 0.0))
                log.info(
                    "Macro features: spy_20d_ret=%.3f spy_20d_vol=%.3f vix_lvl=%.2f "
                    "vix_slope=%.3f yc_slope=%.3f breadth=%.2f",
                    latest_regime.get("spy_20d_return", 0),
                    latest_regime.get("spy_20d_vol", 0),
                    latest_regime.get("vix_level", 0),
                    latest_regime.get("vix_term_slope", 0),
                    latest_regime.get("yield_curve_slope", 0),
                    latest_regime.get("market_breadth", 0),
                )
    except Exception as e:
        # Preflight upstream already validates macro data freshness, so this
        # block should never fire in the happy path. Log at ERROR so CloudWatch
        # surfaces it loudly; fallback leaves zero-fill macro features so we
        # don't take the whole day offline, but the ridge's macro coefficients
        # will see a degenerate row — predictions that day are effectively
        # equivalent to the pre-macro-features v3.0 model. Tracked via log scan.
        log.error("Macro feature build failed: %s — using zero-fill defaults", e)

    # Stage 1c: today's z-scored macro vector for the macro-aug volatility
    # GBM. Time-series z-score over the full regime_features_df history;
    # take the last row. Broadcast to every ticker on the per-ticker
    # feature concat. None when history is too thin (< MACRO_NORM_WINDOW
    # min_periods) or the regime build failed — parallel field rides as
    # null in that case.
    today_macros_zscored: np.ndarray | None = None
    if not regime_features_df.empty and len(regime_features_df) >= 60:
        try:
            from data.dataset import time_series_zscore_normalize
            macro_subset = regime_features_df[
                list(cfg.MACRO_NORM_FEATURES)
            ].to_numpy(dtype=np.float64)
            macro_dates_list = list(pd.DatetimeIndex(regime_features_df.index))
            macro_zscored_full = time_series_zscore_normalize(
                macro_subset, macro_dates_list,
                window=cfg.MACRO_NORM_WINDOW,
            )
            if macro_zscored_full.size and not np.isnan(macro_zscored_full[-1]).all():
                today_macros_zscored = macro_zscored_full[-1]
                log.info(
                    "Macro z-score for vol-aug GBM (Stage 1c): "
                    "today's vector finite=%d/%d (window=%d)",
                    int(np.isfinite(today_macros_zscored).sum()),
                    today_macros_zscored.shape[0],
                    cfg.MACRO_NORM_WINDOW,
                )
        except Exception as e:
            log.warning(
                "Macro z-score for vol-aug GBM failed (parallel observe-only, "
                "non-blocking): %s", e,
            )

    # ── Step 2: Load research signals for calibrator ─────────────────────────
    # Read signals/latest.json for composite scores, conviction, and the
    # top-level sector_modifiers dict. Both the per-ticker view and the full
    # payload are kept — `extract_research_features` reads sector_modifiers
    # from the top level (not from per-ticker entries; the latter was the
    # `run_inference.py:293` bug that always returned 0.0 sector_modifier
    # for every ticker, regardless of research's sector ratings).
    import json
    research_signals = {}
    research_signals_payload: dict | None = None
    try:
        import boto3 as _b3_sig
        _s3_sig = _b3_sig.client("s3")
        try:
            sig_obj = _s3_sig.get_object(
                Bucket=ctx.bucket, Key="signals/latest.json"
            )
            sig_data = json.loads(sig_obj["Body"].read())
            research_signals_payload = sig_data
            for sig in sig_data.get("universe", []):
                ticker = sig.get("ticker")
                if ticker:
                    research_signals[ticker] = sig
            log.info(
                "Loaded %d research signals from signals/latest.json (date=%s)",
                len(research_signals), sig_data.get("date", "?"),
            )
        except Exception:
            log.info("signals/latest.json not found — no research signals for calibrator")
    except Exception as e:
        log.warning("Research signal loading failed: %s", e)

    # ── Step 3: Load pre-computed features from ArcticDB ──────────────────────
    # PR #5 cutover: ArcticDB is the single feature source. The prior
    # three-tier read (S3 parquet feature store → inline compute) was
    # deleted because:
    # (1) It was also living inside the dead `_run_gbm_inference` function,
    #     which PR #3 meant to target but never actually reached production.
    # (2) Silent fallbacks turned a broken feature source into "degraded
    #     mode" warnings that no one saw, which is exactly the pattern
    #     that masked the 2026-04-14 ArcticDB outage.
    # Preflight (inference/preflight.py) verifies macro/SPY freshness
    # before we get here. If ArcticDB is broken, cold start fails loud.
    max_r = getattr(cfg, "LABEL_CLIP", 0.15)
    precomputed = _load_precomputed_features_from_arcticdb(ctx)

    # ── Cross-sectional rank-normalize today's batch ────────────────────────
    # Training rank-normalizes via ``cross_sectional_rank_normalize`` per-date
    # (``data/dataset.py``) before fitting the L1 GBMs; the GBM's split
    # thresholds are learned on percentile ranks in (0, 1). Inference must
    # apply the same transform across today's full ticker batch — without
    # this, raw feature values land in ranges the GBM never saw, and tree
    # splits learned at e.g. ``rank < 0.42`` route every ticker into one
    # leaf, producing a small number of degenerate predictions.
    #
    # Origin: 2026-05-09 audit. Friday's production predictions
    # degenerated to 7 unique p_up values across 27 tickers (5 saturated
    # at 0.010); the SaturdayHealthCheck DriftDetection alarm fired
    # "93% predict DOWN" — both signatures of leaf-clustering caused by
    # the inference-vs-training feature distribution gap. The
    # alpha-engine-predictor PR #107 closed the NaN→1.0 rank-norm bug
    # (training-side); this closes the inference-side counterpart.
    #
    # Momentum is the deterministic baseline (model/momentum_scorer.py)
    # which operates on raw feature units (5d/20d returns, MA50 ratio,
    # RSI 0-100) per-ticker, so no rank-norm batch is built here.
    present_tickers = [t for t in ctx.tickers if t in precomputed]
    ticker_to_batch_idx = {t: i for i, t in enumerate(present_tickers)}
    n_present = len(present_tickers)
    today_dates = [ctx.date_str] * n_present  # one cross-section per inference run

    X_vol_ranked: np.ndarray | None = None
    if vol_scorer is not None and n_present >= 1:
        from data.dataset import cross_sectional_rank_normalize as _rank_norm
        X_vol_raw = np.stack([
            precomputed[t][cfg.VOLATILITY_FEATURES].to_numpy(dtype=np.float32)
            for t in present_tickers
        ])
        X_vol_ranked = _rank_norm(X_vol_raw, today_dates).astype(np.float32)

    # Stage 1c: macro-augmented rank+z-score batch for the parallel
    # vol-aug GBM. Schema must match training feature order:
    # [VOLATILITY_FEATURES rank-normed, MACRO_NORM_FEATURES z-scored].
    # Same n_features the aug GBM was trained on (12 = 6 + 6) per
    # Stage 1b's VOL_AUG_FEATURES list. Today's z-scored macros are
    # broadcast to every ticker because macros are constant cross-
    # sectionally on a given date.
    vol_scorer_aug = ctx.meta_models.get("volatility_macro_aug")
    X_vol_aug_ranked: np.ndarray | None = None
    if (
        vol_scorer_aug is not None
        and X_vol_ranked is not None
        and today_macros_zscored is not None
    ):
        macro_block = np.tile(
            today_macros_zscored.astype(np.float32), (n_present, 1),
        )
        X_vol_aug_ranked = np.concatenate(
            [X_vol_ranked, macro_block], axis=1,
        ).astype(np.float32)

    # Stage 2b: risk-augmented rank-norm batch for the parallel vol-risk
    # GBM. Schema must match training feature order:
    # [VOLATILITY_FEATURES rank-normed, RISK_AUG_FEATURES rank-normed].
    # Both halves are per-ticker (cross-sectionally varying) so the same
    # rank-norm pipeline applies to both. Total 11 features (6 vol + 5 risk).
    # Tolerant of missing risk features per ticker (NaN-fill via reindex);
    # LightGBM handles NaN natively.
    vol_scorer_risk_aug = ctx.meta_models.get("volatility_risk_aug")
    X_vol_risk_aug_ranked: np.ndarray | None = None
    if vol_scorer_risk_aug is not None and X_vol_ranked is not None:
        try:
            X_risk_raw = np.stack([
                precomputed[t]
                .reindex(columns=cfg.RISK_AUG_FEATURES)
                .to_numpy(dtype=np.float32)
                for t in present_tickers
            ])
            X_risk_ranked = _rank_norm(X_risk_raw, today_dates).astype(np.float32)
            X_vol_risk_aug_ranked = np.concatenate(
                [X_vol_ranked, X_risk_ranked], axis=1,
            ).astype(np.float32)
        except Exception as e:
            log.warning(
                "Risk feature batch build failed (parallel observe-only, "
                "non-blocking): %s", e,
            )

    for ticker in ctx.tickers:
        latest = precomputed.get(ticker)
        if latest is None:
            # Ticker not in ArcticDB (new constituent not yet backfilled, or
            # transient per-ticker read failure below the 5% threshold).
            # Inline compute fallback was removed in PR #5 — a production
            # feature store with per-ticker holes is the upstream team's
            # job to fix, not ours to mask.
            ctx.n_skipped += 1
            continue

        # Layer 1A: Momentum L1 component — deterministic weighted-blend
        # baseline (model/momentum_scorer.py). NaN / None / non-finite
        # inputs degrade to neutral defaults inside predict_dict.
        momentum_score = _momentum_scorer_predict_dict(latest)

        # Layer 1B: Volatility model. LightGBM handles NaN natively; if
        # predict raises that's a real load/inference fault we want loud,
        # not a per-ticker silent zero.
        expected_move = 0.0
        if vol_scorer is not None and X_vol_ranked is not None:
            idx = ticker_to_batch_idx[ticker]
            vol_x = X_vol_ranked[idx:idx + 1]  # rank-normed slice, shape (1, n_features)
            expected_move = float(vol_scorer.predict(vol_x)[0])

        # Stage 1c parallel observation: macro-augmented vol GBM rides
        # alongside the plain vol GBM. ``expected_move_macro_aug`` is None
        # when the aug GBM isn't loaded, today's macros couldn't be
        # z-scored (insufficient history), or the per-ticker predict raises.
        # Plain ``expected_move`` is the L2 ridge input until Stage 1d
        # cuts over.
        expected_move_macro_aug: float | None = None
        if vol_scorer_aug is not None and X_vol_aug_ranked is not None:
            idx_aug = ticker_to_batch_idx.get(ticker)
            if idx_aug is not None:
                try:
                    expected_move_macro_aug = float(
                        vol_scorer_aug.predict(
                            X_vol_aug_ranked[idx_aug:idx_aug + 1]
                        )[0]
                    )
                except Exception as e:
                    log.debug(
                        "volatility_macro_aug.predict failed for %s "
                        "(parallel observe-only, non-blocking): %s",
                        ticker, e,
                    )

        # Stage 2b parallel observation: risk-augmented vol GBM. Same
        # pattern as macro-aug — None when the aug GBM isn't loaded or
        # the per-ticker predict raises. Plain expected_move is canonical
        # until Stage 2d cuts over.
        expected_move_risk_aug: float | None = None
        if vol_scorer_risk_aug is not None and X_vol_risk_aug_ranked is not None:
            idx_risk = ticker_to_batch_idx.get(ticker)
            if idx_risk is not None:
                try:
                    expected_move_risk_aug = float(
                        vol_scorer_risk_aug.predict(
                            X_vol_risk_aug_ranked[idx_risk:idx_risk + 1]
                        )[0]
                    )
                except Exception as e:
                    log.debug(
                        "volatility_risk_aug.predict failed for %s "
                        "(parallel observe-only, non-blocking): %s",
                        ticker, e,
                    )

        # Layer 1C: Research calibrator + research-feature extraction
        # Centralized in ``model.research_features.extract_research_features``
        # so this call site stays in lockstep with the meta-trainer's
        # row-construction lookup. Pre-2026-04-28 this block reimplemented
        # the lookup inline and contained a bug where ``sector_modifiers``
        # was read from the per-ticker dict (always missing) instead of
        # the top-level signals payload — the helper reads from the full
        # payload, fixing it. None return triggers the same neutral
        # defaults the legacy inline path used so a missing ticker
        # doesn't break inference.
        from model.research_features import extract_research_features

        sig = research_signals.get(ticker)
        rf = extract_research_features(
            research_signals_payload, ticker, research_cal,
        )
        if rf is not None:
            research_cal_prob = rf["research_calibrator_prob"]
            research_score_norm = rf["research_composite_score"]
            research_conviction = rf["research_conviction"]
            sector_modifier = rf["sector_macro_modifier"]
        else:
            research_cal_prob = 0.5  # neutral default
            research_score_norm = 0.5
            research_conviction = 0.0
            sector_modifier = 0.0

        # Audit Phase 3 PR 4/5 (2026-05-09): ResearchGBMScorer is now the
        # canonical source for research_calibrator_prob. Build the GBM
        # input dict (3 research features + 6 macro features = 9 inputs
        # per RESEARCH_GBM_FEATURES) and override research_cal_prob with
        # the LGB output when the GBM is loaded. Bucket-lookup output
        # (already in research_cal_prob from extract_research_features)
        # remains the fallback when the GBM isn't loaded.
        if research_gbm is not None and getattr(research_gbm, "fitted", False):
            try:
                research_cal_prob = float(research_gbm.predict_from_dict({
                    "research_composite_score": research_score_norm,
                    "research_conviction": research_conviction,
                    "sector_macro_modifier": sector_modifier,
                    **macro_row_for_meta,
                }))
            except Exception as e:
                # Non-blocking — log per-ticker and fall back to the
                # bucket-lookup's research_cal_prob already populated above.
                log.debug("ResearchGBMScorer.predict_from_dict failed for %s: %s", ticker, e)

        # Layer 2: Meta-model
        meta_features = {
            "research_calibrator_prob": research_cal_prob,
            "momentum_score": momentum_score,
            "expected_move": expected_move,
            "research_composite_score": research_score_norm,
            "research_conviction": research_conviction,
            "sector_macro_modifier": sector_modifier,
            **macro_row_for_meta,  # raw macro features
        }

        # Track A PR 5/6 cutover (2026-05-09): canonical_predicted_alpha
        # parallel-output retired. predicted_alpha IS the canonical alpha
        # (meta_model is now trained on canonical labels).

        # NaN-aware sanitization at the ridge boundary. Ridge is NaN-poison;
        # data layer ships partial-NaN features for short-history tickers
        # (2026-04-21 evening graceful-degrade policy). Impute neutral and
        # log per-ticker rather than quarantine the ticker.
        meta_features, nan_keys = _sanitize_meta_features(meta_features)
        if nan_keys:
            log.warning(
                "ticker=%s: %d/%d meta-features were NaN — neutral-imputed: %s. "
                "Continuing with degraded prediction per graceful-degrade policy.",
                ticker, len(nan_keys), len(meta_features), nan_keys,
            )
            ctx.n_nan_imputed_tickers += 1

        if meta_model is not None and meta_model.is_fitted:
            alpha = float(meta_model.predict_single(meta_features))
        else:
            # Fallback: weighted average of Layer 1 outputs. Prior regime-based
            # term dropped along with the Tier 0 classifier removal (2026-04-16).
            alpha = (
                0.4 * momentum_score
                + 0.3 * (research_cal_prob - 0.5) * 0.1
                + 0.2 * expected_move * np.sign(momentum_score)
            )

        # The manual research-signal adjustment that lived here pre-2026-04-28
        # was a workaround for the meta-trainer hardcoding research features
        # to constants — Ridge zeroed those coefficients, so the inference
        # path patched in a small `(research_cal_prob - 0.5) * 0.01` boost
        # post hoc. PR #55 wired real research features into training; the
        # 2026-04-28 retraining run promoted a meta-model with non-zero
        # coefficients on all four research features (research_calibrator_prob:
        # +0.0064, research_composite_score: +0.0117, research_conviction:
        # -0.0015, sector_macro_modifier: -0.0411). Keeping the manual
        # adjustment now would double-count research signal; it's been
        # removed.

        alpha = float(np.clip(alpha, -max_r, max_r))

        # Calibrated confidence
        _cal = getattr(ctx, "calibrator", None)
        if _cal is not None and _cal.is_fitted:
            _cal_result = _cal.calibrate_prediction(alpha, label_clip=max_r)
            p_up = _cal_result["p_up"]
            p_down = _cal_result["p_down"]
            predicted_direction = _cal_result["predicted_direction"]
            confidence = _cal_result["prediction_confidence"]
        else:
            p_up = float(np.clip(0.5 + alpha / (2.0 * max_r), 0.0, 1.0))
            p_down = float(np.clip(0.5 - alpha / (2.0 * max_r), 0.0, 1.0))
            if alpha >= 0:
                predicted_direction = "UP"
                confidence = p_up
            else:
                predicted_direction = "DOWN"
                confidence = p_down

        result = {
            "ticker": ticker,
            "predicted_direction": predicted_direction,
            "prediction_confidence": round(confidence, 4),
            "predicted_alpha": round(alpha, 6),
            "p_up": round(p_up, 4),
            "p_flat": 0.0,
            "p_down": round(p_down, 4),
            "combined_rank": None,
            # Meta-model detail (new fields, additive)
            # research_calibrator_prob is now sourced from ResearchGBMScorer
            # when loaded (Phase 3 PR 4/5 cutover 2026-05-09); falls back to
            # the bucket-lookup ResearchCalibrator otherwise. The legacy
            # `research_gbm_prob` parallel field was retired with the
            # cutover — it now equals research_calibrator_prob whenever the
            # GBM is loaded, and dashboards key off `research_calibrator_prob`.
            "research_calibrator_prob": round(research_cal_prob, 4),
            # canonical_predicted_alpha parallel output retired with the
            # Track A PR 5/6 cutover (2026-05-09). predicted_alpha IS the
            # canonical alpha now (meta_model trained on canonical labels).
            "momentum_confirmation": round(momentum_score, 6),
            "expected_move": round(expected_move, 6),
            # Stage 1c parallel-observation field: macro-augmented vol
            # GBM prediction. Rides alongside ``expected_move`` for offline
            # ``analysis.variant_cutover_gate`` evaluation. None when the
            # aug GBM isn't loaded, today's macros couldn't be z-scored
            # (history too thin or regime build failed), or the per-ticker
            # predict raised. The plain ``expected_move`` remains the L2
            # ridge input until Stage 1d cuts over after the gate clears.
            "expected_move_macro_aug": (
                round(expected_move_macro_aug, 6)
                if expected_move_macro_aug is not None else None
            ),
            # Stage 2b parallel-observation field: risk-augmented vol GBM
            # prediction. Same evaluation framework as
            # expected_move_macro_aug — variant_cutover_gate compares
            # against expected_move (plain). Cutover is Stage 2d.
            "expected_move_risk_aug": (
                round(expected_move_risk_aug, 6)
                if expected_move_risk_aug is not None else None
            ),
            # regime_bull/regime_bear removed from per-ticker output 2026-04-16
            # (Tier 0 classifier retired). Downstream consumers (dashboard,
            # executor veto gate) must not expect these keys; the LLM macro
            # agent in research still emits a market_regime string via
            # signals.json, which remains the regime input the executor's
            # position sizer consumes.
            "meta_model_version": "v3.0",
        }

        if ticker in ctx.ticker_data_age:
            result["price_data_age_days"] = ctx.ticker_data_age[ticker]
        if ctx.ticker_sources:
            result["watchlist_source"] = ctx.ticker_sources.get(ticker, "unknown")

        ctx.predictions.append(result)

    # Sort by predicted_alpha descending (best first)
    ctx.predictions.sort(key=lambda p: -(p.get("predicted_alpha") or 0))
    # Assign combined_rank
    for i, p in enumerate(ctx.predictions):
        p["combined_rank"] = i + 1

    _rescale_cross_sectional(ctx)
    log.info(
        "Meta-inference complete: %d predictions, %d skipped, "
        "%d tickers had NaN meta-features (neutral-imputed)",
        len(ctx.predictions), ctx.n_skipped, ctx.n_nan_imputed_tickers,
    )
    # Always emit the gauge — including value=0 — so alarm baselines
    # are continuous. CloudWatch missing-data is harder to alarm on
    # than a steady 0 stream.
    _emit_nan_feature_tickers_metric(ctx.n_nan_imputed_tickers)


_MIN_UNIQUE_P_UP_BINS = 3
"""Variance-fallback threshold for the calibrator-active branch of
_rescale_cross_sectional. If the calibrator's batch outputs collapse to
fewer than this many unique p_up bins, fall through to the linear
heuristic. K=3 catches the 2026-04-28 collapse class (1 unique bin)
without false-engaging on small healthy batches that legitimately
cluster (e.g. a 5-ticker holiday batch where isotonic plateaus could
naturally produce 2 bins). Paired with `_MIN_BATCH_SIZE_FOR_VARIANCE_GATE`
below — the gate is suppressed for tiny batches where N < threshold is
structurally expected."""

_MIN_BATCH_SIZE_FOR_VARIANCE_GATE = 5


def _rescale_cross_sectional(ctx: "PipelineContext") -> None:
    """Heuristic cross-sectional confidence rescaling (calibrator-guarded).

    When the isotonic calibrator is loaded (post-2026-04-15 binary UP/DOWN
    migration), per-ticker calls to ``calibrator.calibrate_prediction`` in
    ``_run_meta_inference`` produce properly-calibrated, absolute-scale
    probabilities. Rescaling here would overwrite those with a linear
    heuristic — a silent regression of the calibration work. In that case
    this function is a no-op.

    **Variance fallback (2026-04-29):** when the calibrator IS loaded but
    its outputs on the current batch collapse to fewer than
    ``_MIN_UNIQUE_P_UP_BINS`` unique p_up values (the 2026-04-28
    calibrator-collapse pathology), the skip is suppressed and the batch
    falls through to the linear heuristic so cross-sectional variance is
    recovered for today's predictions. The fallback firing is a loud
    signal that calibrator training has degenerated; investigate at
    leisure but don't block today's inference. Closes ROADMAP P1
    "calibrator collapse" investigation step #5.

    Without a calibrator (legacy fallback path), rescaling is mandatory:
    meta ridge outputs cluster in ~[-0.01, +0.01], narrower than
    LABEL_CLIP (±0.15). Linear p_up values collapse toward 0.5 for
    everything. META_ALPHA_CLIP floors the batch max_abs so tiny, noisy
    alpha spreads don't inflate to extreme confidences.
    """
    _cal = getattr(ctx, "calibrator", None)
    _calibrated = _cal is not None and getattr(_cal, "is_fitted", False)
    if _calibrated:
        # Variance gate: count unique p_up bins (rounded to inference
        # precision — matches the 4dp rounding done at write time).
        p_up_values = [p.get("p_up", 0.5) or 0.5 for p in ctx.predictions]
        unique_count = len({round(v, 4) for v in p_up_values})
        n_preds = len(p_up_values)

        if (
            n_preds >= _MIN_BATCH_SIZE_FOR_VARIANCE_GATE
            and unique_count < _MIN_UNIQUE_P_UP_BINS
        ):
            log.error(
                "VARIANCE FALLBACK ENGAGED: calibrator outputs collapsed to "
                "%d unique p_up bins across %d tickers (threshold=%d). "
                "This is the 2026-04-28 calibrator-collapse pathology — "
                "investigate calibrator + meta-model training. Falling "
                "through to linear heuristic rescale to recover variance "
                "for today's batch.",
                unique_count, n_preds, _MIN_UNIQUE_P_UP_BINS,
            )
            # Fall through to the linear rescaling block below.
        else:
            log.info(
                "Skipping cross-sectional rescaling — isotonic calibrator "
                "active (method=%s, ECE_after=%.4f, unique_p_up_bins=%d)",
                _cal.method, _cal._ece_after or 0.0, unique_count,
            )
            return
    else:
        log.warning(
            "No calibrator loaded — applying linear heuristic cross-sectional "
            "rescaling. This path should only fire before the first "
            "post-migration retrain ships an isotonic calibrator to S3."
        )
    _META_ALPHA_CLIP = 0.02  # 2% — reasonable expected range for 5d alpha
    alphas = [p.get("predicted_alpha", 0) or 0 for p in ctx.predictions]
    if not alphas:
        return
    max_abs = max(abs(a) for a in alphas)
    meta_clip = max(max_abs, _META_ALPHA_CLIP)
    log.info(
        "Meta confidence rescaling: max_abs_alpha=%.6f  meta_clip=%.6f  (floor=%.3f)",
        max_abs, meta_clip, _META_ALPHA_CLIP,
    )
    for p in ctx.predictions:
        a = p.get("predicted_alpha", 0) or 0
        p_up = float(np.clip(0.5 + a / (2.0 * meta_clip), 0.0, 1.0))
        p_down = 1.0 - p_up
        if a >= 0:
            direction = "UP"
            confidence = p_up
        else:
            direction = "DOWN"
            confidence = p_down
        p["p_up"] = round(p_up, 4)
        p["p_down"] = round(p_down, 4)
        p["predicted_direction"] = direction
        p["prediction_confidence"] = round(confidence, 4)


