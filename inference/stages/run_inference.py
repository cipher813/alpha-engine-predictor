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

log = logging.getLogger(__name__)


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

    # Compute combined_rank
    for p in ctx.predictions:
        mse_r = p.get("mse_rank")
        lr_r = p.get("model_rank")
        if mse_r is not None and lr_r is not None:
            p["combined_rank"] = round((mse_r + lr_r) / 2, 1)

    # Sort by combined_rank (ascending = best first), fall back to mse_rank
    ctx.predictions.sort(key=lambda p: p.get("combined_rank") or p.get("mse_rank") or 999)

    # Feature store writes removed — standalone compute.py handles this now.
    # See feature_store/compute.py and alpha-engine-data-feature-store-260402.md


def _run_meta_inference(ctx: PipelineContext) -> None:
    """Run Layer 1 specialized models → meta-model → predictions."""
    # Note: `data.feature_engineer.compute_features` is no longer imported here
    # after PR #5 removed the per-ticker inline compute fallback. Features now
    # come exclusively from ArcticDB via _load_precomputed_features_from_arcticdb.
    # Training still imports it — only inference was migrated.
    from model.meta_model import META_FEATURES, MACRO_FEATURE_META_MAP

    mom_scorer = ctx.meta_models.get("momentum")
    vol_scorer = ctx.meta_models.get("volatility")
    # regime_model removed from ctx.meta_models lookup 2026-04-16 — Tier 0
    # classifier no longer loaded by load_model.py.
    research_cal = ctx.meta_models.get("research_calibrator")
    meta_model = ctx.meta_models.get("meta")

    if mom_scorer is None and vol_scorer is None and meta_model is None:
        # Per feedback_hard_fail_until_stable: this is a cold-start load
        # failure, not a transient condition. Silent fallback to the old v2
        # path (deleted 2026-04-15) would degrade quality without alerting.
        raise RuntimeError(
            "All Layer-1 and meta-model scorers failed to load. Check "
            "load_model.py diagnostics and verify predictor/weights/meta/ "
            "in S3 is populated and readable."
        )

    _mom_ic = getattr(mom_scorer, "_val_ic", 0) if mom_scorer else 0
    _mom_mode = "GBM" if _mom_ic >= 0.02 else "direct (GBM IC=%.4f < 0.02)" % _mom_ic
    log.info("Momentum scoring mode: %s", _mom_mode)

    # ── Step 1: Compute raw macro features (once, market-wide) ──────────────
    # RegimePredictor used here as a pure feature-engineering utility via
    # build_features(). The regime classifier itself was retired 2026-04-16
    # (Tier 0 model could not clear an honest baseline — see meta_model.py
    # note and roadmap). The 6 macro features below feed the meta ridge
    # directly; no classifier in the loop.
    macro_row_for_meta: dict[str, float] = {name: 0.0 for name in MACRO_FEATURE_META_MAP.values()}
    try:
        from model.regime_predictor import RegimePredictor as _RPFeatureBuilder
        spy_s = ctx.macro.get("SPY") if ctx.macro else None
        vix_s = ctx.macro.get("VIX") if ctx.macro else None
        vix3m_s = ctx.macro.get("VIX3M") if ctx.macro else None
        tnx_s = ctx.macro.get("TNX") if ctx.macro else None
        irx_s = ctx.macro.get("IRX") if ctx.macro else None

        if spy_s is not None and len(spy_s) >= 20:
            _close_prices = {}
            for _tk, _df in (ctx.price_data or {}).items():
                if _df is not None and not _df.empty and "Close" in _df.columns:
                    _close_prices[_tk] = _df["Close"].astype(float)
            regime_features_df = _RPFeatureBuilder().build_features(
                spy_s, vix_s, vix3m_s, tnx_s, irx_s, _close_prices,
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

        # Layer 1A: Momentum model
        # If the momentum GBM has low quality (IC < 0.02 or best_iter <= 1),
        # fall back to a direct weighted average of raw momentum features.
        # This avoids a near-constant output from a barely-trained model.
        _mom_ic = getattr(mom_scorer, "_val_ic", 0) if mom_scorer else 0
        momentum_score = 0.0
        if mom_scorer is not None and _mom_ic >= 0.02:
            try:
                mom_x = latest[cfg.MOMENTUM_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
                momentum_score = float(mom_scorer.predict(mom_x)[0])
            except Exception:
                pass
        else:
            try:
                _m5 = float(latest.get("momentum_5d", 0) or 0)
                _m20 = float(latest.get("momentum_20d", 0) or 0)
                _ma50 = float(latest.get("price_vs_ma50", 0) or 0)
                _rsi = float(latest.get("rsi_14", 50) or 50)
                momentum_score = (
                    0.4 * _m5 + 0.3 * _m20 + 0.2 * _ma50 + 0.1 * (_rsi - 50) / 100
                )
            except Exception:
                pass

        # Layer 1B: Volatility model
        expected_move = 0.0
        if vol_scorer is not None:
            try:
                vol_x = latest[cfg.VOLATILITY_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
                expected_move = float(vol_scorer.predict(vol_x)[0])
            except Exception as _vol_exc:
                if ticker == ctx.tickers[0]:
                    log.warning("Volatility model predict failed for %s: %s", ticker, _vol_exc)

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

        # Research signal adjustment: meta-model was trained without research
        # data, so its ridge coefficients for research features are zero.
        # Apply a direct adjustment until the model is retrained with signals.
        # Scale: research_cal_prob in [0,1] → adjustment in [-0.005, +0.005].
        # Conviction amplifies: rising=1.5x, declining=0.5x.
        if sig:
            _cal_adj = (research_cal_prob - 0.5) * 0.01
            _conv_mult = {1.0: 1.5, 0.0: 1.0, -1.0: 0.5}.get(research_conviction, 1.0)
            alpha += _cal_adj * _conv_mult

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
            "mse_rank": None,
            "model_rank": None,
            "combined_rank": None,
            # Meta-model detail (new fields, additive)
            "research_calibrator_prob": round(research_cal_prob, 4),
            "momentum_confirmation": round(momentum_score, 6),
            "expected_move": round(expected_move, 6),
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
    log.info("Meta-inference complete: %d predictions, %d skipped", len(ctx.predictions), ctx.n_skipped)


def _rescale_cross_sectional(ctx: "PipelineContext") -> None:
    """Heuristic cross-sectional confidence rescaling (calibrator-guarded).

    When the isotonic calibrator is loaded (post-2026-04-15 binary UP/DOWN
    migration), per-ticker calls to ``calibrator.calibrate_prediction`` in
    ``_run_meta_inference`` produce properly-calibrated, absolute-scale
    probabilities. Rescaling here would overwrite those with a linear
    heuristic — a silent regression of the calibration work. In that case
    this function is a no-op.

    Without a calibrator (legacy fallback path), rescaling is mandatory:
    meta ridge outputs cluster in ~[-0.01, +0.01], narrower than
    LABEL_CLIP (±0.15). Linear p_up values collapse toward 0.5 for
    everything. META_ALPHA_CLIP floors the batch max_abs so tiny, noisy
    alpha spreads don't inflate to extreme confidences.
    """
    _cal = getattr(ctx, "calibrator", None)
    _calibrated = _cal is not None and getattr(_cal, "is_fitted", False)
    if _calibrated:
        log.info(
            "Skipping cross-sectional rescaling — isotonic calibrator active "
            "(method=%s, ECE_after=%.4f)",
            _cal.method, _cal._ece_after or 0.0,
        )
        return

    log.warning(
        "No calibrator loaded — applying linear heuristic cross-sectional "
        "rescaling. This path should only fire before the first post-migration "
        "retrain ships an isotonic calibrator to S3."
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


