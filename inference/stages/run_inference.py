"""Stage: run_inference — Compute features and run GBM or MLP inference."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as cfg
from inference.pipeline import PipelineContext, PipelineAbort

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Run model inference across all tickers."""
    from inference.daily_predict import write_predictions

    # Timeout gate
    if ctx.near_timeout():
        log.warning("Soft timeout before inference — writing partial predictions")
        write_predictions(ctx.predictions, ctx.date_str, ctx.bucket,
                          {"model_version": "timeout", "timed_out": True},
                          dry_run=ctx.dry_run, fd=ctx.fd)
        raise PipelineAbort("soft timeout before inference")

    ctx.predictions = []
    ctx.n_skipped = 0

    if ctx.inference_mode == "meta" and ctx.meta_models:
        _run_meta_inference(ctx)
    elif ctx.model_type == "gbm":
        _run_gbm_inference(ctx)
    else:
        _run_mlp_inference(ctx)

    log.info("Inference complete: %d predictions  %d skipped", len(ctx.predictions), ctx.n_skipped)

    # Compute combined_rank
    for p in ctx.predictions:
        mse_r = p.get("mse_rank")
        lr_r = p.get("model_rank")
        if mse_r is not None and lr_r is not None:
            p["combined_rank"] = round((mse_r + lr_r) / 2, 1)

    # Sort by combined_rank (ascending = best first), fall back to mse_rank
    ctx.predictions.sort(key=lambda p: p.get("combined_rank") or p.get("mse_rank") or 999)

    # ── Feature store: expand to full universe ───────────────────────────────
    if (cfg.FEATURE_STORE_ENABLED and cfg.FEATURE_STORE_FULL_UNIVERSE
            and cfg.FEATURE_STORE_WRITE_ON_INFERENCE
            and ctx.model_type == "gbm" and not ctx.near_timeout()):
        _expand_feature_store(ctx)

    # Write feature store snapshot
    if cfg.FEATURE_STORE_ENABLED and cfg.FEATURE_STORE_WRITE_ON_INFERENCE and ctx.store_rows:
        try:
            from feature_store.writer import write_feature_snapshot
            from feature_store.registry import upload_registry

            _fs_df = pd.DataFrame(ctx.store_rows)
            write_feature_snapshot(ctx.date_str, _fs_df, ctx.bucket, prefix=cfg.FEATURE_STORE_PREFIX)
            upload_registry(ctx.bucket, prefix=cfg.FEATURE_STORE_PREFIX)
        except Exception as _fs_exc:
            log.warning("Feature store write failed (non-fatal): %s", _fs_exc)


def _run_meta_inference(ctx: PipelineContext) -> None:
    """Run Layer 1 specialized models → meta-model → predictions."""
    from data.feature_engineer import compute_features as _compute_features
    from model.meta_model import META_FEATURES

    mom_scorer = ctx.meta_models.get("momentum")
    vol_scorer = ctx.meta_models.get("volatility")
    regime_model = ctx.meta_models.get("regime")
    research_cal = ctx.meta_models.get("research_calibrator")
    meta_model = ctx.meta_models.get("meta")

    if mom_scorer is None and vol_scorer is None and meta_model is None:
        log.error("No meta-models available — falling back to v2.0")
        _run_gbm_inference(ctx)
        return

    # ── Step 1: Compute regime (once, market-wide) ───────────────────────────
    regime_probs = {"regime_bear": 0.33, "regime_neutral": 0.34, "regime_bull": 0.33}
    if regime_model is not None and regime_model.is_fitted:
        try:
            spy_s = ctx.macro.get("SPY") if ctx.macro else None
            vix_s = ctx.macro.get("VIX") if ctx.macro else None
            vix3m_s = ctx.macro.get("VIX3M") if ctx.macro else None
            tnx_s = ctx.macro.get("TNX") if ctx.macro else None
            irx_s = ctx.macro.get("IRX") if ctx.macro else None

            if spy_s is not None and len(spy_s) >= 20:
                # build_features expects dict of Close Series, not DataFrames
                _close_prices = {}
                for _tk, _df in (ctx.price_data or {}).items():
                    if _df is not None and not _df.empty and "Close" in _df.columns:
                        _close_prices[_tk] = _df["Close"].astype(float)
                regime_features_df = regime_model.build_features(
                    spy_s, vix_s, vix3m_s, tnx_s, irx_s, _close_prices,
                )
                if not regime_features_df.empty:
                    latest_regime = regime_features_df.iloc[-1]
                    regime_probs = regime_model.predict_single(latest_regime.to_dict())
                    log.info("Regime: bull=%.2f  neutral=%.2f  bear=%.2f",
                             regime_probs["regime_bull"], regime_probs["regime_neutral"],
                             regime_probs["regime_bear"])
        except Exception as e:
            log.warning("Regime prediction failed: %s — using uniform priors", e)

    # ── Step 2: Load research signals for calibrator ─────────────────────────
    # Read signals.json for composite scores and conviction
    research_signals = {}
    try:
        import boto3 as _b3_sig
        _s3_sig = _b3_sig.client("s3")
        sig_key = f"signals/{ctx.date_str}/signals.json"
        try:
            sig_obj = _s3_sig.get_object(Bucket=ctx.bucket, Key=sig_key)
            sig_data = json.loads(sig_obj["Body"].read())
            for sig in sig_data.get("universe", []):
                ticker = sig.get("ticker")
                if ticker:
                    research_signals[ticker] = sig
            log.info("Loaded %d research signals for calibrator", len(research_signals))
        except Exception:
            # Try previous day
            from datetime import datetime as _dt, timedelta as _td
            prev = (_dt.strptime(ctx.date_str, "%Y-%m-%d") - _td(days=1)).strftime("%Y-%m-%d")
            try:
                sig_obj = _s3_sig.get_object(Bucket=ctx.bucket, Key=f"signals/{prev}/signals.json")
                sig_data = json.loads(sig_obj["Body"].read())
                for sig in sig_data.get("universe", []):
                    ticker = sig.get("ticker")
                    if ticker:
                        research_signals[ticker] = sig
                log.info("Loaded %d research signals (previous day %s)", len(research_signals), prev)
            except Exception:
                log.info("No research signals available for calibrator")
    except Exception as e:
        log.warning("Research signal loading failed: %s", e)

    import json

    # ── Step 3: Per-ticker feature computation + Layer 1 inference ────────────
    max_r = getattr(cfg, "LABEL_CLIP", 0.15)

    for ticker in ctx.tickers:
        df = ctx.price_data.get(ticker, pd.DataFrame())
        if df.empty or len(df) < cfg.MIN_ROWS_FOR_FEATURES:
            ctx.n_skipped += 1
            continue

        sector_etf_sym = ctx.sector_map.get(ticker)
        sector_etf_series = ctx.macro.get(sector_etf_sym) if sector_etf_sym else None

        try:
            featured_df = _compute_features(
                df,
                spy_series=ctx.macro.get("SPY") if ctx.macro else None,
                vix_series=ctx.macro.get("VIX") if ctx.macro else None,
                sector_etf_series=sector_etf_series,
                tnx_series=ctx.macro.get("TNX") if ctx.macro else None,
                irx_series=ctx.macro.get("IRX") if ctx.macro else None,
                gld_series=ctx.macro.get("GLD") if ctx.macro else None,
                uso_series=ctx.macro.get("USO") if ctx.macro else None,
                vix3m_series=ctx.macro.get("VIX3M") if ctx.macro else None,
            )
        except Exception as exc:
            log.warning("Feature computation failed for %s: %s", ticker, exc)
            ctx.n_skipped += 1
            continue

        if featured_df.empty:
            ctx.n_skipped += 1
            continue

        latest = featured_df.iloc[-1]

        # Layer 1A: Momentum model
        momentum_score = 0.0
        if mom_scorer is not None:
            try:
                mom_x = latest[cfg.MOMENTUM_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
                momentum_score = float(mom_scorer.predict(mom_x)[0])
            except Exception:
                pass

        # Layer 1B: Volatility model
        expected_move = 0.0
        if vol_scorer is not None:
            try:
                vol_x = latest[cfg.VOLATILITY_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)
                expected_move = float(vol_scorer.predict(vol_x)[0])
            except Exception:
                pass

        # Layer 1C: Research calibrator
        research_cal_prob = 0.5  # neutral default
        research_score_norm = 0.5
        research_conviction = 0.0
        sector_modifier = 0.0
        sig = research_signals.get(ticker)
        if sig:
            raw_score = sig.get("score", 50)
            research_score_norm = raw_score / 100.0
            conv = sig.get("conviction", "stable")
            research_conviction = {"rising": 1.0, "stable": 0.0, "declining": -1.0}.get(conv, 0.0)
            sector_modifier = sig.get("sector_modifiers", {}).get(sig.get("sector", ""), 1.0) - 1.0
            if research_cal is not None and research_cal.is_fitted:
                research_cal_prob = research_cal.predict(raw_score)

        # Layer 2: Meta-model
        meta_features = {
            "research_calibrator_prob": research_cal_prob,
            "momentum_score": momentum_score,
            "expected_move": expected_move,
            "regime_bull": regime_probs["regime_bull"],
            "regime_bear": regime_probs["regime_bear"],
            "research_composite_score": research_score_norm,
            "research_conviction": research_conviction,
            "sector_macro_modifier": sector_modifier,
        }

        if meta_model is not None and meta_model.is_fitted:
            alpha = float(meta_model.predict_single(meta_features))
        else:
            # Fallback: weighted average of Layer 1 outputs
            alpha = 0.4 * momentum_score + 0.3 * (research_cal_prob - 0.5) * 0.1 + 0.2 * expected_move * np.sign(momentum_score) + 0.1 * (regime_probs["regime_bull"] - regime_probs["regime_bear"]) * 0.05

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
            "regime_bull": round(regime_probs["regime_bull"], 4),
            "regime_bear": round(regime_probs["regime_bear"], 4),
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

    # ── Cross-sectional confidence rescaling ────────────────────────────────
    # Meta-model ridge outputs cluster in ~[-0.01, +0.01] — far narrower than
    # LABEL_CLIP (0.15). The linear mapping p_up = 0.5 + alpha/(2*0.15)
    # compresses all confidences to ~0.48-0.53, disabling the veto gate.
    # Fix: rescale using the empirical alpha range from THIS batch so the
    # full [0, 1] confidence range is used.
    alphas = [p.get("predicted_alpha", 0) or 0 for p in ctx.predictions]
    if alphas:
        max_abs = max(abs(a) for a in alphas)
        meta_clip = max(max_abs, 1e-6)  # floor to avoid div-by-zero
        log.info(
            "Meta confidence rescaling: max_abs_alpha=%.6f  meta_clip=%.6f  (LABEL_CLIP=%.3f)",
            max_abs, meta_clip, max_r,
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

    log.info("Meta-inference complete: %d predictions, %d skipped", len(ctx.predictions), ctx.n_skipped)


def _run_gbm_inference(ctx: PipelineContext) -> None:
    """Batch GBM inference with cross-sectional rank normalization."""
    from data.feature_engineer import compute_features as _compute_features
    from scipy.stats import rankdata as _rankdata_inf

    gbm_feature_cols = list(cfg.GBM_FEATURES)

    # Validate feature alignment
    for label, sc in [("MSE", ctx.mse_scorer), ("Rank", ctx.rank_scorer)]:
        if sc is None or sc._booster is None:
            continue
        expected = sc._booster.num_feature()
        if len(gbm_feature_cols) != expected:
            if sc._feature_names and len(sc._feature_names) == expected:
                gbm_feature_cols = list(sc._feature_names)
                log.warning(
                    "%s model expects %d features but config has %d — "
                    "using model's feature list for inference",
                    label, expected, len(cfg.GBM_FEATURES),
                )
                break
            else:
                gbm_feature_cols = cfg.GBM_FEATURES[:expected]
                log.warning(
                    "%s model expects %d features but config has %d — "
                    "truncating to first %d features",
                    label, expected, len(cfg.GBM_FEATURES), expected,
                )
                break
        if sc._feature_names and list(sc._feature_names) != list(gbm_feature_cols):
            log.warning(
                "Feature ORDER mismatch — using %s model's feature order",
                label,
            )
            gbm_feature_cols = list(sc._feature_names)
            break

    ctx.gbm_feature_cols = gbm_feature_cols

    # Compute features per ticker
    raw_vectors: dict[str, np.ndarray] = {}
    for ticker in ctx.tickers:
        df = ctx.price_data.get(ticker, pd.DataFrame())
        if df.empty or len(df) < cfg.MIN_ROWS_FOR_FEATURES:
            ctx.n_skipped += 1
            continue
        sector_etf_sym = ctx.sector_map.get(ticker)
        sector_etf_series = ctx.macro.get(sector_etf_sym) if sector_etf_sym else None
        try:
            featured_df = _compute_features(
                df,
                spy_series=ctx.macro.get("SPY") if ctx.macro else None,
                vix_series=ctx.macro.get("VIX") if ctx.macro else None,
                sector_etf_series=sector_etf_series,
                tnx_series=ctx.macro.get("TNX") if ctx.macro else None,
                irx_series=ctx.macro.get("IRX") if ctx.macro else None,
                gld_series=ctx.macro.get("GLD") if ctx.macro else None,
                uso_series=ctx.macro.get("USO") if ctx.macro else None,
                earnings_data=ctx.earnings_all.get(ticker),
                revision_data=ctx.revision_all.get(ticker),
                options_data=ctx.options_all.get(ticker),
                fundamental_data=ctx.fundamental_all.get(ticker),
                vix3m_series=ctx.macro.get("VIX3M") if ctx.macro else None,
            )
        except Exception as exc:
            log.warning("Feature computation failed for %s: %s", ticker, exc)
            ctx.n_skipped += 1
            continue
        if featured_df.empty:
            ctx.n_skipped += 1
            continue
        latest = featured_df.iloc[-1]
        try:
            raw_vectors[ticker] = latest[gbm_feature_cols].to_numpy(dtype=np.float32)
        except KeyError:
            ctx.n_skipped += 1
            continue
        # Debug: log first ticker
        if len(raw_vectors) == 1:
            _fv = raw_vectors[ticker]
            log.info(
                "Feature debug %s: last_date=%s rsi=%.4f macd_cross=%.4f mom20d=%.4f hash=%s",
                ticker, featured_df.index[-1].date(),
                float(_fv[0]), float(_fv[1]), float(_fv[6]),
                hash(_fv.tobytes()),
            )
        # Feature store row
        try:
            row = {"ticker": ticker}
            for f in cfg.FEATURES:
                row[f] = float(latest[f]) if f in latest.index else 0.0
            ctx.store_rows.append(row)
        except Exception:
            pass

    if not raw_vectors:
        return

    # Compute cross-sectional dispersion from today's returns across the batch
    _xsect_disp_val = 0.0
    try:
        _today_rets = []
        for _tk, _df in ctx.price_data.items():
            if _df is not None and not _df.empty and "Close" in _df.columns and len(_df) >= 2:
                _last_ret = float((_df["Close"].iloc[-1] / _df["Close"].iloc[-2]) - 1.0)
                if np.isfinite(_last_ret):
                    _today_rets.append(_last_ret)
        if len(_today_rets) >= 10:
            _xsect_disp_val = float(np.std(_today_rets))
    except Exception:
        pass  # leave at 0.0

    # Inject xsect_dispersion into feature vectors and store_rows
    _disp_idx = None
    if "xsect_dispersion" in gbm_feature_cols:
        _disp_idx = gbm_feature_cols.index("xsect_dispersion")
    if _disp_idx is not None:
        for _tk in raw_vectors:
            raw_vectors[_tk][_disp_idx] = _xsect_disp_val
    for _sr in ctx.store_rows:
        _sr["xsect_dispersion"] = _xsect_disp_val

    # Cross-sectional rank normalization
    ordered_tickers = list(raw_vectors.keys())
    X_batch = np.stack([raw_vectors[t] for t in ordered_tickers])
    n_tickers = X_batch.shape[0]
    if n_tickers > 1:
        for f in range(X_batch.shape[1]):
            vals = X_batch[:, f]
            order = vals.argsort()
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(n_tickers, dtype=np.float32)
            unique_vals, inverse = np.unique(vals, return_inverse=True)
            if len(unique_vals) < n_tickers:
                for uv_idx in range(len(unique_vals)):
                    mask = inverse == uv_idx
                    if mask.sum() > 1:
                        ranks[mask] = ranks[mask].mean()
            X_batch[:, f] = ranks / max(n_tickers - 1, 1)
        log.info("Rank-normalized GBM features across %d tickers", n_tickers)
    else:
        X_batch[:, :] = 0.5

    log.info(
        "Ranked features debug %s: hash=%s first5=%s",
        ordered_tickers[0], hash(X_batch[0].tobytes()),
        X_batch[0, :5].tolist(),
    )

    # Run both models
    alpha_scores = np.full(n_tickers, np.nan)
    rank_model_ranks = np.full(n_tickers, np.nan)

    if ctx.mse_scorer is not None:
        try:
            alpha_scores = ctx.mse_scorer.predict(X_batch)
            log.info("MSE inference complete: %d tickers", n_tickers)
        except Exception as exc:
            log.error("MSE inference failed: %s", exc)

    if ctx.rank_scorer is not None:
        try:
            rank_raw = ctx.rank_scorer.predict(X_batch)
            valid_r = ~np.isnan(rank_raw)
            if valid_r.sum() > 0:
                ranks = _rankdata_inf(-rank_raw[valid_r], method="average")
                rank_model_ranks[valid_r] = ranks
            log.info("Lambdarank inference complete: %d tickers", n_tickers)
        except Exception as exc:
            log.warning("Lambdarank inference failed: %s", exc)

    # CatBoost inference + blending (if lgb_cat_blend mode)
    if getattr(ctx, "cat_scorer", None) is not None and ctx.inference_mode == "lgb_cat_blend":
        try:
            cat_scores = ctx.cat_scorer.predict(X_batch)
            bw = getattr(ctx, "blend_weights", None) or {"lgb": 0.5, "cat": 0.5}
            w_lgb = bw.get("lgb", 0.5)
            w_cat = bw.get("cat", 0.5)
            # Rank-normalize then blend (same approach as training)
            valid_lgb = ~np.isnan(alpha_scores)
            valid_cat = ~np.isnan(cat_scores)
            valid_both = valid_lgb & valid_cat
            if valid_both.sum() > 1:
                lgb_r = np.full(n_tickers, np.nan)
                cat_r = np.full(n_tickers, np.nan)
                lgb_r[valid_both] = _rankdata_inf(alpha_scores[valid_both], method="average") / valid_both.sum()
                cat_r[valid_both] = _rankdata_inf(cat_scores[valid_both], method="average") / valid_both.sum()
                # Map blended rank back to alpha scale: center and scale to match LGB range
                blended_rank = w_lgb * lgb_r + w_cat * cat_r
                # Re-scale to alpha range using LGB's mean/std as reference
                lgb_valid = alpha_scores[valid_both]
                blended_alpha = np.full(n_tickers, np.nan)
                # Simple approach: use rank-weighted average of actual alpha scores
                blended_alpha[valid_both] = w_lgb * alpha_scores[valid_both] + w_cat * cat_scores[valid_both]
                alpha_scores = blended_alpha
                log.info("LGB-Cat blend applied: w_lgb=%.2f w_cat=%.2f (%d tickers)", w_lgb, w_cat, valid_both.sum())
        except Exception as exc:
            log.warning("CatBoost blending failed — using LGB-only: %s", exc)

    if np.all(np.isnan(alpha_scores)) and not np.all(np.isnan(rank_model_ranks)):
        log.warning("MSE unavailable — no calibrated alpha scores this run")

    max_r = getattr(cfg, "LABEL_CLIP", 0.15)
    alpha_scores = np.clip(alpha_scores, -max_r, max_r)

    # MSE rank for comparison
    mse_ranks = np.full(n_tickers, np.nan)
    valid_a = ~np.isnan(alpha_scores)
    if valid_a.sum() > 0:
        mse_ranks[valid_a] = _rankdata_inf(-alpha_scores[valid_a], method="average")

    # Build prediction dicts
    for i, ticker in enumerate(ordered_tickers):
        alpha = float(alpha_scores[i])
        if np.isnan(alpha):
            ctx.n_skipped += 1
            continue

        # Calibrated confidence (Platt scaling if available, linear fallback)
        _cal = getattr(ctx, "calibrator", None)
        if _cal is not None and _cal.is_fitted:
            _cal_result = _cal.calibrate_prediction(alpha, label_clip=max_r)
            p_up = _cal_result["p_up"]
            p_down = _cal_result["p_down"]
            p_flat = _cal_result["p_flat"]
            predicted_direction = _cal_result["predicted_direction"]
            confidence = _cal_result["prediction_confidence"]
        else:
            p_up   = float(np.clip(0.5 + alpha / (2.0 * max_r), 0.0, 1.0))
            p_down = float(np.clip(0.5 - alpha / (2.0 * max_r), 0.0, 1.0))
            p_flat = float(max(0.0, 1.0 - p_up - p_down))
            if alpha >= 0:
                predicted_direction = "UP"
                confidence = p_up
            else:
                predicted_direction = "DOWN"
                confidence = p_down

        result = {
            "ticker":                ticker,
            "predicted_direction":   predicted_direction,
            "prediction_confidence": round(confidence, 4),
            "predicted_alpha":       round(alpha, 6),
            "p_up":                  round(p_up, 4),
            "p_flat":               round(p_flat, 4),
            "p_down":                round(p_down, 4),
            "mse_rank":              int(mse_ranks[i]) if not np.isnan(mse_ranks[i]) else None,
            "model_rank":            int(rank_model_ranks[i]) if not np.isnan(rank_model_ranks[i]) else None,
            "combined_rank":         None,
        }
        if ticker in ctx.ticker_data_age:
            result["price_data_age_days"] = ctx.ticker_data_age[ticker]
        if ctx.ticker_sources:
            result["watchlist_source"] = ctx.ticker_sources.get(ticker, "unknown")
        earnings_info = ctx.earnings_all.get(ticker, {})
        if earnings_info.get("next_earnings_days") is not None:
            result["next_earnings_days"] = earnings_info["next_earnings_days"]

        # Multi-horizon predictions (if auxiliary models loaded)
        if ctx.horizon_scorers:
            horizon_alphas = {}
            horizon_directions = []
            primary_dir = 1 if alpha >= 0 else -1
            horizon_directions.append(primary_dir)
            for h, h_scorer in ctx.horizon_scorers.items():
                try:
                    h_alpha = float(h_scorer.predict(X_batch[i:i+1])[0])
                    horizon_alphas[f"alpha_{h}d"] = round(h_alpha, 6)
                    horizon_directions.append(1 if h_alpha >= 0 else -1)
                except Exception:
                    pass
            if horizon_alphas:
                result.update(horizon_alphas)
                # Horizon agreement: fraction of horizons agreeing on direction
                n_agree = sum(1 for d in horizon_directions if d == primary_dir)
                result["horizon_agreement"] = round(n_agree / len(horizon_directions), 2)

        ctx.predictions.append(result)


def _run_mlp_inference(ctx: PipelineContext) -> None:
    """Per-ticker MLP inference (legacy path)."""
    from inference.daily_predict import predict_ticker

    norm_stats = ctx.checkpoint.get("norm_stats", {})
    for ticker in ctx.tickers:
        df = ctx.price_data.get(ticker, pd.DataFrame())
        sector_etf_sym = ctx.sector_map.get(ticker)
        sector_etf_series = ctx.macro.get(sector_etf_sym) if sector_etf_sym else None
        result = predict_ticker(
            ticker, df, ctx.model, norm_stats,
            macro=ctx.macro,
            sector_etf_series=sector_etf_series,
        )
        if result is not None:
            if ticker in ctx.ticker_data_age:
                result["price_data_age_days"] = ctx.ticker_data_age[ticker]
            if ctx.ticker_sources:
                result["watchlist_source"] = ctx.ticker_sources.get(ticker, "unknown")
            ctx.predictions.append(result)
        else:
            ctx.n_skipped += 1


def _expand_feature_store(ctx: PipelineContext) -> None:
    """Compute technical features for non-watchlist tickers in the slim cache."""
    try:
        from data.feature_engineer import compute_features as _fs_compute

        _watchlist_set = set(ctx.tickers)
        _skip_set = {"SPY", "^VIX", "^VIX3M", "^TNX", "^IRX", "GLD", "USO", "VIX3M"}
        _skip_set.update(s for s in ctx.price_data if s.startswith("XL"))

        _universe_tickers = [
            t for t in ctx.price_data
            if t not in _watchlist_set and t not in _skip_set
            and ctx.price_data[t] is not None and len(ctx.price_data[t]) >= cfg.MIN_ROWS_FOR_FEATURES
        ]
        _n_universe = len(_universe_tickers)

        if _n_universe > 0:
            log.info("Feature store universe expansion: %d non-watchlist tickers", _n_universe)
            _fs_spy = ctx.macro.get("SPY") if ctx.macro else None
            _fs_vix = ctx.macro.get("VIX") if ctx.macro else None
            _fs_tnx = ctx.macro.get("TNX") if ctx.macro else None
            _fs_irx = ctx.macro.get("IRX") if ctx.macro else None
            _fs_gld = ctx.macro.get("GLD") if ctx.macro else None
            _fs_uso = ctx.macro.get("USO") if ctx.macro else None
            _fs_vix3m = ctx.macro.get("VIX3M") if ctx.macro else None

            _fs_ok = 0
            _fs_err = 0
            for _fs_ticker in _universe_tickers:
                if ctx.near_timeout():
                    log.warning("Soft timeout during universe expansion — wrote %d/%d", _fs_ok, _n_universe)
                    break
                try:
                    _fs_etf_sym = ctx.sector_map.get(_fs_ticker)
                    _fs_etf_series = ctx.macro.get(_fs_etf_sym) if _fs_etf_sym else None
                    _fs_featured = _fs_compute(
                        ctx.price_data[_fs_ticker],
                        spy_series=_fs_spy, vix_series=_fs_vix,
                        sector_etf_series=_fs_etf_series,
                        tnx_series=_fs_tnx, irx_series=_fs_irx,
                        gld_series=_fs_gld, uso_series=_fs_uso,
                        vix3m_series=_fs_vix3m,
                    )
                    if not _fs_featured.empty:
                        _fs_latest = _fs_featured.iloc[-1]
                        _fs_row = {"ticker": _fs_ticker}
                        for f in cfg.FEATURES:
                            _fs_row[f] = float(_fs_latest[f]) if f in _fs_latest.index else 0.0
                        ctx.store_rows.append(_fs_row)
                        _fs_ok += 1
                except Exception:
                    _fs_err += 1

            log.info(
                "Feature store universe expansion: %d OK, %d errors, %d total store rows",
                _fs_ok, _fs_err, len(ctx.store_rows),
            )
    except Exception as _fs_univ_exc:
        log.warning("Feature store universe expansion failed (non-fatal): %s", _fs_univ_exc)
