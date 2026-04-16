"""
training/train_handler.py — Weekly GBM retraining pipeline for Lambda.

Called by inference/handler.py when event["action"] == "train".

Pipeline:
  1. Download per-ticker OHLCV Parquet files + sector_map.json from S3.
  1b. Refresh any stale parquets with recent yfinance data and upload back to S3.
  2. Build regression arrays (29 features) and apply 70/15/15 time-based split
     with purge gaps of FORWARD_DAYS between train/val and val/test.
  3. Train GBMScorer with Optuna-tuned params from config.GBM_TUNED_PARAMS
     (n_estimators=2000, early_stopping=50).
  4. Evaluate on test set: IC, IC IR, positive-period rate.
  5. Upload dated backup unconditionally; promote to gbm_latest.txt only if IC gate passes.
  5b. Write slim 2-year price cache to predictor/price_cache_slim/ so the daily
      inference Lambda can skip the 2y yfinance fetch and read from S3 instead.
  6. Send training summary email with results.

S3 layout:
  predictor/price_cache/*.parquet       — adjusted OHLCV price history per ticker (10y,
                                          auto_adjust=True; rewritten weekly by this handler)
  predictor/price_cache/sector_map.json — ticker → sector ETF symbol
  predictor/weights/gbm_latest.txt      — (output) active inference weights
  predictor/weights/gbm_{date}.txt      — (output) dated backup

  predictor/daily_closes/{date}.parquet — Backward-split-adjusted OHLCV snapshot per
                                          trading day.  Written by DataPhase1 in
                                          alpha-engine-data/collectors/daily_closes.py
                                          as the first step of the weekday Step Function,
                                          before the predictor inference Lambda runs.
                                          Used as the Mon–Fri delta source by
                                          load_price_data_from_cache(), reducing daily
                                          fetches to only the ~1-2 split tickers per week.
  predictor/price_cache_slim/*.parquet  — 2-year slice of each ticker (written by
                                          write_slim_cache() after each weekly training run).
                                          The inference Lambda downloads this (~9 MB) instead
                                          of re-fetching 2y from yfinance each morning.

Lambda environment variables (same as predictor email):
  EMAIL_SENDER / EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD / AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import tempfile
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from ssm_secrets import load_secrets

load_secrets()

import numpy as np

log = logging.getLogger(__name__)


# ── S3 data download ───────────────────────────────────────────────────────────

def download_price_cache(bucket: str, prefix: str, local_dir: Path) -> int:
    """
    Download all objects under S3 prefix (Parquets + sector_map.json) to local_dir.
    Returns number of files downloaded.

    DEAD CODE — flagged 2026-04-14 (PR #6).

    This was the S3-parquet fallback path used by ``main()`` when ArcticDB
    was unavailable. PR #6 removed that fallback — ArcticDB is now the
    single source of training data. Zero callers remain after this PR;
    ``store/arctic_reader.py`` docstring still references the signature
    but doesn't invoke it.

    Slated for deletion in a follow-up cleanup PR once PR #6's training
    cutover has soaked for 2+ Saturday runs against live ArcticDB.
    """
    import boto3
    s3 = boto3.client("s3")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]   # strip prefix, keep filename
            if not filename or filename.endswith("/"):
                continue
            local_path = local_dir / filename
            s3.download_file(bucket, key, str(local_path))
            n += 1

    log.info("Downloaded %d files from s3://%s/%s → %s", n, bucket, prefix, local_dir)
# ── Training email ─────────────────────────────────────────────────────────────

def _build_ic_table_html(result, is_meta, ic_color, ic_label, promoted_mode,
                         promo_color, promo_label, val_ic, mse_ic, test_ic,
                         rank_ic, ensemble_ic, ensemble_on, ic_ir, ic_pos):
    """Build the IC metrics table HTML for the training email."""
    _bg_style = ' style="background:#f9f9f9;"'
    _row = lambda label, value, bg=False: (
        f'<tr{_bg_style if bg else ""}>'
        f'<td style="padding:5px 10px; color:#555; width:160px;">{label}</td>'
        f'<td style="padding:5px 10px; font-family:monospace; font-weight:bold;">{value}</td></tr>'
    )

    rows = '<table style="border-collapse:collapse; width:100%; margin-bottom:12px;">'

    if is_meta:
        meta_ic = result.get("meta_model_ic", test_ic)
        mom_ic = result.get("momentum_test_ic", mse_ic)
        vol_ic = result.get("volatility_test_ic", 0)
        rows += _row("Meta-Model IC", f'{meta_ic:.4f}', bg=True)
        rows += _row("Momentum IC", f'{mom_ic:.4f}')
        rows += _row("Volatility IC", f'{vol_ic:.4f}', bg=True)
        # Regime classifier removed from the critical path 2026-04-16 (Tier 0
        # model retired; raw macro features now feed the ridge directly).
        # Tier 1 regime model will re-introduce regime rows here when it
        # ships, with metrics tied to a named baseline.
        coefs = result.get("meta_coefficients", {})
        if coefs:
            coef_str = " | ".join(
                f'{k}={v:+.3f}' for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))
                if k != "intercept" and abs(v) > 0.0001
            )
            rows += _row("Meta Coefficients", coef_str, bg=True)
        # Feature importance (Phase 4): show standardized coefficients +
        # permutation IC drop side-by-side. Standardized reveals which features
        # drive predictions independent of raw scale; permutation cross-checks
        # against ridge's linearity assumption. Together they inform the
        # classifier-output-vs-raw-macro pruning decision.
        importance = result.get("meta_importance", {})
        std_coef = importance.get("standardized_coef", {}) if importance else {}
        perm = importance.get("permutation", {}) if importance else {}
        if std_coef:
            top = sorted(std_coef.items(), key=lambda x: -abs(x[1]))[:8]
            imp_rows = "".join(
                f'<tr><td style="padding:2px 8px; font-family:monospace; color:#555;">{name}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; text-align:right;">{std:+.3f}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; text-align:right;">{perm.get(name, 0):+.4f}</td></tr>'
                for name, std in top
            )
            imp_html = (
                '<table style="border-collapse:collapse; font-size:11px; margin-top:4px;">'
                '<tr style="color:#555;">'
                '<th style="padding:2px 8px; text-align:left;">feature</th>'
                '<th style="padding:2px 8px; text-align:right;">std_coef</th>'
                '<th style="padding:2px 8px; text-align:right;">perm_ic_drop</th>'
                '</tr>'
                f'{imp_rows}</table>'
            )
            rows += (
                '<tr>'
                '<td style="padding:5px 10px; color:#555; vertical-align:top;">Feature Importance</td>'
                f'<td style="padding:5px 10px;">{imp_html}</td>'
                '</tr>'
            )
    else:
        rows += _row("Val IC", f'{val_ic:.4f}', bg=True)
        rows += _row("MSE Model IC",
                      f'{mse_ic:.4f}{f" — {ic_label}" if promoted_mode == "mse" else ""}')
        if ensemble_on and rank_ic is not None:
            rows += _row("Lambdarank IC",
                          f'{rank_ic:.4f}{f" — {ic_label}" if promoted_mode == "rank" else ""}', bg=True)
            rows += _row("Ensemble IC",
                          f'{ensemble_ic:.4f}{f" — {ic_label}" if promoted_mode == "ensemble" else ""}')
        else:
            rows += _row("Test IC", f'<span style="color:{ic_color};">{test_ic:.4f} — {ic_label}</span>')
        rows += _row("IC IR", f'{ic_ir:.3f} ({ic_pos}/20 positive)', bg=True)

    rows += (
        f'<tr style="background:#f9f9f9;">'
        f'<td style="padding:5px 10px; color:#555; font-weight:bold;">Promotion</td>'
        f'<td style="padding:5px 10px; font-weight:bold; color:{promo_color};">{promo_label}</td></tr>'
    )
    rows += '</table>'
    return rows


def send_training_email(result: dict, date_str: str) -> bool:
    """
    Send GBM training summary email via Gmail SMTP (primary) or SES (fallback).
    Returns True on success. Never raises.
    """
    import config as cfg

    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info("Training email skipped — EMAIL_SENDER/EMAIL_RECIPIENTS not set")
        return False

    promoted     = result.get("promoted", False)
    promoted_mode = result.get("promoted_mode")
    passes_ic    = result.get("passes_ic_gate", False)
    val_ic       = result.get("val_ic", 0.0)
    test_ic      = result.get("test_ic", 0.0)
    mse_ic       = result.get("mse_ic", test_ic)
    rank_ic      = result.get("rank_ic")
    ensemble_ic  = result.get("ensemble_ic")
    ensemble_on  = result.get("ensemble_enabled", False)
    ic_ir        = result.get("ic_ir", 0.0)
    version      = result.get("model_version", "unknown")
    is_meta      = "meta" in str(version).lower()
    elapsed_s    = result.get("elapsed_s", 0)
    n_train      = result.get("n_train", 0)
    ic_pos       = result.get("ic_positive_20", 0)
    top10        = result.get("feature_importance_top10", [])

    ic_color    = "#2e7d32" if passes_ic else "#c62828"
    ic_label    = "PASS ✓" if passes_ic else "FAIL ✗"
    promo_color = "#2e7d32" if promoted else "#c62828"

    # Build the promotion label. Previously this was hardcoded to
    # "NOT promoted (IC gate failed ✗)" for all non-promotion cases, which
    # was actively misleading: the 2026-04-11 v3.0-meta run produced
    # Meta-Model IC 0.0525 (well above the 0.03 gate) but was blocked by
    # the walk-forward validation on the momentum base model. The email
    # said "IC gate failed" when the IC gate actually passed.
    def _build_failure_reason() -> str:
        wf = result.get("walk_forward") or {}
        if not wf:
            return "IC gate failed"  # fallback — no wf info available
        mom = wf.get("momentum_median_ic")
        vol = wf.get("volatility_median_ic")
        reasons: list[str] = []
        if mom is not None and mom <= 0:
            reasons.append(f"momentum median IC {mom:+.4f}")
        if vol is not None and vol <= 0:
            reasons.append(f"volatility median IC {vol:+.4f}")
        if reasons:
            return "walk-forward failed: " + ", ".join(reasons)
        # wf section present but no negative medians — must be the
        # in-sample IC gate that failed
        return "IC gate failed"

    promo_label = (
        f"Promoted → weights/meta/ ✓" if promoted and is_meta
        else f"Promoted → gbm_latest ({promoted_mode}) ✓" if promoted
        else f"NOT promoted ({_build_failure_reason()}) ✗"
    )
    status_str  = "PASS" if passes_ic else "FAIL"

    # CatBoost metrics for email
    cat_enabled  = result.get("catboost_enabled", False)
    cat_ic_val   = result.get("catboost_ic")
    blend_ic_val = result.get("lgb_cat_blend_ic")
    blend_wts    = result.get("blend_weights")
    cal_metrics  = result.get("calibration")
    mh_data      = result.get("multi_horizon")

    subject = (
        f"Alpha Engine Training | {date_str} | "
        f"{'Meta IC' if is_meta else 'IC'} {test_ic:.4f} {status_str} | "
        f"{'Promoted (' + promoted_mode + ')' if promoted else 'Not promoted'}"
    )

    # Feature importance bar chart (top 5)
    max_gain  = max((r["gain"] for r in top10), default=1)
    feat_rows = "".join(
        f'<tr>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["feature"]}</td>'
        f'<td style="padding:2px 8px; width:130px;">'
        f'<div style="background:#1976d2; height:10px; width:{int(r["gain"]/max_gain*100)}%;"></div>'
        f'</td>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["gain"]:.1f}</td>'
        f'</tr>'
        for r in top10[:5]
    )

    # Walk-forward section for email
    wf_data = result.get("walk_forward")
    wf_html = ""
    wf_plain = ""
    is_meta = "meta" in str(result.get("model_version", "")).lower()
    if wf_data and wf_data.get("folds"):
        # Meta-model uses per-model median ICs; v2.0 uses single median_ic
        wf_median = wf_data.get("median_ic") or wf_data.get("momentum_median_ic", 0.0)
        wf_pct = wf_data.get("pct_positive", 0.0)
        wf_pass = wf_data.get("passes_wf", False)
        wf_color = "#2e7d32" if wf_pass else "#c62828"
        wf_label = "PASS ✓" if wf_pass else "FAIL ✗"

        if is_meta:
            mom_median = wf_data.get("momentum_median_ic", "n/a")
            vol_median = wf_data.get("volatility_median_ic", "n/a")
            fold_rows = "".join(
                f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["fold"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_start"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_end"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:{"#2e7d32" if f.get("mom_ic", 0) > 0 else "#c62828"};">{f.get("mom_ic", 0):.4f}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:#2e7d32;">{f.get("vol_ic", 0):.4f}</td>'
                f'</tr>'
                for i, f in enumerate(wf_data["folds"])
            )
            wf_html = (
                f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
                f'({len(wf_data["folds"])} folds)</h3>'
                f'<p style="font-size:12px; margin:2px 0;">Momentum median IC: <b>{mom_median}</b> '
                f'&nbsp;|&nbsp; Volatility median IC: <b>{vol_median}</b> '
                f'&nbsp;|&nbsp; Status: <b style="color:{wf_color};">{wf_label}</b></p>'
                f'<table style="border-collapse:collapse; width:100%; font-size:11px;">'
                f'<tr style="background:#e0e0e0;">'
                f'<th style="padding:3px 6px;">Fold</th>'
                f'<th style="padding:3px 6px;">Test Start</th>'
                f'<th style="padding:3px 6px;">Test End</th>'
                f'<th style="padding:3px 6px;">Mom IC</th>'
                f'<th style="padding:3px 6px;">Vol IC</th></tr>'
                f'{fold_rows}</table>'
            )
            wf_plain = (
                f"\n--- Walk-Forward ({len(wf_data['folds'])} folds) ---"
                f"\nMomentum median IC: {mom_median}  |  Volatility median IC: {vol_median}"
                f"\nStatus: {wf_label}\n"
                + "\n".join(
                    f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                    f"mom={f.get('mom_ic', 0):.4f}  vol={f.get('vol_ic', 0):.4f}"
                    for f in wf_data["folds"]
                ) + "\n"
            )
        else:
            fold_rows = "".join(
                f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["fold"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_start"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_end"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["n_train"]:,}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:{"#2e7d32" if f["ic"] > 0 else "#c62828"};">{f["ic"]:.4f}</td>'
                f'</tr>'
                for i, f in enumerate(wf_data["folds"])
            )
            wf_html = (
                f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
                f'({len(wf_data["folds"])} folds)</h3>'
                f'<p style="font-size:12px; margin:2px 0;">Median IC: <b style="color:{wf_color};">'
                f'{wf_median:.4f}</b> — {wf_label} &nbsp;|&nbsp; '
                f'Positive folds: <b>{wf_pct*100:.0f}%</b></p>'
                f'<table style="border-collapse:collapse; width:100%; font-size:11px;">'
                f'<tr style="background:#e0e0e0;">'
                f'<th style="padding:3px 6px;">Fold</th>'
                f'<th style="padding:3px 6px;">Test Start</th>'
                f'<th style="padding:3px 6px;">Test End</th>'
                f'<th style="padding:3px 6px;">Train N</th>'
                f'<th style="padding:3px 6px;">IC</th></tr>'
                f'{fold_rows}</table>'
            )
            wf_plain = (
                f"\n--- Walk-Forward ({len(wf_data['folds'])} folds) ---"
                f"\nMedian IC: {wf_median:.4f} — {wf_label}"
                f"\nPositive folds: {wf_pct*100:.0f}%\n"
                + "\n".join(
                    f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                    f"train={f['n_train']:,}  IC={f['ic']:.4f}"
                    for f in wf_data["folds"]
                ) + "\n"
            )

    # ── SHAP vs Gain comparison section ──────────────────────────────────────
    shap_top10    = result.get("feature_importance_shap_top10", [])
    shap_stability = result.get("shap_rank_stability")

    shap_html = ""
    shap_plain = ""
    if shap_top10 and top10:
        # Build rank lookup: feature → rank (1-based)
        gain_rank = {r["feature"]: i + 1 for i, r in enumerate(top10)}
        shap_rank = {r["feature"]: i + 1 for i, r in enumerate(shap_top10)}
        all_features = list(dict.fromkeys(
            [r["feature"] for r in top10] + [r["feature"] for r in shap_top10]
        ))

        comparison_rows = ""
        comparison_plain_lines = []
        for feat in all_features[:10]:
            g_rank = gain_rank.get(feat, "-")
            s_rank = shap_rank.get(feat, "-")
            divergence = ""
            if isinstance(g_rank, int) and isinstance(s_rank, int):
                diff = abs(g_rank - s_rank)
                if diff > 3:
                    divergence = f' style="color:#c62828; font-weight:bold;"'
            comparison_rows += (
                f'<tr>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{feat}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;">{g_rank}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;"'
                f'{divergence}>{s_rank}</td>'
                f'</tr>'
            )
            flag = " ***" if isinstance(g_rank, int) and isinstance(s_rank, int) and abs(g_rank - s_rank) > 3 else ""
            comparison_plain_lines.append(f"  {feat:<22} Gain:{g_rank}  SHAP:{s_rank}{flag}")

        stability_note = ""
        stability_plain = ""
        if shap_stability is not None:
            stab_color = "#2e7d32" if shap_stability >= 0.80 else "#c62828"
            stab_label = "stable" if shap_stability >= 0.80 else "DRIFT WARNING"
            stability_note = (
                f'<p style="font-size:12px; margin:4px 0;">SHAP rank stability (vs last week): '
                f'<b style="color:{stab_color};">rho={shap_stability:.4f} — {stab_label}</b></p>'
            )
            stability_plain = f"\nSHAP rank stability: rho={shap_stability:.4f} — {stab_label}"

        shap_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Importance: Gain vs SHAP</h3>'
            f'{stability_note}'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">Gain Rank</th>'
            f'<th style="padding:3px 8px;">SHAP Rank</th></tr>'
            f'{comparison_rows}</table>'
            f'<p style="font-size:10px; color:#888;">Features with rank divergence &gt;3 highlighted in red.</p>'
        )
        shap_plain = (
            "\n--- Gain vs SHAP Rank ---"
            + stability_plain
            + "\n" + "\n".join(comparison_plain_lines)
            + "\n  (*** = rank divergence > 3)\n"
        )

    # ── Feature Health section (per-feature IC + noise detection) ──────────
    feat_ics = result.get("feature_ics", {})
    noise_cands = result.get("noise_candidates", [])
    feat_health_html = ""
    feat_health_plain = ""
    if feat_ics:
        sorted_ics = sorted(feat_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        ic_rows = "".join(
            f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px;">{fname}</td>'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px; '
            f'color:{"#2e7d32" if fic > 0 else "#c62828"};">{fic:.4f}</td>'
            f'</tr>'
            for i, (fname, fic) in enumerate(sorted_ics[:10])
        )
        noise_note = ""
        if noise_cands:
            noise_note = (
                f'<p style="font-size:11px; color:#c62828; margin:4px 0;">'
                f'Noise candidates ({len(noise_cands)}): {", ".join(noise_cands)}</p>'
            )
        feat_health_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Health</h3>'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">IC vs Forward</th></tr>'
            f'{ic_rows}</table>'
            f'{noise_note}'
        )
        feat_health_plain = (
            "\n--- Feature Health (top 10 by |IC|) ---\n"
            + "\n".join(f"  {fname:<22} IC={fic:.4f}" for fname, fic in sorted_ics[:10])
            + (f"\nNoise candidates: {', '.join(noise_cands)}" if noise_cands else "")
            + "\n"
        )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:600px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Training — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{version}</b> &nbsp;|&nbsp;'
        f'Training samples: <b>{n_train:,}</b> &nbsp;|&nbsp;'
        f'Elapsed: <b>{elapsed_s:.0f}s ({elapsed_s/60:.1f} min)</b></p>'

        + _build_ic_table_html(result, is_meta, ic_color, ic_label, promoted_mode,
                               promo_color, promo_label, val_ic, mse_ic, test_ic,
                               rank_ic, ensemble_ic, ensemble_on, ic_ir, ic_pos) +

        f'{wf_html}'

        + (
            f'<h3 style="margin-bottom:4px;">Top 5 Features by Gain</h3>'
            f'<table>{feat_rows}</table>'
            if feat_rows else ""
        ) +

        f'{shap_html}'

        f'{feat_health_html}'

        + (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Confidence Calibration</h3>'
            f'<p style="font-size:12px;">Method: <b>{cal_metrics["method"]}</b> &nbsp;|&nbsp; '
            f'Samples: <b>{cal_metrics["n_samples"]:,}</b> &nbsp;|&nbsp; '
            f'ECE: <b>{cal_metrics["ece_before"]:.4f} → {cal_metrics["ece_after"]:.4f}</b> '
            f'({(1 - cal_metrics["ece_after"] / max(cal_metrics["ece_before"], 1e-8)) * 100:.0f}% reduction)</p>'
            if cal_metrics and cal_metrics.get("fitted") else ""
        )
        + (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Multi-Horizon Models</h3>'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Horizon</th>'
            f'<th style="padding:3px 8px;">IC</th>'
            f'<th style="padding:3px 8px;">Promoted</th></tr>'
            + "".join(
                f'<tr><td style="padding:2px 8px; font-family:monospace;">{h}d</td>'
                f'<td style="padding:2px 8px; font-family:monospace;">{v.get("test_ic", "err")}</td>'
                f'<td style="padding:2px 8px; font-family:monospace;">{v.get("promoted", False)}</td></tr>'
                for h, v in mh_data["auxiliary"].items() if isinstance(v, dict) and "error" not in v
            )
            + f'</table>'
            if mh_data and mh_data.get("auxiliary") else ""
        ) +

        f'<p style="font-size:11px; color:#aaa; margin-top:20px;">'
        # Meta (v3.0) trainer uses a simpler walk-forward gate:
        # both base models must have strictly positive median IC.
        # See meta_trainer.py:483. The v2 single-model path uses
        # cfg.WF_MEDIAN_IC_GATE + WF_MIN_FOLDS_POSITIVE. Describe
        # whichever one actually gates this run.
        + (
            f'IC gate: ≥{cfg.MIN_IC:.2f} meta IC to promote &nbsp;|&nbsp; '
            f'Walk-forward: momentum &amp; volatility median IC both &gt; 0</p>'
            if is_meta else
            f'IC gate: ≥{cfg.MIN_IC:.2f} to promote &nbsp;|&nbsp; '
            f'Walk-forward: median IC ≥{cfg.WF_MEDIAN_IC_GATE:.2f}, '
            f'{cfg.WF_MIN_FOLDS_POSITIVE*100:.0f}%+ positive folds</p>'
        )
        + f'</body></html>'
    )

    if is_meta:
        meta_ic = result.get("meta_model_ic", test_ic)
        mom_ic = result.get("momentum_test_ic", mse_ic)
        vol_ic_val = result.get("volatility_test_ic", 0)
        plain_body = (
            f"Alpha Engine Training — {date_str}\n"
            f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
            f"\nMeta-Model IC:      {meta_ic:.4f} — {ic_label}"
            f"\nMomentum IC:        {mom_ic:.4f}"
            f"\nVolatility IC:      {vol_ic_val:.4f}"
            f"\nPromotion:          {promo_label}\n"
            f"{wf_plain}"
        )
        coefs = result.get("meta_coefficients", {})
        if coefs:
            plain_body += "\nMeta coefficients:\n" + "\n".join(
                f"  {k:<28} {v:+.4f}"
                for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))
                if k != "intercept" and abs(v) > 0.0001
            ) + "\n"
    else:
        _mse_mark  = " ✓" if promoted_mode == "mse" else ""
        _rank_mark = " ✓" if promoted_mode == "rank" else ""
        _ens_mark  = " ✓" if promoted_mode == "ensemble" else ""
        plain_body = (
            f"Alpha Engine Training — {date_str}\n"
            f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
            f"\nVal IC:             {val_ic:.4f}"
            f"\nMSE Model IC:       {mse_ic:.4f}{' — ' + ic_label if promoted_mode == 'mse' else ''}{_mse_mark}"
            + (
                f"\nLambdarank Model IC: {rank_ic:.4f}{' — ' + ic_label if promoted_mode == 'rank' else ''}{_rank_mark}"
                f"\nEnsemble IC:        {ensemble_ic:.4f}{' — ' + ic_label if promoted_mode == 'ensemble' else ''}{_ens_mark}"
                if ensemble_on and rank_ic is not None and ensemble_ic is not None else
                f"\nTest IC:            {test_ic:.4f} — {ic_label}"
            )
            + (
                f"\nCatBoost IC:        {cat_ic_val:.4f}"
                f"\nLGB-Cat Blend IC:   {blend_ic_val:.4f} (w_lgb={blend_wts['lgb']:.1f})"
                if cat_enabled and cat_ic_val is not None else ""
            ) +
            f"\nPromoted:           {promoted_mode if promoted else 'none'}"
            f"\nIC IR:              {ic_ir:.3f} ({ic_pos}/20 positive)"
            f"\nPromotion:          {promo_label}\n"
            f"{wf_plain}"
            f"\nTop features: " + ", ".join(r["feature"] for r in top10[:5])
            + f"\n{shap_plain}"
            + f"{feat_health_plain}\n"
        )

    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if app_password:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body,  "html",  "utf-8"))
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender, app_password.replace(" ", ""))
                server.sendmail(sender, recipients, msg.as_string())
            log.info("Training email sent via Gmail SMTP: '%s'", subject)
            return True
        except Exception as exc:
            log.warning("Gmail SMTP failed (%s) — trying SES fallback", exc)

    # SES fallback
    try:
        import boto3
        ses = boto3.client("ses", region_name=cfg.AWS_REGION)
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body,  "Charset": "UTF-8"},
                },
            },
        )
        log.info("Training email sent via SES: '%s'", subject)
        return True
    except Exception as exc:
        log.warning("SES failed: %s — training email not delivered", exc)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Entry point called from inference/handler.py for the "train" action.

    1.  Download price cache from S3 to /tmp.
    1b. Refresh any stale parquets with recent yfinance data → upload back to S3.
    2.  Run GBM training pipeline on the now-current local cache.
    2b. Write a 2-year slim cache to S3 (predictor/price_cache_slim/).
        The inference Lambda uses this slim cache + daily_closes delta instead
        of fetching 2 years from yfinance every morning, reducing daily yfinance
        API calls by ~125×.
    3.  Send training summary email.

    Returns the result dict from run_gbm_training().
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    fd = None

    # Structured logging + flow-doctor via alpha-engine-lib (same as
    # inference handler). Training runs on EC2 spot which doesn't
    # configure logging elsewhere, so this needs to happen at entry.
    from alpha_engine_lib.logging import setup_logging
    setup_logging(
        "predictor-training",
        flow_doctor_yaml=str(Path(__file__).parent.parent / "flow-doctor-training.yaml"),
    )
    logging.getLogger().setLevel(logging.INFO)

    # Preflight — fail fast on env / connectivity / ArcticDB staleness
    # before a 90-minute training run commits to stale data. See PR #6
    # and training/preflight.py.
    from training.preflight import TrainingPreflight
    TrainingPreflight(bucket=bucket).run()

    log.info("GBM training run: date=%s  bucket=%s  dry_run=%s", date_str, bucket, dry_run)

    # Step 1: Load price cache from ArcticDB. PR #6 removed the S3 parquet
    # fallback — it masked the PR #3 miswiring for a week (same root cause
    # as the inference-path issue fixed in PR #5). Preflight above already
    # verified macro/SPY freshness, so ArcticDB is known reachable here.
    tmp_cache = Path(tempfile.mkdtemp()) / "cache"
    from store.arctic_reader import download_from_arctic
    log.info("[data_source=arcticdb] Loading universe from ArcticDB...")
    n_files = download_from_arctic(bucket=bucket, local_dir=tmp_cache)

    if n_files == 0:
        raise RuntimeError(
            f"ArcticDB returned zero files for training — "
            f"universe library is empty or unreachable at bucket={bucket}. "
            "Check Saturday DataPhase1 + weekly backfill ran cleanly."
        )

    # Step 1b: Price cache refresh now handled by alpha-engine-data (Phase 1).
    # The data repo runs weekly_collector.py --phase 1 before training,
    # so S3 parquets are already current.
    log.info("Price cache refresh: skipped (handled by alpha-engine-data)")

    # Step 2: Train + upload (v3 meta-model only — v2 single-GBM and
    # multi-horizon dispatch branches removed 2026-04-13; v2 machinery
    # itself still in-tree, full rip-out tracked in ROADMAP).
    from training.meta_trainer import run_meta_training
    result = run_meta_training(
        data_dir=str(tmp_cache),
        bucket=bucket,
        date_str=date_str,
        dry_run=dry_run,
    )

    # Step 2b: Slim cache write now handled by alpha-engine-data (Phase 1).
    # The data repo writes price_cache_slim/ from the full cache after refresh.
    log.info("Slim cache write: skipped (handled by alpha-engine-data)")

    # Feature store registry upload removed — alpha-engine-data handles this now.

    # Step 2d: Write training summary to S3 (works for all modes)
    if not dry_run:
        try:
            import boto3 as _b3_sum
            _s3_sum = _b3_sum.client("s3")
            _sum_body = json.dumps(result, indent=2, default=str).encode()
            _s3_sum.put_object(
                Bucket=bucket,
                Key=f"predictor/metrics/training_summary_{date_str}.json",
                Body=_sum_body, ContentType="application/json",
            )
            _s3_sum.put_object(
                Bucket=bucket,
                Key="predictor/metrics/training_summary_latest.json",
                Body=_sum_body, ContentType="application/json",
            )
            log.info("Training summary written to S3 (dated + latest)")
        except Exception as _sum_err:
            log.warning("Training summary write failed (non-blocking): %s", _sum_err)

    # Step 3: Email
    if not dry_run:
        send_training_email(result, date_str)

    # Step 4: Health status
    if not dry_run:
        try:
            from health_status import write_health
            write_health(
                bucket=bucket,
                module_name="predictor_training",
                status="ok",
                run_date=date_str,
                duration_seconds=result.get("train_time_seconds", 0),
                summary={
                    "promoted": result.get("promoted", False),
                    "ic_30d": result.get("ic_30d"),
                    "n_train": result.get("n_train"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
            )
        except Exception as _he:
            log.warning("Health status write failed: %s", _he)

        # Data manifest
        try:
            from health_status import write_data_manifest
            write_data_manifest(
                bucket=bucket,
                module_name="predictor_training",
                run_date=date_str,
                manifest={
                    "promoted": result.get("promoted", False),
                    "promoted_mode": result.get("promoted_mode"),
                    "test_ic": result.get("test_ic"),
                    "n_train": result.get("n_train"),
                    "n_test": result.get("n_test"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
            )
        except Exception as _me:
            log.warning("Data manifest write failed: %s", _me)

    return result

