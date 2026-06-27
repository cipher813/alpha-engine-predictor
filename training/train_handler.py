"""
training/train_handler.py — Weekly GBM retraining pipeline for Lambda.

Called by inference/handler.py when event["action"] == "train".

Pipeline:
  1. Load the universe (OHLCV + pre-computed feature columns) from the
     ArcticDB universe library. ArcticDB is the single, canonical source of
     training data — the legacy S3-parquet / yfinance-refresh fallback was
     removed by PR #6 (training cutover 2026-04-16). No data is fetched over
     yfinance and no parquets are written back to S3 by this handler.
  2. Build regression arrays (29 features) and apply 70/15/15 time-based split
     with purge gaps of FORWARD_DAYS between train/val and val/test.
  3. Train GBMScorer with Optuna-tuned params from config.GBM_TUNED_PARAMS
     (n_estimators=2000, early_stopping=50).
  4. Evaluate on test set: IC, IC IR, positive-period rate.
  5. Upload dated backup unconditionally; promote to gbm_latest.txt only if IC gate passes.
  5b. Write slim 2-year price cache to predictor/price_cache_slim/ for the
      daily inference Lambda's price reads.
  6. Send training summary email with results.

S3 layout:
  predictor/weights/gbm_latest.txt      — (output) active inference weights
  predictor/weights/gbm_{date}.txt      — (output) dated backup

  predictor/price_cache/*.parquet       — LEGACY 10y per-ticker OHLCV parquets.
                                          No longer read or written by training —
                                          ArcticDB is canonical since PR #6
                                          (2026-04-16). Retained on S3 only as a
                                          historical fallback artifact; orphaned.
  predictor/daily_closes/{date}.parquet — Backward-split-adjusted OHLCV snapshot per
                                          trading day.  Written by DataPhase1 in
                                          alpha-engine-data/collectors/daily_closes.py
                                          as the first step of the weekday Step Function,
                                          before the predictor inference Lambda runs.
                                          Historically read by inference as the Mon–Fri
                                          delta to the slim cache; inference now reads
                                          directly from ArcticDB (Phase 7a, 2026-04-16).
                                          These parquets are now orphaned — candidates
                                          for removal in Phase 7e.
  predictor/price_cache_slim/*.parquet  — 2-year slice of each ticker (written by
                                          write_slim_cache() after each weekly training run)
                                          for the daily inference Lambda's price reads.

Lambda environment variables (same as predictor email):
  EMAIL_SENDER / EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD / AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from krepis.secrets import get_secret

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# Module-top so import-time errors in numpy / lightgbm / catboost below
# (heavy GBM imports) are also captured by flow-doctor's ERROR handler.
# Training runs on EC2 spot (NOT Lambda) — no LAMBDA_TASK_ROOT, so the
# yaml path resolves via two-dirs-up from this file (training/train_handler.py
# → repo root, where flow-doctor-training.yaml lives).
#
# exclude_patterns starts empty by deliberate convention; add patterns
# only after observing real ERROR-level noise during training runs.
from krepis.logging import setup_logging, guard_entrypoint
from nousergon_lib.phase_registry import PhaseRegistry
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = str(
    Path(__file__).resolve().parent.parent / "flow-doctor-training.yaml"
)
setup_logging(
    "predictor-training",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

import numpy as np

log = logging.getLogger(__name__)


_SUBSAMPLE_NOISE_FLOOR_PCT = 30.0


def _annotate_subsample_noise_floor(result: dict, s3, bucket: str) -> None:
    """Annotate `result["short_history_subsample"][*]` with `n_pct_change_vs_prior`
    and WARN-log on >|`_SUBSAMPLE_NOISE_FLOOR_PCT`| swing.

    Closes 5/23-SF P0 sweep (L980 noise-floor WARN). Reads the prior cycle's
    `training_summary_latest.json` from S3 BEFORE the current cycle's write
    overwrites it, computes the per-component subsample-`n` percent change,
    and surfaces a WARN when the swing is large enough that a per-week
    margin comparison is no longer apples-to-apples (the 5/24 incident:
    `train_start_date` clamp shrunk vol subsample n 821→263 = -68%,
    ratcheting baseline IC +0.057 while no underlying capability changed).

    Best-effort: any failure (no prior, parse error, S3 hiccup) logs DEBUG
    and returns without raising. The current-cycle write proceeds normally.
    """
    try:
        sh = result.get("short_history_subsample") or {}
        if not sh:
            return

        try:
            prior_obj = s3.get_object(
                Bucket=bucket,
                Key="predictor/metrics/training_summary_latest.json",
            )
            prior = json.loads(prior_obj["Body"].read())
        except Exception as e:
            log.debug("subsample noise-floor: no prior summary (%s); skipping", e)
            return

        prior_sh = prior.get("short_history_subsample") or {}
        for component, current in sh.items():
            if not isinstance(current, dict):
                continue
            current_n = current.get("n")
            prior_component = prior_sh.get(component) or {}
            prior_n = prior_component.get("n")
            if current_n is None or prior_n is None:
                continue
            if not isinstance(current_n, (int, float)) or not isinstance(
                prior_n, (int, float)
            ) or prior_n == 0:
                continue
            pct_change = float((current_n - prior_n) / prior_n * 100.0)
            current["n_pct_change_vs_prior"] = round(pct_change, 2)
            current["prior_n"] = int(prior_n)
            if abs(pct_change) > _SUBSAMPLE_NOISE_FLOOR_PCT:
                log.warning(
                    "subsample noise-floor: %s.n changed by %+.1f%% vs prior cycle "
                    "(prior=%d, current=%d). Per-week margin comparison on this "
                    "component is no longer apples-to-apples — baseline IC may have "
                    "ratcheted mechanically rather than from a real capability shift. "
                    "See ROADMAP L980 / 5/24 train_start_date clamp incident.",
                    component, pct_change, int(prior_n), int(current_n),
                )
    except Exception as e:
        log.debug("subsample noise-floor annotation failed (non-blocking): %s", e)


# ── Training email ─────────────────────────────────────────────────────────────


def _build_shadow_calibration_html(sm: dict | None) -> str:
    """Render the Shadow Calibrator observability row.

    Returns the empty string when no shadow calibrator was fit (shadow
    disabled via ``calibration.shadow_method: null`` OR the shadow fit
    failed — both surface as ``fitted: False`` in the dict). Mirrors the
    Confidence Calibration row's style; adds shadow-specific knobs
    (``class_weight``, ``C``) and gate-pass fingerprints so the operator
    can tell at a glance whether shadow is ready to promote → live.
    """
    if not sm or not sm.get("fitted"):
        return ""

    def _gate(passed_key: str, failed_key: str) -> str:
        passed = sm.get(passed_key)
        label = "PASS" if passed else ("FAIL" if passed is False else "n/a")
        failed_check = sm.get(failed_key)
        suffix = f" ({failed_check})" if failed_check else ""
        return f"<b>{label}{suffix}</b>"

    ece_before = sm["ece_before"]
    ece_after = sm["ece_after"]
    return (
        '<h3 style="margin-top:16px; margin-bottom:4px;">'
        'Shadow Calibrator (observability)</h3>'
        '<p style="font-size:12px;">'
        f'Method: <b>{sm["method"]}</b> &nbsp;|&nbsp; '
        f'class_weight: <b>{sm.get("class_weight")}</b> &nbsp;|&nbsp; '
        f'C: <b>{sm.get("C")}</b> &nbsp;|&nbsp; '
        f'Samples: <b>{sm["n_samples"]:,}</b> &nbsp;|&nbsp; '
        f'ECE: <b>{ece_before:.4f} → {ece_after:.4f}</b> &nbsp;|&nbsp; '
        'output_dist: '
        + _gate("output_distribution_gate_passed",
                "output_distribution_gate_failed_check")
        + ' &nbsp;|&nbsp; stratified: '
        + _gate("stratified_per_regime_passed",
                "stratified_per_regime_failed_check")
        + '</p>'
    )


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
        # ── Headline ICs are OOS (held-out) per 2026-05-13 email facelift ──
        # Pre-facelift the email led with the L2 Ridge's IN-SAMPLE Pearson
        # (e.g. 0.4634) — `np.corrcoef(model.predict(X), y)` over the same
        # pooled-OOF rows the Ridge just fit on. PR #133 added the honest
        # OOS field; this rewrite makes it the headline. The in-sample
        # value rides underneath as a secondary line for context.
        #
        # Meta-Model OOS IC sources:
        #   1. meta_model_oos_ic (top-level, populated PR #133 onward)
        #   2. horizon_diagnostic.curve.21d.spearman (fallback)
        #   3. None — no OOS measurement, fall back to in-sample with a
        #      "(in-sample — no OOS reference)" caveat.
        hd = result.get("horizon_diagnostic") or {}
        curve = hd.get("curve") or {}
        h21 = curve.get("21d") or {}
        meta_oos_ic = result.get("meta_model_oos_ic")
        if meta_oos_ic is None:
            meta_oos_ic = h21.get("spearman")
        meta_in_sample = result.get("meta_model_in_sample_ic") or result.get(
            "meta_model_ic", test_ic
        )
        meta_oos_ci_lo = h21.get("spearman_ci_lo")
        meta_oos_ci_hi = h21.get("spearman_ci_hi")

        if meta_oos_ic is not None:
            ci_str = (
                f' [95% CI {meta_oos_ci_lo:+.4f}, {meta_oos_ci_hi:+.4f}]'
                if meta_oos_ci_lo is not None and meta_oos_ci_hi is not None
                else ""
            )
            rows += _row("Meta-Model OOS IC (Spearman 21d)",
                         f'{meta_oos_ic:+.4f}{ci_str}', bg=True)
            rows += _row("Meta-Model in-sample IC (reference)",
                         f'{meta_in_sample:+.4f}')
        else:
            rows += _row("Meta-Model IC (in-sample — no OOS reference)",
                         f'{meta_in_sample:+.4f}', bg=True)

        # Volatility OOS IC: walk-forward median across 16 held-out folds
        # is the honest OOS measurement. Falls back to held-out test_ic
        # if walk_forward isn't populated (legacy summaries).
        wf = result.get("walk_forward") or {}
        vol_oos_median = wf.get("volatility_median_ic")
        if vol_oos_median is not None:
            rows += _row("Volatility OOS IC (WF median, 16 folds)",
                         f'{vol_oos_median:+.4f}', bg=False)
        else:
            vol_ic = result.get("volatility_test_ic", 0)
            rows += _row("Volatility IC (held-out)", f'{vol_ic:+.4f}', bg=False)

        # Research GBM OOS IC — exposed at top level per the 2026-05-13
        # email facelift. ResearchGBMScorer is the canonical source for
        # research_calibrator_prob META_FEATURE (Phase 3 PR 4/5 cutover
        # 2026-05-09). val_ic is the held-out 20%-OOS measurement; the
        # train_ic appears parenthetically so a widening train/val gap
        # (~2× ratio is the typical overfitting signal) surfaces inline.
        res_val = result.get("research_gbm_val_ic")
        res_train = result.get("research_gbm_train_ic")
        res_n = result.get("research_calibrator_n", 0)
        if res_val is not None:
            train_suffix = (
                f' (train_ic {res_train:+.4f})'
                if res_train is not None else ""
            )
            rows += _row("Research OOS IC (GBM val, held-out 20%)",
                         f'{res_val:+.4f}{train_suffix}', bg=True)
        else:
            # Bucket-lookup fallback (no GBM fit this cycle). Surface the
            # calibrator hit-rate instead of an IC.
            res_metrics = result.get("research_calibrator_metrics") or {}
            res_hr = res_metrics.get("overall_hit_rate")
            if res_hr is not None:
                rows += _row("Research signal (bucket lookup)",
                             f'hit_rate={res_hr:.4f} (n={res_n})', bg=True)

        # Momentum L1 is the deterministic weighted-blend baseline — there's
        # no GBM fit IC. The WF median IC is observability-only post the
        # 2026-05-13 gate change (per IC study, momentum standalone IC at
        # 21d ≈ 0 but contributes Ridge-stack interaction value). Surface
        # it dimmed so operators can spot regression without it being read
        # as a headline.
        mom_wf = wf.get("momentum_median_ic")
        if mom_wf is not None:
            rows += _row("Momentum WF median IC (observability)",
                         f'<span style="color:#888;">{mom_wf:+.4f} '
                         f'(deterministic baseline; not gating)</span>',
                         bg=False)
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
    # `passes_ic` is now the QUALITY verdict (all gates passed), decoupled from
    # the promotion decision (L4540). Training is ALWAYS challenger-first since
    # config#1052/#679 (the dead TRAINING_AUTO_PROMOTE_ENABLED flag was retired):
    # a run can pass every gate and NOT promote — that is a registered challenger,
    # NOT a failure; the model-zoo `select_winner` step owns promotion. Legacy
    # pre-challenger-first archive manifests may carry `auto_promote_enabled=True`
    # (where promoted == gate_passed); honor that for old-archive reads, but a
    # current run is challenger-registered whenever it passes the gate without
    # promoting.
    passes_ic    = result.get("passes_ic_gate", False)
    auto_promote = result.get("auto_promote_enabled", False)
    challenger_registered = bool(passes_ic and not promoted and not auto_promote)
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
        # Prefer the authoritative gate-resolution block (added 2026-05-14
        # to close ROADMAP L1668). `promotion_gate_detail.promoted_blocker_reason`
        # is computed off the SAME boolean values the live `promoted` formula
        # uses, so the email's failure-reason can never disagree with the
        # actual blocker. Falls back to the legacy walk_forward scan only
        # for older training_summary readers (pre-2026-05-14 archive runs).
        gate_detail = result.get("promotion_gate_detail") or {}
        blocker = gate_detail.get("promoted_blocker_reason")
        if blocker:
            # Decorate each named gate with the most informative scalar so
            # the email reader doesn't have to S3-grep training_summary for
            # context. The blocker string is +-joined when multiple gates
            # fired (e.g. "meta_ic+output_dist").
            parts: list[str] = []
            for name in blocker.split("+"):
                if name == "meta_ic":
                    meta_ic_val = result.get("meta_model_ic")
                    threshold = gate_detail.get("meta_ic_threshold")
                    if meta_ic_val is not None and threshold is not None:
                        parts.append(
                            f"meta IC {meta_ic_val:+.4f} < gate {threshold:+.4f}"
                        )
                    else:
                        parts.append("meta IC gate")
                elif name == "subsample":
                    sh = result.get("short_history_subsample") or {}
                    vol_sh = (sh.get("volatility") or {})
                    cic = vol_sh.get("component_ic")
                    bic = vol_sh.get("baseline_ic")
                    if cic is not None and bic is not None:
                        parts.append(
                            f"volatility subsample (component {cic:+.4f} ≤ baseline {bic:+.4f})"
                        )
                    else:
                        parts.append("volatility subsample gate")
                elif name == "output_dist":
                    parts.append("output-distribution gate (calibrator shape)")
                elif name == "stratified":
                    parts.append("stratified per-regime gate")
                else:
                    parts.append(name)
            return "promotion blocked: " + "; ".join(parts)
        # Legacy fallback path for pre-2026-05-14 archive runs that lack
        # the gate_detail block. Same heuristic as before — accurate only
        # when walk_forward has a negative volatility median.
        wf = result.get("walk_forward") or {}
        if not wf:
            return "IC gate failed"  # fallback — no wf info available
        vol = wf.get("volatility_median_ic")
        reasons: list[str] = []
        if vol is not None and vol <= 0:
            reasons.append(f"volatility median IC {vol:+.4f}")
        if reasons:
            return "walk-forward failed: " + ", ".join(reasons)
        return "IC gate failed"

    promo_label = (
        f"Promoted → weights/meta/ ✓" if promoted and is_meta
        else f"Promoted → gbm_latest ({promoted_mode}) ✓" if promoted
        else "Registered challenger (gates passed; model-zoo select_winner decides promotion) ◆"
        if challenger_registered
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

    # Subject IC: prefer OOS Spearman over in-sample Ridge fit. PR #133's
    # meta_model_oos_ic field is the honest measurement; fall back to the
    # horizon_diagnostic curve at the active forward horizon, then to
    # in-sample as a last resort (with explicit "(in-sample)" suffix so
    # the deprecation is visible in subject-line search results).
    if is_meta:
        _subj_oos = result.get("meta_model_oos_ic")
        if _subj_oos is None:
            _subj_oos = (
                (result.get("horizon_diagnostic") or {}).get("curve") or {}
            ).get("21d", {}).get("spearman")
        if _subj_oos is not None:
            _subj_ic_text = f"Meta OOS IC {_subj_oos:+.4f}"
        else:
            _subj_ic_text = f"Meta IC {test_ic:+.4f} (in-sample)"
    else:
        _subj_ic_text = f"IC {test_ic:.4f}"
    _subj_promo = (
        f"Promoted ({promoted_mode})" if promoted
        else "Registered challenger" if challenger_registered
        else "Not promoted"
    )
    subject = (
        f"Alpha Engine Training | {date_str} | "
        f"{_subj_ic_text} {status_str} | "
        f"{_subj_promo}"
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
            # Walk-forward header — vol gate result + momentum observability.
            # Per the 2026-05-13 gate change, only vol gates promotion;
            # momentum WF median is informational (the L1 stays in the
            # Ridge for interaction value but is not gated).
            mom_fmt = (
                f'{mom_median:+.4f}'
                if isinstance(mom_median, (int, float)) else str(mom_median)
            )
            vol_fmt = (
                f'{vol_median:+.4f}'
                if isinstance(vol_median, (int, float)) else str(vol_median)
            )
            wf_html = (
                f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
                f'({len(wf_data["folds"])} folds)</h3>'
                f'<p style="font-size:12px; margin:2px 0;">'
                f'Volatility median IC: <b>{vol_fmt}</b> '
                f'&nbsp;|&nbsp; Status: <b style="color:{wf_color};">{wf_label}</b>'
                f'<br><span style="color:#888;">Momentum median IC (observability, '
                f'not gating): {mom_fmt}</span></p>'
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
                f"\nVolatility median IC: {vol_fmt}  |  Status: {wf_label}"
                f"\nMomentum median IC (observability, not gating): {mom_fmt}\n"
                + "\n".join(
                    f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                    f"mom={f.get('mom_ic', 0):+.4f}  vol={f.get('vol_ic', 0):+.4f}"
                    for f in wf_data["folds"]
                ) + "\n"
            )
        else:
            # config#1083 n_train landmine fix: a fold dict can lack n_train / ic
            # (e.g. a fold that errored or a CPCV-only summary). f["n_train"]:,
            # KeyError'd and crashed the WHOLE training email — the 2nd of the two
            # botched rotations. Default-read every fold field so the email never
            # crashes on a malformed fold (fail-soft on the OBSERVABILITY email; the
            # training result itself already persisted to S3).
            fold_rows = "".join(
                f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f.get("fold", "?")}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f.get("test_start", "?")}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f.get("test_end", "?")}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f.get("n_train", 0):,}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:{"#2e7d32" if f.get("ic", 0) > 0 else "#c62828"};">{f.get("ic", 0):.4f}</td>'
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
                    f"  Fold {f.get('fold', '?')}: [{f.get('test_start', '?')} → {f.get('test_end', '?')}] "
                    f"train={f.get('n_train', 0):,}  IC={f.get('ic', 0):.4f}"
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
        # Shadow calibrator row — observability only. Surfaces shadow's
        # method, ECE, gate-pass fingerprint so operators can decide when
        # shadow is safe to promote → live (flip calibration.method in
        # predictor.yaml). Empty when shadow disabled.
        + _build_shadow_calibration_html(result.get("shadow_calibration"))
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
            # 2026-05-13 facelift: walk-forward gate dropped momentum check.
            # Momentum's standalone IC at 21d ≈ 0 (per IC study) but the
            # Ridge stack extracts interaction value, so the L1 stays in
            # META_FEATURES while the gate only requires vol median > 0.
            # Subject + headline ICs are OOS (Spearman 21d) per PR #133.
            f'IC gate: ≥{cfg.MIN_IC:.2f} in-sample meta IC to promote &nbsp;|&nbsp; '
            f'Walk-forward: volatility median IC &gt; 0 (momentum observability-only)</p>'
            if is_meta else
            f'IC gate: ≥{cfg.MIN_IC:.2f} to promote &nbsp;|&nbsp; '
            f'Walk-forward: median IC ≥{cfg.WF_MEDIAN_IC_GATE:.2f}, '
            f'{cfg.WF_MIN_FOLDS_POSITIVE*100:.0f}%+ positive folds</p>'
        )
        + f'</body></html>'
    )

    if is_meta:
        # ── Plain-text body: mirror the HTML facelift (OOS-headline) ──
        _hd = result.get("horizon_diagnostic") or {}
        _curve = _hd.get("curve") or {}
        _h21 = _curve.get("21d") or {}
        _meta_oos = (
            result.get("meta_model_oos_ic")
            or _h21.get("spearman")
        )
        _meta_in_sample = (
            result.get("meta_model_in_sample_ic")
            or result.get("meta_model_ic", test_ic)
        )
        _ci_lo = _h21.get("spearman_ci_lo")
        _ci_hi = _h21.get("spearman_ci_hi")
        _ci_text = (
            f" [95% CI {_ci_lo:+.4f}, {_ci_hi:+.4f}]"
            if _ci_lo is not None and _ci_hi is not None else ""
        )
        _meta_oos_line = (
            f"Meta-Model OOS IC (Spearman 21d): {_meta_oos:+.4f}{_ci_text} — {ic_label}"
            if _meta_oos is not None else
            f"Meta-Model IC (in-sample, no OOS reference): {_meta_in_sample:+.4f} — {ic_label}"
        )

        _wf = result.get("walk_forward") or {}
        _vol_oos_median = _wf.get("volatility_median_ic")
        _vol_oos_line = (
            f"Volatility OOS IC (WF median, 16 folds): {_vol_oos_median:+.4f}"
            if _vol_oos_median is not None
            else f"Volatility IC (held-out): {result.get('volatility_test_ic', 0):+.4f}"
        )

        _res_val = result.get("research_gbm_val_ic")
        _res_train = result.get("research_gbm_train_ic")
        if _res_val is not None:
            _train_suffix = (
                f" (train_ic {_res_train:+.4f})" if _res_train is not None else ""
            )
            _research_line = (
                f"Research OOS IC (GBM val, held-out 20%): "
                f"{_res_val:+.4f}{_train_suffix}"
            )
        else:
            _res_metrics = result.get("research_calibrator_metrics") or {}
            _res_hr = _res_metrics.get("overall_hit_rate")
            _res_n = result.get("research_calibrator_n", 0)
            _research_line = (
                f"Research signal (bucket lookup): hit_rate={_res_hr:.4f} (n={_res_n})"
                if _res_hr is not None else "Research signal: n/a"
            )

        _mom_wf_median = _wf.get("momentum_median_ic")
        _mom_line = (
            f"Momentum WF median IC (observability, not gating): "
            f"{_mom_wf_median:+.4f}"
            if _mom_wf_median is not None else
            "Momentum: deterministic baseline (no GBM fit IC)"
        )

        plain_body = (
            f"Alpha Engine Training — {date_str}\n"
            f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
            f"\n{_meta_oos_line}"
            f"\nMeta-Model in-sample IC (reference): {_meta_in_sample:+.4f}"
            f"\n{_vol_oos_line}"
            f"\n{_research_line}"
            f"\n{_mom_line}"
            f"\nPromotion: {promo_label}\n"
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
            f"\nPromoted:           {promoted_mode if promoted else ('challenger' if challenger_registered else 'none')}"
            f"\nIC IR:              {ic_ir:.3f} ({ic_pos}/20 positive)"
            f"\nPromotion:          {promo_label}\n"
            f"{wf_plain}"
            f"\nTop features: " + ", ".join(r["feature"] for r in top10[:5])
            + f"\n{shap_plain}"
            + f"{feat_health_plain}\n"
        )

    # SMTP/SES dispatch via the krepis.email_sender chokepoint
    # (L4356 — Gmail SMTP primary, SES fallback). Same semantics as the
    # pre-consolidation inline path.
    from krepis.email_sender import send_email as _send_email
    return _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=cfg.AWS_REGION,
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────

_TRAINING_SUMMARY_KEY = "predictor/metrics/training_summary_{date}.json"
# config#1170 — per-challenger-spec training summary (the meta_training phase's
# L4524 reload artifact). Namespaced under the spec's MODEL_VERSION_LABEL so a
# zoo rerun reloads ITS OWN summary, never a sibling spec's (consequence #3).
_TRAINING_SUMMARY_KEY_SPEC = "predictor/model_zoo/{spec}/metrics/training_summary_{date}.json"


def _spec_namespace() -> "str | None":
    """The current zoo-challenger spec namespace, or ``None`` for the base champion.

    config#1170 — phase markers + the training-summary reload artifact were
    keyed at DATE granularity (``predictor/{date}/.phases/meta_training.json``),
    a SINGLE file shared by every model-zoo spec for that date. That made
    same-date reruns skip ALL specs (any spec's marker auto-skipped every spec)
    and blocked a FAILED spec from retrying same-day (it saw a sibling's success
    marker). The fix namespaces the markers + reload artifact per challenger spec.

    The namespace is derived from ``cfg.MODEL_VERSION_LABEL`` — the model_zoo
    sets this per spec via ``spec_overrides`` around the ``main()`` call
    (model_zoo.train_spec). When it equals the base champion default
    (``v3.0-meta`` / config ``model_version_label``) the run IS the base champion,
    so we return None and the legacy spec-less paths are preserved (the issue
    requires the base champion keeps its current path). A challenger returns a
    filesystem-safe slug of its label so each spec's skip/reload is independent.
    """
    try:
        import config as _cfg
    except Exception:  # config unreadable (dev/CI w/o predictor.yaml)
        return None
    label = getattr(_cfg, "MODEL_VERSION_LABEL", None)
    base = _cfg_default_label(_cfg)
    if not label or label == base:
        return None
    # Slugify so an arbitrary spec label is a safe single S3 path segment.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return slug or None


def _cfg_default_label(_cfg) -> str:
    """The base champion's model_version_label (the spec-less identity).

    config sets ``MODEL_VERSION_LABEL`` from ``model_version_label`` (default
    ``v3.0-meta``); a zoo spec then OVERRIDES the module attr at train time. So
    the live attr can't tell us the base — we read the underlying yaml default
    via ``config._cfg`` when present, else fall back to the documented default.
    """
    try:
        raw = getattr(_cfg, "_cfg", None)
        if isinstance(raw, dict):
            return raw.get("model_version_label", "v3.0-meta") or "v3.0-meta"
    except Exception:
        pass
    return "v3.0-meta"


def _training_summary_key(date_str: str, spec: "str | None") -> str:
    """The training-summary S3 key for this run — spec-namespaced for a zoo
    challenger (config#1170), the legacy date-only key for the base champion."""
    if spec:
        return _TRAINING_SUMMARY_KEY_SPEC.format(spec=spec, date=date_str)
    return _TRAINING_SUMMARY_KEY.format(date=date_str)


def _build_registry(
    bucket: str,
    date_str: str,
    dry_run: bool,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    spec: "str | None" = None,
) -> "PhaseRegistry | None":
    """Construct the training :class:`PhaseRegistry` (markers under
    ``predictor/{date}/.phases/`` + watchdog), or ``None`` in dry-run.

    Predictor is the 3rd consumer of the lib phase framework (after the
    backtester + alpha-engine-data). Dry-run returns None so the smoke pass
    (``spot_train.sh`` calls ``main(dry_run=True)``) writes no markers. Recovery
    controls come from explicit args OR, on the spot, the ``PREDICTOR_SKIP_PHASES``
    / ``PREDICTOR_FORCE_PHASES`` / ``PREDICTOR_FORCE`` env vars (args win).
    Watchdog caps read from ``config.FULL_RUN_HARD_CAPS_SECONDS`` — best-effort,
    so a dev/CI run with no ``predictor.yaml`` degrades to an unwatchdogged
    registry (logged) rather than failing the import.

    config#1170 — ``spec`` namespaces the markers per zoo challenger:
    ``predictor/model_zoo/{spec}/{date}/.phases/{phase}.json`` instead of the
    date-only ``predictor/{date}/.phases/{phase}.json``. The base champion
    (spec is None) keeps its legacy path, so each challenger's skip/auto-skip is
    independent: a same-date rerun re-runs only the not-yet-completed / failed
    specs, and a failed spec can retry same-day without a sibling's marker
    short-circuiting it.
    """
    if dry_run:
        return None
    import os as _os

    def _csv(s: str) -> list[str]:
        return [p.strip() for p in (s or "").split(",") if p.strip()]

    skip = _csv(skip_phases) or _csv(_os.environ.get("PREDICTOR_SKIP_PHASES", ""))
    fphases = _csv(force_phases) or _csv(_os.environ.get("PREDICTOR_FORCE_PHASES", ""))
    force_all = bool(force) or _os.environ.get("PREDICTOR_FORCE", "").lower() in ("1", "true", "yes")
    try:
        import config as _cfg
        hard_caps = getattr(_cfg, "FULL_RUN_HARD_CAPS_SECONDS", {}) or {}
    except Exception as _cap_err:  # config unreadable (dev/CI w/o predictor.yaml)
        log.warning("phase watchdog caps unavailable (%s) — running unwatchdogged", _cap_err)
        hard_caps = {}
    # config#1170 — per-spec marker prefix for zoo challengers; the base champion
    # (spec is None) keeps the spec-less "predictor" prefix.
    marker_prefix = f"predictor/model_zoo/{spec}" if spec else "predictor"
    return PhaseRegistry(
        date=date_str,
        bucket=bucket,
        marker_prefix=marker_prefix,
        skip_phases=skip,
        force=force_all,
        force_phases=fphases,
        hard_caps=hard_caps,
    )


def _maybe_phase(reg: "PhaseRegistry | None", name: str, **log_ctx):
    """Marker-only phase wrapper (watchdog + START/END marker, no auto-skip) for
    stages whose body manages its own hard-fail/best-effort posture inline.
    Returns :func:`contextlib.nullcontext` in dry-run (reg is None)."""
    if reg is None:
        return nullcontext()
    return reg.phase(name, supports_auto_skip=False, **log_ctx)


def _load_training_result(bucket: str, date_str: str, spec: "str | None" = None) -> "dict | None":
    """Reload a prior run's ``result`` dict from the persisted training summary
    so a recovery that AUTO-SKIPS the ~90-min ``meta_training`` phase can still
    drive the downstream summary/risk-model/gate/email/health stages. The summary
    (``predictor/metrics/training_summary_{date}.json``, or the spec-namespaced
    key for a zoo challenger — config#1170) IS the json-serialized ``result`` and
    is the meta_training phase's recorded artifact — so L4524 only auto-skips when
    it exists, and this reload then succeeds. Mirrors the backtester's
    ``_load_predictor_artifacts``. 404 → None (caller refuses to proceed on a
    half-trained state); any other S3 error → raise (fail loud)."""
    import boto3
    from botocore.exceptions import ClientError

    key = _training_summary_key(date_str, spec)
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket", "NotFound"):
            log.warning(
                "meta_training auto-skip: training summary s3://%s/%s absent — "
                "cannot reload result", bucket, key,
            )
            return None
        raise


def _write_training_summary(
    result: dict, bucket: str, date_str: str, spec: "str | None" = None,
) -> None:
    """Annotate + persist the training ``result`` to S3 (dated + latest). Extracted
    from the inline Step-2d block so the ``meta_training`` phase can record the
    dated summary as its L4524 artifact (the reload source). Non-blocking: a write
    failure logs a WARN — the training itself already succeeded.

    config#1170 — for a zoo challenger (``spec`` set) the dated summary is written
    to the spec-namespaced key so it's that spec's OWN reload artifact, and the
    shared ``training_summary_latest.json`` champion pointer is NOT overwritten
    (only the base champion advances "latest")."""
    try:
        import boto3 as _b3_sum
        _s3_sum = _b3_sum.client("s3")
        # Annotate subsample-noise-floor BEFORE the write — reads the PRIOR
        # cycle's latest.json so the percent-change reference is genuinely
        # prior-cycle rather than this-cycle.
        _annotate_subsample_noise_floor(result, _s3_sum, bucket)
        _sum_body = json.dumps(result, indent=2, default=str).encode()
        _s3_sum.put_object(
            Bucket=bucket,
            Key=_training_summary_key(date_str, spec),
            Body=_sum_body, ContentType="application/json",
        )
        if not spec:
            _s3_sum.put_object(
                Bucket=bucket,
                Key="predictor/metrics/training_summary_latest.json",
                Body=_sum_body, ContentType="application/json",
            )
        log.info(
            "Training summary written to S3 (%s)",
            "dated, spec-namespaced" if spec else "dated + latest",
        )
    except Exception as _sum_err:
        log.warning("Training summary write failed (non-blocking): %s", _sum_err)


def _env_truthy(val: Optional[str]) -> bool:
    """A standard truthy-env reading: set + not one of {"", "0", "false", "no",
    "off"} (case-insensitive). Used by the PREDICTOR_DEFER_TRAINING_EMAIL gate."""
    if val is None:
        return False
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def main(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    send_email: bool = True,
) -> dict:
    """Top-level training entrypoint.

    Thin wrapper around :func:`_main_impl` that runs the body inside
    ``guard_entrypoint()`` so an uncaught ``raise`` from the spot-train
    run is reported to flow-doctor (email) and re-raised. ``guard_entrypoint``
    no-ops when flow-doctor is inactive (dev / CI / pytest), so this is
    safe everywhere; it activates only on the deployed spot (where
    ``ALPHA_ENGINE_DEPLOYED=1`` is exported by ``infrastructure/spot_train.sh``).
    The spot driver calls ``train_handler.main()`` directly, so wrapping
    here is the true top-level entrypoint.

    ``send_email`` gates the per-run training summary email. The model-zoo
    rotation trains every challenger via ``train_spec`` → ``main()``; without
    this gate each challenger emails, producing the per-challenger email storm
    the consolidated zoo digest replaces. The zoo passes ``send_email=False``;
    the base ``--full-only`` retrain defers its own email to the digest via the
    ``PREDICTOR_DEFER_TRAINING_EMAIL`` env (honored below) so the SF/spot can
    defer without threading the arg through the spot heredoc. The dated
    ``training_summary_{date}.json`` S3 write is ALWAYS performed regardless of
    ``send_email`` (the email is observability over an artifact that already
    persisted).
    """
    # Env override so the spot/SF can defer the base retrain's email to the
    # zoo digest without arg-threading: when PREDICTOR_DEFER_TRAINING_EMAIL is
    # set/truthy, default send_email to False. An explicit send_email=False
    # caller (the zoo) is already covered; the env only flips the default-True
    # callers (the base --full-only spot heredoc, which passes no send_email).
    if send_email and _env_truthy(os.environ.get("PREDICTOR_DEFER_TRAINING_EMAIL")):
        log.info(
            "PREDICTOR_DEFER_TRAINING_EMAIL set — deferring the per-run training "
            "email to the model-zoo digest (training_summary S3 write unaffected)."
        )
        send_email = False
    with guard_entrypoint():
        return _main_impl(
            bucket, date_str=date_str, dry_run=dry_run,
            skip_phases=skip_phases, force_phases=force_phases, force=force,
            send_email=send_email,
        )


def _main_impl(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    *,
    skip_phases: str = "",
    force_phases: str = "",
    force: bool = False,
    send_email: bool = True,
) -> dict:
    """
    Entry point called from inference/handler.py for the "train" action.

    1.  Load the universe (OHLCV + features) from the ArcticDB universe
        library to /tmp. ArcticDB is canonical since PR #6 (2026-04-16);
        there is no yfinance fetch and no S3-parquet refresh/upload.
    2.  Run GBM training pipeline on the loaded cache.
    2b. Write a 2-year slim cache to S3 (predictor/price_cache_slim/)
        for the daily inference Lambda's price reads.
    3.  Send training summary email.

    Returns the result dict from run_meta_training().
    """
    if date_str is None:
        # Trading-day axis (config#1015): the Saturday SF runs on a calendar
        # date that is NOT a trading day, so a calendar now() keyed the training
        # summary / meta-archive / oos_rows / risk_model artifacts on Saturday
        # instead of Friday's close — contaminating the predictor artifact chain.
        # Mirror model_zoo's default: resolve the trading_day via now_dual,
        # falling back to calendar only if the lib lookup fails.
        try:
            from krepis.dates import now_dual
            _td = now_dual().trading_day
            date_str = _td.isoformat() if hasattr(_td, "isoformat") else str(_td)
            log.info("train_handler: no date passed — defaulting to trading_day %s", date_str)
        except Exception:  # noqa: BLE001 — date defaulting must not block training
            date_str = datetime.now().strftime("%Y-%m-%d")
            log.warning("train_handler: could not resolve trading_day via now_dual; "
                        "fell back to calendar date %s", date_str, exc_info=True)

    # setup_logging already ran at module-top (see comment near the
    # krepis.logging import). Apply standard log level here.
    logging.getLogger().setLevel(logging.INFO)

    # config#1170 — the zoo-challenger spec namespace (None for the base
    # champion). Derived from cfg.MODEL_VERSION_LABEL, which model_zoo sets per
    # spec via spec_overrides around this main() call. Namespaces the phase
    # markers + the meta_training reload artifact PER SPEC so a same-date rerun
    # re-runs only the not-yet-completed / failed specs (and a failed spec can
    # retry same-day) instead of every spec skipping on the first spec's marker.
    spec_ns = _spec_namespace()
    if spec_ns:
        log.info("model-zoo challenger run: phase markers namespaced under spec=%s", spec_ns)

    # Phase registry: markers under predictor/{date}/.phases/ (or, for a zoo
    # challenger, predictor/model_zoo/{spec}/{date}/.phases/) + watchdog +
    # meta_training auto-skip-reload (L4528, 3rd lib consumer). None in dry-run.
    reg = _build_registry(
        bucket, date_str, dry_run,
        skip_phases=skip_phases, force_phases=force_phases, force=force,
        spec=spec_ns,
    )

    # Preflight — fail fast on env / connectivity / ArcticDB staleness
    # before a 90-minute training run commits to stale data. See PR #6
    # and training/preflight.py.
    from training.preflight import TrainingPreflight
    with _maybe_phase(reg, "preflight"):
        TrainingPreflight(bucket=bucket).run()

    log.info("GBM training run: date=%s  bucket=%s  dry_run=%s", date_str, bucket, dry_run)

    # Step 1: Load price cache from ArcticDB. PR #6 removed the S3 parquet
    # fallback — it masked the PR #3 miswiring for a week (same root cause
    # as the inference-path issue fixed in PR #5). Preflight above already
    # verified macro/SPY freshness, so ArcticDB is known reachable here.
    # supports_auto_skip=False (via _maybe_phase): the cache lands in local
    # /tmp, which is empty on a fresh recovery spot, so it must always re-run.
    tmp_cache = Path(tempfile.mkdtemp()) / "cache"
    from store.arctic_reader import download_from_arctic
    log.info("[data_source=arcticdb] Loading universe from ArcticDB...")
    with _maybe_phase(reg, "data_load"):
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

    # Step 2 (+ 2d): Train + upload (v3 meta-model only — v2 single-GBM and
    # multi-horizon dispatch branches removed 2026-04-13; v2 machinery itself
    # still in-tree, full rip-out tracked in ROADMAP) AND the training-summary
    # write, folded into ONE `meta_training` phase. The dated summary
    # (predictor/metrics/training_summary_{date}.json) IS this phase's recorded
    # L4524 artifact AND the reload source — so on a recovery the phase
    # AUTO-SKIPS only when training fully completed (weights + summary on S3),
    # and `result` is reloaded from the summary instead of re-running the
    # ~90-min long pole. A crash between training and the summary write leaves
    # the artifact absent → the marker is invalid → a clean re-train. The
    # watchdog hard-cap on this phase converts an OOM (today an opaque SSM
    # TimedOut, L4511) into a PhaseTimeoutError + faulthandler dump.
    from training.meta_trainer import run_meta_training

    def _train_and_summarize() -> dict:
        _r = run_meta_training(
            data_dir=str(tmp_cache), bucket=bucket, date_str=date_str, dry_run=dry_run,
        )
        # Slim cache write + feature-store registry upload are handled by
        # alpha-engine-data (Phase 1) — nothing to do here.
        log.info("Slim cache write: skipped (handled by alpha-engine-data)")
        if not dry_run:
            _write_training_summary(_r, bucket, date_str, spec_ns)
        return _r

    if reg is None:
        result = _train_and_summarize()
    else:
        with reg.phase("meta_training", supports_auto_skip=True) as ctx:
            if ctx.skipped:
                result = _load_training_result(bucket, date_str, spec_ns)
                if result is None:
                    raise RuntimeError(
                        "meta_training marker is ok but the training summary is absent "
                        "— refusing to proceed on a half-trained state. Re-run with "
                        "--force-phases=meta_training (PREDICTOR_FORCE_PHASES=meta_training)."
                    )
                log.info(
                    "meta_training auto-skipped (%s) — reloaded result from summary",
                    ctx.skip_reason,
                )
            else:
                result = _train_and_summarize()
                ctx.record_artifact(_training_summary_key(date_str, spec_ns))

    # Step 2d2: Factor-risk-model F + D weekly persistence (ROADMAP C.2b).
    # Reads the per-ticker parquets the train_handler already populated
    # to ``tmp_cache`` from ArcticDB, extracts close → log returns + the
    # 8 ``*_zscore`` factor-loading columns (C.1, alpha-engine-data
    # #324), runs the Fama-MacBeth cross-sectional regression
    # (C.2a, ``risk_model.build_factor_risk_model``), and persists F + D
    # parquets to ``s3://{bucket}/risk_model/{date}/``. Non-blocking:
    # failure here doesn't abort the training pipeline. Auto-skips
    # when loading-column coverage is insufficient (pre-#324 cache).
    #
    # The actual wiring of Σ = B · F · Bᵀ + D into
    # ``executor.portfolio_optimizer.solve_target_weights`` is C.3 —
    # under a hard sequencing constraint per the plan doc: do NOT
    # wire until B.5 (α̂-uncertainty cutover gate) passes. Two
    # simultaneous Σ-substrate changes make backtester regressions
    # untraceable. C.2b ships the weekly persistence so by the time
    # C.3 reads ``risk_model/{date}/``, there are ≥4 weeks of F + D
    # accumulated.
    if not dry_run:
        try:
            from training.risk_model_persist import (
                build_and_persist_risk_model,
            )
            with _maybe_phase(reg, "risk_model_persist"):
                _rm_payload = build_and_persist_risk_model(
                    data_dir=str(tmp_cache),
                    bucket=bucket,
                    date_str=date_str,
                )
            result["risk_model"] = _rm_payload
            log.info(
                "risk_model_persist: status=%s",
                _rm_payload.get("status"),
            )
        except Exception as _rm_err:
            log.warning(
                "risk_model_persist failed (non-blocking): %s",
                _rm_err, exc_info=True,
            )

    # Step 2e: Triple-barrier cutover gate (Stage 3 PR 5 SF wiring).
    # Runs after model upload + training summary so the gate has access
    # to the freshly-promoted meta_model_tb.pkl and the most recent ~6
    # weeks of inference predictions. Result lands at
    # ``s3://{bucket}/predictor/variant_gates/triple_barrier_{date}.json``
    # and is merged into the training summary dict so the email and
    # downstream consumers can surface it.
    #
    # Failure is non-blocking — the gate is observe-only until
    # ``triple_barrier.enforce_cutover`` flips in alpha-engine-config.
    # Until ~6-9 weeks of parallel-observation data accumulate (PR 3
    # inference path needs to write meta_alpha_tb fields + the 21d
    # forward window needs to close), the gate naturally returns an
    # insufficient-data verdict per ``min_pairs=100``.
    if not dry_run:
        try:
            from analysis.triple_barrier_cutover_runner import run_gate as _run_tb_gate
            with _maybe_phase(reg, "triple_barrier_gate"):
                _gate_payload = _run_tb_gate(
                    bucket=bucket,
                    n_days=42,
                    horizon_days=21,
                    threshold=1.15,
                    write_to_s3=True,
                )
            result["triple_barrier_cutover_gate"] = {
                "passed": _gate_payload.get("passed"),
                "n_pairs": _gate_payload.get("n_pairs"),
                "baseline_ic": _gate_payload.get("baseline_ic"),
                "variant_ic": _gate_payload.get("variant_ic"),
                "relative_lift": _gate_payload.get("relative_lift"),
                "n_realized_filled": _gate_payload.get("n_realized_filled"),
                "reason": _gate_payload.get("reason"),
            }
            log.info(
                "Stage 3 cutover gate: passed=%s n_pairs=%d lift=%s",
                _gate_payload.get("passed"),
                _gate_payload.get("n_pairs", 0),
                _gate_payload.get("relative_lift"),
            )
        except Exception as _gate_err:
            log.warning(
                "Stage 3 cutover gate failed (non-blocking, observe-only): %s",
                _gate_err,
            )

    # Step 3: Email — gated on send_email (model-zoo passes False so each
    # challenger does NOT email; the base --full-only retrain defers via
    # PREDICTOR_DEFER_TRAINING_EMAIL → the consolidated zoo digest). The dated
    # training_summary_{date}.json S3 write happened above regardless, so a
    # suppressed email loses no DATA — only the per-run notification.
    if not dry_run and send_email:
        send_training_email(result, date_str)
    elif not dry_run:
        log.info(
            "Per-run training email suppressed (send_email=False) — summary "
            "persisted to predictor/metrics/training_summary_%s.json; the "
            "model-zoo digest reports champion + challengers + promotion.",
            date_str,
        )

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

