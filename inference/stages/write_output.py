"""Stage: write_output — Build metrics, apply veto, write predictions, send email, write health."""

from __future__ import annotations

import json
import logging
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config as cfg
from inference.pipeline import PipelineContext
from inference.s3_io import _s3_put_json

log = logging.getLogger(__name__)


# ── S3-delivered predictor params (veto threshold) ────────────────────────────

_predictor_params_cache: dict | None = None
_predictor_params_loaded: bool = False
# Local cache persists last known optimal across Lambda cold-starts (via /tmp)
# and EC2 restarts (via project dir).
_PREDICTOR_PARAMS_CACHE_PATH = Path(
    os.environ.get("PREDICTOR_PARAMS_CACHE", "/tmp/predictor_params_cache.json")
)


def _load_predictor_params_from_s3(s3_bucket: str) -> dict | None:
    """Read config/predictor_params.json from S3. Cache per cold-start.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    params survive transient S3 failures.
    """
    global _predictor_params_cache, _predictor_params_loaded
    if _predictor_params_loaded:
        return _predictor_params_cache
    _predictor_params_loaded = True

    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key="config/predictor_params.json")
        data = json.loads(obj["Body"].read())
        if "veto_confidence" in data:
            _predictor_params_cache = data
            log.info("Loaded predictor params from S3: veto_confidence=%.2f", data["veto_confidence"])
            # Persist to local cache for fault tolerance
            try:
                _PREDICTOR_PARAMS_CACHE_PATH.write_text(json.dumps(data, indent=2))
            except Exception:
                pass  # best-effort
        return _predictor_params_cache
    except Exception as e:
        log.warning("Could not read predictor params from S3: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if _PREDICTOR_PARAMS_CACHE_PATH.exists():
            data = json.loads(_PREDICTOR_PARAMS_CACHE_PATH.read_text())
            if "veto_confidence" in data:
                _predictor_params_cache = data
                log.info(
                    "Loaded predictor params from local cache (last known optimal): veto_confidence=%.2f",
                    data["veto_confidence"],
                )
                return _predictor_params_cache
    except Exception as e2:
        log.debug("Could not read local predictor params cache: %s", e2)

    return None


def get_veto_threshold(s3_bucket: str, market_regime: str = "") -> float:
    """
    Return the active veto confidence threshold, adjusted by market regime.

    In bear/caution regimes, the threshold is lowered (more aggressive vetoing)
    to protect capital. In bull regimes, the threshold is raised (more permissive)
    to avoid missing opportunities.

    Regime adjustments (applied to the base threshold from S3 or config):
      bear:    -0.10  (e.g., 0.65 → 0.55 — veto more aggressively)
      caution: -0.05  (e.g., 0.65 → 0.60)
      neutral:  0.00  (no adjustment)
      bullish: +0.05  (e.g., 0.65 → 0.70 — allow more entries)
    """
    params = _load_predictor_params_from_s3(s3_bucket)
    if params and "veto_confidence" in params:
        base = float(params["veto_confidence"])
    else:
        base = cfg.MIN_CONFIDENCE

    # Regime-adaptive adjustment
    regime = market_regime.lower().strip() if market_regime else ""
    regime_adjustments = {
        "bear": -0.10,
        "bearish": -0.10,
        "caution": -0.05,
        "neutral": 0.0,
        "bull": 0.05,
        "bullish": 0.05,
    }
    adjustment = regime_adjustments.get(regime, 0.0)
    adjusted = max(0.40, min(0.90, base + adjustment))

    if adjustment != 0.0:
        log.info(
            "Veto threshold regime-adjusted: base=%.2f %+.2f (%s) → %.2f",
            base, adjustment, regime, adjusted,
        )

    return adjusted


# ── Output writing (migrated from daily_predict.py) ──────────────────────────

def write_predictions(
    predictions: list[dict],
    date_str: str,
    s3_bucket: str,
    metrics: dict,
    dry_run: bool = False,
    veto_threshold: float | None = None,
    fd=None,
) -> None:
    """
    Write predictions JSON to S3 at both the dated key and latest.json.
    Also writes metrics/latest.json. All S3 operations are best-effort.

    Parameters
    ----------
    predictions : List of per-ticker prediction dicts.
    date_str :    Date string YYYY-MM-DD.
    s3_bucket :   S3 bucket name.
    metrics :     Metrics dict to write to predictor/metrics/latest.json.
    dry_run :     If True, print to stdout instead of writing to S3.
    veto_threshold : Confidence threshold for veto gate. Defaults to cfg.MIN_CONFIDENCE.
    """
    threshold = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    # Build the predictions envelope
    n_high_confidence = sum(
        1 for p in predictions
        if p.get("prediction_confidence", 0) >= threshold
    )

    output = {
        "date": date_str,
        "model_version": metrics.get("model_version", "unknown"),
        "model_hit_rate_30d": metrics.get("hit_rate_30d_rolling", None),
        "n_predictions": len(predictions),
        "n_high_confidence": n_high_confidence,
        "predictions": predictions,
    }

    metrics_out = {
        **metrics,
        "n_predictions_today": len(predictions),
        "n_high_confidence": n_high_confidence,
        "last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
    }

    predictions_json = json.dumps(output, indent=2)
    metrics_json = json.dumps(metrics_out, indent=2)

    if dry_run:
        print("=== PREDICTIONS (dry-run) ===")
        print(predictions_json)
        print("\n=== METRICS (dry-run) ===")
        print(metrics_json)
        return

    dated_key = cfg.PREDICTIONS_KEY.format(date=date_str)
    latest_key = cfg.PREDICTIONS_LATEST_KEY
    metrics_key = cfg.METRICS_KEY

    import boto3
    s3 = boto3.client("s3")

    # Write each S3 object independently so partial failures don't block others
    writes = [
        (dated_key, predictions_json, "predictions (dated)"),
        (latest_key, predictions_json, "predictions (latest)"),
        (metrics_key, metrics_json, "metrics"),
    ]
    n_ok = 0
    for key, body, label in writes:
        try:
            _s3_put_json(s3, s3_bucket, key, body)
            log.info("Written s3://%s/%s", s3_bucket, key)
            n_ok += 1
        except Exception as exc:
            log.error("S3 write failed for %s: %s", label, exc)
    if n_ok < len(writes):
        log.error(
            "Partial S3 write: %d/%d succeeded. Check IAM permissions for s3://%s",
            n_ok, len(writes), s3_bucket,
        )


# ── Predictor email ────────────────────────────────────────────────────────────

def _build_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
) -> tuple[str, str, str]:
    """
    Build subject, HTML body, and plain-text body for the combined morning briefing.

    When signals_data is supplied (the raw signals.json payload from the research
    pipeline), a research section is prepended containing market regime, buy
    candidates, and sector ratings. The GBM predictions follow as the second half.

    Returns
    -------
    (subject, html_body, plain_body)
    """
    import datetime as _dt

    _vt = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    model_version = metrics.get("model_version", "unknown")
    val_ic        = metrics.get("ic_30d")        # 30-day information coefficient
    n_total       = len(predictions)
    is_meta       = metrics.get("inference_mode") == "meta" or "meta" in model_version.lower()

    # Single list sorted by combined_rank (best first)
    sorted_preds = sorted(predictions, key=lambda p: p.get("combined_rank") or p.get("mse_rank") or 999)

    # Counts for subject line
    ups   = [p for p in predictions if p.get("predicted_direction") == "UP"]
    downs = [p for p in predictions if p.get("predicted_direction") == "DOWN"]

    # Vetoes: negative predicted alpha AND combined_rank in bottom half
    n_preds = len(predictions)
    vetoes = [
        p for p in predictions
        if (p.get("predicted_alpha", 0) or 0) < 0
        and p.get("combined_rank") is not None
        and p["combined_rank"] > n_preds / 2
    ]
    n_vetoed = len(vetoes)

    # ── Research data extraction ───────────────────────────────────────────────
    sd = signals_data or {}
    market_regime    = sd.get("market_regime", "")
    population       = sd.get("universe", []) or sd.get("population", [])
    buy_candidates   = sd.get("buy_candidates", []) or []
    sector_ratings   = sd.get("sector_ratings", {})
    sorted_sectors: list = []

    # Tickers the executor can act on that the GBM did NOT score — surface these
    # prominently so a stale/short predictions run can't silently hide buys.
    _pred_tickers     = {p.get("ticker") for p in predictions}
    unscored_buys     = [
        c for c in buy_candidates
        if isinstance(c, dict) and c.get("ticker") not in _pred_tickers
    ]
    unscored_tickers  = {c.get("ticker") for c in unscored_buys}

    # ── Subject ───────────────────────────────────────────────────────────────
    veto_str    = f" | {n_vetoed} veto{'es' if n_vetoed != 1 else ''}" if n_vetoed else ""
    regime_str  = f" | {market_regime.upper()}" if market_regime else ""
    cand_str    = f" | {len(population)} stocks" if population else ""
    subject = (
        f"Alpha Engine Brief | {date_str}{regime_str}{cand_str} | "
        f"{len(ups)} UP / {len(downs)} DOWN"
        f"{veto_str}"
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    _pt = _dt.timezone(_dt.timedelta(hours=-7))  # PDT
    run_time = _dt.datetime.now(_pt).strftime("%-I:%M %p PT")
    ic_str   = f"{val_ic:.4f}" if isinstance(val_ic, (int, float)) else "—"

    def _source_tag(p: dict) -> str:
        return ""  # buy_candidates merged into universe — no separate source tag

    def _alpha_str(p: dict) -> str:
        a = p.get("predicted_alpha")
        if a is None:
            return "—"
        return f"{'+' if a >= 0 else ''}{a * 100:.2f}%"

    def _conf_pct(p: dict) -> str:
        return f"{p.get('prediction_confidence', 0) * 100:.0f}%"

    # ── HTML ──────────────────────────────────────────────────────────────────
    TH = 'style="background:#f0f0f0; padding:4px 8px; text-align:left; border:1px solid #ccc;"'
    TD = 'style="padding:4px 8px; border:1px solid #ddd;"'
    TDR = 'style="padding:4px 8px; border:1px solid #ddd; text-align:right;"'
    TABLE = 'style="border-collapse:collapse; width:100%; font-family:monospace; font-size:12px;"'

    def _rank_str(p: dict, key: str = "model_rank") -> str:
        r = p.get(key)
        return f"{r}" if r is not None else "—"

    def _dir_badge(p: dict) -> str:
        d = p.get("predicted_direction", "")
        is_veto = (d == "DOWN" and p.get("prediction_confidence", 0) >= _vt)
        if is_veto:
            return '<span style="color:#c62828; font-weight:bold;">⚠ VETO</span>'
        colors = {"UP": "#2e7d32", "DOWN": "#c62828"}
        return f'<span style="color:{colors.get(d, "#888")}; font-weight:bold;">{d}</span>'

    def _meta_cols(p: dict) -> str:
        """Extra columns for meta-model predictions: momentum, vol, research.
        Regime column removed 2026-04-16 (Tier 0 classifier retired). The
        LLM-derived market_regime from research still renders in the header
        pill above the table — that's the regime signal the executor consumes."""
        mom = p.get("momentum_confirmation")
        vol = p.get("expected_move")
        rscore = p.get("research_calibrator_prob")
        return (
            f'<td {TDR}>{f"{mom:+.3f}" if mom is not None else "—"}</td>'
            f'<td {TDR}>{f"{vol:.3f}" if vol is not None else "—"}</td>'
            f'<td {TDR}>{f"{rscore:.0%}" if rscore is not None else "—"}</td>'
        )

    def _html_prediction_table(preds: list[dict]) -> str:
        if not preds:
            return '<p style="color:#888; font-style:italic;">No predictions available.</p>'
        n_preds = len(preds)
        rows = []
        for p in preds:
            cr = p.get("combined_rank")
            cr_str = f'{cr:.1f}' if cr is not None else "—"
            is_vetoed = (
                p.get("predicted_alpha", 0) is not None
                and (p.get("predicted_alpha", 0) or 0) < 0
                and cr is not None
                and cr > n_preds / 2
            )
            veto_tag = ' <span style="color:#d32f2f;">⚠ VETO</span>' if is_vetoed else ""
            rows.append(
                f'<tr>'
                f'<td {TD}><b>{p["ticker"]}{_source_tag(p)}</b></td>'
                f'<td {TDR}>{_alpha_str(p)}</td>'
                f'<td {TDR}>{cr_str}</td>'
                f'<td {TDR}>{_conf_pct(p)}</td>'
                f'<td {TD} style="text-align:center;">{_dir_badge(p)}{veto_tag}</td>'
                + (_meta_cols(p) if is_meta else "")
                + f'<td {TD}>{p.get("watchlist_source", "—")}</td>'
                f'</tr>'
            )
        meta_headers = (
            f'<th {TH}>Mom</th>'
            f'<th {TH}>Vol</th>'
            f'<th {TH}>Res.Cal</th>'
        ) if is_meta else ""
        return (
            f'<table {TABLE}>'
            f'<tr>'
            f'<th {TH}>Ticker</th>'
            f'<th {TH}>Alpha</th>'
            f'<th {TH}>Rank</th>'
            f'<th {TH}>Conf</th>'
            f'<th {TH}>Signal</th>'
            + meta_headers
            + f'<th {TH}>Source</th>'
            f'</tr>'
            + "\n".join(rows)
            + f'</table>'
        )

    veto_section_html = ""
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        veto_section_html = (
            f'<hr style="border:1px solid #eee; margin:16px 0;">'
            f'<h3 style="color:#c62828;">⚠ Vetoes ({n_vetoed})</h3>'
            f'<p style="font-size:12px; margin:4px 0;">'
            f'Negative predicted α + bottom-half combined rank — executor will override ENTER → HOLD:</p>'
            f'<p style="font-family:monospace; font-size:13px;"><b>{veto_tickers}</b></p>'
        )

    # ── Research section HTML ─────────────────────────────────────────────────
    research_html = ""
    if sd:
        # Market regime pill
        regime_color = {"bullish": "#2e7d32", "bearish": "#c62828"}.get(
            market_regime.lower(), "#555"
        )
        regime_pill = (
            f'<span style="display:inline-block; background:{regime_color}; color:#fff; '
            f'font-size:11px; padding:2px 8px; border-radius:3px; font-weight:bold;">'
            f'{market_regime.upper() if market_regime else "NEUTRAL"}</span>'
        )

        def _render_research_row(c: dict) -> str:
            score      = c.get("score") or c.get("long_term_score") or "—"
            conviction = c.get("conviction", "—")
            signal     = c.get("signal") or c.get("long_term_rating") or "—"
            sector     = c.get("sector", "—")
            gbm_veto   = c.get("gbm_veto", False)
            score_str  = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            badges = ""
            if gbm_veto:
                badges += ' <span style="color:#c62828; font-weight:bold; font-size:10px;">GBM⚠</span>'
            if c.get("ticker") in unscored_tickers:
                badges += ' <span style="color:#d84315; font-weight:bold; font-size:10px;" title="Actionable but no GBM prediction">NO PRED</span>'
            return (
                f'<tr>'
                f'<td {TD}><b>{c.get("ticker","?")}</b>{badges}</td>'
                f'<td {TDR}>{score_str}</td>'
                f'<td {TD}>{conviction}</td>'
                f'<td {TD}>{signal}</td>'
                f'<td {TD}>{sector}</td>'
                f'</tr>'
            )

        def _render_research_table(rows_src: list) -> str:
            rows = "".join(_render_research_row(c) for c in rows_src if isinstance(c, dict))
            if not rows:
                rows = f'<tr><td colspan="5" style="padding:4px 8px; color:#888; font-style:italic;">none</td></tr>'
            return (
                f'<table {TABLE}>'
                f'<tr><th {TH}>Ticker</th><th {TH}>Score</th><th {TH}>Conviction</th>'
                f'<th {TH}>Signal</th><th {TH}>Sector</th></tr>'
                f'{rows}'
                f'</table>'
            )

        # Buy Candidates = the actionable set (signal == ENTER). This is what the
        # executor sizes positions on; render it explicitly so every tradeable
        # ticker is visible in the morning brief even if it isn't in predictions.
        buy_table = _render_research_table(buy_candidates) if buy_candidates else ""
        cand_table = _render_research_table(population)

        # Sector ratings (top sectors sorted by rating desc, skip empty)
        sector_rows = ""
        sorted_sectors = sorted(
            [(s, v) for s, v in sector_ratings.items() if isinstance(v, dict)],
            key=lambda x: x[1].get("rating", 0),
            reverse=True,
        )
        for sector, v in sorted_sectors[:8]:
            rating   = v.get("rating", "—")
            modifier = v.get("modifier", "—")
            rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
            modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
            sector_rows += f'<tr><td {TD}>{sector}</td><td {TDR}>{rating_str}</td><td {TDR}>{modifier_str}</td></tr>'
        sector_table = ""
        if sector_rows:
            sector_table = (
                f'<table {TABLE}>'
                f'<tr><th {TH}>Sector</th><th {TH}>Rating</th><th {TH}>Modifier</th></tr>'
                f'{sector_rows}'
                f'</table>'
            )

        buy_block = ""
        if buy_table:
            unscored_note = ""
            if unscored_buys:
                _names = ", ".join(sorted(t for t in unscored_tickers if t))
                unscored_note = (
                    f'<p style="margin:4px 0 0 0; font-size:11px; color:#d84315;">'
                    f'⚠ {len(unscored_buys)} buy candidate{"s" if len(unscored_buys) != 1 else ""} '
                    f'not scored by GBM (executor can still size them): <b>{_names}</b>'
                    f'</p>'
                )
            buy_block = (
                f'<h4 style="margin:8px 0 4px 0; font-size:12px; color:#2e7d32;">'
                f'Buy Candidates ({len(buy_candidates)}) — actionable</h4>'
                f'{buy_table}'
                f'{unscored_note}'
            )

        research_html = (
            f'<div style="background:#f8f9fa; border-left:3px solid #555; padding:12px 16px; margin-bottom:16px;">'
            f'<h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">Research Brief</h3>'
            f'<p style="margin:0 0 8px 0;">Market Regime: {regime_pill}</p>'
            f'{buy_block}'
            f'<h4 style="margin:8px 0 4px 0; font-size:12px; color:#555;">Population ({len(population)})</h4>'
            f'{cand_table}'
            f'{"<h4 style=margin:8px 0 4px 0; font-size:12px; color:#555;>Sector Ratings</h4>" + sector_table if sector_table else ""}'
            f'</div>'
        )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:700px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Brief — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{model_version}</b> &nbsp;|&nbsp;'
        f'IC (val): <b>{ic_str}</b> &nbsp;|&nbsp;'
        f'Mode: <b>{metrics.get("inference_mode", "mse")}</b> &nbsp;|&nbsp;'
        f'Universe: <b>{n_total}</b> tickers &nbsp;|&nbsp;'
        f'Run at <b>{run_time}</b></p>'
        f'{research_html}'
        f'<h3 style="font-size:13px; color:#333; margin-bottom:4px;">{"Predictions" if is_meta else "GBM Predictions"}</h3>'
        f'{_html_prediction_table(sorted_preds)}'
        f'{veto_section_html}'
        f'<p style="font-size:11px; color:#aaa; margin-top:24px;">'
        f'⚠ VETO = negative α + bottom-half rank'
        + (' &nbsp;|&nbsp; Mom = momentum &nbsp;|&nbsp; Vol = expected move &nbsp;|&nbsp; Res.Cal = research calibrator P(correct)' if is_meta else '')
        + f'</p>'
        f'</body></html>'
    )

    # ── Plain text ────────────────────────────────────────────────────────────
    def _plain_prediction_list(preds: list[dict]) -> str:
        if not preds:
            return "  (none)\n"
        _n = len(preds)
        lines = []
        for p in preds:
            cr = p.get("combined_rank")
            cr_str = f"{cr:.1f}" if cr is not None else "—"
            is_vetoed = (
                (p.get("predicted_alpha", 0) or 0) < 0
                and cr is not None
                and cr > _n / 2
            )
            veto = " [VETO]" if is_vetoed else ""
            lines.append(
                f"  {p['ticker']:<6}  α={_alpha_str(p):>7}  Rank {cr_str:<5}"
                f"  {_conf_pct(p):>4}  {p.get('predicted_direction','—'):<4}"
                f"  {p.get('watchlist_source', '—')}{_source_tag(p)}{veto}"
            )
        return "\n".join(lines) + "\n"

    def _plain_research_row(c: dict) -> str:
        score = c.get("score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "—"
        flags = ""
        if c.get("gbm_veto"):
            flags += " [GBM VETO]"
        if c.get("ticker") in unscored_tickers:
            flags += " [NO PRED]"
        return (
            f"  {c.get('ticker','?'):<6}  score={score_str:>5}  "
            f"{c.get('conviction','—'):<10}  {c.get('signal','—'):<8}  "
            f"{c.get('sector','—')}{flags}\n"
        )

    # Research plain section
    research_plain = ""
    if sd:
        research_plain = (
            f"\n{'='*60}\n"
            f"RESEARCH BRIEF\n"
            f"{'='*60}\n"
            f"Market Regime: {market_regime.upper() if market_regime else 'NEUTRAL'}\n"
        )
        if buy_candidates:
            research_plain += f"\nBuy Candidates ({len(buy_candidates)}) — actionable:\n"
            for c in buy_candidates:
                if isinstance(c, dict):
                    research_plain += _plain_research_row(c)
            if unscored_buys:
                _names = ", ".join(sorted(t for t in unscored_tickers if t))
                research_plain += (
                    f"  ⚠ {len(unscored_buys)} not scored by GBM: {_names}\n"
                )
        if population:
            research_plain += f"\nPopulation ({len(population)}):\n"
            for c in population:
                if isinstance(c, dict):
                    research_plain += _plain_research_row(c)
        if sector_ratings:
            research_plain += "\nSector Ratings:\n"
            for sector, v in sorted_sectors[:8]:
                rating = v.get("rating", "—")
                modifier = v.get("modifier", "—")
                rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
                modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
                research_plain += f"  {sector:<20}  rating={rating_str:>3}  modifier={modifier_str}\n"

    plain_body = (
        f"Alpha Engine Brief — {date_str}\n"
        f"Model: {model_version}  IC(val): {ic_str}  Mode: {metrics.get('inference_mode', 'mse')}  Universe: {n_total}  Run: {run_time}\n"
        f"{research_plain}"
        f"\n{'='*60}\n"
        f"{'PREDICTIONS' if is_meta else 'GBM PREDICTIONS'}\n"
        f"{'='*60}\n"
        f"\nPredictions (sorted by combined rank, {len(ups)} UP / {len(downs)} DOWN)\n"
        f"{_plain_prediction_list(sorted_preds)}"
    )
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        plain_body += (
            f"\nOPTION A VETOES ({n_vetoed}): {veto_tickers}\n"
            f"(DOWN + conf >= {int(_vt * 100)}% → executor HOLD override)\n"
        )

    return subject, html_body, plain_body


def send_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
) -> bool:
    """
    Send combined morning briefing email via Gmail SMTP (primary) or SES (fallback).

    When signals_data is provided (research pipeline's signals.json payload),
    the email includes a research section (market regime, buy candidates, sector
    ratings) followed by the GBM predictions — one complete morning briefing.

    Reads from environment / config:
        EMAIL_SENDER        — from-address
        EMAIL_RECIPIENTS    — list of recipient addresses
        GMAIL_APP_PASSWORD  — enables Gmail SMTP path (recommended)
        AWS_REGION          — SES region fallback

    Returns True on success, False on any failure. Never raises.
    """
    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info(
            "Predictor email skipped — set EMAIL_SENDER and EMAIL_RECIPIENTS "
            "env vars in the Lambda to enable"
        )
        return False

    try:
        subject, html_body, plain_body = _build_predictor_email(
            predictions, metrics, date_str, signals_data=signals_data,
            veto_threshold=veto_threshold,
        )
    except Exception as exc:
        log.warning("Failed to build predictor email body: %s", exc)
        return False

    # Morning briefing HTML archival removed — no consumers read it (email delivers the content).

    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if app_password:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

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
            log.info("Predictor email sent via Gmail SMTP: '%s'", subject)
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
        log.info("Predictor email sent via SES: '%s'", subject)
        return True
    except Exception as exc:
        log.warning("SES send failed: %s — predictor email not delivered", exc)
        return False


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Write predictions, metrics, email, and health status."""

    # ── Build metrics ────────────────────────────────────────────────────────
    gbm_meta = _load_gbm_meta(ctx)

    if ctx.inference_mode == "meta":
        last_trained = gbm_meta.get("trained_date", "unknown")
    elif ctx.model_type == "gbm":
        last_trained = gbm_meta.get("trained_date", getattr(ctx.scorer, "_best_iteration", "unknown"))
    else:
        last_trained = ctx.checkpoint.get("epoch", "unknown")

    metrics = {
        "model_version": ctx.model_version,
        "model_type": ctx.model_type,
        "inference_mode": ctx.inference_mode,
        "last_trained": last_trained,
        "training_samples": gbm_meta.get("n_train") if ctx.model_type == "gbm" else None,
        "val_loss": round(float(ctx.val_loss), 6) if isinstance(ctx.val_loss, (int, float)) else None,
        "ic_30d": gbm_meta.get("test_ic") if ctx.model_type == "gbm" else None,
        "ic_ir_30d": gbm_meta.get("ic_ir") if ctx.model_type == "gbm" else None,
        "hit_rate_30d_rolling": None,
        "price_freshness": {
            "max_age_days": max(ctx.ticker_data_age.values()) if ctx.ticker_data_age else -1,
            "n_stale": sum(1 for d in ctx.ticker_data_age.values() if d > 1),
        },
    }

    # ── Veto logic ───────────────────────────────────────────────────────────
    market_regime = ctx.signals_data.get("market_regime", "") if ctx.signals_data else ""
    veto_thresh = get_veto_threshold(ctx.bucket, market_regime=market_regime)

    n_preds = len(ctx.predictions)
    for p in ctx.predictions:
        cr = p.get("combined_rank")
        alpha = p.get("predicted_alpha", 0) or 0
        p["gbm_veto"] = (alpha < 0 and cr is not None and cr > n_preds / 2)

    # ── Write predictions ────────────────────────────────────────────────────
    write_predictions(ctx.predictions, ctx.date_str, ctx.bucket, metrics,
                      dry_run=ctx.dry_run, veto_threshold=veto_thresh, fd=ctx.fd)

    # ── Send email ───────────────────────────────────────────────────────────
    if not ctx.dry_run:
        email_sent = send_predictor_email(
            ctx.predictions, metrics, ctx.date_str,
            signals_data=ctx.signals_data, veto_threshold=veto_thresh,
        )
        if not email_sent:
            log.warning("Predictor email failed to send (Gmail + SES both failed)")

    # ── Health status ────────────────────────────────────────────────────────
    try:
        from health_status import write_health
        n_up = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP")
        n_down = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN")
        write_health(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            status="ok",
            run_date=ctx.date_str,
            duration_seconds=ctx.elapsed_seconds(),
            summary={
                "n_predictions": len(ctx.predictions),
                "n_up": n_up,
                "n_down": n_down,
            },
        )
    except Exception as _he:
        log.warning("Health status write failed: %s", _he)

    # ── Data manifest ────────────────────────────────────────────────────────
    try:
        from health_status import write_data_manifest
        write_data_manifest(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            run_date=ctx.date_str,
            manifest={
                "n_predictions": len(ctx.predictions),
                "n_up": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP"),
                "n_down": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN"),
                "n_tickers_failed": len(getattr(cfg, 'FAILED_TICKERS', [])),
                "model_version": getattr(cfg, 'GBM_VERSION', 'unknown'),
            },
        )
    except Exception as _me:
        log.warning("Data manifest write failed: %s", _me)

    log.info("Predictor run complete for %s", ctx.date_str)


def _load_gbm_meta(ctx: PipelineContext) -> dict:
    """Load GBM training metadata from S3 (best-effort)."""
    if ctx.model_type != "gbm" or ctx.local:
        return {}
    try:
        import boto3 as _boto3
        _s3 = _boto3.client("s3")
        _resp = _s3.get_object(Bucket=ctx.bucket, Key=cfg.GBM_WEIGHTS_META_KEY)
        meta = json.loads(_resp["Body"].read())
        log.info("GBM weights meta loaded: trained_date=%s  n_train=%s",
                 meta.get("trained_date"), meta.get("n_train"))
        return meta
    except Exception as _exc:
        log.debug("GBM weights meta not found or unreadable: %s", _exc)
        return {}
