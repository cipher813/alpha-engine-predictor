"""
analysis/horizon_battery.py — offline horizon-IC measurement battery.

Reads the OOS rows persisted by meta_trainer at
``s3://{bucket}/predictor/diagnostics/oos_rows/{latest|YYYY-MM-DD}.parquet``
and computes the same overlap / non-overlap / per-regime / bootstrap-CI
breakdown that the training pipeline emits. Lets operators iterate on
horizon decisions without waiting for the next Saturday training cycle:
vary regime thresholds, horizon sets, bootstrap iteration counts, etc.

Usage:
    python -m analysis.horizon_battery
    python -m analysis.horizon_battery --date 2026-05-09 --output report.json
    python -m analysis.horizon_battery --bootstrap-iter 5000

Per the 2026-05-07 predictor audit Track B (PR 4/N): the audit's
"Phase 1 deliverable B" requires multi-regime IC ratios at 5d / 10d /
21d / 60d / 90d using non-overlapping windows + bootstrap CIs. This
module is the offline harness that makes those measurements iteratable
on demand.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from training.meta_trainer import (
    _DIAGNOSTIC_HORIZONS,
    _bootstrap_ic_ci_by_date,
    _classify_regime,
    _nonoverlapping_date_mask,
)

OOS_ROWS_PREFIX = "predictor/diagnostics/oos_rows/"


def load_oos_rows(bucket: str, date: str | None = None):
    """Download the OOS rows parquet from S3 and return a DataFrame.

    ``date`` of ``None`` reads ``latest.parquet``. Otherwise reads
    ``{date}.parquet``. Returns ``None`` if the object doesn't exist —
    caller decides whether that's a hard fail or a "training hasn't run
    yet" warning.
    """
    import boto3
    import io
    import pandas as pd

    key = f"{OOS_ROWS_PREFIX}{'latest' if date is None else date}.parquet"
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded %d OOS rows from s3://%s/%s", len(df), bucket, key)
        return df
    except s3.exceptions.NoSuchKey:
        log.error("No OOS rows at s3://%s/%s — has training run since the persistence PR landed?", bucket, key)
        return None


def compute_horizon_battery(
    df,
    horizons: list[int] | None = None,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute the full horizon IC battery against the loaded OOS rows.

    Mirrors the diagnostic computed during training in meta_trainer's
    Step 7.1, but operates on the persisted rows so no training rerun
    is needed. Useful for iterating on horizon parameters: bootstrap
    iterations, regime thresholds (via custom classifier), etc.

    Args:
        df: DataFrame from ``load_oos_rows``. Must have columns
            ``date``, ``actual_fwd_{h}d`` for each horizon h, and the
            META_FEATURES used by the meta-model. The meta-model
            predictions are not in the rows, so this function fits a
            fresh Ridge to the rows and computes IC against that — same
            pattern as training, just standalone.
        horizons: list of horizon ints. Default: _DIAGNOSTIC_HORIZONS.
        bootstrap_iter: number of bootstrap iterations per horizon.
        seed: RNG seed for bootstrap reproducibility.

    Returns:
        dict with shape:
            {
                "horizons": [...],
                "regime_distribution": {"bull": N, "neutral": N, "bear": N},
                "curve": {
                    "5d": {"spearman": ..., "n": ..., "spearman_ci_lo": ...,
                           "spearman_ci_hi": ..., "spearman_nonoverlap": ...,
                           "n_nonoverlap": ..., "spearman_nonoverlap_ci_lo": ...,
                           "spearman_nonoverlap_ci_hi": ...,
                           "by_regime": {"bull": {...}, "neutral": {...}, "bear": {...}}},
                    ...
                },
            }
    """
    import numpy as np
    from scipy.stats import spearmanr
    from model.meta_model import MetaModel, META_FEATURES

    horizons = horizons or list(_DIAGNOSTIC_HORIZONS)

    # Refit the meta-model on the OOS rows so we get predictions to
    # correlate against. The training pipeline does this same step at
    # meta_trainer.py:866 — same Ridge alpha=1.0, same META_FEATURES.
    log.info("Fitting Ridge on %d OOS rows × %d META_FEATURES", len(df), len(META_FEATURES))
    meta_X = df[META_FEATURES].to_numpy(dtype=np.float32)
    meta_y = df["actual_fwd"].to_numpy(dtype=np.float32)
    model = MetaModel(alpha=1.0)
    model.fit(meta_X, meta_y, feature_names=META_FEATURES)
    preds = model.predict(meta_X).ravel()

    # Classify regime per row using the same heuristic as training.
    regimes = [_classify_regime(r) for r in df.to_dict(orient="records")]
    regime_dist = {
        "bull": regimes.count("bull"),
        "neutral": regimes.count("neutral"),
        "bear": regimes.count("bear"),
    }
    log.info("Regime distribution: %s", regime_dist)

    dates = df["date"].tolist()

    curve: dict = {}
    for h in horizons:
        col = f"actual_fwd_{h}d"
        if col not in df.columns:
            log.warning("Column %s not in OOS rows — skipping horizon %dd", col, h)
            continue
        y_h = df[col].to_numpy()
        mask = np.isfinite(y_h)

        # Overall overlapping IC.
        if mask.sum() >= 100:
            sp = spearmanr(preds[mask], y_h[mask])
            ic = float(sp.correlation) if np.isfinite(sp.correlation) else 0.0
        else:
            ic = float("nan")

        # Bootstrap CI on overlapping IC.
        ic_ci_lo, ic_ci_hi = float("nan"), float("nan")
        if mask.sum() >= 100:
            ic_ci_lo, ic_ci_hi = _bootstrap_ic_ci_by_date(
                preds[mask], y_h[mask],
                [d for d, m in zip(dates, mask) if m],
                n_iter=bootstrap_iter, ci=0.95, seed=seed + h,
            )

        # Non-overlapping subsample IC + CI.
        nonoverlap_ic, nonoverlap_n = float("nan"), 0
        nonoverlap_ci_lo, nonoverlap_ci_hi = float("nan"), float("nan")
        nonoverlap_mask = _nonoverlapping_date_mask(dates, h)
        combined_mask = np.array(nonoverlap_mask) & mask
        if combined_mask.sum() >= 10:
            sp_no = spearmanr(preds[combined_mask], y_h[combined_mask])
            nonoverlap_ic = (
                float(sp_no.correlation)
                if np.isfinite(sp_no.correlation) else float("nan")
            )
            nonoverlap_n = int(combined_mask.sum())
            nonoverlap_ci_lo, nonoverlap_ci_hi = _bootstrap_ic_ci_by_date(
                preds[combined_mask], y_h[combined_mask],
                [d for d, m in zip(dates, combined_mask) if m],
                n_iter=bootstrap_iter, ci=0.95, seed=seed + h + 1000,
            )

        # Per-regime IC slice.
        regime_ics: dict = {}
        for regime in ("bull", "neutral", "bear"):
            r_mask = np.array([rr == regime for rr in regimes]) & mask
            if r_mask.sum() >= 30:
                sp_r = spearmanr(preds[r_mask], y_h[r_mask])
                ic_r = (
                    float(sp_r.correlation)
                    if np.isfinite(sp_r.correlation) else float("nan")
                )
            else:
                ic_r = float("nan")
            regime_ics[regime] = {
                "spearman": _round_or_none(ic_r),
                "n": int(r_mask.sum()),
            }

        curve[f"{h}d"] = {
            "spearman": _round_or_none(ic),
            "n": int(mask.sum()),
            "spearman_ci_lo": _round_or_none(ic_ci_lo),
            "spearman_ci_hi": _round_or_none(ic_ci_hi),
            "spearman_nonoverlap": _round_or_none(nonoverlap_ic),
            "n_nonoverlap": nonoverlap_n,
            "spearman_nonoverlap_ci_lo": _round_or_none(nonoverlap_ci_lo),
            "spearman_nonoverlap_ci_hi": _round_or_none(nonoverlap_ci_hi),
            "by_regime": regime_ics,
        }

    return {
        "horizons": [f"{h}d" for h in horizons],
        "regime_distribution": regime_dist,
        "curve": curve,
        "n_rows": len(df),
        "bootstrap_iter": bootstrap_iter,
    }


def _round_or_none(x: float, ndigits: int = 6):
    """Round if finite, else None (JSON-friendly)."""
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), ndigits)


def format_report(report: dict) -> str:
    """Human-readable summary for console output."""
    lines = [
        f"Horizon battery — {report['n_rows']} OOS rows, "
        f"{report['bootstrap_iter']}-iter bootstrap CI",
        f"Regime distribution: {report['regime_distribution']}",
        "",
        f"{'Horizon':<8}{'IC':<10}{'CI':<22}{'IC (no-ovr)':<14}{'CI (no-ovr)':<22}"
        f"{'IC bull':<10}{'IC neutral':<12}{'IC bear':<10}",
    ]
    for hkey, h in report["curve"].items():
        ic = _fmt(h.get("spearman"))
        ci = _fmt_ci(h.get("spearman_ci_lo"), h.get("spearman_ci_hi"))
        ic_no = _fmt(h.get("spearman_nonoverlap"))
        ci_no = _fmt_ci(h.get("spearman_nonoverlap_ci_lo"), h.get("spearman_nonoverlap_ci_hi"))
        ic_bull = _fmt(h["by_regime"]["bull"]["spearman"])
        ic_neutral = _fmt(h["by_regime"]["neutral"]["spearman"])
        ic_bear = _fmt(h["by_regime"]["bear"]["spearman"])
        lines.append(
            f"{hkey:<8}{ic:<10}{ci:<22}{ic_no:<14}{ci_no:<22}"
            f"{ic_bull:<10}{ic_neutral:<12}{ic_bear:<10}"
        )
    return "\n".join(lines)


def _fmt(x):
    return f"{x:+.4f}" if isinstance(x, (int, float)) else "—"


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "[—]"
    return f"[{lo:+.3f},{hi:+.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--date", default=None,
        help="OOS rows date (YYYY-MM-DD). Default: latest.",
    )
    parser.add_argument(
        "--bucket", default=cfg.RESEARCH_BUCKET,
        help=f"S3 bucket. Default: {cfg.RESEARCH_BUCKET}",
    )
    parser.add_argument(
        "--bootstrap-iter", type=int, default=1000,
        help="Bootstrap iterations per IC. Default: 1000.",
    )
    parser.add_argument(
        "--horizons", default=None,
        help="Comma-separated horizons (e.g. 5,10,21,60). Default: all diagnostic horizons.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path. Default: print summary only.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap RNG seed. Default: 42.",
    )
    args = parser.parse_args()

    horizons = None
    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    df = load_oos_rows(args.bucket, date=args.date)
    if df is None or len(df) == 0:
        log.error("No OOS rows available — exiting with nonzero status.")
        sys.exit(1)

    report = compute_horizon_battery(
        df, horizons=horizons,
        bootstrap_iter=args.bootstrap_iter,
        seed=args.seed,
    )

    print(format_report(report))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("JSON report written to %s", args.output)


if __name__ == "__main__":
    main()
