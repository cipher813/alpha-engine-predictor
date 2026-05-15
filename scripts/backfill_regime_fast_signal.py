#!/usr/bin/env python3
"""
scripts/backfill_regime_fast_signal.py — Stage F1 calibration backfill.

Recalibrates the **daily** BOCPD hazard + run-length window against a 10y
daily ``intensity_z`` reconstruction, and reports the F1 gate metrics:

  • T1-analog timing  — median lead time (trading days) of forced_bear
    onset vs the retrospective weekly smoothed-HMM bear-stretch onset,
    plus the false-positive rate (forced_bear days inside no bear
    stretch). Positive lead = the fast leg fired *before* the slow leg
    (the entire point); ≤0 = no value over weekly.
  • Whipsaw / churn   — toggles/yr, mean forced_bear dwell (days),
    fraction of episodes shorter than ``min_clear_days``. A combo that
    toggles > ~6×/yr or dwells < ~15 trading days is miscalibrated
    *regardless of downstream Sortino* — fail F1, do not promote.

Design notes
------------
- Daily ``intensity_z`` is reconstructed from the SAME price-cache
  parquets the live path uses, via ``fetch_macro_feature_history_daily``
  (the weekly fetcher already builds a daily frame then downsamples;
  this skips the downsample). No new data plumbing.
- Ground truth is the **weekly** smoothed-HMM bear stretch. That is the
  correct resolution for a latent regime — NBER itself dates cycles at
  monthly resolution; sub-week "regime truth" does not exist. Lead time
  measured in trading days against a weekly-resolution onset is the
  honest metric, not a limitation.
- Bounded compute (per feedback_no_resource_intensive_local_scripts):
  one hmmlearn fit on ~520 weekly obs (seconds) + a microsecond-per-step
  daily replay over ~2520 obs × a small sweep grid. No torch, no
  all-core loads. Ground truth is best-effort — whipsaw/episode stats
  still print if the HMM can't be built (e.g. offline).

Usage
-----
    python scripts/backfill_regime_fast_signal.py \
        [--bucket alpha-engine-research] \
        [--from-parquet-dir /local/dir/with/SPY.parquet/...] \
        [--out report.md]
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from regime.bocpd import BOCPDDetector  # noqa: E402
from regime.composite import compute_composite_intensity  # noqa: E402
from regime.fast_signal import FastSignalState, FastSignalTunables, step  # noqa: E402

# HMM feature subset for the ground-truth fit — mirrors
# regime/handler.py:DEFAULT_HMM_FEATURE_COLUMNS (inlined to avoid
# importing the Lambda handler module here).
HMM_FEATURE_COLUMNS = ("spy_20d_return", "vix_level", "hy_oas_bps")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backfill_fast_signal")
log.setLevel(logging.INFO)

# Sweep grid. Daily hazard ≈ 1/N trading days mean regime; the daily
# run-length window in trading days.
HAZARD_GRID = (1 / 250.0, 1 / 500.0, 1 / 750.0, 1 / 1000.0)
MIN_RUNLENGTH_GRID = (5, 10, 15)
TRADING_DAYS_PER_YEAR = 252
# Daily z-score window for the composite — daily analog of the live
# path's 260-week (~5y) AQR convention.
DAILY_ZSCORE_WINDOW = 1260


# ── Local-parquet stub so the script runs without S3/network ─────────────
class _LocalParquetS3:
    """Minimal ``get_object`` stub reading ``{dir}/{TICKER}.parquet``."""

    def __init__(self, directory: str) -> None:
        self._dir = directory.rstrip("/")

    class exceptions:  # noqa: N801 — mimic botocore client.exceptions
        class NoSuchKey(Exception):
            pass

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        path = f"{self._dir}/{Key.split('/')[-1]}"
        try:
            with open(path, "rb") as fh:
                return {"Body": io.BytesIO(fh.read())}
        except FileNotFoundError as e:  # pragma: no cover - dev convenience
            raise _LocalParquetS3.exceptions.NoSuchKey(str(e))


def _daily_intensity_series(daily_features: pd.DataFrame) -> pd.Series:
    """Per-day composite intensity_z over a rolling 5y window. Replays
    the same ``compute_composite_intensity`` the live path uses so the
    backfill is apples-to-apples with production."""
    out: list[float] = []
    idx: list[pd.Timestamp] = []
    cols = list(daily_features.columns)
    values = daily_features.to_dict("records")
    stamps = list(daily_features.index)
    for i, (ts, row) in enumerate(zip(stamps, values)):
        lo = max(0, i - DAILY_ZSCORE_WINDOW)
        hist = daily_features.iloc[lo : i + 1]
        if len(hist) < 60:  # not enough history for a stable z-score yet
            continue
        res = compute_composite_intensity(
            current={c: row[c] for c in cols},
            history=hist,
            rolling_window_weeks=None,  # already windowed by the slice
        )
        # Live inference convention: negative = risk-off. The composite
        # returns stress orientation (positive = risk-off); invert to
        # match ctx.regime_intensity_z exactly.
        out.append(-float(res["intensity_z"]))
        idx.append(ts)
    return pd.Series(out, index=pd.DatetimeIndex(idx), name="intensity_z")


@dataclass
class ComboResult:
    hazard: float
    min_runlength: int
    n_days: int
    n_forced_bear_days: int
    forced_bear_fraction: float
    n_episodes: int
    toggles_per_year: float
    mean_dwell_days: float
    frac_episodes_below_clear: float
    median_lead_days: float | None
    false_positive_rate: float | None


def _episodes(forced: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    eps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for ts, on in forced.items():
        if on and start is None:
            start = ts
        elif not on and start is not None:
            eps.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        eps.append((start, forced.index[-1]))
    return eps


def _replay(iz: pd.Series, hazard: float, min_rl: int) -> pd.Series:
    det = BOCPDDetector.for_daily(hazard=hazard)
    tun = FastSignalTunables(min_runlength_for_change=min_rl)
    state: FastSignalState | None = None
    forced: list[bool] = []
    for ts, val in iz.items():
        d = ts.strftime("%Y-%m-%d")
        state, art = step(
            state, intensity_z=float(val), trading_day=d,
            calendar_date=d, run_id="backfill", detector=det, tunables=tun,
        )
        forced.append(bool(art["forced_bear"]))
    return pd.Series(forced, index=iz.index, name="forced_bear")


def _bear_stretches(s3, bucket: str) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
    """Weekly smoothed-HMM bear stretches (best-effort ground truth)."""
    try:
        from regime.features import fetch_macro_feature_history
        from regime.hmm import HMMRegimeClassifier

        wk = fetch_macro_feature_history(s3_client=s3, bucket=bucket)
        cols = tuple(
            c for c in HMM_FEATURE_COLUMNS
            if c in wk.columns and not wk[c].dropna().empty
        )
        fit_df = wk[list(cols)].dropna()
        hmm = HMMRegimeClassifier(n_states=3, random_state=42).fit(fit_df, cols)
        proba = hmm.predict_proba_smoother(fit_df)  # cols: (bear,neutral,bull)
        is_bear = pd.Series(proba.argmax(axis=1) == 0, index=fit_df.index)
        stretches: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        start = None
        for ts, b in is_bear.items():
            if b and start is None:
                start = ts
            elif not b and start is not None:
                stretches.append((start, prev))
                start = None
            prev = ts
        if start is not None:
            stretches.append((start, is_bear.index[-1]))
        return stretches
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.warning("Ground-truth HMM unavailable (%s) — lead-time omitted.", exc)
        return None


def _lead_metrics(
    forced: pd.Series,
    stretches: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[float | None, float]:
    eps = _episodes(forced)
    leads: list[float] = []
    for s, _ in stretches:
        prior = [e for e in eps if e[0] <= s + pd.Timedelta(days=21)]
        if not prior:
            continue
        nearest = min(prior, key=lambda e: abs((e[0] - s).days))
        leads.append((s - nearest[0]).days * (TRADING_DAYS_PER_YEAR / 365.0))
    in_bear = pd.Series(False, index=forced.index)
    for s, e in stretches:
        in_bear.loc[(in_bear.index >= s) & (in_bear.index <= e)] = True
    fb_days = forced[forced]
    if len(fb_days) == 0:
        fpr = 0.0
    else:
        fpr = float((~in_bear.reindex(fb_days.index).fillna(False)).mean())
    median_lead = float(np.median(leads)) if leads else None
    return median_lead, fpr


def _evaluate(iz: pd.Series, stretches) -> list[ComboResult]:
    results: list[ComboResult] = []
    years = max(len(iz) / TRADING_DAYS_PER_YEAR, 1e-9)
    for hz in HAZARD_GRID:
        for mr in MIN_RUNLENGTH_GRID:
            forced = _replay(iz, hz, mr)
            eps = _episodes(forced)
            dwell = [
                (e[1] - e[0]).days * (TRADING_DAYS_PER_YEAR / 365.0)
                for e in eps
            ]
            below = sum(1 for d in dwell if d < FastSignalTunables().min_clear_days)
            if stretches is not None:
                lead, fpr = _lead_metrics(forced, stretches)
            else:
                lead, fpr = None, None
            results.append(ComboResult(
                hazard=hz, min_runlength=mr, n_days=len(iz),
                n_forced_bear_days=int(forced.sum()),
                forced_bear_fraction=float(forced.mean()),
                n_episodes=len(eps),
                toggles_per_year=len(eps) / years,
                mean_dwell_days=float(np.mean(dwell)) if dwell else 0.0,
                frac_episodes_below_clear=(below / len(eps)) if eps else 0.0,
                median_lead_days=lead, false_positive_rate=fpr,
            ))
    return results


def _report(iz: pd.Series, results: list[ComboResult]) -> str:
    lines = [
        "# Stage F1 — Daily BOCPD fast-signal calibration backfill",
        "",
        f"Daily intensity_z series: **{len(iz)} obs**, "
        f"{iz.index.min().date()} → {iz.index.max().date()} "
        f"(~{len(iz)/TRADING_DAYS_PER_YEAR:.1f}y).",
        "",
        "Gate reminders (regime-fast-signal-260515.md §5): promote a combo "
        "only if **median lead > 0** *and* whipsaw is sane "
        "(toggles/yr ≲ 6, mean dwell ≳ 15d, few sub-clear episodes). "
        "Lead/FPR omitted ⇒ ground-truth HMM unavailable this run.",
        "",
        "| hazard | min_rl | fb_days | fb_frac | episodes | toggles/yr | "
        "mean_dwell_d | %ep<clear | median_lead_d | FPR |",
        "|--------|--------|---------|---------|----------|------------|"
        "--------------|-----------|---------------|-----|",
    ]
    for r in results:
        lead = "—" if r.median_lead_days is None else f"{r.median_lead_days:+.1f}"
        fpr = "—" if r.false_positive_rate is None else f"{r.false_positive_rate:.2f}"
        lines.append(
            f"| 1/{1/r.hazard:.0f} | {r.min_runlength} | "
            f"{r.n_forced_bear_days} | {r.forced_bear_fraction:.3f} | "
            f"{r.n_episodes} | {r.toggles_per_year:.2f} | "
            f"{r.mean_dwell_days:.1f} | {r.frac_episodes_below_clear:.2f} | "
            f"{lead} | {fpr} |"
        )
    lines += [
        "",
        "**Interpretation.** Pick the combo with positive median lead and "
        "the lowest whipsaw that still fires on the known bear stretches. "
        "If every combo whipsaws, the composite intensity_z is too noisy "
        "for a daily detector — escalate before F2 (do not relax the "
        "confirmation gate to force a pass).",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--from-parquet-dir", default=None,
                    help="Local dir with {TICKER}.parquet to skip S3.")
    ap.add_argument("--out", default=None, help="Write the report here too.")
    args = ap.parse_args()

    if args.from_parquet_dir:
        s3 = _LocalParquetS3(args.from_parquet_dir)
    else:
        import boto3
        s3 = boto3.client("s3")

    from regime.features import fetch_macro_feature_history_daily

    daily = fetch_macro_feature_history_daily(s3_client=s3, bucket=args.bucket)
    if daily.empty:
        log.error("No daily feature history — cannot calibrate. Check the "
                  "price-cache parquets.")
        return 1
    iz = _daily_intensity_series(daily)
    if len(iz) < 252:
        log.error("Only %d daily intensity_z obs — need ≥1y to calibrate.",
                  len(iz))
        return 1

    stretches = _bear_stretches(s3, args.bucket)
    results = _evaluate(iz, stretches)
    report = _report(iz, results)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        log.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
