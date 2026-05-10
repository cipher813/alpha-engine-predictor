"""
analysis/variant_cutover_gate.py — variant-agnostic cutover gate validator.

Generic version of the audit §8 Phase 4 cutover criterion. Compares any
two parallel prediction streams against realized alpha and applies a
relative-lift threshold:

> "Gate: variant only ships if variant IC > baseline IC by
> ≥ ``relative_lift_threshold`` (default 1.15 = 15% relative)."

Originally implemented as ``analysis/regime_cutover_gate.py`` for the
regime-conditioned cutover (single-Ridge baseline vs regime-conditioned
variant). Refactored here into a variant-agnostic module so the same
gate framework can validate every future cutover under the regime-
conditioning rebuild — Stage 1 (macros into L1), Stage 2 (per-ticker
risk features), Stage 3 (triple-barrier alpha labels), Stage 4 (continuous
regime feature in L2). Each cutover provides:

  - a baseline alpha field (the current production stream)
  - a variant alpha field (the parallel-observation stream)
  - optionally, a slice field for sub-aggregated diagnostic IC

The 15% relative-lift threshold is the institutional default; callers
override per-stage as needed.

Usage (programmatic):

    from analysis.variant_cutover_gate import validate_cutover_gate
    result = validate_cutover_gate(
        pairs=...,
        baseline_field="predicted_alpha",
        variant_field="predicted_alpha_macro_aug",  # Stage 1 example
        relative_lift_threshold=1.15,
    )

Usage (CLI):

    python -m analysis.variant_cutover_gate \
        --baseline predicted_alpha \
        --variant predicted_alpha_triple_barrier \
        --window 30 --output gate.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


@dataclass(frozen=True)
class VariantCutoverGateResult:
    """Result of a variant cutover gate evaluation.

    ``passed``: True if variant IC > baseline IC by at least
    ``relative_lift_threshold`` (default 1.15 = 15% relative lift).

    ``baseline_ic`` / ``variant_ic``: Pearson correlation of each
    prediction stream against realized alpha over the validation window.

    ``relative_lift``: ``variant_ic / baseline_ic``, or None when
    ``baseline_ic`` is below ``min_baseline_ic_floor`` (the gate stays
    neutral because lift ratios become meaningless near zero).

    ``n_pairs``: number of (date, ticker) pairs that contributed both
    a baseline and a variant alpha along with a finite realized alpha
    (after filtering NaN tail rows).

    ``reason``: human-readable explanation. Populated for both pass and
    fail outcomes so operators can audit the decision later.

    ``baseline_field`` / ``variant_field``: the names of the streams
    compared. Persisted in the result so downstream readers know what
    was evaluated without having to track CLI args.

    ``slice_field``: optional sub-slicing key (e.g., ``"predicted_regime"``).
    None when no per-slice diagnostic was requested.

    ``per_slice_ic``: optional per-slice IC of the variant stream.
    Populated when ``slice_field`` is set; otherwise empty dict.
    """
    passed: bool
    baseline_ic: float
    variant_ic: float
    relative_lift: float | None
    n_pairs: int
    reason: str
    relative_lift_threshold: float
    baseline_field: str
    variant_field: str
    slice_field: str | None = None
    per_slice_ic: dict = field(default_factory=dict)


def compute_parity_ic(
    pairs: list[dict],
    baseline_field: str,
    variant_field: str,
    realized_field: str = "realized_alpha",
    min_pairs_for_ic: int = 10,
) -> tuple[float, float, int]:
    """Compute Pearson IC for both prediction streams against realized alpha.

    Args:
        pairs: list of dicts each carrying ``baseline_field``,
            ``variant_field``, and ``realized_field``. Optional keys
            (preserved for filtering): ``ticker``, ``date``.
        baseline_field: dict key for the baseline alpha stream
            (e.g., ``"predicted_alpha"``).
        variant_field: dict key for the variant alpha stream
            (e.g., ``"regime_conditioned_alpha"``).
        realized_field: dict key for realized alpha. Defaults to
            ``"realized_alpha"``.
        min_pairs_for_ic: minimum number of (baseline, variant, realized)
            triples with finite values required to compute meaningful IC.
            Defaults to 10. Below this, returns NaN ICs.

    Returns:
        ``(baseline_ic, variant_ic, n_pairs)`` — both ICs as Pearson
        correlations on the filtered set of pairs that have all three
        fields finite. Returns ``(NaN, NaN, n)`` when fewer than
        ``min_pairs_for_ic`` pairs have valid data.
    """
    import numpy as np

    if not pairs:
        return float("nan"), float("nan"), 0

    baseline = np.array([
        p.get(baseline_field) if p.get(baseline_field) is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)
    variant = np.array([
        p.get(variant_field) if p.get(variant_field) is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)
    realized = np.array([
        p.get(realized_field) if p.get(realized_field) is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)

    mask = (
        np.isfinite(baseline)
        & np.isfinite(variant)
        & np.isfinite(realized)
    )
    n = int(mask.sum())
    if n < min_pairs_for_ic:
        return float("nan"), float("nan"), n

    # Use Pearson correlation. Spearman would be more robust to outliers
    # but the existing meta-Ridge IC reporting (manifest's meta_model.ic)
    # uses Pearson, so this matches and stays cross-comparable.
    baseline_ic = float(np.corrcoef(baseline[mask], realized[mask])[0, 1])
    variant_ic = float(np.corrcoef(variant[mask], realized[mask])[0, 1])
    return baseline_ic, variant_ic, n


def compute_per_slice_ic(
    pairs: list[dict],
    variant_field: str,
    slice_field: str,
    realized_field: str = "realized_alpha",
    min_pairs_per_slice: int = 10,
) -> dict:
    """Slice variant IC by an arbitrary categorical field.

    Args:
        pairs: as in ``compute_parity_ic``.
        variant_field: dict key for the variant alpha stream.
        slice_field: dict key with categorical slice values
            (e.g., ``"predicted_regime"``, ``"sector"``).
        realized_field: dict key for realized alpha.
        min_pairs_per_slice: minimum pairs in a slice to compute IC.
            Below this, slice returns ``{"spearman": None, "n": <count>}``.

    Returns:
        dict keyed by slice value with sub-dicts of ``{spearman, n}``.
        Uses Spearman (rank IC) here because per-slice samples can be
        small enough that Pearson is sensitive to outliers.
    """
    import numpy as np
    from scipy.stats import spearmanr

    by_slice: dict[str, list[dict]] = {}
    for p in pairs:
        s = p.get(slice_field)
        if s is None:
            continue
        by_slice.setdefault(str(s), []).append(p)

    result: dict = {}
    for slice_val, slice_pairs in by_slice.items():
        if len(slice_pairs) < min_pairs_per_slice:
            result[slice_val] = {"spearman": None, "n": len(slice_pairs)}
            continue
        variant = np.array([
            p.get(variant_field) if p.get(variant_field) is not None
            else float("nan") for p in slice_pairs
        ], dtype=np.float64)
        realized = np.array([
            p.get(realized_field) if p.get(realized_field) is not None
            else float("nan") for p in slice_pairs
        ], dtype=np.float64)
        mask = np.isfinite(variant) & np.isfinite(realized)
        if mask.sum() < min_pairs_per_slice:
            result[slice_val] = {"spearman": None, "n": int(mask.sum())}
            continue
        sp = spearmanr(variant[mask], realized[mask])
        ic = float(sp.correlation) if np.isfinite(sp.correlation) else None
        result[slice_val] = {"spearman": ic, "n": int(mask.sum())}
    return result


def validate_cutover_gate(
    pairs: list[dict],
    baseline_field: str,
    variant_field: str,
    realized_field: str = "realized_alpha",
    slice_field: str | None = None,
    relative_lift_threshold: float = 1.15,
    min_pairs: int = 100,
    min_baseline_ic_floor: float = 0.005,
) -> VariantCutoverGateResult:
    """Apply the variant-agnostic cutover criterion.

    Pass conditions (ALL must hold):
      1. ``n_pairs >= min_pairs`` (default 100) — enough data for a reliable
         comparison. Below this, the gate stays neutral (passed=False
         with a sample-size reason).
      2. ``baseline_ic > min_baseline_ic_floor`` — the comparison is only
         meaningful if the baseline has *some* predictive power. If
         baseline IC is essentially zero, dividing by it produces
         meaningless lift ratios; gate stays neutral.
      3. ``relative_lift >= relative_lift_threshold`` — the audit's core
         criterion. Default 1.15 = 15% relative lift.

    Args:
        pairs: list of dicts as documented in ``compute_parity_ic``.
        baseline_field: baseline alpha stream key.
        variant_field: variant alpha stream key.
        realized_field: realized alpha key. Defaults to ``"realized_alpha"``.
        slice_field: optional sub-slicing key. When provided,
            ``per_slice_ic`` is populated; otherwise empty dict.
        relative_lift_threshold: minimum variant/baseline ratio for pass.
        min_pairs: sample-size floor.
        min_baseline_ic_floor: minimum baseline IC for meaningful lift.

    Returns:
        ``VariantCutoverGateResult`` with ``passed`` set + ``reason``
        populated for both outcomes so operators can audit.
    """
    bl_ic, var_ic, n = compute_parity_ic(
        pairs, baseline_field, variant_field, realized_field=realized_field,
    )
    per_slice: dict = {}
    if slice_field is not None:
        per_slice = compute_per_slice_ic(
            pairs, variant_field, slice_field, realized_field=realized_field,
        )

    common_kwargs: dict = dict(
        baseline_ic=bl_ic,
        variant_ic=var_ic,
        n_pairs=n,
        relative_lift_threshold=relative_lift_threshold,
        baseline_field=baseline_field,
        variant_field=variant_field,
        slice_field=slice_field,
        per_slice_ic=per_slice,
    )

    if n < min_pairs:
        return VariantCutoverGateResult(
            passed=False,
            relative_lift=None,
            reason=(
                f"insufficient data: {n} pairs (need >= {min_pairs}). "
                f"Wait for more inference cycles + realized windows to "
                f"close before re-running the gate."
            ),
            **common_kwargs,
        )

    if bl_ic <= min_baseline_ic_floor:
        return VariantCutoverGateResult(
            passed=False,
            relative_lift=None,
            reason=(
                f"baseline IC {bl_ic:.4f} below floor "
                f"{min_baseline_ic_floor:.4f} — baseline ({baseline_field}) "
                f"is too weak to compute a meaningful relative lift. "
                f"Investigate baseline health before deciding on cutover."
            ),
            **common_kwargs,
        )

    lift = var_ic / bl_ic
    if lift >= relative_lift_threshold:
        reason = (
            f"PASS: variant ({variant_field}) IC {var_ic:.4f} exceeds "
            f"baseline ({baseline_field}) IC {bl_ic:.4f} by "
            f"{(lift - 1) * 100:.1f}% relative "
            f"(threshold {(relative_lift_threshold - 1) * 100:.0f}%). "
            f"Cutover is safe to flip; n={n} pairs."
        )
        return VariantCutoverGateResult(
            passed=True,
            relative_lift=lift,
            reason=reason,
            **common_kwargs,
        )

    reason = (
        f"FAIL: variant ({variant_field}) IC {var_ic:.4f} only "
        f"{(lift - 1) * 100:+.1f}% relative to baseline ({baseline_field}) "
        f"IC {bl_ic:.4f} (threshold +{(relative_lift_threshold - 1) * 100:.0f}%). "
        f"Hold cutover; n={n} pairs."
    )
    return VariantCutoverGateResult(
        passed=False,
        relative_lift=lift,
        reason=reason,
        **common_kwargs,
    )


def compute_realized_alpha_for_pairs(
    pairs: list[dict],
    horizon_days: int,
    prices_by_ticker: dict,
    sector_map: dict[str, str],
) -> int:
    """Fill ``realized_alpha`` on each pair via forward sector-neutral residual log-return.

    For each (date, ticker) pair, computes the realized alpha as
    ``log(stock_close[t+H] / stock_close[t]) - log(bench_close[t+H] / bench_close[t])``
    where ``H = horizon_days`` and ``bench`` is the ticker's sector ETF
    (per ``sector_map``) or SPY when no sector mapping exists. Pairs whose
    forward window hasn't closed yet (``t+H`` past end of data) get
    ``realized_alpha = None`` and are filtered downstream by the gate's
    ``min_pairs`` floor.

    The horizon should match the canonical fixed-horizon label
    (``cfg.FORWARD_DAYS``) so the gate measures predictive power on the
    institutionally-defined ground truth — if triple-barrier predictions
    win on the fixed-21d axis, the lift is real (not measurement-axis
    bias from grading the variant on its own training target).

    Args:
        pairs: list of dicts each carrying ``date`` (ISO YYYY-MM-DD or
            pd.Timestamp) and ``ticker``. Mutated in place — each dict
            gets its ``realized_alpha`` key set.
        horizon_days: forward window in trading days (e.g., 21).
        prices_by_ticker: ``{ticker: pd.Series of Close prices indexed
            by trading day}``. Caller supplies — runner orchestration
            handles the S3/ArcticDB load.
        sector_map: ``{ticker: sector_etf_symbol}``. SPY fallback when
            ticker not in map. The benchmark price series must be present
            in ``prices_by_ticker`` keyed by its symbol.

    Returns:
        Number of pairs filled with a finite realized alpha (i.e., where
        the forward window had closed). Operators/tests use this to
        confirm enough realized data exists before invoking the gate.
    """
    import datetime
    import math

    import pandas as pd

    n_filled = 0
    for p in pairs:
        date_raw = p.get("date")
        ticker = p.get("ticker")
        if date_raw is None or ticker is None:
            continue
        # Accept both ISO strings and pd.Timestamp inputs (the SF training
        # path emits Timestamps; the CLI loads ISO strings from JSON).
        if isinstance(date_raw, str):
            try:
                pred_date = pd.Timestamp(date_raw).normalize()
            except Exception:
                continue
        elif isinstance(date_raw, (datetime.date, datetime.datetime, pd.Timestamp)):
            pred_date = pd.Timestamp(date_raw).normalize()
        else:
            continue

        stock_series = prices_by_ticker.get(ticker)
        if stock_series is None or stock_series.empty:
            continue
        bench_symbol = sector_map.get(ticker, "SPY")
        bench_series = prices_by_ticker.get(bench_symbol)
        if bench_series is None or bench_series.empty:
            # Fall back to SPY if the sector ETF isn't in prices_by_ticker
            bench_series = prices_by_ticker.get("SPY")
            if bench_series is None or bench_series.empty:
                continue

        # Find the prediction date's row in stock + bench. Use ``method='ffill'``
        # so a non-trading day (e.g., holiday) snaps back to the prior session.
        try:
            stock_idx = stock_series.index.get_indexer([pred_date], method="ffill")[0]
            bench_idx = bench_series.index.get_indexer([pred_date], method="ffill")[0]
        except Exception:
            continue
        if stock_idx < 0 or bench_idx < 0:
            continue

        stock_fwd_idx = stock_idx + horizon_days
        bench_fwd_idx = bench_idx + horizon_days
        # Forward window hasn't closed → leave realized_alpha as None
        if stock_fwd_idx >= len(stock_series) or bench_fwd_idx >= len(bench_series):
            continue

        try:
            stock_t = float(stock_series.iloc[stock_idx])
            stock_t_plus = float(stock_series.iloc[stock_fwd_idx])
            bench_t = float(bench_series.iloc[bench_idx])
            bench_t_plus = float(bench_series.iloc[bench_fwd_idx])
        except Exception:
            continue
        if (
            stock_t <= 0 or stock_t_plus <= 0
            or bench_t <= 0 or bench_t_plus <= 0
        ):
            continue

        stock_log_ret = math.log(stock_t_plus / stock_t)
        bench_log_ret = math.log(bench_t_plus / bench_t)
        p["realized_alpha"] = stock_log_ret - bench_log_ret
        n_filled += 1
    return n_filled


def _load_prediction_history(
    bucket: str,
    n_days: int,
    baseline_field: str,
    variant_field: str,
    slice_field: str | None = None,
) -> list[dict]:
    """Download last ``n_days`` of prediction JSONs from S3 and flatten.

    Returns a list of per-(date, ticker) dicts each carrying:
        ``date`` / ``ticker`` / ``baseline_field`` / ``variant_field`` /
        (optional) ``slice_field`` / ``realized_alpha`` (None — caller
        fills via downstream realized-alpha computation).

    Pre-cutover dates won't have ``variant_field`` (None values) — those
    rows still load cleanly and are filtered out by the IC computation
    when None.
    """
    import boto3
    import datetime

    s3 = boto3.client("s3")
    today = datetime.date.today()
    pairs: list[dict] = []
    for d_offset in range(n_days):
        d = today - datetime.timedelta(days=d_offset)
        key = f"predictor/predictions/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except s3.exceptions.NoSuchKey:
            continue
        except Exception as e:
            log.warning("Failed to load %s: %s", key, e)
            continue
        for p in data.get("predictions", []):
            row: dict = {
                "date": data.get("date"),
                "ticker": p.get("ticker"),
                baseline_field: p.get(baseline_field),
                variant_field: p.get(variant_field),
                "realized_alpha": None,
            }
            if slice_field is not None:
                row[slice_field] = p.get(slice_field)
            pairs.append(row)
    log.info(
        "Loaded %d (date, ticker) prediction rows over %d days "
        "(baseline=%s, variant=%s)",
        len(pairs), n_days, baseline_field, variant_field,
    )
    return pairs


def main():
    """CLI entry point for offline gate evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bucket", default=cfg.RESEARCH_BUCKET,
        help=f"S3 bucket. Default: {cfg.RESEARCH_BUCKET}",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Baseline alpha field name (e.g., predicted_alpha).",
    )
    parser.add_argument(
        "--variant", required=True,
        help="Variant alpha field name (e.g., regime_conditioned_alpha).",
    )
    parser.add_argument(
        "--slice", default=None, dest="slice_field",
        help="Optional categorical field for per-slice IC diagnostic.",
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help="Trailing window of days to load prediction history. Default: 30.",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.15,
        help="Relative lift threshold for cutover. Default: 1.15 (15%%).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path for the gate result.",
    )
    args = parser.parse_args()

    pairs = _load_prediction_history(
        args.bucket, args.window, args.baseline, args.variant,
        slice_field=args.slice_field,
    )
    if not pairs:
        log.error(
            "No prediction history loaded — verify the variant field "
            "(%s) is being written to predictions JSONs.",
            args.variant,
        )
        sys.exit(1)

    log.warning(
        "TODO: realized_alpha computation not implemented in this PR; "
        "gate will return insufficient-data verdict until a downstream "
        "step fills realized_alpha from price data + horizon alignment."
    )

    result = validate_cutover_gate(
        pairs,
        baseline_field=args.baseline,
        variant_field=args.variant,
        slice_field=args.slice_field,
        relative_lift_threshold=args.threshold,
    )

    print(json.dumps(asdict(result), indent=2, default=str))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        log.info("Result written to %s", args.output)

    sys.exit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
