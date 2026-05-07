"""
analysis/regime_cutover_gate.py — Phase 4 cutover gate validator.

Computes the audit §8 Phase 4 cutover criterion:

> "Gate: this only ships if regime-conditioned ensemble IC > single-Ridge
> IC by ≥ 15% relative across the validation period."

Reads historical predictions from S3 (predictor/predictions/{date}.json
over a window), pairs each prediction with the realized 5d sector-relative
alpha (computed from price data at date+5d), and computes:

  - IC of ``predicted_alpha`` vs realized 5d alpha (single-Ridge canonical)
  - IC of ``regime_conditioned_alpha`` vs realized 5d alpha (parallel)

Returns a structured ``CutoverGateResult`` carrying both ICs, the relative
lift, the pass/fail decision, and a reason string for operator review.

Typical use is offline before flipping the cutover feature flag (PR 5b
adds the inference-side feature flag that consumes this verdict). The
validator is also designed to run inside the SF as a Phase 4 gate state
once the cutover is in production — same module, two execution sites.

Usage:
    python -m analysis.regime_cutover_gate --window 30
    python -m analysis.regime_cutover_gate --window 60 --output gate.json

This is PR 5a of the Phase 4 sequence — module + tests, no production
wiring. PR 5b adds the inference-side feature flag and the SF gate state.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


@dataclass(frozen=True)
class CutoverGateResult:
    """Result of the Phase 4 regime-conditioned cutover gate.

    ``passed``: True if regime-conditioned IC > single-Ridge IC by at
    least ``relative_lift_threshold`` (default 1.15 = 15% relative lift).

    ``single_ridge_ic`` / ``regime_conditioned_ic``: Pearson correlation
    of each prediction stream against realized 5d sector-relative alpha
    over the validation window.

    ``relative_lift``: regime_conditioned_ic / single_ridge_ic, or None
    when single_ridge_ic is too small to compute meaningfully.

    ``n_pairs``: number of (date, ticker) pairs that contributed both
    a single-Ridge and a regime-conditioned alpha along with a finite
    realized alpha (after filtering NaN tail rows).

    ``reason``: human-readable explanation. Populated for both pass and
    fail outcomes so operators can audit the decision later.

    ``per_regime_ic``: bonus diagnostic — IC of regime_conditioned_alpha
    sliced by predicted_regime label. Helps operators see whether the
    overall lift is concentrated in one regime or distributed.
    """
    passed: bool
    single_ridge_ic: float
    regime_conditioned_ic: float
    relative_lift: float | None
    n_pairs: int
    reason: str
    relative_lift_threshold: float
    per_regime_ic: dict


def compute_parity_ic(
    pairs: list[dict],
) -> tuple[float, float, int]:
    """Compute Pearson IC for both prediction streams against realized alpha.

    Args:
        pairs: list of dicts each carrying:
            - ``predicted_alpha`` (single-Ridge canonical)
            - ``regime_conditioned_alpha`` (Phase 4 parallel)
            - ``realized_alpha`` (forward 5d sector-relative, computed
              from price data at date+5d)
            Optional keys (preserved for filtering): ``ticker``, ``date``.

    Returns:
        (single_ridge_ic, regime_conditioned_ic, n_pairs) — both ICs as
        Pearson correlations on the filtered set of pairs that have all
        three fields finite. Returns ``(NaN, NaN, 0)`` when fewer than
        10 pairs have valid data.
    """
    import numpy as np

    if not pairs:
        return float("nan"), float("nan"), 0

    pred_alpha = np.array([
        p.get("predicted_alpha") if p.get("predicted_alpha") is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)
    rcm_alpha = np.array([
        p.get("regime_conditioned_alpha") if p.get("regime_conditioned_alpha") is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)
    realized = np.array([
        p.get("realized_alpha") if p.get("realized_alpha") is not None else float("nan")
        for p in pairs
    ], dtype=np.float64)

    mask = (
        np.isfinite(pred_alpha)
        & np.isfinite(rcm_alpha)
        & np.isfinite(realized)
    )
    n = int(mask.sum())
    if n < 10:
        return float("nan"), float("nan"), n

    # Use Pearson correlation. Spearman would be more robust to outliers
    # but the existing meta-Ridge IC reporting (manifest's meta_model.ic)
    # uses Pearson, so this matches and stays cross-comparable.
    single_ridge_ic = float(np.corrcoef(pred_alpha[mask], realized[mask])[0, 1])
    regime_ic = float(np.corrcoef(rcm_alpha[mask], realized[mask])[0, 1])
    return single_ridge_ic, regime_ic, n


def compute_per_regime_ic(pairs: list[dict]) -> dict:
    """Slice regime_conditioned_alpha vs realized IC by predicted_regime.

    Returns a dict keyed by regime ("bear" / "neutral" / "bull") with
    sub-dicts of {spearman, n}. Regimes with fewer than 10 finite pairs
    return None for the IC.
    """
    import numpy as np
    from scipy.stats import spearmanr

    by_regime: dict[str, list[dict]] = {"bull": [], "neutral": [], "bear": []}
    for p in pairs:
        r = p.get("predicted_regime")
        if r in by_regime:
            by_regime[r].append(p)

    result = {}
    for regime, regime_pairs in by_regime.items():
        if len(regime_pairs) < 10:
            result[regime] = {"spearman": None, "n": len(regime_pairs)}
            continue
        rcm = np.array([
            p.get("regime_conditioned_alpha") if p.get("regime_conditioned_alpha") is not None
            else float("nan") for p in regime_pairs
        ], dtype=np.float64)
        realized = np.array([
            p.get("realized_alpha") if p.get("realized_alpha") is not None
            else float("nan") for p in regime_pairs
        ], dtype=np.float64)
        mask = np.isfinite(rcm) & np.isfinite(realized)
        if mask.sum() < 10:
            result[regime] = {"spearman": None, "n": int(mask.sum())}
            continue
        sp = spearmanr(rcm[mask], realized[mask])
        ic = float(sp.correlation) if np.isfinite(sp.correlation) else None
        result[regime] = {"spearman": ic, "n": int(mask.sum())}
    return result


def validate_cutover_gate(
    pairs: list[dict],
    relative_lift_threshold: float = 1.15,
    min_pairs: int = 100,
    min_single_ridge_ic_floor: float = 0.005,
) -> CutoverGateResult:
    """Apply the audit §8 Phase 4 cutover criterion.

    Pass conditions (ALL must hold):
      1. n_pairs >= min_pairs (default 100) — enough data for a reliable
         comparison. Below this, the gate stays neutral (passed=False
         with a sample-size reason).
      2. single_ridge_ic > min_single_ridge_ic_floor — the comparison
         is only meaningful if the baseline has *some* predictive power.
         If single-Ridge IC is essentially zero, dividing by it produces
         meaningless lift ratios; gate stays neutral.
      3. relative_lift >= relative_lift_threshold — the audit's core
         criterion. Default 1.15 = 15% relative lift.

    Returns CutoverGateResult with ``passed`` set + ``reason`` populated
    for both outcomes so operators can audit. ``per_regime_ic`` always
    populated as a bonus diagnostic.
    """
    sr_ic, rcm_ic, n = compute_parity_ic(pairs)
    per_regime = compute_per_regime_ic(pairs)

    if n < min_pairs:
        return CutoverGateResult(
            passed=False,
            single_ridge_ic=sr_ic,
            regime_conditioned_ic=rcm_ic,
            relative_lift=None,
            n_pairs=n,
            reason=(
                f"insufficient data: {n} pairs (need >= {min_pairs}). "
                f"Wait for more weekday inference cycles + realized 5d windows "
                f"to close before re-running the gate."
            ),
            relative_lift_threshold=relative_lift_threshold,
            per_regime_ic=per_regime,
        )

    if sr_ic <= min_single_ridge_ic_floor:
        return CutoverGateResult(
            passed=False,
            single_ridge_ic=sr_ic,
            regime_conditioned_ic=rcm_ic,
            relative_lift=None,
            n_pairs=n,
            reason=(
                f"single-Ridge IC {sr_ic:.4f} below floor {min_single_ridge_ic_floor:.4f} "
                f"— baseline is too weak to compute a meaningful relative lift. "
                f"Investigate whether the underlying single-Ridge model is healthy "
                f"before deciding on regime-conditioned cutover."
            ),
            relative_lift_threshold=relative_lift_threshold,
            per_regime_ic=per_regime,
        )

    lift = rcm_ic / sr_ic
    if lift >= relative_lift_threshold:
        reason = (
            f"PASS: regime-conditioned IC {rcm_ic:.4f} exceeds single-Ridge "
            f"IC {sr_ic:.4f} by {(lift - 1) * 100:.1f}% relative "
            f"(threshold {(relative_lift_threshold - 1) * 100:.0f}%). "
            f"Cutover is safe to flip; n={n} pairs."
        )
        return CutoverGateResult(
            passed=True,
            single_ridge_ic=sr_ic,
            regime_conditioned_ic=rcm_ic,
            relative_lift=lift,
            n_pairs=n,
            reason=reason,
            relative_lift_threshold=relative_lift_threshold,
            per_regime_ic=per_regime,
        )

    reason = (
        f"FAIL: regime-conditioned IC {rcm_ic:.4f} only "
        f"{(lift - 1) * 100:+.1f}% relative to single-Ridge IC {sr_ic:.4f} "
        f"(threshold +{(relative_lift_threshold - 1) * 100:.0f}%). "
        f"Hold cutover; n={n} pairs."
    )
    return CutoverGateResult(
        passed=False,
        single_ridge_ic=sr_ic,
        regime_conditioned_ic=rcm_ic,
        relative_lift=lift,
        n_pairs=n,
        reason=reason,
        relative_lift_threshold=relative_lift_threshold,
        per_regime_ic=per_regime,
    )


def _load_prediction_history(bucket: str, n_days: int) -> list[dict]:
    """Download the last ``n_days`` of predictions JSONs from S3 and flatten.

    Returns a list of per-(date, ticker) dicts each carrying:
        ``date`` / ``ticker`` / ``predicted_alpha`` /
        ``regime_conditioned_alpha`` / ``predicted_regime``.

    Pre-Phase 4 dates won't have regime_conditioned_alpha / predicted_regime
    (None values) — those rows still load cleanly and are filtered out by
    the IC computation when None.
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
            pairs.append({
                "date": data.get("date"),
                "ticker": p.get("ticker"),
                "predicted_alpha": p.get("predicted_alpha"),
                "regime_conditioned_alpha": p.get("regime_conditioned_alpha"),
                "predicted_regime": p.get("predicted_regime"),
                # realized_alpha is None — caller fills via _compute_realized_alphas.
                "realized_alpha": None,
            })
    log.info("Loaded %d (date, ticker) prediction rows over %d days", len(pairs), n_days)
    return pairs


def main():
    """CLI entry point for offline gate evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bucket", default=cfg.RESEARCH_BUCKET,
        help=f"S3 bucket. Default: {cfg.RESEARCH_BUCKET}",
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

    pairs = _load_prediction_history(args.bucket, args.window)
    if not pairs:
        log.error(
            "No prediction history loaded — has Phase 4 PR 4 (#100) deployed "
            "to inference Lambda yet? Manifest must include "
            "regime_conditioned_alpha + predicted_regime fields."
        )
        sys.exit(1)

    log.warning(
        "TODO: realized_alpha computation not implemented in this PR; "
        "gate will return insufficient-data verdict. PR 5b wires the "
        "realized-alpha computation from price data + the inference-side "
        "feature flag that consumes this verdict."
    )

    result = validate_cutover_gate(
        pairs, relative_lift_threshold=args.threshold,
    )

    print(json.dumps(asdict(result), indent=2, default=str))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        log.info("Result written to %s", args.output)

    sys.exit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
