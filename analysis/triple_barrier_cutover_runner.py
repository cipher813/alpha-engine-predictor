"""
analysis/triple_barrier_cutover_runner.py — Stage 3 cutover gate runner.

Orchestrates the variant_cutover_gate weekly evaluation for the triple-
barrier L2 Ridge cutover (Stage 3 PR 4 of the regime-conditioning rebuild
arc). Designed to run from Saturday SF after PredictorTraining completes
and writes meta_model_tb.pkl, OR as a standalone CLI for ad-hoc operator
evaluation.

Pipeline:

  1. Load past N days of ``predictor/predictions/{date}.json`` from S3
  2. Extract (date, ticker, predicted_alpha, meta_alpha_tb) per row
  3. Load price + sector_map from S3 to compute realized 21d sector-
     neutral residual log-return per (date, ticker)
  4. Invoke ``validate_cutover_gate`` with 15% relative-lift threshold
  5. Write result JSON to
     ``s3://{bucket}/predictor/variant_gates/triple_barrier_{date}.json``

The realized alpha uses the same fixed-21d sector-neutral residual that
the canonical Ridge trains on — NOT the triple-barrier path-dependent
label. This is a deliberate institutional choice (LdP-conservative): the
variant must demonstrate predictive power on the BASELINE'S own ground
truth axis to flip the cutover. If we evaluated the variant on its own
training target, the gate would be biased toward the variant by
construction — circular.

Cutover (PR 5) is gated on ≥3 consecutive Saturday gate evaluations
showing ``passed: true`` with ``relative_lift >= 1.15`` and
``n_pairs >= 100``. Below those floors the gate stays neutral.

Plan: alpha-engine-docs/private/triple-barrier-260510.md
ROADMAP: alpha-engine-config/private-docs/ROADMAP.md line ~1501.

Usage (programmatic):

    from analysis.triple_barrier_cutover_runner import run_gate
    result = run_gate(bucket="alpha-engine-research", n_days=42)

Usage (CLI):

    python -m analysis.triple_barrier_cutover_runner \
        --bucket alpha-engine-research --window 42

The default 42-day window covers ~6 weeks of weekday inferences (a
21-day forward window plus one full 21-day evaluation window after
labels close). 4 weeks of evaluable predictions is the minimum the
gate's 100-pair floor will typically clear with ~25 universe tickers
per day.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
from analysis.variant_cutover_gate import (  # noqa: E402
    _load_prediction_history,
    compute_realized_alpha_for_pairs,
    validate_cutover_gate,
)


# Field names — pinned to the production predictions JSON schema. The
# baseline is ``predicted_alpha`` (the canonical Ridge's emitted field
# from inference/stages/run_inference.py); the variant is
# ``meta_alpha_tb`` (added in Stage 3 PR 3).
BASELINE_FIELD = "predicted_alpha"
VARIANT_FIELD = "meta_alpha_tb"

# Default forward-realized horizon — matches cfg.FORWARD_DAYS (the
# canonical label the canonical Ridge trains on).
DEFAULT_HORIZON = 21

# Default trailing window for prediction history. ~6 weeks gives the
# rolling-4-weeks gate a buffer for the 21-day realized-window close.
DEFAULT_WINDOW_DAYS = 42

# Default S3 prefix where gate result JSONs land. Operator dashboards
# and the eventual evaluator-email surfacer read from here.
DEFAULT_OUTPUT_PREFIX = "predictor/variant_gates"


def _load_prices_from_s3(
    bucket: str,
    tickers: set[str],
    s3_client=None,
) -> dict:
    """Download per-ticker price parquets from S3 and return as
    ``{ticker: pd.Series of Close indexed by date}``.

    Tickers absent from S3 are silently skipped — caller's downstream
    realized-alpha join filters those rows.
    """
    import boto3
    import pandas as pd

    s3 = s3_client or boto3.client("s3")
    prices: dict = {}
    for ticker in tickers:
        key = f"predictor/price_cache/{ticker}.parquet"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(BytesIO(obj["Body"].read()))
        except Exception as e:
            log.debug("Skipping price load for %s: %s", ticker, e)
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        if df.index.name != "date" and not isinstance(df.index, pd.DatetimeIndex):
            # Some parquets use a 'date' column rather than a DatetimeIndex
            if "date" in df.columns:
                df = df.set_index("date")
        try:
            df.index = pd.DatetimeIndex(df.index).normalize()
        except Exception:
            continue
        prices[ticker] = df["Close"].astype(float).sort_index()
    log.info("Loaded prices for %d/%d tickers from s3://%s", len(prices), len(tickers), bucket)
    return prices


def _load_sector_map_from_s3(bucket: str, s3_client=None) -> dict:
    import boto3

    s3 = s3_client or boto3.client("s3")
    for key in ("data/sector_map.json", "predictor/price_cache/sector_map.json"):
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:
            continue
    log.warning("sector_map.json not found at expected S3 keys — defaulting all tickers to SPY benchmark")
    return {}


def run_gate(
    bucket: str = None,
    n_days: int = DEFAULT_WINDOW_DAYS,
    horizon_days: int = DEFAULT_HORIZON,
    threshold: float = 1.15,
    output_s3_key: str | None = None,
    write_to_s3: bool = True,
    s3_client=None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
    prediction_pairs: list[dict] | None = None,
) -> dict:
    """End-to-end Stage 3 cutover-gate evaluation.

    Args:
        bucket: S3 bucket. Defaults to ``cfg.RESEARCH_BUCKET``.
        n_days: trailing prediction-history window. Default 42.
        horizon_days: forward-realized window. Default 21 (matches
            ``cfg.FORWARD_DAYS``).
        threshold: relative-lift threshold for pass. Default 1.15 (15%).
        output_s3_key: S3 key for the result JSON. When None, defaults
            to ``predictor/variant_gates/triple_barrier_{today}.json``.
        write_to_s3: when False, skip the S3 write (local-only run for
            CLI / testing).
        s3_client: optional pre-configured boto3 S3 client (testing).
        prices_by_ticker / sector_map / prediction_pairs: optional
            pre-loaded inputs for testing — when provided, the runner
            skips the corresponding S3 load.

    Returns:
        dict-form gate result (asdict of VariantCutoverGateResult) plus
        runner metadata: ``run_date``, ``window_days``, ``horizon_days``,
        ``n_realized_filled``, ``n_pairs_loaded``.
    """
    bucket = bucket or cfg.RESEARCH_BUCKET
    run_date = datetime.date.today().isoformat()

    if prediction_pairs is None:
        prediction_pairs = _load_prediction_history(
            bucket, n_days, BASELINE_FIELD, VARIANT_FIELD,
        )
    n_pairs_loaded = len(prediction_pairs)
    if n_pairs_loaded == 0:
        log.warning("No prediction history loaded — gate will report insufficient data")

    if prices_by_ticker is None or sector_map is None:
        tickers_in_pairs = {p["ticker"] for p in prediction_pairs if p.get("ticker")}
        sector_map = sector_map or _load_sector_map_from_s3(bucket, s3_client=s3_client)
        # Always include SPY as the universal fallback benchmark
        bench_symbols = set(sector_map.values()) | {"SPY"}
        prices_by_ticker = prices_by_ticker or _load_prices_from_s3(
            bucket, tickers_in_pairs | bench_symbols, s3_client=s3_client,
        )

    n_realized_filled = compute_realized_alpha_for_pairs(
        prediction_pairs,
        horizon_days=horizon_days,
        prices_by_ticker=prices_by_ticker,
        sector_map=sector_map,
    )
    log.info(
        "Realized alpha filled for %d/%d pairs (horizon=%dd; remainder are "
        "either unrealized — forward window not closed — or missing price data)",
        n_realized_filled, n_pairs_loaded, horizon_days,
    )

    gate_result = validate_cutover_gate(
        prediction_pairs,
        baseline_field=BASELINE_FIELD,
        variant_field=VARIANT_FIELD,
        relative_lift_threshold=threshold,
    )
    log.info("Gate verdict: %s — %s", gate_result.passed, gate_result.reason)

    payload = {
        **asdict(gate_result),
        "run_date": run_date,
        "window_days": n_days,
        "horizon_days": horizon_days,
        "n_pairs_loaded": n_pairs_loaded,
        "n_realized_filled": n_realized_filled,
    }

    if write_to_s3:
        import boto3

        s3 = s3_client or boto3.client("s3")
        key = output_s3_key or f"{DEFAULT_OUTPUT_PREFIX}/triple_barrier_{run_date}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        log.info("Gate result written to s3://%s/%s", bucket, key)

    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bucket", default=None,
        help=f"S3 bucket. Default: cfg.RESEARCH_BUCKET ({cfg.RESEARCH_BUCKET}).",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"Trailing prediction-history window in days. Default: {DEFAULT_WINDOW_DAYS}.",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Forward-realized window in trading days. Default: {DEFAULT_HORIZON}.",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.15,
        help="Relative-lift threshold for cutover. Default: 1.15 (15%%).",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Skip S3 write; print result to stdout only (local / test mode).",
    )
    parser.add_argument(
        "--output-key", default=None,
        help="Override default S3 key for the result JSON.",
    )
    args = parser.parse_args()

    payload = run_gate(
        bucket=args.bucket,
        n_days=args.window,
        horizon_days=args.horizon,
        threshold=args.threshold,
        output_s3_key=args.output_key,
        write_to_s3=not args.no_write,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
