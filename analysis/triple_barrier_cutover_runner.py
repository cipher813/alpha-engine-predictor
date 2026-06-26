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
  5. Write result JSON via the canonical lib helpers
     (``alpha_engine_lib.eval_artifacts``):
     ``s3://{bucket}/predictor/variant_gates/triple_barrier/{run_id}.json``
     plus a ``latest.json`` operator-fetch sidecar at the prefix root.

S3 layout (canonical eval-style flat layout codified in
``alpha_engine_lib.eval_artifacts`` v0.8.0)::

    predictor/variant_gates/triple_barrier/
      ├── 2605160200.json                # YYMMDDHHMM run_id
      ├── 2605161430.json                # second run same day → distinct key
      ├── 2605230200.json
      └── latest.json                    # most-recent run — operator UX

Same-day re-runs land at distinct ``run_id`` keys instead of overwriting,
preserving forensic capture. The ``latest.json`` sidecar provides a
single-fetch endpoint for dashboards / evaluator email rendering. Per
the Alpha Engine ``DATE_CONVENTIONS.md`` dual-tracking convention, every
payload carries both ``calendar_date`` (wall-clock UTC) and
``trading_day`` (last closed NYSE session) so downstream consumers can
query along either axis.

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
from alpha_engine_lib.dates import now_dual  # noqa: E402
from alpha_engine_lib.eval_artifacts import (  # noqa: E402
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
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
# and the eventual evaluator-email surfacer read from here. The
# ``triple_barrier/`` sub-prefix isolates this gate from any future
# variant gates (Stage 4 continuous regime feature, Stage 5 meta-
# labeling) that will land their own sub-prefixes here.
DEFAULT_OUTPUT_PREFIX = "predictor/variant_gates/triple_barrier"


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
    # Wave-3 PR4 cutover (DONE): price_cache parquets live at
    # ``reference/price_cache/`` only — the legacy ``predictor/price_cache/``
    # tree is removed live via ``aws s3 rm``. The legacy fallback is dropped.
    # Mirrors regime/features._read_parquet_close's chain without importing
    # from a deployed-Lambda module — analysis tools stay decoupled.
    price_cache_prefixes: tuple[str, ...] = (
        "reference/price_cache/",
    )
    for ticker in tickers:
        df = None
        for prefix in price_cache_prefixes:
            key = f"{prefix}{ticker}.parquet"
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                df = pd.read_parquet(BytesIO(obj["Body"].read()))
                break
            except Exception as e:
                last_exc = e
        if df is None:
            log.debug("Skipping price load for %s: %s", ticker, last_exc)
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
    # sector_map.json now lives at 3 keys post alpha-engine-data #272
    # (write-both gap fix that landed alongside Wave-3 PR3-wave-1): the
    # constituents collector writes to data/, predictor/, AND the new
    # reference/ home. Read order = ``reference/`` first (post-PR4 sole
    # survivor), then the two legacy keys we've historically tolerated.
    for key in (
        "reference/price_cache/sector_map.json",
        "data/sector_map.json",
        "predictor/price_cache/sector_map.json",
    ):
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
    write_latest: bool = True,
    s3_client=None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
    prediction_pairs: list[dict] | None = None,
    run_id: str | None = None,
) -> dict:
    """End-to-end Stage 3 cutover-gate evaluation.

    Default S3 layout — canonical eval-style flat layout codified in
    ``alpha_engine_lib.eval_artifacts`` v0.8.0::

        predictor/variant_gates/triple_barrier/
          {run_id}.json   ← per-invocation artifact (YYMMDDHHMM encodes date)
          latest.json     ← single-fetch operator UX (mirror)

    Same-minute re-runs collide by design (production cron cadence makes
    this impossible; tests inject explicit ``run_id``). Both
    ``calendar_date`` (UTC wall-clock) and ``trading_day`` (last closed
    NYSE session) are embedded in every payload per
    ``DATE_CONVENTIONS.md``.

    Args:
        bucket: S3 bucket. Defaults to ``cfg.S3_BUCKET``.
        n_days: trailing prediction-history window. Default 42.
        horizon_days: forward-realized window. Default 21 (matches
            ``cfg.FORWARD_DAYS``).
        threshold: relative-lift threshold for pass. Default 1.15 (15%).
        output_s3_key: S3 key for the result JSON. When None, formats
            via ``eval_artifact_key(DEFAULT_OUTPUT_PREFIX, run_id)``.
        write_to_s3: when False, skip the S3 write (local / test mode).
        write_latest: when True (default), also write a copy at
            ``eval_latest_key(DEFAULT_OUTPUT_PREFIX)`` for single-fetch
            operator UX.
        s3_client: optional pre-configured boto3 S3 client (testing).
        prices_by_ticker / sector_map / prediction_pairs: optional
            pre-loaded inputs for testing — when provided, the runner
            skips the corresponding S3 load.
        run_id: optional pre-minted run identifier (testing). When None,
            a fresh ``YYMMDDHHMM`` id is minted via
            ``alpha_engine_lib.eval_artifacts.new_eval_run_id``.

    Returns:
        dict-form gate result (asdict of VariantCutoverGateResult) plus
        runner metadata: ``calendar_date``, ``trading_day``, ``run_id``,
        ``window_days``, ``horizon_days``, ``n_pairs_loaded``,
        ``n_realized_filled``, ``s3_key`` (when ``write_to_s3=True``).
    """
    bucket = bucket or cfg.S3_BUCKET
    dual = now_dual()
    if run_id is None:
        run_id = new_eval_run_id()

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
        # DATE_CONVENTIONS.md dual-tracking — facts about the artifact
        # for downstream consumers that key off either UTC wall-clock
        # or NYSE-session axes. The S3 path uses the YYMMDDHHMM run_id
        # (which encodes the date itself) — these payload fields are
        # for join queries, not addressing.
        "calendar_date": dual.calendar_date,
        "trading_day": dual.trading_day,
        # First-class run identifier — same-minute collisions are by
        # design (production cron cadence makes them effectively
        # impossible; tests inject explicit run_ids). Per
        # alpha_engine_lib.eval_artifacts.new_eval_run_id contract.
        "run_id": run_id,
        "window_days": n_days,
        "horizon_days": horizon_days,
        "n_pairs_loaded": n_pairs_loaded,
        "n_realized_filled": n_realized_filled,
    }

    if write_to_s3:
        import boto3

        s3 = s3_client or boto3.client("s3")
        key = output_s3_key or eval_artifact_key(DEFAULT_OUTPUT_PREFIX, run_id)
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        log.info("Gate result written to s3://%s/%s", bucket, key)
        payload["s3_key"] = key
        # Operator-UX sidecar — a stable single-fetch key. Pure copy of
        # the dated artifact; the dated key remains the source of truth
        # so re-runs preserve forensic capture under their own run_id.
        if write_latest:
            latest_key = eval_latest_key(DEFAULT_OUTPUT_PREFIX)
            s3.put_object(
                Bucket=bucket,
                Key=latest_key,
                Body=body,
                ContentType="application/json",
            )
            log.info("Gate result also mirrored to s3://%s/%s", bucket, latest_key)

    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bucket", default=None,
        help=f"S3 bucket. Default: cfg.S3_BUCKET ({cfg.S3_BUCKET}).",
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
