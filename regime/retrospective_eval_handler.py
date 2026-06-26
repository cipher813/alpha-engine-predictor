"""
regime/retrospective_eval_handler.py — Lambda entry point for the
weekly T1 retrospective HMM smoothing eval.

Closes the Lambda+SF wiring side of Stage C.2 T1 per
regime-v3-260514.md §5.3.3. The substrate module (regime/retrospective_eval.py)
already ships the pure-logic pieces — this handler glues them together
into a Lambda-callable entry point and wires the S3 I/O.

What this Lambda does
---------------------
1. Fetch the historical macro feature DataFrame (regime/features.py).
2. Load the macro agent's historical regime calls from
   ``signals/{date}/signals.json`` archive (one S3 GET per Saturday
   in the window; ~104 GETs for a 2y window — tolerable for a weekly
   batch job).
3. For each agent call timestamp at least ``lag_weeks`` (default 8)
   prior to the most recent feature observation:
     a. Locate that timestamp in the feature panel as an integer index.
     b. Run the HMM **smoother** over the full feature history.
     c. Take the smoother's argmax at the eval index as the
        retrospective label.
4. Pair agent calls with retrospective labels, compute the
   asymmetric-loss-weighted agreement score (bear-miss weighted 2×).
5. Write the canonical eval-artifact JSON to
   ``s3://{bucket}/regime/retrospective/{YYMMDDHHMM}.json`` +
   ``latest.json`` sidecar.

Look-ahead discipline
---------------------
The smoother is invoked here ONLY — the live substrate path
(``regime/handler.py``) uses the filter exclusively. Pinned by
regression test ``test_regime_hmm.py::test_filter_vs_smoother``.

Invocation modes
----------------
- ``action == "produce"`` (default) — compute + write to S3.
- ``action == "dry_run"`` — compute + return payload without writing.

Event payload (all optional):

    {
      "action": "produce" | "dry_run",
      "bucket": "alpha-engine-research",
      "price_cache_prefix": "predictor/price_cache/",
      "signals_prefix": "signals/",
      "retrospective_prefix": "regime/retrospective",
      "lookback_weeks": 520,
      "lag_weeks": 8,
      "rolling_window_weeks": 26,
      "agent_calls_window_weeks": 104,
      "bear_miss_weight": 2.0,
      "hmm_feature_columns": ["spy_20d_return", "vix_level", "hy_oas_bps"]
    }
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Sequence

import pandas as pd
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.logging import monitor_handler, setup_logging
from regime.features import (
    DEFAULT_PRICE_CACHE_PREFIX,
    fetch_macro_feature_history,
)
from regime.handler import DEFAULT_HMM_FEATURE_COLUMNS
from regime.retrospective_eval import (
    DEFAULT_BEAR_MISS_WEIGHT,
    DEFAULT_LAG_WEEKS,
    DEFAULT_ROLLING_WEEKS,
    assemble_t1_eval_payload,
    compute_retrospective_labels,
    pair_agent_calls_with_retrospective,
    score_agreement,
)
from regime.substrate import DEFAULT_S3_BUCKET


# Structured logging + flow-doctor. setup_logging attaches a FlowDoctorHandler
# at ERROR (off under pytest) so every log.error() in this regime Lambda routes
# through flow-doctor's capture -> dedupe -> alert dispatch. flow-doctor-regime.yaml
# ships at the Lambda task root (Dockerfile COPY); ${VAR} secrets come from the
# Lambda --environment block. Mirrors inference/handler.py.
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor-regime.yaml",
)
setup_logging(
    "predictor-regime",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)


DEFAULT_SIGNALS_PREFIX: str = "signals/"
DEFAULT_RETROSPECTIVE_PREFIX: str = "regime/retrospective"
DEFAULT_AGENT_CALLS_WINDOW_WEEKS: int = 104


def load_agent_calls_history(
    s3_client: Any,
    *,
    bucket: str = DEFAULT_S3_BUCKET,
    signals_prefix: str = DEFAULT_SIGNALS_PREFIX,
    window_weeks: int = DEFAULT_AGENT_CALLS_WINDOW_WEEKS,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Walk ``s3://{bucket}/{signals_prefix}`` and return a DataFrame
    of agent macro_regime calls over the trailing ``window_weeks``.

    Each row corresponds to one ``signals/{YYYY-MM-DD}/signals.json``
    artifact. The ``trading_day`` is the date in the key (research
    writes signals weekly on Saturdays for the prior Friday close).

    Tolerates per-file read errors: a malformed signals.json logs at
    INFO and is skipped — the rolling agreement score absorbs sparse
    history rather than crashing the eval. Returns an empty frame
    when no agent calls land in the window (cold start).
    """
    # tz-naive throughout — the parsed YYYY-MM-DD keys are tz-naive,
    # and pd.Timestamp.utcnow() is tz-aware. Normalize to avoid the
    # "Cannot compare tz-naive and tz-aware timestamps" error.
    if as_of is None:
        ref = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        ref_ts = pd.Timestamp(as_of)
        ref = (ref_ts.tz_localize(None) if ref_ts.tz is not None else ref_ts).normalize()
    cutoff = ref - pd.Timedelta(weeks=window_weeks)

    paginator = s3_client.get_paginator("list_objects_v2")
    dated_keys: list[tuple[pd.Timestamp, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=signals_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            # signals/{YYYY-MM-DD}/signals.json — skip everything else
            if not key.endswith("/signals.json"):
                continue
            parts = key[len(signals_prefix):].split("/")
            if len(parts) != 2:
                continue
            try:
                td = pd.Timestamp(parts[0])
            except (ValueError, TypeError):
                continue
            if td < cutoff:
                continue
            dated_keys.append((td, key))

    if not dated_keys:
        logger.info(
            "[T1] no agent calls under s3://%s/%s within %d-week window — "
            "returning empty history",
            bucket, signals_prefix, window_weeks,
        )
        return pd.DataFrame(columns=["trading_day", "market_regime"])

    rows: list[dict[str, Any]] = []
    for td, key in sorted(dated_keys, key=lambda x: x[0]):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except (ClientError, json.JSONDecodeError, KeyError) as e:
            logger.info(
                "[T1] skipping signals at s3://%s/%s (%s) — proceeding",
                bucket, key, type(e).__name__,
            )
            continue
        regime = data.get("market_regime")
        if not regime:
            continue
        rows.append({"trading_day": td, "market_regime": regime})

    return pd.DataFrame(rows)


def _build_eval_indices(
    feature_history: pd.DataFrame,
    hmm_feature_columns: Sequence[str],
    agent_call_timestamps: Sequence[pd.Timestamp],
    *,
    lag_weeks: int,
) -> tuple[dict[pd.Timestamp, int], list[int]]:
    """Map each agent call's timestamp to an integer position in the
    dropna'd feature panel, filtering out:
      * calls whose timestamp doesn't align to any feature row
      * calls within ``lag_weeks`` of the feature history's tail (the
        smoother needs lag_weeks of future observations to refine)

    Returns ``({timestamp: eval_index}, sorted_eval_indices)``. The
    second value feeds ``compute_retrospective_labels`` directly.
    """
    # Tolerate fully empty / missing-column panels (cold start). The
    # production path always has all columns by this point; this guard
    # lets callers (test fixtures, ad-hoc replays) pass minimal frames.
    if feature_history.empty or not all(
        c in feature_history.columns for c in hmm_feature_columns
    ):
        return {}, []

    hmm_input = feature_history[list(hmm_feature_columns)].dropna()
    if hmm_input.empty:
        return {}, []

    # Calendar-aware lag: caller works in trading-day terms ("8 weeks
    # ago") but the feature panel is indexed by row. The feature panel
    # is weekly, so row-count and week-count agree; pinning here lets
    # us swap to a non-weekly panel later without surprising callers.
    max_eval_position = len(hmm_input) - lag_weeks
    if max_eval_position <= 0:
        return {}, []

    # Build a date → row-index lookup once. Agent calls fall on Saturdays;
    # feature panel rows fall on the nearest preceding business day. The
    # match here uses an exact date match (the feature panel is resampled
    # weekly upstream so a Saturday call lands on the Friday row); if
    # exact-match returns empty we fall back to the nearest-prior row.
    date_to_position: dict[pd.Timestamp, int] = {}
    panel_index = hmm_input.index
    panel_dates_arr = panel_index.normalize()

    for ts in agent_call_timestamps:
        ts_norm = pd.Timestamp(ts).normalize()
        # Exact match
        matches = panel_index[panel_dates_arr == ts_norm]
        if len(matches) >= 1:
            row_position = panel_index.get_loc(matches[0])
        else:
            # Nearest prior row: find the last panel date <= ts
            mask = panel_dates_arr <= ts_norm
            if not mask.any():
                continue
            last_idx = panel_index[mask][-1]
            row_position = panel_index.get_loc(last_idx)

        # Treat int / np.int64 as ints; reject slice / boolean / out-of-range
        if not isinstance(row_position, (int,)) and not (
            hasattr(row_position, "__index__") and not isinstance(row_position, slice)
        ):
            continue
        position_int = int(row_position)
        if position_int >= max_eval_position:
            continue
        date_to_position[ts_norm] = position_int

    eval_indices = sorted(set(date_to_position.values()))
    return date_to_position, eval_indices


def produce_t1_eval(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    price_cache_prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    signals_prefix: str = DEFAULT_SIGNALS_PREFIX,
    retrospective_prefix: str = DEFAULT_RETROSPECTIVE_PREFIX,
    lookback_weeks: int = 520,
    lag_weeks: int = DEFAULT_LAG_WEEKS,
    rolling_window_weeks: int = DEFAULT_ROLLING_WEEKS,
    agent_calls_window_weeks: int = DEFAULT_AGENT_CALLS_WINDOW_WEEKS,
    bear_miss_weight: float = DEFAULT_BEAR_MISS_WEIGHT,
    hmm_feature_columns: tuple[str, ...] = DEFAULT_HMM_FEATURE_COLUMNS,
    write: bool = True,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """End-to-end T1 eval pipeline.

    Returns the assembled eval payload dict. When ``write=True``,
    also publishes the canonical eval-artifact (run_id.json + latest.json
    sidecar) and includes the S3 keys in the return value.

    Designed for both live Saturday SF invocation (default args) and
    ad-hoc replays via the CLI (caller overrides individual params).
    """
    feature_history = fetch_macro_feature_history(
        s3_client=s3_client,
        bucket=bucket,
        prefix=price_cache_prefix,
        lookback_weeks=lookback_weeks,
        as_of=as_of,
    )
    if feature_history.empty:
        raise RuntimeError(
            f"No macro feature history at s3://{bucket}/{price_cache_prefix} — "
            f"upstream FRED ingestion likely failed; T1 eval cannot run."
        )

    available_hmm_cols = tuple(
        c for c in hmm_feature_columns
        if c in feature_history.columns and not feature_history[c].dropna().empty
    )
    if "spy_20d_return" not in available_hmm_cols:
        raise RuntimeError(
            "spy_20d_return is required for HMM state pinning and was missing "
            "from the fetched history."
        )

    agent_calls = load_agent_calls_history(
        s3_client,
        bucket=bucket,
        signals_prefix=signals_prefix,
        window_weeks=agent_calls_window_weeks,
        as_of=as_of,
    )

    date_to_position, eval_indices = _build_eval_indices(
        feature_history,
        available_hmm_cols,
        list(agent_calls["trading_day"]) if not agent_calls.empty else [],
        lag_weeks=lag_weeks,
    )

    if not eval_indices:
        logger.info(
            "[T1] no eval indices land within lag window — emitting empty "
            "eval artifact (n_pairings=0)"
        )
        retrospective_labels: dict[pd.Timestamp, str] = {}
        pairings = []
    else:
        index_labels = compute_retrospective_labels(
            feature_history,
            hmm_feature_columns=available_hmm_cols,
            eval_indices=eval_indices,
        )
        position_to_date = {v: k for k, v in date_to_position.items()}
        retrospective_labels = {
            position_to_date[pos]: label
            for pos, label in index_labels.items()
            if pos in position_to_date
        }
        pairings = pair_agent_calls_with_retrospective(
            agent_calls=agent_calls,
            retrospective_labels=retrospective_labels,
        )

    score = score_agreement(
        pairings,
        bear_miss_weight=bear_miss_weight,
        rolling_window_weeks=rolling_window_weeks,
    )

    from nousergon_lib.dates import now_dual
    from nousergon_lib.eval_artifacts import new_eval_run_id

    dual = now_dual()
    run_id = new_eval_run_id()
    payload = assemble_t1_eval_payload(
        pairings=pairings,
        score=score,
        run_id=run_id,
        calendar_date=str(dual.calendar_date),
        trading_day=str(dual.trading_day),
        feature_window_start=str(feature_history.index.min().date()) if not feature_history.empty else None,
        feature_window_end=str(feature_history.index.max().date()) if not feature_history.empty else None,
        lag_weeks=lag_weeks,
        hmm_feature_columns=list(available_hmm_cols),
    )

    if not write:
        return {"payload": payload, "wrote": False}

    keys = _write_t1_eval_artifact(
        payload,
        s3_client=s3_client,
        bucket=bucket,
        prefix=retrospective_prefix,
    )
    return {"payload": payload, "wrote": True, **keys}


def _write_t1_eval_artifact(
    payload: dict[str, Any],
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> dict[str, str]:
    """Publish a T1 eval payload to S3 in canonical eval_artifacts shape:
    forensic ``{prefix}/{run_id}.json`` always; ``{prefix}/latest.json``
    sidecar with summary headlines.
    """
    from nousergon_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
    )
    from datetime import datetime, timezone

    run_id = payload["run_id"]
    artifact_key = eval_artifact_key(prefix, run_id)
    body = json.dumps(payload, default=str).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=artifact_key,
        Body=body,
        ContentType="application/json",
    )

    latest_key = eval_latest_key(prefix)
    # Use .get() with None defaults — score_agreement returns a
    # slimmed-down dict when pairings is empty (cold start), missing
    # rolling_window_size etc. Sidecar tolerates that.
    score_block = payload["score"]
    sidecar = {
        "run_id": run_id,
        "artifact_key": artifact_key,
        "calendar_date": payload["calendar_date"],
        "trading_day": payload["trading_day"],
        "schema_version": payload["schema_version"],
        "eval_tier": payload["eval_tier"],
        "n_pairings": score_block.get("n_pairings", 0),
        "symmetric_agreement_rate": score_block.get("symmetric_agreement_rate"),
        "asymmetric_weighted_agreement_rate": score_block.get("asymmetric_weighted_agreement_rate"),
        "rolling_window_score": score_block.get("rolling_window_score"),
        "rolling_window_size": score_block.get("rolling_window_size"),
        "lag_weeks": payload["lag_weeks"],
        "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=latest_key,
        Body=json.dumps(sidecar).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(
        "[T1] wrote run_id=%s → s3://%s/%s (latest=%s) | "
        "n_pairings=%d | rolling_score=%s",
        run_id, bucket, artifact_key, latest_key,
        sidecar["n_pairings"], sidecar["rolling_window_score"],
    )
    return {"artifact_key": artifact_key, "latest_key": latest_key}


@monitor_handler
def lambda_handler(event: dict | None, context: Any) -> dict[str, Any]:
    """AWS Lambda entry point.

    Routes ``event["action"]`` to either ``produce`` (default) or
    ``dry_run``. Event payload schema documented in the module
    docstring above.
    """
    event = event or {}
    action = event.get("action", "produce")

    import boto3
    s3_client = boto3.client("s3")

    kwargs = {
        "s3_client": s3_client,
        "bucket": event.get("bucket", DEFAULT_S3_BUCKET),
        "price_cache_prefix": event.get("price_cache_prefix", DEFAULT_PRICE_CACHE_PREFIX),
        "signals_prefix": event.get("signals_prefix", DEFAULT_SIGNALS_PREFIX),
        "retrospective_prefix": event.get("retrospective_prefix", DEFAULT_RETROSPECTIVE_PREFIX),
        "lookback_weeks": int(event.get("lookback_weeks", 520)),
        "lag_weeks": int(event.get("lag_weeks", DEFAULT_LAG_WEEKS)),
        "rolling_window_weeks": int(event.get("rolling_window_weeks", DEFAULT_ROLLING_WEEKS)),
        "agent_calls_window_weeks": int(event.get("agent_calls_window_weeks", DEFAULT_AGENT_CALLS_WINDOW_WEEKS)),
        "bear_miss_weight": float(event.get("bear_miss_weight", DEFAULT_BEAR_MISS_WEIGHT)),
        "hmm_feature_columns": tuple(event.get("hmm_feature_columns", DEFAULT_HMM_FEATURE_COLUMNS)),
    }

    if action == "produce":
        result = produce_t1_eval(**kwargs, write=True)
        return {
            "statusCode": 200,
            "action": action,
            "run_id": result["payload"]["run_id"],
            "artifact_key": result["artifact_key"],
            "latest_key": result["latest_key"],
            "n_pairings": result["payload"]["score"]["n_pairings"],
            "asymmetric_weighted_agreement_rate": result["payload"]["score"]["asymmetric_weighted_agreement_rate"],
            "rolling_window_score": result["payload"]["score"]["rolling_window_score"],
        }
    elif action == "dry_run":
        result = produce_t1_eval(**kwargs, write=False)
        return {
            "statusCode": 200,
            "action": action,
            "payload": result["payload"],
        }
    else:
        return {
            "statusCode": 400,
            "error": f"unknown action {action!r}; expected 'produce' or 'dry_run'",
        }
