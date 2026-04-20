"""Coverage-delta check: compare signals.json buy_candidates against
predictions/{date}.json and return the missing tickers.

Used by the weekday Step Function's coverage-gap Choice state to decide
whether to re-invoke the predictor (`action=predict` with `tickers` payload)
before allowing the executor to run.

Returns a JSON-serializable dict so it can be consumed directly from the
Step Function state without extra parsing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


def _extract_tickers(entries: list, field: str = "ticker") -> set[str]:
    """Pull uppercase ticker names from a list of dicts."""
    out: set[str] = set()
    for e in entries or []:
        if isinstance(e, dict):
            t = e.get(field)
            if t:
                out.add(str(t).upper())
    return out


def _read_s3_json(bucket: str, key: str) -> Optional[dict]:
    """Best-effort S3 JSON read. Returns None on NoSuchKey/404/AccessDenied.

    Any other boto/network error raises so the Step Function task fails and
    `HandleFailure` fires — we don't want to silently treat a transient S3
    outage as "no predictions exist".
    """
    import boto3
    from botocore.exceptions import ClientError
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "AccessDenied", "404", "403"):
            log.info("s3://%s/%s not present (%s)", bucket, key, code)
            return None
        raise


def compute_coverage_delta(
    bucket: str,
    date_str: Optional[str] = None,
) -> dict:
    """
    Compare signals.json `buy_candidates` ⊆ predictions.json predictions.

    Returns
    -------
    {
        "date":              YYYY-MM-DD,
        "missing_tickers":   sorted list of buy_candidate tickers missing a prediction,
        "missing_count":     len(missing_tickers),
        "n_buy_candidates":  total buy_candidates in signals.json,
        "n_predictions":     total predictions in predictions.json,
        "has_gap":           missing_count > 0 (convenience for SF Choice state),
        "signals_present":   True iff signals.json could be read,
        "predictions_present": True iff predictions.json could be read,
    }

    Semantics
    ---------
    - If signals.json is absent: treat as empty buy_candidates (no gap). The
      upstream pipeline will have failed elsewhere before we reach the
      coverage check.
    - If predictions.json is absent: buy_candidates are all "missing". The
      Choice state should then re-invoke with `tickers=buy_candidates`.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    signals = _read_s3_json(bucket, f"signals/{date_str}/signals.json")
    predictions_doc = _read_s3_json(bucket, f"predictor/predictions/{date_str}.json")

    buy_tickers = _extract_tickers((signals or {}).get("buy_candidates") or [])
    pred_tickers = _extract_tickers((predictions_doc or {}).get("predictions") or [])

    missing = sorted(buy_tickers - pred_tickers)

    return {
        "date": date_str,
        "missing_tickers": missing,
        "missing_count": len(missing),
        "n_buy_candidates": len(buy_tickers),
        "n_predictions": len(pred_tickers),
        "has_gap": len(missing) > 0,
        "signals_present": signals is not None,
        "predictions_present": predictions_doc is not None,
    }
