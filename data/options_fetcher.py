"""
data/options_fetcher.py — Options data reader for predictor features (O12).

Provides options-derived signals (put/call ratio, IV rank) for GBM features.
Both inference and training read the same S3 archive snapshot written by the
upstream alternative-data collector (nousergon-data ``collectors/alternative.py``)
at ``archive/options/{date}.json``.

yfinance retired (yfinance-centralization PR4b, config#874): the live
``yfinance.Ticker().option_chain()`` inference fetch was a hard cutover to the
archive read once the producer write-both soak completed (write-both merged
nousergon-data#252, 2026-05-17). No yfinance fallback remains — the predictor
is now yfinance-free at runtime. When the archive snapshot is missing for a
day, callers neutral-fill (see ``_neutral_features``) exactly as the legacy
path did on a yfinance miss.

Graceful degradation: returns ``None`` when the archive key is absent so the
caller can decide (the inference stage neutral-fills; training skips).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def load_historical_options(
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> Optional[dict[str, dict]]:
    """Load options data from the S3 archive snapshot for ``date_str``.

    Reads the single flat file the upstream collector writes at
    ``archive/options/{date}.json`` — a ``{ticker: {put_call_ratio, iv_rank,
    atm_iv}}`` mapping where ``put_call_ratio`` is a raw OI ratio and
    ``iv_rank`` is on a 0-100 scale (verified against the producer,
    nousergon-data ``collectors/alternative.py::_build_predictor_options_mirror``).

    Used by BOTH the inference and training feature paths — the producer's
    ``run_date`` axis is the trading day, so reading ``ctx.date_str`` resolves
    the current trading day's snapshot at inference time.

    Returns the per-ticker feature dict in predictor units (``put_call_ratio``
    log-transformed, ``iv_rank`` normalized to [0, 1], ``atm_iv`` raw), or
    ``None`` if the key is absent / unreadable so the caller can neutral-fill.
    """
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/options/{date_str}.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = json.loads(obj["Body"].read())
        # Convert archive-format to predictor features.
        result = {}
        for ticker, data in raw.items():
            pc = data.get("put_call_ratio", 1.0)
            result[ticker] = {
                "put_call_ratio": float(np.log(max(pc, 0.01))),
                "iv_rank": data.get("iv_rank", 50.0) / 100.0,
                "atm_iv": data.get("atm_iv", 0.0),
            }
        return result
    except Exception:
        return None


def _neutral_features() -> dict:
    """Neutral values when options data unavailable."""
    return {
        "put_call_ratio": 0.0,   # log(1.0) = 0
        "iv_rank": 0.5,          # 50th percentile
        "atm_iv": 0.0,
    }
