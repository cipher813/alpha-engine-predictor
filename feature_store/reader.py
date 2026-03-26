"""
feature_store/reader.py — Read feature snapshots from S3.

Provides functions for reading single-date snapshots and date ranges.
Used by predictor (future training from store), executor, research,
and backtester as S3-contract consumers.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "features/"


def read_feature_snapshot(
    date_str: str,
    group: str,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client=None,
) -> Optional[pd.DataFrame]:
    """
    Read a single group's Parquet file for a given date.

    Returns None if the file does not exist.
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    key = f"{prefix}{date_str}/{group}.parquet"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")
        logger.debug("Read %s: %d rows, %d cols", key, len(df), len(df.columns))
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.debug("Feature snapshot not found: s3://%s/%s", bucket, key)
        return None
    except Exception as exc:
        logger.warning("Failed to read feature snapshot %s: %s", key, exc)
        return None


def read_feature_range(
    start_date: str,
    end_date: str,
    group: str,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client=None,
) -> pd.DataFrame:
    """
    Read and concatenate feature snapshots for a date range (inclusive).

    Skips dates where the snapshot does not exist (weekends, holidays).
    Returns an empty DataFrame if no snapshots are found.
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    frames = []

    current = start
    while current <= end:
        ds = current.isoformat()
        df = read_feature_snapshot(ds, group, bucket, prefix, s3_client)
        if df is not None:
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        logger.info("No feature snapshots found for %s in %s to %s", group, start_date, end_date)
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Read %s features: %s to %s, %d snapshots, %d total rows",
        group, start_date, end_date, len(frames), len(result),
    )
    return result


def latest_available_date(
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client=None,
) -> Optional[str]:
    """
    Find the most recent date that has feature snapshots in S3.

    Scans the features/ prefix for date-like subdirectories and returns
    the lexicographically largest one.
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, Delimiter="/"
        )
        prefixes = response.get("CommonPrefixes", [])
        dates = []
        for p in prefixes:
            # prefix looks like "features/2026-03-26/"
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)
        if dates:
            return sorted(dates)[-1]
    except Exception as exc:
        logger.warning("Failed to list feature dates: %s", exc)

    return None


def read_registry(
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client=None,
) -> Optional[dict]:
    """Read registry.json from S3."""
    import json

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    key = f"{prefix}registry.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as exc:
        logger.warning("Failed to read feature registry: %s", exc)
        return None
