"""
feature_store/writer.py — Write feature snapshots to S3 as dated Parquet files.

Each snapshot is split by feature group (technical, macro, interaction,
alternative, fundamental) so consumers can read only what they need.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd

from feature_store.registry import GROUPS

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "features/"


def write_feature_snapshot(
    date_str: str,
    features_df: pd.DataFrame,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client=None,
) -> dict[str, int]:
    """
    Write a feature DataFrame to S3, split by group.

    Parameters
    ----------
    date_str : YYYY-MM-DD date for this snapshot.
    features_df : DataFrame with a 'ticker' column plus feature columns.
                  One row per ticker. Missing feature columns are skipped.
    bucket : S3 bucket name.
    prefix : S3 key prefix (default "features/").
    s3_client : Optional boto3 S3 client (for testing / reuse).

    Returns
    -------
    dict mapping group name → number of rows written.
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    written = {}

    for group, feature_names in GROUPS.items():
        # Find which features from this group exist in the DataFrame
        available = [f for f in feature_names if f in features_df.columns]
        if not available:
            logger.debug("Skipping group %s — no columns present in DataFrame", group)
            continue

        # Build the group DataFrame
        if group == "macro":
            # Macro features are identical across tickers — write one row per date
            # Take the first row's values (they're all the same)
            row = features_df[available].iloc[0:1].copy()
            row.insert(0, "date", date_str)
            group_df = row
        else:
            # Per-ticker features
            id_cols = []
            if "ticker" in features_df.columns:
                id_cols.append("ticker")
            group_df = features_df[id_cols + available].copy()
            group_df.insert(len(id_cols), "date", date_str)

        # Write to S3 as Parquet
        buf = io.BytesIO()
        group_df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)

        key = f"{prefix}{date_str}/{group}.parquet"
        s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

        written[group] = len(group_df)
        logger.debug("Wrote %s: %d rows, %d features", key, len(group_df), len(available))

    total_groups = len(written)
    total_rows = sum(written.values())
    logger.info(
        "Feature snapshot written for %s: %d groups, %d total rows",
        date_str, total_groups, total_rows,
    )
    return written
