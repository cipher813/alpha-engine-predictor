"""
inference/s3_io.py — Shared S3 write helpers with retry.

Used by multiple pipeline stages (load_prices for daily closes,
write_output for predictions). Extracted to avoid cross-stage imports.
"""

from __future__ import annotations

from retry import retry


@retry(max_attempts=3, retryable=(Exception,), label="s3_put_json")
def _s3_put_json(s3, bucket: str, key: str, body: str) -> None:
    """Write a JSON string to S3 with retry."""
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/json")


@retry(max_attempts=3, retryable=(Exception,), label="s3_put_bytes")
def _s3_put_bytes(s3, bucket: str, key: str, body: bytes) -> None:
    """Write bytes to S3 with retry."""
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/octet-stream")
