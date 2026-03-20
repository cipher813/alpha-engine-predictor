"""
Centralized health status writer/reader for Alpha Engine modules.

Each module writes a health JSON to S3 after every run. Downstream modules
can check upstream health to detect stale or failed dependencies.
"""

import json
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)


def write_health(
    bucket: str,
    module_name: str,
    status: str,
    run_date: str,
    duration_seconds: float,
    summary: dict | None = None,
    warnings: list | None = None,
    error: str | None = None,
) -> None:
    """Write health status JSON to S3 at health/{module_name}.json."""
    payload = {
        "module": module_name,
        "status": status,  # "ok" | "degraded" | "failed"
        "last_success": datetime.now(timezone.utc).isoformat() if status != "failed" else None,
        "run_date": run_date,
        "duration_seconds": round(duration_seconds, 1),
        "summary": summary or {},
        "warnings": warnings or [],
        "error": error,
    }
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"health/{module_name}.json",
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Health status written: %s → %s", module_name, status)
    except Exception as e:
        logger.warning("Failed to write health status for %s: %s", module_name, e)


def read_health(bucket: str, module_name: str) -> dict | None:
    """Read health JSON for a module. Returns None if not found."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=f"health/{module_name}.json")
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def check_upstream_health(
    bucket: str,
    modules: list[str],
    max_age_hours: float = 48,
) -> dict:
    """Check health of multiple upstream modules.

    Returns {module: {"status": str, "age_hours": float, "stale": bool}}.
    """
    results = {}
    now = datetime.now(timezone.utc)
    for mod in modules:
        health = read_health(bucket, mod)
        if health is None:
            results[mod] = {"status": "unknown", "age_hours": -1, "stale": True}
            continue
        age_hours = -1.0
        if health.get("last_success"):
            try:
                last = datetime.fromisoformat(health["last_success"])
                age_hours = (now - last).total_seconds() / 3600
            except (ValueError, TypeError):
                pass
        results[mod] = {
            "status": health.get("status", "unknown"),
            "age_hours": round(age_hours, 1),
            "stale": age_hours < 0 or age_hours > max_age_hours,
        }
    return results
