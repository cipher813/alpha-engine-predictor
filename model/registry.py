"""Phase 0 — self-contained, immutable, content-addressed model registry.

The champion/challenger governance arc (plan:
``alpha-engine-docs/private/model-shadow-comparison-260601.md``) requires that
every model version be reproducible *in isolation*. The legacy
``predictor/weights/meta/archive/{date}/`` store is a PARTIAL subset — it omits
``feature_list.json`` (the feature contract) and ``manifest.json`` (lineage), so
pointing inference at an old archive would run that meta-model against the
*current* feature ordering → silent feature misalignment → invalid comparison
(found 2026-06-01).

This module snapshots the COMPLETE live inference contract (all weights +
feature_list + manifest) into an immutable, content-addressed bundle under
``predictor/registry/{version_id}/``. It is purely additive — a server-side S3
copy of objects already uploaded by training; it never touches the live weights,
the promotion gate, or the archive. It doubles as the backfill tool (snapshot
the current live model now, before the next retrain overwrites it).

``version_id = {model_version}-{date}-{fingerprint8}`` where the fingerprint is a
stable hash over the contract files' ETags — so an identical contract collapses
to the same id (idempotent), and any weight/feature change yields a new id.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

DEFAULT_SOURCE_PREFIX = "predictor/weights/meta/"
DEFAULT_REGISTRY_PREFIX = "predictor/registry/"

# Files that MUST be present for a snapshot to be a valid, reproducible bundle.
# feature_list = the feature contract; manifest = lineage/metrics. Without them
# the version is not independently runnable, so we refuse to register it.
REQUIRED_CONTRACT_FILES = ("feature_list.json", "manifest.json")


class RegistryError(RuntimeError):
    """Raised when a snapshot cannot be produced as a complete, valid bundle."""


def _list_contract_keys(s3, bucket: str, source_prefix: str) -> list[str]:
    """Return the FLAT live-contract object keys directly under ``source_prefix``.

    Excludes the ``archive/`` and ``registry/`` subtrees (anything with a ``/``
    in the suffix after ``source_prefix``) so only the live inference contract —
    the weight files + feature_list + manifest — is captured.
    """
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            suffix = key[len(source_prefix):]
            if not suffix or "/" in suffix:
                continue  # subdir (archive/, registry/) or the prefix marker
            keys.append(key)
    return sorted(keys)


def _etag(s3, bucket: str, key: str) -> str:
    return s3.head_object(Bucket=bucket, Key=key)["ETag"].strip('"')


def _fingerprint(name_etags: list[tuple[str, str]]) -> str:
    """8-hex content fingerprint over sorted (filename, etag) pairs."""
    h = hashlib.sha256()
    for name, etag in sorted(name_etags):
        h.update(name.encode())
        h.update(b"\x00")
        h.update(etag.encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]


def snapshot_to_registry(
    s3,
    bucket: str,
    *,
    model_version: str,
    date: str,
    source_prefix: str = DEFAULT_SOURCE_PREFIX,
    registry_prefix: str = DEFAULT_REGISTRY_PREFIX,
    code_sha: str | None = None,
    created_utc: str | None = None,
) -> str:
    """Snapshot the live inference contract under ``source_prefix`` into an
    immutable content-addressed bundle ``{registry_prefix}{version_id}/``.

    Refuses (``RegistryError``) if the required contract files are missing — a
    bundle that can't be run in isolation must never be registered. Idempotent:
    if the same-fingerprint bundle already has a ``_lineage.json``, returns its
    ``version_id`` without recopying. Returns the ``version_id``.
    """
    keys = _list_contract_keys(s3, bucket, source_prefix)
    names = {k.rsplit("/", 1)[-1] for k in keys}
    missing = [f for f in REQUIRED_CONTRACT_FILES if f not in names]
    if missing:
        raise RegistryError(
            f"refusing to snapshot {source_prefix!r} — missing required contract "
            f"files {missing}; the version would not be reproducible in isolation."
        )

    name_etags = [(k.rsplit("/", 1)[-1], _etag(s3, bucket, k)) for k in keys]
    fp = _fingerprint(name_etags)
    version_id = f"{model_version}-{date}-{fp}"
    dest = f"{registry_prefix}{version_id}/"

    # Idempotency: a completed bundle is marked by its _lineage.json.
    try:
        s3.head_object(Bucket=bucket, Key=f"{dest}_lineage.json")
        return version_id  # already snapshotted — immutable, leave as-is
    except Exception:
        pass  # not present → write it

    for key in keys:
        filename = key.rsplit("/", 1)[-1]
        s3.copy_object(
            Bucket=bucket,
            Key=f"{dest}{filename}",
            CopySource={"Bucket": bucket, "Key": key},
        )

    lineage = {
        "version_id": version_id,
        "model_version": model_version,
        "date": date,
        "code_sha": code_sha,
        "source_prefix": source_prefix,
        "fingerprint": fp,
        "files": sorted(names),
        "file_etags": {n: e for n, e in name_etags},
        "created_utc": created_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"{dest}_lineage.json",
        Body=json.dumps(lineage, indent=2).encode(),
        ContentType="application/json",
    )
    return version_id


def _cli() -> None:
    """Backfill CLI: snapshot the current live model (or any prefix) into the
    registry. Usage: ``python -m model.registry --bucket B --model-version V
    --date YYYY-MM-DD [--source-prefix P]``.
    """
    import argparse

    import boto3

    p = argparse.ArgumentParser(description="Snapshot a model contract into the registry.")
    p.add_argument("--bucket", required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    p.add_argument("--code-sha", default=None)
    args = p.parse_args()

    vid = snapshot_to_registry(
        boto3.client("s3"),
        args.bucket,
        model_version=args.model_version,
        date=args.date,
        source_prefix=args.source_prefix,
        code_sha=args.code_sha,
    )
    print(f"registry snapshot: predictor/registry/{vid}/")


if __name__ == "__main__":
    _cli()
