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
    stage: str | None = None,
) -> str:
    """Snapshot the live inference contract under ``source_prefix`` into an
    immutable content-addressed bundle ``{registry_prefix}{version_id}/``.

    Refuses (``RegistryError``) if the required contract files are missing — a
    bundle that can't be run in isolation must never be registered. Idempotent:
    if the same-fingerprint bundle already has a ``_lineage.json``, returns its
    ``version_id`` without recopying. Returns the ``version_id``.

    ``stage`` (``champion`` | ``challenger`` | ``archived``) is recorded in the
    lineage so the champion/challenger arc can tell, from the registry alone,
    which versions were promoted to live (champion) vs captured as observe-only
    candidates (challenger) — the Phase-0 capture gap fix (L4469): every trained
    candidate is registered, not only the promoted one, so a challenger exists
    to shadow.
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

    # Idempotency: a completed bundle is marked by its _lineage.json. The
    # bundle's BYTES are immutable, but the ``stage`` can legitimately advance
    # (a challenger captured observe-only later gets promoted — same content,
    # same fingerprint, same version_id). On that one transition patch the
    # recorded stage in place; otherwise leave the bundle untouched.
    _existing = None
    try:
        _existing_raw = s3.get_object(
            Bucket=bucket, Key=f"{dest}_lineage.json"
        )["Body"].read()
        _existing = json.loads(_existing_raw)
    except Exception:
        _existing = None  # missing / unreadable → write a fresh bundle below
    if _existing is not None:
        if (
            stage is not None
            and stage != _existing.get("stage")
            and stage == "champion"  # only promote-in-place; never demote
        ):
            _existing["stage"] = stage
            s3.put_object(
                Bucket=bucket,
                Key=f"{dest}_lineage.json",
                Body=json.dumps(_existing, indent=2).encode(),
                ContentType="application/json",
            )
        return version_id  # already snapshotted — immutable bytes

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
        "stage": stage,
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


def list_versions(
    s3,
    bucket: str,
    *,
    stage: str | None = None,
    registry_prefix: str = DEFAULT_REGISTRY_PREFIX,
) -> list[dict]:
    """List registered versions by reading each bundle's ``_lineage.json``.

    Returns lineage dicts (each carries ``version_id``, ``stage``, ``date``,
    ``created_utc``, ...), filtered to ``stage`` when given, sorted newest-first
    by ``(date, created_utc)``. Used by the Phase-1 shadow runner to enumerate
    challengers to shadow. Best-effort: a bundle whose ``_lineage.json`` is
    missing or unreadable is skipped, not fatal.
    """
    out: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=registry_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/_lineage.json"):
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                lineage = json.loads(body)
            except Exception:
                continue
            if stage is not None and lineage.get("stage") != stage:
                continue
            out.append(lineage)
    out.sort(
        key=lambda d: (str(d.get("date") or ""), str(d.get("created_utc") or "")),
        reverse=True,
    )
    return out


def _patch_stage(s3, bucket: str, dest: str, stage: str) -> None:
    """Read ``{dest}_lineage.json``, set its ``stage``, re-put. Best-effort on a
    missing lineage (logs nothing here — caller decides)."""
    raw = s3.get_object(Bucket=bucket, Key=f"{dest}_lineage.json")["Body"].read()
    lineage = json.loads(raw)
    lineage["stage"] = stage
    s3.put_object(
        Bucket=bucket, Key=f"{dest}_lineage.json",
        Body=json.dumps(lineage, indent=2).encode(), ContentType="application/json",
    )


def promote_to_champion(
    s3,
    bucket: str,
    version_id: str,
    *,
    registry_prefix: str = DEFAULT_REGISTRY_PREFIX,
    live_prefix: str = DEFAULT_SOURCE_PREFIX,
) -> dict:
    """Promote a registered version to the live champion (L4469, operator step).

    Copies the version's immutable bundle into the live weights prefix (so
    inference loads it), demotes any prior champion to ``archived``, and marks
    this version ``champion`` in its lineage. This is the DELIBERATE, operator-
    gated promotion that replaces training auto-promote (challenger-first): a
    challenger earns champion only after the leaderboard shows realized OOS edge.

    Refuses (``RegistryError``) if the bundle is missing or incomplete — never
    point live inference at an unreproducible contract. Returns a summary dict.
    NOTE: also update MODEL_REGISTRY.yaml (the doc SoT) to match — printed by
    the CLI for the operator to commit.
    """
    src = f"{registry_prefix}{version_id}/"
    try:
        json.loads(s3.get_object(Bucket=bucket, Key=f"{src}_lineage.json")["Body"].read())
    except Exception as e:
        raise RegistryError(
            f"cannot promote {version_id!r}: no registry bundle / _lineage.json ({e})"
        )

    contract = [
        k for k in _list_contract_keys(s3, bucket, src)
        if k.rsplit("/", 1)[-1] != "_lineage.json"
    ]
    names = {k.rsplit("/", 1)[-1] for k in contract}
    missing = [f for f in REQUIRED_CONTRACT_FILES if f not in names]
    if missing:
        raise RegistryError(
            f"refusing to promote {version_id!r} — incomplete bundle, missing "
            f"{missing}; live inference must never run an unreproducible contract."
        )

    for k in contract:
        fname = k.rsplit("/", 1)[-1]
        s3.copy_object(
            Bucket=bucket, Key=f"{live_prefix}{fname}",
            CopySource={"Bucket": bucket, "Key": k},
        )

    # Demote any prior champion(s), then mark this one champion (exactly one).
    for other in list_versions(s3, bucket, stage="champion", registry_prefix=registry_prefix):
        ovid = other.get("version_id")
        if ovid and ovid != version_id:
            _patch_stage(s3, bucket, f"{registry_prefix}{ovid}/", "archived")
    _patch_stage(s3, bucket, src, "champion")

    return {
        "version_id": version_id,
        "files_copied": sorted(names),
        "live_prefix": live_prefix,
    }


def _cli() -> None:
    """Model registry CLI — snapshot / promote / list (L4469).

    Snapshot (backfill):
        python -m model.registry --bucket B --model-version V --date YYYY-MM-DD [--source-prefix P]
    Promote a challenger to the live champion (operator step):
        python -m model.registry --bucket B --promote VERSION_ID
    List registered versions:
        python -m model.registry --bucket B --list [--stage challenger]
    """
    import argparse

    import boto3

    p = argparse.ArgumentParser(description="Model registry: snapshot / promote / list.")
    p.add_argument("--bucket", required=True)
    p.add_argument("--model-version", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    p.add_argument("--code-sha", default=None)
    p.add_argument("--promote", default=None, metavar="VERSION_ID",
                   help="Promote this registered version to the live champion.")
    p.add_argument("--list", action="store_true", help="List registered versions.")
    p.add_argument("--stage", default=None, help="Filter --list by stage.")
    args = p.parse_args()
    s3 = boto3.client("s3")

    if args.list:
        for v in list_versions(s3, args.bucket, stage=args.stage):
            print(f"{(v.get('stage') or '?'):10s} {v.get('version_id')}  ({v.get('date')})")
        return

    if args.promote:
        result = promote_to_champion(s3, args.bucket, args.promote)
        print(f"PROMOTED {result['version_id']} → live champion "
              f"({len(result['files_copied'])} files copied to {result['live_prefix']}).")
        print("NEXT: update MODEL_REGISTRY.yaml — set this version stage: champion, "
              "prior champion → archived (the doc SoT).")
        return

    if not (args.model_version and args.date):
        p.error("snapshot requires --model-version and --date (or use --promote / --list)")
    vid = snapshot_to_registry(
        s3, args.bucket,
        model_version=args.model_version, date=args.date,
        source_prefix=args.source_prefix, code_sha=args.code_sha,
    )
    print(f"registry snapshot: predictor/registry/{vid}/")


if __name__ == "__main__":
    _cli()
