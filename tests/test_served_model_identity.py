"""L4540 part 2: the live manifest is the LATEST-RUN record (refreshes every
Saturday for the freshness SLA), so `served_date`/`served_version` carry the
identity of the model ACTUALLY served — equal to the run on a promoting run,
carried forward on a non-promoting run, restamped on a manual promote. Inference
reads `served_date` for "last trained" so a non-promoting run never reports its
own date over stale served weights.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _read_live_served_identity
from model.registry import promote_to_champion

_MANIFEST_KEY = "predictor/weights/meta/manifest.json"


def _s3_with_manifest(manifest: dict | None):
    s3 = MagicMock()
    if manifest is None:
        s3.get_object.side_effect = Exception("NoSuchKey")
    else:
        body = MagicMock()
        body.read.return_value = json.dumps(manifest).encode()
        s3.get_object.return_value = {"Body": body}
    return s3


# ── Part 2a: _read_live_served_identity carry-forward ────────────────────────

def test_served_identity_prefers_explicit_fields():
    s3 = _s3_with_manifest(
        {"date": "2026-06-13", "promoted": False,
         "served_version": "v3.0-meta", "served_date": "2026-06-06"}
    )
    assert _read_live_served_identity(s3, "bkt", _MANIFEST_KEY) == ("v3.0-meta", "2026-06-06")


def test_served_identity_falls_back_to_prior_date_only_if_promoted():
    # Old-format manifest (no served_* fields) that DID promote → its date is served.
    s3 = _s3_with_manifest({"date": "2026-05-30", "promoted": True, "version": "v3.0-meta"})
    assert _read_live_served_identity(s3, "bkt", _MANIFEST_KEY) == ("v3.0-meta", "2026-05-30")


def test_served_identity_none_when_prior_was_nonpromoting_old_format():
    # Old-format non-promoting manifest: its `date` overwrote without changing
    # live weights (the exact bug) → untrustworthy → None (consumer falls back).
    s3 = _s3_with_manifest({"date": "2026-06-06", "promoted": False, "version": "v3.0-meta"})
    assert _read_live_served_identity(s3, "bkt", _MANIFEST_KEY) == (None, None)


def test_served_identity_none_when_no_prior_manifest():
    s3 = _s3_with_manifest(None)
    assert _read_live_served_identity(s3, "bkt", _MANIFEST_KEY) == (None, None)


# ── Part 2b: promote_to_champion restamps served identity on the live manifest ─

def _promote_s3_with_live_manifest(*, live_manifest):
    """promote_to_champion mock that ALSO serves a readable live manifest so the
    L4540 restamp fires. Bundle is complete; one prior champion present."""
    vid = "spec-residual-mom-2026-06-13-ab12cd34"
    reg = "predictor/registry/"
    src = f"{reg}{vid}/"
    live_key = "predictor/weights/meta/manifest.json"
    s3 = MagicMock()

    def _paginate(Bucket, Prefix):
        if Prefix == src:
            return [{"Contents": [{"Key": src + f} for f in
                                  ("meta_model.pkl", "feature_list.json",
                                   "manifest.json", "_lineage.json")]}]
        return [{"Contents": [{"Key": f"{src}_lineage.json"}]}]

    pag = MagicMock()
    pag.paginate.side_effect = _paginate
    s3.get_paginator.return_value = pag

    bodies = {
        f"{src}_lineage.json": {"version_id": vid, "stage": "challenger", "date": "2026-06-13"},
        live_key: live_manifest,
    }

    def _get(Bucket, Key):
        body = MagicMock()
        body.read.return_value = json.dumps(bodies[Key]).encode()
        return {"Body": body}

    s3.get_object.side_effect = _get
    return s3, vid, live_key


def test_promote_restamps_served_identity_on_live_manifest():
    # The live manifest pre-promote is the challenger's own training record
    # (promoted False, served carried from the prior champion). After promote it
    # must report THIS version as served.
    s3, vid, live_key = _promote_s3_with_live_manifest(
        live_manifest={"date": "2026-06-13", "promoted": False,
                       "served_version": "v3.0-meta", "served_date": "2026-06-06"}
    )
    promote_to_champion(s3, "bkt", vid)
    # Find the put_object that rewrote the live manifest.
    manifest_puts = [
        json.loads(c.kwargs["Body"])
        for c in s3.put_object.call_args_list
        if c.kwargs.get("Key") == live_key
    ]
    assert manifest_puts, "promote did not restamp the live manifest"
    m = manifest_puts[-1]
    assert m["served_version"] == vid
    assert m["served_date"] == "2026-06-13"   # the lineage date of the promoted version
    assert m["promoted"] is True


# ── Part 2c: inference reads served_date for the trained-date surface ─────────

def test_load_gbm_meta_reports_served_date_not_run_date():
    from inference.stages import write_output

    manifest = {
        "date": "2026-06-13",            # latest RUN (non-promoting)
        "promoted": False,
        "version": "spec-residual-mom",
        "served_version": "v3.0-meta",   # the model actually served
        "served_date": "2026-06-06",
        "models": {"meta_model": {"ic": 0.058}},
    }
    body = MagicMock()
    body.read.return_value = json.dumps(manifest).encode()
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": body}

    ctx = MagicMock()
    ctx.bucket = "bkt"
    ctx.local = False
    ctx.inference_mode = "meta"

    import boto3
    _saved = boto3.client
    boto3.client = lambda *a, **k: s3
    try:
        out = write_output._load_gbm_meta(ctx)
    finally:
        boto3.client = _saved

    assert out["trained_date"] == "2026-06-06"   # served, NOT the 06-13 run date
    assert out["run_date"] == "2026-06-13"
    assert out["served_version"] == "v3.0-meta"


def test_load_gbm_meta_falls_back_to_run_date_for_old_manifest():
    # An old manifest without served_* still reports a sensible trained_date.
    from inference.stages import write_output

    manifest = {"date": "2026-05-30", "promoted": True, "version": "v3.0-meta",
                "models": {"meta_model": {"ic": 0.09}}}
    body = MagicMock()
    body.read.return_value = json.dumps(manifest).encode()
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": body}
    ctx = MagicMock()
    ctx.bucket, ctx.local, ctx.inference_mode = "bkt", False, "meta"

    import boto3
    _saved = boto3.client
    boto3.client = lambda *a, **k: s3
    try:
        out = write_output._load_gbm_meta(ctx)
    finally:
        boto3.client = _saved
    assert out["trained_date"] == "2026-05-30"
