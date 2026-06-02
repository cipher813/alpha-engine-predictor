"""Tests for model.registry — Phase 0 content-addressed model registry snapshot.

Snapshots the complete live inference contract (weights + feature_list +
manifest) into an immutable, content-addressed bundle. Refuses incomplete
contracts; idempotent on re-run; excludes the archive/ subtree; records the
champion/challenger ``stage`` and promotes-in-place (L4469 capture-gap fix).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from model.registry import RegistryError, list_versions, snapshot_to_registry

_PREFIX = "predictor/weights/meta/"


def _live_keys(names):
    return [_PREFIX + n for n in names]


def _make_s3(contract_names, *, lineage_exists=False, lineage_stage=None, etags=None):
    """MagicMock S3 with a paginator over the flat live contract + an archive/
    subdir (which must be excluded), head_object for etags, and get_object for
    the _lineage.json idempotency / stage-patch read."""
    etags = etags or {}
    s3 = MagicMock()

    contents = [{"Key": k} for k in _live_keys(contract_names)]
    # An archive subdir object that MUST be excluded (has a '/' in the suffix).
    contents.append({"Key": _PREFIX + "archive/2026-05-24/meta_model.pkl"})
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": contents}]
    s3.get_paginator.return_value = paginator

    def _head(Bucket, Key):
        name = Key.rsplit("/", 1)[-1]
        return {"ETag": f'"{etags.get(name, name + "-etag")}"'}

    s3.head_object.side_effect = _head

    def _get(Bucket, Key):
        if Key.endswith("_lineage.json") and lineage_exists:
            body = MagicMock()
            body.read.return_value = json.dumps(
                {"version_id": "x", "stage": lineage_stage}
            ).encode()
            return {"Body": body}
        raise Exception("NoSuchKey")

    s3.get_object.side_effect = _get
    return s3


def test_happy_path_copies_contract_and_writes_lineage():
    names = ["meta_model.pkl", "volatility_model.txt", "feature_list.json", "manifest.json"]
    s3 = _make_s3(names)
    vid = snapshot_to_registry(s3, "bkt", model_version="meta-v3.0-8models", date="2026-05-30")

    assert vid.startswith("meta-v3.0-8models-2026-05-30-")
    # Every flat contract file copied, archive subdir excluded.
    copied = {c.kwargs["CopySource"]["Key"] for c in s3.copy_object.call_args_list}
    assert copied == set(_live_keys(names))
    assert not any("archive/" in k for k in copied)
    # Lineage written under the version_id dir.
    put_keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
    assert put_keys == [f"predictor/registry/{vid}/_lineage.json"]


def test_refuses_incomplete_contract_missing_feature_list():
    names = ["meta_model.pkl", "manifest.json"]  # no feature_list.json
    s3 = _make_s3(names)
    with pytest.raises(RegistryError, match="feature_list.json"):
        snapshot_to_registry(s3, "bkt", model_version="v", date="2026-05-30")
    s3.copy_object.assert_not_called()


def test_idempotent_when_lineage_exists():
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    s3 = _make_s3(names, lineage_exists=True)
    vid = snapshot_to_registry(s3, "bkt", model_version="v", date="2026-05-30")
    assert vid.startswith("v-2026-05-30-")
    s3.copy_object.assert_not_called()  # already snapshotted → no recopy
    s3.put_object.assert_not_called()  # no stage arg → nothing to patch


def test_fingerprint_changes_with_content():
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    v1 = snapshot_to_registry(_make_s3(names, etags={"meta_model.pkl": "AAA"}),
                              "bkt", model_version="v", date="2026-05-30")
    v2 = snapshot_to_registry(_make_s3(names, etags={"meta_model.pkl": "BBB"}),
                              "bkt", model_version="v", date="2026-05-30")
    assert v1 != v2  # different weights → different version_id


def test_stage_recorded_in_lineage():
    # L4469: a non-promoted candidate registers as a `challenger`.
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    s3 = _make_s3(names)
    snapshot_to_registry(s3, "bkt", model_version="v", date="2026-06-02",
                         stage="challenger")
    body = s3.put_object.call_args_list[-1].kwargs["Body"]
    assert json.loads(body)["stage"] == "challenger"


def test_promote_in_place_patches_stage_when_bundle_exists():
    # A challenger captured earlier, later promoted (same content → same id):
    # patch the recorded stage in place, do NOT recopy the immutable bytes.
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    s3 = _make_s3(names, lineage_exists=True, lineage_stage="challenger")
    snapshot_to_registry(s3, "bkt", model_version="v", date="2026-06-02",
                         stage="champion")
    s3.copy_object.assert_not_called()  # bundle bytes immutable
    body = s3.put_object.call_args_list[-1].kwargs["Body"]
    assert json.loads(body)["stage"] == "champion"


def test_list_versions_filters_by_stage_and_sorts_newest_first():
    s3 = MagicMock()
    pag = MagicMock()
    pag.paginate.return_value = [{"Contents": [
        {"Key": "predictor/registry/v-2026-06-01-aaa/_lineage.json"},
        {"Key": "predictor/registry/v-2026-06-02-bbb/_lineage.json"},
        {"Key": "predictor/registry/v-2026-05-30-ccc/_lineage.json"},
        {"Key": "predictor/registry/v-2026-06-02-bbb/meta_model.pkl"},  # not a lineage
    ]}]
    s3.get_paginator.return_value = pag
    lineages = {
        "predictor/registry/v-2026-06-01-aaa/_lineage.json":
            {"version_id": "v-2026-06-01-aaa", "stage": "challenger", "date": "2026-06-01"},
        "predictor/registry/v-2026-06-02-bbb/_lineage.json":
            {"version_id": "v-2026-06-02-bbb", "stage": "challenger", "date": "2026-06-02"},
        "predictor/registry/v-2026-05-30-ccc/_lineage.json":
            {"version_id": "v-2026-05-30-ccc", "stage": "champion", "date": "2026-05-30"},
    }

    def _get(Bucket, Key):
        body = MagicMock()
        body.read.return_value = json.dumps(lineages[Key]).encode()
        return {"Body": body}

    s3.get_object.side_effect = _get

    challengers = list_versions(s3, "bkt", stage="challenger")
    assert [v["version_id"] for v in challengers] == [
        "v-2026-06-02-bbb", "v-2026-06-01-aaa",  # newest first, champion excluded
    ]
    assert len(list_versions(s3, "bkt")) == 3  # no filter → all


def test_no_demote_when_bundle_exists():
    # Never demote champion → challenger on a re-snapshot (e.g. a later
    # observe-only run of the same content).
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    s3 = _make_s3(names, lineage_exists=True, lineage_stage="champion")
    snapshot_to_registry(s3, "bkt", model_version="v", date="2026-06-02",
                         stage="challenger")
    s3.copy_object.assert_not_called()
    s3.put_object.assert_not_called()  # no demote
