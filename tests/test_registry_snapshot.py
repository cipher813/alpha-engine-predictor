"""Tests for model.registry — Phase 0 content-addressed model registry snapshot.

Snapshots the complete live inference contract (weights + feature_list +
manifest) into an immutable, content-addressed bundle. Refuses incomplete
contracts; idempotent on re-run; excludes the archive/ subtree.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model.registry import RegistryError, snapshot_to_registry

_PREFIX = "predictor/weights/meta/"


def _live_keys(names):
    return [_PREFIX + n for n in names]


def _make_s3(contract_names, *, lineage_exists=False, etags=None):
    """MagicMock S3 with a paginator over the flat live contract + an archive/
    subdir (which must be excluded), head_object for etags + idempotency."""
    etags = etags or {}
    s3 = MagicMock()

    contents = [{"Key": k} for k in _live_keys(contract_names)]
    # An archive subdir object that MUST be excluded (has a '/' in the suffix).
    contents.append({"Key": _PREFIX + "archive/2026-05-24/meta_model.pkl"})
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": contents}]
    s3.get_paginator.return_value = paginator

    def _head(Bucket, Key):
        if Key.endswith("_lineage.json"):
            if lineage_exists:
                return {"ETag": '"lineage"'}
            raise Exception("NoSuchKey")
        name = Key.rsplit("/", 1)[-1]
        return {"ETag": f'"{etags.get(name, name + "-etag")}"'}

    s3.head_object.side_effect = _head
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
    s3.put_object.assert_not_called()


def test_fingerprint_changes_with_content():
    names = ["meta_model.pkl", "feature_list.json", "manifest.json"]
    v1 = snapshot_to_registry(_make_s3(names, etags={"meta_model.pkl": "AAA"}),
                              "bkt", model_version="v", date="2026-05-30")
    v2 = snapshot_to_registry(_make_s3(names, etags={"meta_model.pkl": "BBB"}),
                              "bkt", model_version="v", date="2026-05-30")
    assert v1 != v2  # different weights → different version_id
