"""Tests for momentum GBM params loading + S3 override."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_config_exposes_momentum_defaults():
    import config as cfg

    assert cfg.MOMENTUM_GBM_N_ESTIMATORS == 300
    assert cfg.MOMENTUM_GBM_EARLY_STOPPING_ROUNDS == 30
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["num_leaves"] == 7
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["max_depth"] == 2
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["min_child_samples"] == 500
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["lambda_l1"] == 1.0
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["lambda_l2"] == 1.0


def test_momentum_defaults_inherit_shared_non_overridden_keys():
    """Momentum params should carry over shared keys like objective/metric
    that are not in the momentum-specific override."""
    import config as cfg

    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["objective"] == cfg.GBM_TUNED_PARAMS["objective"]
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["metric"] == cfg.GBM_TUNED_PARAMS["metric"]
    assert cfg.MOMENTUM_GBM_TUNED_PARAMS["seed"] == cfg.GBM_TUNED_PARAMS["seed"]


def test_s3_override_helper_returns_none_when_key_absent():
    from training.meta_trainer import _load_momentum_params_from_s3

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("NoSuchKey")
    with patch("boto3.client", return_value=mock_s3):
        result = _load_momentum_params_from_s3("test-bucket")

    assert result is None


def test_s3_override_helper_parses_valid_payload():
    from training.meta_trainer import _load_momentum_params_from_s3

    payload = {
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "tuned_params": {"num_leaves": 15, "max_depth": 3},
    }
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(payload).encode()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": mock_body}

    with patch("boto3.client", return_value=mock_s3):
        result = _load_momentum_params_from_s3("test-bucket")

    assert result == payload


def test_s3_override_helper_rejects_non_dict_payload():
    from training.meta_trainer import _load_momentum_params_from_s3

    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps([1, 2, 3]).encode()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": mock_body}

    with patch("boto3.client", return_value=mock_s3):
        result = _load_momentum_params_from_s3("test-bucket")

    assert result is None
