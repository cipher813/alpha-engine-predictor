"""Tests for ssm_secrets.load_secrets — SSM Parameter Store → env hydration."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_ssm_secrets_module(monkeypatch):
    """Reload ssm_secrets between tests so the `_loaded` flag doesn't bleed."""
    if "ssm_secrets" in sys.modules:
        del sys.modules["ssm_secrets"]
    yield


def _ssm_pages(items):
    """items: list of (name, value). Returns one-page paginator response."""
    return [{"Parameters": [{"Name": name, "Value": value} for name, value in items]}]


def test_load_secrets_populates_env_from_ssm(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([
        ("/alpha-engine/POLYGON_API_KEY", "polygon-secret"),
        ("/alpha-engine/FRED_API_KEY", "fred-secret"),
    ])
    fake_client.get_paginator.return_value = fake_paginator

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client) as boto_mock:
            count = ssm_secrets.load_secrets()

    assert count == 2
    assert os.environ["POLYGON_API_KEY"] == "polygon-secret"
    assert os.environ["FRED_API_KEY"] == "fred-secret"
    boto_mock.assert_called_once_with("ssm", region_name="us-east-1")
    fake_client.get_paginator.assert_called_once_with("get_parameters_by_path")
    fake_paginator.paginate.assert_called_once_with(
        Path="/alpha-engine/", Recursive=False, WithDecryption=True,
    )


def test_load_secrets_respects_explicit_env(monkeypatch):
    """Explicit env vars must NOT be overwritten by SSM values."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-set-value")

    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([
        ("/alpha-engine/ANTHROPIC_API_KEY", "ssm-value-should-lose"),
    ])
    fake_client.get_paginator.return_value = fake_paginator

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client):
            count = ssm_secrets.load_secrets()

    assert count == 0  # nothing was loaded (preserved value)
    assert os.environ["ANTHROPIC_API_KEY"] == "operator-set-value"


def test_load_secrets_idempotent(monkeypatch):
    """Second call within the same process returns 0 (already loaded)."""
    monkeypatch.delenv("TEST_VAR", raising=False)

    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([
        ("/alpha-engine/TEST_VAR", "from-ssm"),
    ])
    fake_client.get_paginator.return_value = fake_paginator

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client) as boto_mock:
            first = ssm_secrets.load_secrets()
            second = ssm_secrets.load_secrets()

    assert first == 1
    assert second == 0  # idempotent — short-circuit on _loaded flag
    boto_mock.assert_called_once()  # boto3 only called once


def test_load_secrets_uses_explicit_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)

    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([])
    fake_client.get_paginator.return_value = fake_paginator

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client) as boto_mock:
            ssm_secrets.load_secrets(region="eu-west-1")

    boto_mock.assert_called_once_with("ssm", region_name="eu-west-1")


def test_load_secrets_skips_empty_env_key(monkeypatch):
    """A parameter whose name equals the prefix exactly produces an empty
    env_key and must be skipped (don't try to set os.environ[''])."""
    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([
        ("/alpha-engine/", "noise"),
        ("/alpha-engine/REAL_KEY", "real-value"),
    ])
    fake_client.get_paginator.return_value = fake_paginator

    monkeypatch.delenv("REAL_KEY", raising=False)
    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client):
            count = ssm_secrets.load_secrets()

    assert count == 1
    assert os.environ.get("REAL_KEY") == "real-value"


def test_load_secrets_boto3_unavailable_returns_zero(monkeypatch):
    """ImportError on boto3 must not raise — falls back silently."""
    import ssm_secrets

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("simulated missing boto3")
        return real_import(name, *args, **kwargs)

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("builtins.__import__", side_effect=fake_import):
            count = ssm_secrets.load_secrets()

    assert count == 0


def test_load_secrets_ssm_failure_returns_zero(monkeypatch):
    """Any SSM error (auth, network, etc.) must be caught and return 0."""
    import ssm_secrets

    fake_client = MagicMock()
    fake_client.get_paginator.side_effect = RuntimeError("SSM unreachable")

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client):
            count = ssm_secrets.load_secrets()

    assert count == 0


def test_load_secrets_picks_up_aws_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    import ssm_secrets

    fake_client = MagicMock()
    fake_paginator = MagicMock()
    fake_paginator.paginate.return_value = _ssm_pages([])
    fake_client.get_paginator.return_value = fake_paginator

    with patch.object(ssm_secrets, "_loaded", False):
        with patch("boto3.client", return_value=fake_client) as boto_mock:
            ssm_secrets.load_secrets()

    boto_mock.assert_called_once_with("ssm", region_name="us-west-2")
