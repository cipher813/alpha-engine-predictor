"""Repo-root pytest fixtures and env defaults.

Pins ``ALPHA_ENGINE_SECRETS_SOURCE=env`` for the test process so
``alpha_engine_lib.secrets.get_secret()`` (post 2026-05-12 .env→SSM
migration, PR 4 of the arc) reads from monkeypatched env vars only —
never the real SSM Parameter Store. Set at module-import time (not
just inside a fixture body) so it's in place before any test module
imports ``training/train_handler.py``, ``polygon_client.py``, etc.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")


@pytest.fixture(autouse=True)
def _isolate_secrets_from_ssm(monkeypatch):
    """Re-pin ``ALPHA_ENGINE_SECRETS_SOURCE=env`` per test + clear the
    per-process secret cache. See
    ``alpha-engine-docs/private/env-to-ssm-260512.md`` § Risks.
    """
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    try:
        from alpha_engine_lib.secrets import clear_cache
    except ImportError:
        yield
        return
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _block_real_alert_publish(monkeypatch):
    """Stub ``alpha_engine_lib.alerts.publish`` so NO predictor test fans out a
    real SNS / Telegram operator alert (L4571 added a promotion alert to the
    model-zoo cutover path). Mirrors the alpha-engine executor conftest; the
    lib's own ``PYTEST_CURRENT_TEST`` guard is the backup. See
    [[reference_alpha_engine_tests_alerts_publish_autostubbed]].
    """
    try:
        from unittest.mock import MagicMock
        from alpha_engine_lib import alerts
    except ImportError:
        yield
        return
    monkeypatch.setattr(alerts, "publish", MagicMock(name="alerts.publish"))
    yield
