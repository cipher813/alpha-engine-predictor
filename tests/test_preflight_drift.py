"""Unit tests for inference.preflight deploy-drift check.

Verifies that:
- A match between baked SHA and GitHub branch HEAD passes silently.
- A mismatch hard-fails with a RuntimeError naming both SHAs.
- Missing stamp file, "unknown" stamp, and GitHub API outage all degrade
  to warn-and-continue (do not block the Lambda).
"""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import inference.preflight as pf


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_preflight():
    """Construct a PredictorPreflight in a way that doesn't trigger the
    full run() — we isolate check_deploy_drift so S3/ArcticDB calls aren't
    attempted during unit tests.
    """
    return pf.PredictorPreflight(bucket="test-bucket", region="us-east-1")


# ── Happy path ───────────────────────────────────────────────────────────────

@patch.object(pf, "_read_baked_git_sha", return_value="abc123def456")
@patch.object(pf, "_fetch_origin_main_sha", return_value="abc123def456")
def test_passes_when_baked_matches_upstream(mock_fetch, mock_read, caplog):
    import logging
    caplog.set_level(logging.INFO)
    _make_preflight().check_deploy_drift()
    assert any("matches" in r.message for r in caplog.records)


# ── Hard-fail path ───────────────────────────────────────────────────────────

@patch.object(pf, "_read_baked_git_sha", return_value="olddeadbeef1234")
@patch.object(pf, "_fetch_origin_main_sha", return_value="newfreshsha5678")
def test_raises_on_sha_mismatch(mock_fetch, mock_read):
    with pytest.raises(RuntimeError) as exc:
        _make_preflight().check_deploy_drift()
    msg = str(exc.value)
    assert "Deploy drift" in msg
    assert "olddeadbeef" in msg
    assert "newfreshsha" in msg
    # Message should tell the operator how to fix it:
    assert "deploy.yml" in msg or "deploy.sh" in msg


# ── Degraded paths (warn, don't block) ──────────────────────────────────────

@patch.object(pf, "_read_baked_git_sha", return_value=None)
@patch.object(pf, "_fetch_origin_main_sha")
def test_missing_stamp_warns_and_passes(mock_fetch, mock_read, caplog):
    """Legacy image (pre-drift-check) has no baked SHA. Don't block."""
    import logging
    caplog.set_level(logging.WARNING)
    _make_preflight().check_deploy_drift()
    # GitHub never queried — no point if we can't compare
    mock_fetch.assert_not_called()
    assert any("legacy build" in r.message or "build-arg omitted" in r.message
               for r in caplog.records)


@patch.object(pf, "_read_baked_git_sha", return_value="abc123def456")
@patch.object(pf, "_fetch_origin_main_sha", return_value=None)
def test_github_unreachable_warns_and_passes(mock_fetch, mock_read):
    """GitHub API outage is not a reason to block Lambda invocations."""
    _make_preflight().check_deploy_drift()  # must not raise
    mock_fetch.assert_called_once()


# ── Stamp file parsing ───────────────────────────────────────────────────────

def test_read_baked_sha_returns_none_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "GIT_SHA_FILE", tmp_path / "nonexistent")
    assert pf._read_baked_git_sha() is None


def test_read_baked_sha_returns_none_when_unknown(tmp_path, monkeypatch):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text("unknown\n")
    monkeypatch.setattr(pf, "GIT_SHA_FILE", stamp)
    assert pf._read_baked_git_sha() is None


def test_read_baked_sha_strips_whitespace(tmp_path, monkeypatch):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text("  abc123  \n")
    monkeypatch.setattr(pf, "GIT_SHA_FILE", stamp)
    assert pf._read_baked_git_sha() == "abc123"


# ── GitHub API mock ──────────────────────────────────────────────────────────

def test_fetch_origin_main_parses_commit_sha():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"commit": {"sha": "deadbeef1234"}}'
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        sha = pf._fetch_origin_main_sha("cipher813/alpha-engine-predictor")
    assert sha == "deadbeef1234"


def test_fetch_origin_main_returns_none_on_network_error():
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("dns failure")):
        sha = pf._fetch_origin_main_sha("cipher813/alpha-engine-predictor")
    assert sha is None


def test_fetch_origin_main_returns_none_on_read_timeout():
    # Regression: urllib.request.urlopen raises a bare TimeoutError on
    # read-phase timeouts (inside getresponse), not URLError. The
    # 2026-05-07 weekday SF failure was exactly this — the Lambda crashed
    # because the except clause didn't cover TimeoutError. Drift check is
    # designed to warn-and-continue on GitHub outages, so a read timeout
    # must degrade gracefully too.
    with patch("urllib.request.urlopen",
               side_effect=TimeoutError("The read operation timed out")):
        sha = pf._fetch_origin_main_sha("cipher813/alpha-engine-predictor")
    assert sha is None
