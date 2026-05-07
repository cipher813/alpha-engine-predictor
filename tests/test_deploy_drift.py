"""Unit tests for inference.deploy_drift — SF + CF drift probe.

Exercises the pure-Python compare logic + the boto3-call surface with
stubs. The GitHub-fetch helper (``_fetch_origin_main_sha``) is owned
by alpha-engine-lib and tested there; this module re-imports it so
``patch.object(dd, "_fetch_origin_main_sha", ...)`` keeps mocking the
same symbol the production code calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import inference.deploy_drift as dd


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_extract_sf_sha_happy_path():
    assert dd._extract_sf_sha("[git:abc123def456] rest of comment") == "abc123def456"


def test_extract_sf_sha_short_sha_ok():
    # 7-char SHAs (git short hashes) are valid
    assert dd._extract_sf_sha("[git:abc1234] stuff") == "abc1234"


def test_extract_sf_sha_no_prefix_returns_none():
    assert dd._extract_sf_sha("No git prefix here") is None


def test_extract_sf_sha_empty_comment_returns_none():
    assert dd._extract_sf_sha("") is None


def test_extract_sf_sha_handles_whitespace():
    assert dd._extract_sf_sha("   [git:deadbeef12345] some comment") == "deadbeef12345"


def test_extract_sf_sha_rejects_nonhex():
    # 'g' is not hex — prevents accidental match on non-SHA strings
    assert dd._extract_sf_sha("[git:abcdefg] garbage") is None


# ── SHA-match logic ──────────────────────────────────────────────────────────

def test_shas_match_exact():
    assert dd._shas_match("abc123" * 6 + "abcd", "abc123" * 6 + "abcd") is True


def test_shas_match_short_prefix():
    # Short deployed SHA (e.g. 12 chars) matches if it's a prefix of upstream
    assert dd._shas_match("abc123def456", "abc123def456ffffffffffff12345678ffff9999") is True


def test_shas_match_mismatch():
    assert dd._shas_match("abc123def456", "deadbeef56781111111111111111111111111111") is False


def test_shas_match_none_deployed_passes():
    # Missing stamp → can't prove drift → don't block
    assert dd._shas_match(None, "deadbeef" * 5) is True


def test_shas_match_none_upstream_passes():
    assert dd._shas_match("abc123def456", None) is True


def test_shas_match_malformed_deployed_passes():
    # <7 char stamp is malformed, treat as missing
    assert dd._shas_match("abc", "deadbeef" * 5) is True


# ── check_deploy_drift composition ───────────────────────────────────────────

@patch.object(dd, "_read_sf_comment", return_value="[git:deadbeef12345] weekday pipeline")
@patch.object(dd, "_read_stack_tag", return_value="deadbeef12345abcdef0123456789012345abcdef")
@patch.object(dd, "_fetch_origin_main_sha", return_value="deadbeef12345abcdef0123456789012345abcdef")
def test_no_drift_when_everything_matches(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["sf_drift"] is False
    assert result["cf_drift"] is False
    assert result["sf_stamp_present"] is True
    assert result["stack_stamp_present"] is True


@patch.object(dd, "_read_sf_comment", return_value="[git:aaaa111aaaa1] stale")
@patch.object(dd, "_read_stack_tag", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
@patch.object(dd, "_fetch_origin_main_sha", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
def test_sf_drift_detected(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["sf_drift"] is True
    assert result["cf_drift"] is False
    assert result["has_drift"] is True


@patch.object(dd, "_read_sf_comment", return_value="[git:bbbb222bbbb2] ok")
@patch.object(dd, "_read_stack_tag", return_value="aaaa111aaaa1dddddddddddddddddddddddddddd")
@patch.object(dd, "_fetch_origin_main_sha", return_value="bbbb222bbbb2cccccccccccccccccccccccccccc")
def test_cf_drift_detected(mock_fetch, mock_tag, mock_comment):
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["sf_drift"] is False
    assert result["cf_drift"] is True
    assert result["has_drift"] is True


@patch.object(dd, "_read_sf_comment", return_value=None)
@patch.object(dd, "_read_stack_tag", return_value=None)
@patch.object(dd, "_fetch_origin_main_sha", return_value="a" * 40)
def test_missing_stamps_do_not_trigger_drift(mock_fetch, mock_tag, mock_comment):
    # First SF deploy before stamping shipped won't have a prefix; first CF
    # deploy without git-sha tag won't have the tag. Don't block these paths.
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["sf_stamp_present"] is False
    assert result["stack_stamp_present"] is False


@patch.object(dd, "_read_sf_comment", return_value="[git:abc123] old")
@patch.object(dd, "_read_stack_tag", return_value="abc123")
@patch.object(dd, "_fetch_origin_main_sha", return_value=None)
def test_github_outage_is_no_drift(mock_fetch, mock_tag, mock_comment):
    # Can't compare against upstream → report stamps but don't flag drift
    result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["upstream_sha"] is None


# ── AWS read surface ─────────────────────────────────────────────────────────

def test_read_sf_comment_parses_description():
    mock_sfn = MagicMock()
    mock_sfn.describe_state_machine.return_value = {
        "definition": json.dumps({"Comment": "[git:abc123] foo", "States": {}})
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_sfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        comment = dd._read_sf_comment("arn:aws:states:us-east-1:1:stateMachine:x")
    assert comment == "[git:abc123] foo"


def test_read_sf_comment_returns_none_on_error():
    mock_sfn = MagicMock()
    mock_sfn.describe_state_machine.side_effect = Exception("boom")
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_sfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        comment = dd._read_sf_comment("arn:...")
    assert comment is None


def test_read_stack_tag_happy():
    mock_cfn = MagicMock()
    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{
            "Tags": [
                {"Key": "git-sha", "Value": "deadbeef"},
                {"Key": "other", "Value": "thing"},
            ]
        }]
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_cfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        sha = dd._read_stack_tag("alpha-engine-orchestration")
    assert sha == "deadbeef"


def test_read_stack_tag_returns_none_when_tag_absent():
    mock_cfn = MagicMock()
    mock_cfn.describe_stacks.return_value = {
        "Stacks": [{"Tags": [{"Key": "other", "Value": "x"}]}]
    }
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_cfn
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        sha = dd._read_stack_tag("stack")
    assert sha is None
