"""Unit tests for inference.lib_pin_drift — cross-repo lib-pin drift probe (L4517).

Exercises the pure compare logic (parity + floor) + the fail-open degraded mode.
``_fetch_repo_pin`` (the GitHub read) is mocked so tests are hermetic.
"""

from __future__ import annotations

from unittest.mock import patch

import inference.lib_pin_drift as lpd


# Today's aligned fleet: co-install pair matched, all >= floor (v0.39.0).
_ALIGNED = {
    "nousergon/crucible-backtester": "v0.53.0",
    "nousergon/crucible-predictor": "v0.53.0",
    "nousergon/nousergon-data": "v0.39.0",      # == floor → passes
    "nousergon/crucible-research": "v0.42.0",
}


def _patch_pins(mapping):
    """Patch _fetch_repo_pin to resolve from `mapping` (repo → pin|None)."""
    return patch.object(lpd, "_fetch_repo_pin", side_effect=lambda repo, **_: mapping.get(repo))


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_parse_pin_happy():
    line = (
        "alpha-engine-lib[arcticdb,flow_doctor,quant-xs] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.53.0"
    )
    assert lpd._parse_pin(line) == "v0.53.0"


def test_parse_pin_renamed_dist_nousergon_lib():
    # Dist renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
    # (config#1245). The drift probe must parse the new spelling so it keeps
    # working as the fleet crosses one repo at a time.
    line = (
        "nousergon-lib[arcticdb,flow-doctor,quant-xs] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.60.2"
    )
    assert lpd._parse_pin(line) == "v0.60.2"


def test_parse_pin_miss_returns_none():
    assert lpd._parse_pin("requests==2.31.0\nnumpy>=1.26") is None


def test_ge_floor():
    assert lpd._ge_floor("v0.39.0") is True       # == floor
    assert lpd._ge_floor("v0.53.0") is True
    assert lpd._ge_floor("v0.38.9") is False       # below floor
    assert lpd._ge_floor("v0.17.0") is False


# ── check_lib_pin_drift ──────────────────────────────────────────────────────

def test_aligned_fleet_no_drift():
    with _patch_pins(_ALIGNED):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is False
    assert out["parity_ok"] is True
    assert out["floor_ok"] is True
    assert out["reason"] == "in_sync"
    assert out["offenders"] == []


def test_co_install_parity_mismatch_halts():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = "v0.52.1"  # predictor lags backtester
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["parity_ok"] is False
    assert any("co-install parity" in o for o in out["offenders"])
    # the offending repos/versions are named
    assert any("v0.53.0" in o and "v0.52.1" in o for o in out["offenders"])


def test_below_floor_halts_and_names_offender():
    pins = dict(_ALIGNED)
    pins["nousergon/nousergon-data"] = "v0.38.0"  # regressed below floor
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["floor_ok"] is False
    assert any("below floor" in o and "nousergon-data" in o and "v0.38.0" in o
               for o in out["offenders"])


def test_fetch_failure_fails_open():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-research"] = None  # GitHub unreachable / parse miss
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    # The checker's own fragility must NEVER halt the weekly run.
    assert out["has_drift"] is False
    assert out["reason"] == "fetch_failed"


def test_parity_mismatch_below_floor_combined():
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": "v0.49.0",   # parity break
        "nousergon/nousergon-data": "v0.30.0",        # below floor
        "nousergon/crucible-research": "v0.42.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["parity_ok"] is False and out["floor_ok"] is False
    assert len(out["offenders"]) >= 2
