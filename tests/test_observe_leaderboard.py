"""Tests for the realized-edge observe leaderboard (config #671/#702/L4539).

Pins:
  - realized rank-IC scoring of a shadow version vs champion over matured pairs;
  - ``ready_for_full_promotion`` gates on REALIZED edge (>= soak weeks +
    outcomes + beats-champion), NOT training DSR;
  - 'too young' / 'insufficient outcomes' verdicts are explicit, not silent;
  - the leaderboard NEVER promotes / allocates (it is measurement only — there
    is no registry write path in this module).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.observe_leaderboard as ol


def _prices(n_days=120):
    """A rising-AAA / flat-SPY fixture so realized 21d alpha is well-defined and
    positive, with enough forward room for the 21d window to close."""
    idx = pd.date_range("2026-03-01", periods=n_days, freq="B")
    return {
        "AAA": pd.Series(np.linspace(100, 160, n_days), index=idx),
        "BBB": pd.Series(np.linspace(100, 90, n_days), index=idx),
        "SPY": pd.Series(np.linspace(100, 110, n_days), index=idx),
    }


def _pred_rows(alpha_by_ticker, dates):
    """Build (date, ticker, predicted_alpha) rows for a set of dates/tickers."""
    rows = []
    for d in dates:
        for tkr, a in alpha_by_ticker.items():
            rows.append({"date": d, "ticker": tkr,
                         "predicted_alpha": a, "realized_alpha": None})
    return rows


def _weekly_dates(n_weeks, per_week=5):
    """ISO dates spanning ``n_weeks`` distinct weeks (Mon-Fri each), early enough
    that the 21d forward window closes within the fixture's price history."""
    start = pd.Timestamp("2026-03-02")  # a Monday
    out = []
    for w in range(n_weeks):
        for d in range(per_week):
            out.append((start + pd.Timedelta(weeks=w, days=d)).date().isoformat())
    return out


def test_leaderboard_scores_realized_ic_and_promotion_ready():
    # AAA rises (positive realized alpha), BBB falls (negative). A version that
    # ranks AAA > BBB has positive realized rank-IC; the champion that ranks them
    # backwards has negative IC → version beats champion → promotion-ready once
    # the soak floors clear.
    dates = _weekly_dates(5)  # >= 4 weeks
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)      # correct ranking
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)     # inverted ranking
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-good": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    e = res["entries"][0]
    assert e["version_id"] == "v-good"
    assert e["realized_rank_ic"] is not None
    assert e["realized_rank_ic"] > 0                        # ranked correctly
    assert res["champion"]["realized_rank_ic"] < 0          # ranked backwards
    assert e["beats_champion_realized"] is True
    assert e["n_weeks_coverage"] >= ol.MIN_SOAK_WEEKS
    assert e["n_matured_outcomes"] >= ol.MIN_REALIZED_OUTCOMES
    assert e["ready_for_full_promotion"] is True
    assert res["ready_version_ids"] == ["v-good"]


def test_leaderboard_soak_too_young_not_ready():
    # Only 2 weeks of coverage — below MIN_SOAK_WEEKS → not promotion-ready even
    # with a positive realized IC. The verdict says so explicitly.
    dates = _weekly_dates(2)
    good = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    champ = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-young": good},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    e = res["entries"][0]
    assert e["ready_for_full_promotion"] is False
    assert "soak too young" in e["verdict_reason"] or "insufficient" in e["verdict_reason"]
    assert res["ready_version_ids"] == []


def test_leaderboard_loses_to_champion_not_ready():
    # The version ranks backwards (negative IC) while the champion ranks
    # correctly → does NOT beat champion → hold.
    dates = _weekly_dates(5)
    bad = _pred_rows({"AAA": 0.1, "BBB": 0.9}, dates)       # inverted
    champ = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)     # correct
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-bad": bad},
        live_pairs=champ, prices_by_ticker=_prices(), sector_map={},
    )
    e = res["entries"][0]
    assert e["beats_champion_realized"] is False
    assert e["ready_for_full_promotion"] is False
    assert "hold" in e["verdict_reason"]


def test_leaderboard_unmatured_outcomes_reported_not_ready():
    # Predictions whose 21d forward window has NOT closed (dated at the very end
    # of the price history) → no matured outcomes → IC None, explicit verdict.
    last = _prices(120)["AAA"].index[-1]
    near_end = [(last - pd.Timedelta(days=k)).date().isoformat() for k in range(5)]
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, near_end)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v-unmatured": rows},
        live_pairs=rows, prices_by_ticker=_prices(), sector_map={},
    )
    e = res["entries"][0]
    assert e["realized_rank_ic"] is None
    assert e["ready_for_full_promotion"] is False
    assert "insufficient matured outcomes" in e["verdict_reason"]


def test_leaderboard_is_measurement_only_no_registry_writes():
    # The module must never promote/allocate: it has no registry-write surface.
    # Confirm a run with write_to_s3=False touches no S3 and returns a pure dict.
    dates = _weekly_dates(5)
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=False, date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v": rows}, live_pairs=rows,
        prices_by_ticker=_prices(), sector_map={},
    )
    assert "s3_key" not in res                              # nothing written
    # The leaderboard RECOMMENDS an operator promote command in the verdict text
    # but NEVER imports/calls a registry actuator — confirm the module has no
    # promote/register symbol bound at module scope.
    import inspect

    src = inspect.getsource(ol)
    assert "promote_to_champion" not in src
    assert "register_to_observe" not in src
    # source attribution proves the gate is realized-edge, not training DSR.
    assert "NOT training DSR" in res["soak_criteria"]["gate"]


def test_leaderboard_writes_dated_and_latest():
    dates = _weekly_dates(5)
    rows = _pred_rows({"AAA": 0.9, "BBB": 0.1}, dates)

    class _S3:
        def __init__(self):
            self.puts = []

        def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
            self.puts.append(Key)

    s3 = _S3()
    res = ol.build_observe_leaderboard(
        bucket="b", write_to_s3=True, write_latest=True, s3_client=s3,
        date_str="2026-06-13", horizon_days=21,
        shadow_pairs_by_version={"v": rows}, live_pairs=rows,
        prices_by_ticker=_prices(), sector_map={},
    )
    assert f"{ol.OUTPUT_PREFIX}/2026-06-13.json" in s3.puts
    assert f"{ol.OUTPUT_PREFIX}/latest.json" in s3.puts
    assert res["s3_key"] == f"{ol.OUTPUT_PREFIX}/2026-06-13.json"
