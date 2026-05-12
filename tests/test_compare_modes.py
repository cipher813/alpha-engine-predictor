"""Tests for analysis.compare_modes — mode comparison report + CSV export."""

import csv
import json
from unittest.mock import MagicMock, patch

import pytest

from analysis.compare_modes import (
    MODE_HISTORY_KEY,
    export_csv,
    load_mode_history,
    summarize,
)


# ── load_mode_history ──────────────────────────────────────────────────────


def _make_s3_with_object(payload):
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    s3.get_object.return_value = {"Body": body}
    return s3


def test_load_mode_history_returns_list():
    history = [{"date": "2026-04-26", "best_mode": "mse", "mse_ic": 0.1}]
    fake_s3 = _make_s3_with_object(history)
    with patch("boto3.client", return_value=fake_s3):
        result = load_mode_history()
    assert result == history
    fake_s3.get_object.assert_called_once()
    assert fake_s3.get_object.call_args.kwargs["Key"] == MODE_HISTORY_KEY


def test_load_mode_history_non_list_payload_returns_empty():
    fake_s3 = _make_s3_with_object({"not": "a list"})
    with patch("boto3.client", return_value=fake_s3):
        result = load_mode_history()
    assert result == []


def test_load_mode_history_s3_error_returns_empty():
    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("NoSuchKey")
    with patch("boto3.client", return_value=fake_s3):
        result = load_mode_history()
    assert result == []


# ── summarize ──────────────────────────────────────────────────────────────


def test_summarize_empty_history_returns_placeholder():
    report = summarize([])
    assert "Mode Comparison Report" in report
    assert "No mode history data available" in report


def _entry(date_str, best_mode="mse", mse_ic=0.10, rank_ic=0.08, ens_ic=0.12, **extras):
    base = {
        "date": date_str,
        "best_mode": best_mode,
        "mse_ic": mse_ic,
        "rank_ic": rank_ic,
        "ensemble_ic": ens_ic,
        "promoted": True,
        "ic_delta_rank_vs_mse": rank_ic - mse_ic if rank_ic is not None else None,
        "ic_delta_ens_vs_mse": ens_ic - mse_ic if ens_ic is not None else None,
        "ic_ir": 1.5,
        "n_train": 2000,
    }
    base.update(extras)
    return base


def test_summarize_renders_full_report_with_winners_and_deltas():
    history = [
        _entry("2026-03-01", best_mode="mse", mse_ic=0.10, rank_ic=0.08, ens_ic=0.11),
        _entry("2026-03-08", best_mode="ensemble", mse_ic=0.09, rank_ic=0.10, ens_ic=0.13),
        _entry("2026-03-15", best_mode="ensemble", mse_ic=0.11, rank_ic=0.10, ens_ic=0.14),
        _entry("2026-03-22", best_mode="rank", mse_ic=0.10, rank_ic=0.13, ens_ic=0.11),
    ]
    report = summarize(history)

    assert "**Entries:** 4" in report
    assert "2026-03-01 to 2026-03-22" in report
    assert "**ensemble**: 2 (50%)" in report
    assert "**mse**: 1 (25%)" in report
    assert "**rank**: 1 (25%)" in report
    assert "## Average IC Across All Weeks" in report
    assert "## IC Deltas vs MSE Baseline" in report
    assert "Rank vs MSE" in report
    assert "Ensemble vs MSE" in report
    assert "## Weekly Detail" in report
    assert "| 2026-03-22 | 0.1000 | 0.1300 | 0.1100 | rank | yes | 1.50 |" in report


def test_summarize_handles_missing_optional_fields():
    """Missing rank_ic / ensemble_ic / deltas should render as em-dashes,
    not crash."""
    history = [
        {
            "date": "2026-04-01",
            "best_mode": "mse",
            "mse_ic": 0.07,
            "rank_ic": None,
            "ensemble_ic": None,
            "promoted": False,
            "ic_delta_rank_vs_mse": None,
            "ic_delta_ens_vs_mse": None,
            "n_train": 1500,
            "ic_ir": 0.9,
        },
    ]
    report = summarize(history)
    assert "**Entries:** 1" in report
    # Per-mode averages must skip mode with all-None
    assert "**Rank**:" not in report or "n=0" not in report
    # Weekly row uses em-dash for missing fields
    assert "| 0.0700 | — | — |" in report
    assert "| no |" in report


def test_summarize_no_deltas_omits_delta_section():
    """If neither rank nor ensemble deltas are populated, the deltas
    section is omitted entirely."""
    history = [
        {
            "date": "2026-04-01",
            "best_mode": "mse",
            "mse_ic": 0.08,
            "rank_ic": None,
            "ensemble_ic": None,
            "promoted": False,
            "ic_delta_rank_vs_mse": None,
            "ic_delta_ens_vs_mse": None,
            "n_train": 1500,
            "ic_ir": 0.9,
        },
    ]
    report = summarize(history)
    assert "## IC Deltas vs MSE Baseline" not in report


# ── export_csv ──────────────────────────────────────────────────────────────


def test_export_csv_writes_expected_fields(tmp_path):
    history = [
        _entry("2026-03-01", best_mode="mse"),
        _entry("2026-03-08", best_mode="ensemble"),
    ]
    out = tmp_path / "out.csv"
    export_csv(history, str(out))

    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["date"] == "2026-03-01"
    assert rows[0]["best_mode"] == "mse"
    assert rows[1]["best_mode"] == "ensemble"
    expected_fields = {"date", "mse_ic", "rank_ic", "ensemble_ic", "best_mode",
                       "promoted", "ic_delta_rank_vs_mse", "ic_delta_ens_vs_mse",
                       "n_train", "ic_ir"}
    assert set(reader.fieldnames) == expected_fields


def test_export_csv_empty_history_skips_write(tmp_path):
    out = tmp_path / "out.csv"
    export_csv([], str(out))
    # No file should be created for empty history
    assert not out.exists()


def test_export_csv_drops_extra_fields(tmp_path):
    """extrasaction='ignore' means unknown fields silently drop."""
    history = [_entry("2026-03-01", extra_unknown_field="ignored")]
    out = tmp_path / "out.csv"
    export_csv(history, str(out))

    with open(out) as f:
        content = f.read()

    assert "extra_unknown_field" not in content
    assert "ignored" not in content
