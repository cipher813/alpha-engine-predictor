"""Tests for the live-batch variant of the output-distribution gate (Phase 2a-INFER).

Per the 2026-05-07 predictor audit Phase 2a-INFER (companion to the
promotion-time gate covered by test_output_distribution_gate.py):
``validate_live_batch_distribution`` validates today's actual inference
output against the same four invariants the promotion gate checks on
synthetic input. Catches degenerate live-batch output even when the
calibrator itself looks healthy on a synthetic sweep.

Specifically targets the 2026-05-07 incident's failure mode: 16 of 27
tickers all clamped to p_up=0.458, which the promotion-time synthetic
sweep would not catch (the calibrator's flat region happens to overlap
with where the Ridge's live output landed for most of the universe).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.output_distribution_gate import validate_live_batch_distribution


# ── Pass paths ───────────────────────────────────────────────────────────

class TestHealthyLiveBatch:

    def test_diverse_batch_passes(self):
        # 27 tickers spread across p_up range [0.10, 0.85]; healthy.
        preds = []
        for i in range(27):
            p_up = 0.10 + (i / 26) * 0.75  # 0.10 → 0.85
            preds.append({
                "ticker": f"T{i:02d}",
                "p_up": round(p_up, 4),
                "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            })
        result = validate_live_batch_distribution(preds)
        assert result.passed is True
        assert result.failed_check is None
        assert result.metrics["n_unique_p_up"] >= 8

    def test_metrics_carry_batch_size(self):
        preds = [{"ticker": f"T{i}", "p_up": 0.3 + i * 0.02,
                  "predicted_direction": "DOWN" if i < 10 else "UP"} for i in range(20)]
        result = validate_live_batch_distribution(preds)
        assert result.metrics["batch_size"] == 20
        assert result.metrics["n_predictions_total"] == 20


# ── Fail paths — each maps to a real incident ────────────────────────────

class TestPlateauLiveBatch:

    def test_2026_05_07_class_plateau_fails_gate(self):
        # The actual 2026-05-07 incident shape from the morning run:
        # n_unique_p_up=6, with values [0.010, 0.429, 0.458, 0.459, 0.517, 0.521].
        # Distribution: 16 at 0.458, 5 at 0.010, 2 at 0.521, 2 at 0.459,
        # 1 at 0.517, 1 at 0.429.
        from collections import Counter
        actual_distribution = Counter({
            0.458: 16, 0.010: 5, 0.521: 2, 0.459: 2, 0.517: 1, 0.429: 1,
        })
        preds = []
        i = 0
        for p_up, count in actual_distribution.items():
            for _ in range(count):
                direction = "UP" if p_up >= 0.5 else "DOWN"
                preds.append({"ticker": f"T{i:02d}", "p_up": p_up,
                              "predicted_direction": direction})
                i += 1

        result = validate_live_batch_distribution(preds)
        assert result.passed is False
        # The incident exhibited multiple failures — uniqueness is
        # checked first (6 < 8) so that's what fires.
        assert result.failed_check == "unique_p_up"
        assert "2026-05-07-class plateau" in result.reason
        assert result.metrics["n_unique_p_up"] == 6


class TestSaturationLiveBatch:

    def test_floor_saturation_fails(self):
        # 20 tickers all at floor p_up=0.01 (degenerate floor-clipping).
        # Need enough unique values to bypass the uniqueness check first.
        preds = [{"ticker": f"T{i:02d}", "p_up": 0.010 + i * 0.0001,
                  "predicted_direction": "DOWN"} for i in range(15)]
        # Add 5 healthy outputs to push n_unique above the threshold.
        preds += [{"ticker": f"OK{i}", "p_up": 0.4 + i * 0.05,
                   "predicted_direction": "DOWN"} for i in range(5)]
        result = validate_live_batch_distribution(preds, max_saturation_rate=0.5)
        # 15 below 0.011 floor / 20 = 75% saturated → fails.
        assert result.passed is False
        assert result.failed_check == "saturation_rate"


class TestStdevLiveBatch:

    def test_compressed_output_fails_stdev(self):
        # All p_up tightly clustered around 0.50 (4/28 collapse mode).
        preds = []
        for i in range(20):
            # p_up varies by 0.0001, alternating up/down to create n_unique > 8
            p_up = 0.500 + (i - 10) * 0.0001
            preds.append({"ticker": f"T{i:02d}", "p_up": round(p_up, 6),
                          "predicted_direction": "UP" if i >= 10 else "DOWN"})
        result = validate_live_batch_distribution(preds)
        assert result.passed is False
        assert result.failed_check == "stdev"
        assert "compressed" in result.reason


class TestDirectionSkewLiveBatch:

    def test_all_down_fails_direction_skew(self):
        # 5/04 false-collapse: 0 UP / 26 DOWN.
        preds = []
        for i in range(26):
            # Spread p_ups in [0.10, 0.49] so all are DOWN but unique enough
            p_up = 0.10 + i * 0.015
            preds.append({"ticker": f"T{i:02d}", "p_up": round(p_up, 4),
                          "predicted_direction": "DOWN"})
        result = validate_live_batch_distribution(preds)
        assert result.passed is False
        assert result.failed_check == "direction_skew"
        assert result.metrics["n_up"] == 0
        assert result.metrics["n_down"] == 26


# ── Pass-by-default for tiny batches ─────────────────────────────────────

class TestSmallBatchPassthrough:

    def test_empty_batch_passes(self):
        result = validate_live_batch_distribution([])
        assert result.passed is True
        assert "below min_batch_size" in result.reason
        assert result.metrics["batch_size"] == 0

    def test_below_min_batch_size_passes(self):
        # 3 predictions — below default min_batch_size=5.
        preds = [{"ticker": f"T{i}", "p_up": 0.5, "predicted_direction": "UP"}
                 for i in range(3)]
        result = validate_live_batch_distribution(preds, min_batch_size=5)
        assert result.passed is True
        assert "below min_batch_size" in result.reason

    def test_at_min_batch_size_runs_checks(self):
        # 5 predictions at default min_batch_size. The actual checks run
        # but with such a tiny batch the uniqueness threshold (8) cannot
        # be met. So this fails on uniqueness — confirming the gate ran.
        preds = [{"ticker": f"T{i}", "p_up": 0.40 + i * 0.05,
                  "predicted_direction": "DOWN" if i < 2 else "UP"}
                 for i in range(5)]
        result = validate_live_batch_distribution(preds)
        assert result.passed is False
        assert result.failed_check == "unique_p_up"


# ── Defensive on malformed predictions ───────────────────────────────────

class TestDefensiveOnMalformed:

    def test_predictions_with_missing_p_up_filtered_out(self):
        # 5 valid + 5 missing-p_up. Effective batch = 5.
        preds = [{"ticker": f"OK{i}", "p_up": 0.4 + i * 0.05,
                  "predicted_direction": "DOWN" if i < 2 else "UP"}
                 for i in range(5)]
        preds += [{"ticker": f"BAD{i}"} for i in range(5)]  # no p_up
        result = validate_live_batch_distribution(preds)
        # Effective batch size = 5; uniqueness check would fail with min 8.
        # But the helper should at least not crash.
        assert "batch_size" in result.metrics
        # Effective batch_size after filter is 5.
        assert result.metrics["batch_size"] == 5

    def test_none_p_up_filtered_out(self):
        preds = [{"ticker": "OK", "p_up": 0.5, "predicted_direction": "UP"}]
        preds += [{"ticker": "BAD", "p_up": None}]
        result = validate_live_batch_distribution(preds, min_batch_size=1)
        # Only OK is counted.
        assert result.metrics["batch_size"] == 1

    def test_string_p_up_filtered_out(self):
        # Defensive against schema drift — string in numeric field shouldn't crash.
        preds = [{"ticker": "OK", "p_up": 0.5, "predicted_direction": "UP"}]
        preds += [{"ticker": "BAD", "p_up": "0.5"}]
        result = validate_live_batch_distribution(preds, min_batch_size=1)
        assert result.metrics["batch_size"] == 1


# ── Default thresholds match promotion-time gate ─────────────────────────

class TestThresholdsMatchPromotionGate:

    def test_default_thresholds_are_identical_to_promotion_gate(self):
        # Audit invariant: promotion-time and inference-time pass should
        # agree on what "degenerate" means. The shared core
        # (_evaluate_distribution_invariants) enforces this; this test
        # sanity-checks the wrapper defaults.
        from model.output_distribution_gate import (
            validate_calibrator_distribution,
        )
        import inspect

        prom_sig = inspect.signature(validate_calibrator_distribution)
        live_sig = inspect.signature(validate_live_batch_distribution)
        for shared in (
            "min_unique_p_up", "max_saturation_rate", "min_stdev",
            "max_direction_skew", "floor", "ceiling",
        ):
            assert (
                prom_sig.parameters[shared].default
                == live_sig.parameters[shared].default
            ), f"default differs on {shared}"
