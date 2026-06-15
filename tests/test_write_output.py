"""Tests for inference/stages/write_output.py — predictions writing + email builder."""

import json
from unittest.mock import MagicMock, patch

import pytest

import inference.stages.write_output as wo
from inference.stages.write_output import (
    write_predictions, get_veto_threshold,
    _merge_predictions, _read_existing_predictions,
)


class TestWritePredictionsDryRun:
    """Test write_predictions in dry-run mode (no S3)."""

    def test_dry_run_prints(self, capsys):
        predictions = [
            {"ticker": "AAPL", "predicted_direction": "UP", "prediction_confidence": 0.75},
            {"ticker": "MSFT", "predicted_direction": "DOWN", "prediction_confidence": 0.68},
        ]
        metrics = {"model_version": "test-v1", "hit_rate_30d_rolling": 0.55}

        write_predictions(predictions, "2026-04-08", "bucket", metrics, dry_run=True)

        captured = capsys.readouterr()
        assert "PREDICTIONS (dry-run)" in captured.out
        assert "AAPL" in captured.out
        assert "METRICS (dry-run)" in captured.out

    def test_dry_run_counts_high_confidence(self, capsys):
        predictions = [
            {"ticker": "A", "prediction_confidence": 0.80},
            {"ticker": "B", "prediction_confidence": 0.50},
            {"ticker": "C", "prediction_confidence": 0.70},
        ]
        write_predictions(predictions, "2026-04-08", "bucket", {}, dry_run=True, veto_threshold=0.65)

        captured = capsys.readouterr()
        output = json.loads(captured.out.split("=== PREDICTIONS (dry-run) ===\n")[1].split("\n=== METRICS")[0])
        assert output["n_high_confidence"] == 2  # A (0.80) and C (0.70) >= 0.65

    def test_dry_run_includes_date(self, capsys):
        write_predictions([], "2026-04-08", "bucket", {}, dry_run=True)
        captured = capsys.readouterr()
        assert "2026-04-08" in captured.out


class TestWritePredictionsS3:
    """Test write_predictions with mocked S3."""

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    def test_writes_three_keys(self, mock_put):
        predictions = [{"ticker": "AAPL", "prediction_confidence": 0.75}]
        write_predictions(predictions, "2026-04-08", "bucket", {"model_version": "v1"})
        assert mock_put.call_count == 3  # dated, latest, metrics

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json", side_effect=Exception("S3 error"))
    def test_handles_write_failure(self, mock_put):
        # Should not raise — failures are logged but not propagated
        write_predictions([{"ticker": "AAPL"}], "2026-04-08", "bucket", {})


class TestGetVetoThresholdExtended:
    """Additional veto threshold tests."""

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_case_insensitive_regime(self, _mock):
        # Post-2026-05-12: bear adjustment is -0.20 in new confidence units.
        result = get_veto_threshold("bucket", "  BEAR  ")
        assert result == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.30})
    def test_none_regime(self, _mock):
        result = get_veto_threshold("bucket", None)
        assert result == pytest.approx(0.30)


class TestForcedBearClamp:
    """Stage F2 — forced-bear veto clamp (regime-fast-signal-260515.md)."""

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    _ON = {"veto_confidence": 0.30, "regime_forced_bear_enabled": True}
    _OFF = {"veto_confidence": 0.30, "regime_forced_bear_enabled": False}

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_on_forces_bear_floor(self, m):
        m.return_value = self._ON
        # neutral regime → unclamped 0.30; clamp → base-cap = 0.30-0.20 = 0.10
        assert get_veto_threshold("b", "neutral", forced_bear=True) == pytest.approx(0.10)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_off_is_parallel_observe_noop(self, m):
        m.return_value = self._OFF
        # flag off: behavior unchanged even though forced_bear=True
        assert get_veto_threshold("b", "neutral", forced_bear=True) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_no_clamp_when_not_forced_bear(self, m):
        m.return_value = self._ON
        assert get_veto_threshold("b", "neutral", forced_bear=False) == pytest.approx(0.30)

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_forced_bear_overrides_risk_on_wire4(self, m):
        # Wire 4 ON + strong risk-ON intensity_z would raise the threshold
        # to 0.50 (more permissive). Forced-bear must still win — the
        # max-protection property.
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_forced_bear_enabled": True,
        }
        permissive = get_veto_threshold("b", regime_intensity_z=2.0)
        assert permissive == pytest.approx(0.50)  # sanity: risk-on raises it
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_forced_bear_enabled": True,
        }
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False
        clamped = get_veto_threshold("b", regime_intensity_z=2.0, forced_bear=True)
        assert clamped == pytest.approx(0.10)  # bear floor wins

    @patch.object(wo, "_load_predictor_params_from_s3")
    def test_clamp_never_unprotects(self, m):
        # Wire 4 risk-OFF already more aggressive than the bear floor →
        # min() keeps the more-protective value (never raises threshold).
        m.return_value = {
            "veto_confidence": 0.30, "regime_veto_enabled": True,
            "regime_veto_cap": 0.30, "regime_forced_bear_enabled": True,
        }
        # intensity_z=-2 → adj -0.30 → threshold 0.00; bear_floor = 0.30-0.30 = 0.00
        out = get_veto_threshold("b", regime_intensity_z=-2.0, forced_bear=True)
        assert out == pytest.approx(0.0)


class TestMergePredictions:
    """Supplemental-scoring merge: new + existing → union, re-ranked."""

    def _pred(self, ticker: str, alpha: float) -> dict:
        return {
            "ticker": ticker,
            "predicted_alpha": alpha,
            "predicted_direction": "UP" if alpha >= 0 else "DOWN",
            "prediction_confidence": 0.55,
            "combined_rank": None,
        }

    def test_empty_existing_returns_new_with_ranks(self):
        new = [self._pred("A", 0.02), self._pred("B", -0.01)]
        merged = _merge_predictions(new, [])
        assert len(merged) == 2
        # Sorted by alpha desc: A(+0.02) rank 1, B(-0.01) rank 2
        assert merged[0]["ticker"] == "A" and merged[0]["combined_rank"] == 1
        assert merged[1]["ticker"] == "B" and merged[1]["combined_rank"] == 2

    def test_union_preserves_existing_non_overlapping(self):
        existing = [self._pred("X", 0.05), self._pred("Y", 0.03)]
        new = [self._pred("Z", 0.04)]
        merged = _merge_predictions(new, existing)
        tickers = [p["ticker"] for p in merged]
        assert set(tickers) == {"X", "Y", "Z"}
        # Rank recomputed across union: X(0.05), Z(0.04), Y(0.03)
        rank_by_ticker = {p["ticker"]: p["combined_rank"] for p in merged}
        assert rank_by_ticker == {"X": 1, "Z": 2, "Y": 3}

    def test_new_overrides_existing_on_collision(self):
        existing = [self._pred("A", 0.01)]
        new = [self._pred("A", 0.10)]  # new wins
        merged = _merge_predictions(new, existing)
        assert len(merged) == 1
        assert merged[0]["predicted_alpha"] == 0.10
        assert merged[0]["combined_rank"] == 1

    def test_rank_recomputed_across_full_union(self):
        existing = [self._pred(f"E{i}", 0.02 - i * 0.001) for i in range(3)]  # E0 best
        new = [self._pred("N0", 0.025), self._pred("N1", 0.018)]
        merged = _merge_predictions(new, existing)
        # Expected order by alpha desc: N0(.025), E0(.02), E1(.019), N1(.018), E2(.018)
        expected_top = ["N0", "E0", "E1"]
        top3 = [p["ticker"] for p in merged[:3]]
        assert top3 == expected_top
        # Ranks are contiguous 1..N
        ranks = sorted(p["combined_rank"] for p in merged)
        assert ranks == list(range(1, len(merged) + 1))

    def test_case_insensitive_ticker_keys(self):
        existing = [self._pred("aapl", 0.01)]
        new = [self._pred("AAPL", 0.05)]  # should collide
        merged = _merge_predictions(new, existing)
        assert len(merged) == 1  # merged, not both
        assert merged[0]["predicted_alpha"] == 0.05


class TestCoverageGuard:
    """Hard-fail when buy_candidates has tickers not scored by predictor.

    Defense-in-depth: the Step Function coverage-gap Choice state should
    normally re-invoke the predictor with --tickers to fill the gap before
    this write runs. This assertion catches any regression of that wiring
    instead of letting the executor see a partial predictions.json.
    """

    def _ctx(self, predictions, signals_data, explicit=False):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-04-20",
            bucket="bucket",
            dry_run=True,  # skip write/email so we isolate the guard
            predictions=list(predictions),
            signals_data=signals_data,
            explicit_tickers=[],  # first-run path (no merge)
        )
        if explicit:
            ctx.explicit_tickers = [p["ticker"] for p in predictions]
        return ctx

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_passes_when_all_buy_candidates_scored(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": -0.01, "combined_rank": 2,
                 "predicted_direction": "DOWN", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": [{"ticker": "A"}, {"ticker": "B"}]},
        )
        # Should not raise
        wo.run(ctx)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_raises_on_missing_buy_candidates(self, _m1, _m2):
        from inference.pipeline import PipelineHardFail
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={
                "buy_candidates": [
                    {"ticker": "A"},
                    {"ticker": "SNDK"},  # unscored — today's bug
                    {"ticker": "WDC"},   # unscored
                ],
            },
        )
        with pytest.raises(PipelineHardFail) as exc:
            wo.run(ctx)
        assert "SNDK" in str(exc.value)
        assert "WDC" in str(exc.value)
        assert "Coverage gap" in str(exc.value)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_empty_buy_candidates_is_no_op(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": []},
        )
        # Should not raise
        wo.run(ctx)

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_missing_signals_data_is_no_op(self, _m1, _m2):
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={},
        )
        # Should not raise
        wo.run(ctx)


class TestInferenceCoverageDenominator:
    """config#1075: run(ctx) persists the tradable-universe coverage denominator.

    n_universe = |signals.json universe| and n_universe_covered = universe
    tickers that got a prediction, so the report card's inference_coverage can
    grade covered/universe ∈ [0,1] instead of a permanent N/A.
    """

    def _ctx(self, predictions, signals_data):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-06-14",
            bucket="bucket",
            dry_run=True,  # metrics print to stdout; no S3 write
            predictions=list(predictions),
            signals_data=signals_data,
            explicit_tickers=[],
        )

    @staticmethod
    def _metrics(capsys):
        out = capsys.readouterr().out
        return json.loads(out.split("=== METRICS (dry-run) ===\n")[1])

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_counts_universe_and_covered(self, _m1, _m2, capsys):
        # universe of 4; only A/B/C scored → covered 3, universe 4.
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": 0.01, "combined_rank": 2,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "C", "predicted_alpha": 0.01, "combined_rank": 3,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"universe": ["A", "B", "C", "D"], "buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 4
        assert m["n_universe_covered"] == 3  # D unscored

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_universe_dict_entries_normalized(self, _m1, _m2, capsys):
        # universe entries as {ticker: …} dicts must be counted too.
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"universe": [{"ticker": "A"}, {"ticker": "B"}], "buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 2
        assert m["n_universe_covered"] == 1

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_empty_universe_records_zero(self, _m1, _m2, capsys):
        # No universe in signals.json → 0 (evaluator keeps honest N/A, no /0).
        ctx = self._ctx(
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={"buy_candidates": []},
        )
        wo.run(ctx)
        m = self._metrics(capsys)
        assert m["n_universe"] == 0
        assert m["n_universe_covered"] == 0


class TestGbmVetoConfidenceFloor:
    """Pin the confidence floor on gbm_veto computation.

    The binary UP/DOWN output forces every prediction to one extreme even
    when the underlying alpha estimate is near zero. Without a confidence
    floor, low-confidence DOWN predictions (near coin-flip on the calibrator)
    fire ``gbm_veto`` and over-block research's ENTER candidates. 2026-05-11
    incident: 15 of 30 vetos with mean DOWN confidence ~0.75; 8 of 28
    research ENTERs blocked by predictor.

    The fix gates ``gbm_veto`` on ``prediction_confidence >= veto_thresh``
    in addition to the existing ``alpha < 0`` and ``cr > n_preds/2``
    criteria. ``veto_thresh`` is the regime-adjusted value from
    ``get_veto_threshold`` (sourced from ``config/predictor_params.json``'s
    ``veto_confidence`` with ``cfg.MIN_CONFIDENCE`` fallback).

    Confidence semantics post-2026-05-12: ``|p_up - 0.5| * 2`` ∈ [0, 1]
    (ROADMAP L1615). Threshold values below are in the new convention.
    """

    def _ctx(self, predictions, signals_data=None):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-05-11",
            bucket="bucket",
            dry_run=True,
            predictions=list(predictions),
            signals_data=signals_data or {"buy_candidates": []},
            explicit_tickers=[],
        )
        return ctx

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_low_confidence_down_does_not_veto(self, _m1, _m2):
        """A DOWN prediction below the regime-adjusted confidence floor
        must NOT trigger ``gbm_veto`` — near coin-flip is binary noise."""
        # Two DOWN preds both in bottom half; one above floor, one below.
        # Confidence in post-2026-05-12 semantics: |p_up - 0.5| * 2.
        preds = [
            {"ticker": "A", "predicted_alpha": 0.05, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "B", "predicted_alpha": 0.03, "combined_rank": 2,
             "predicted_direction": "UP", "prediction_confidence": 0.60},
            {"ticker": "C", "predicted_alpha": -0.02, "combined_rank": 3,
             "predicted_direction": "DOWN", "prediction_confidence": 0.20},
            {"ticker": "D", "predicted_alpha": -0.08, "combined_rank": 4,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        veto_by_ticker = {p["ticker"]: p["gbm_veto"] for p in ctx.predictions}
        # A, B: UP (alpha>0) → no veto regardless of confidence
        assert veto_by_ticker["A"] is False
        assert veto_by_ticker["B"] is False
        # C: DOWN + bottom half + conf 0.20 < threshold 0.40 → NO veto (the fix)
        assert veto_by_ticker["C"] is False, (
            "DOWN at conf=0.20 below threshold=0.40 must NOT fire gbm_veto — "
            "near coin-flip is binary noise, not a high-conviction bearish call"
        )
        # D: DOWN + bottom half + conf 0.90 >= 0.40 → veto fires
        assert veto_by_ticker["D"] is True, (
            "DOWN at conf=0.90 above threshold=0.40 must fire gbm_veto — "
            "this is the high-conviction bearish signal we still want to block on"
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_top_half_rank_never_vetoes(self, _m1, _m2):
        """The cross-sectional rank gate still applies — top-half ranks
        skip the veto regardless of confidence or direction. Pinned so a
        future refactor doesn't accidentally drop the rank gate while
        adding the confidence gate."""
        preds = [
            # DOWN, top half (rank 1 of 4), high conf → still no veto
            {"ticker": "X", "predicted_alpha": -0.01, "combined_rank": 1,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
            {"ticker": "Y", "predicted_alpha": 0.05, "combined_rank": 2,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "Z", "predicted_alpha": -0.05, "combined_rank": 3,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
            {"ticker": "W", "predicted_alpha": -0.06, "combined_rank": 4,
             "predicted_direction": "DOWN", "prediction_confidence": 0.90},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        veto_by_ticker = {p["ticker"]: p["gbm_veto"] for p in ctx.predictions}
        assert veto_by_ticker["X"] is False  # top half (rank 1 of 4)
        assert veto_by_ticker["Z"] is True   # bottom half, high conf, DOWN
        assert veto_by_ticker["W"] is True

    @patch.object(wo, "get_veto_threshold", return_value=0.40)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_missing_confidence_field_does_not_veto(self, _m1, _m2):
        """If ``prediction_confidence`` is absent (defensive — never
        expected post-PR-#196's coverage gates), treat as 0 → never
        veto. Fail-safe direction: under-veto rather than over-veto
        when the field shape changes."""
        preds = [
            {"ticker": "A", "predicted_alpha": -0.05, "combined_rank": 2,
             "predicted_direction": "DOWN"},  # no prediction_confidence key
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        assert ctx.predictions[0]["gbm_veto"] is False

    @patch.object(wo, "get_veto_threshold", return_value=0.10)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_threshold_is_regime_adapted_via_get_veto_threshold(self, _m1, _m2):
        """The threshold flows from ``get_veto_threshold(market_regime=…)``
        which applies the bull/bear adjustments. Pin that the per-ticker
        veto uses the regime-adjusted value, not a hardcoded constant."""
        preds = [
            # DOWN at conf=0.20 — would NOT veto at threshold 0.40 (bull),
            # but WOULD veto at threshold 0.10 (caution, this test's regime).
            {"ticker": "A", "predicted_alpha": 0.05, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.80},
            {"ticker": "B", "predicted_alpha": -0.02, "combined_rank": 2,
             "predicted_direction": "DOWN", "prediction_confidence": 0.20},
        ]
        ctx = self._ctx(preds)
        wo.run(ctx)
        assert ctx.predictions[1]["gbm_veto"] is True, (
            "At threshold 0.10, conf=0.20 should fire the veto"
        )


class TestReadExistingPredictions:
    """_read_existing_predictions — S3 read with clean 404 handling."""

    def test_returns_empty_on_nosuchkey(self):
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
        )
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = err
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = _read_existing_predictions("bucket", "2026-04-20")
        assert result == []

    def test_reraises_on_other_client_errors(self):
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "InternalError", "Message": "boom"}}, "GetObject"
        )
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = err
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with pytest.raises(ClientError):
                _read_existing_predictions("bucket", "2026-04-20")

    def test_returns_predictions_on_success(self):
        payload = json.dumps({
            "date": "2026-04-20",
            "predictions": [
                {"ticker": "AAPL", "predicted_alpha": 0.02},
                {"ticker": "MSFT", "predicted_alpha": -0.01},
            ],
        }).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = payload
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = _read_existing_predictions("bucket", "2026-04-20")
        assert len(result) == 2
        assert [p["ticker"] for p in result] == ["AAPL", "MSFT"]


class TestSubstrateInventoryKeys:
    """`predictor_decisions` row in alpha_engine_lib.transparency_inventory
    asserts l1_ic, l2_ic, confidence_calibration keys are present in
    predictor/metrics/latest.json. These tests pin the contract."""

    def _ctx(self):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-05-09",
            bucket="bucket",
            dry_run=True,
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
            ],
            signals_data={},
            explicit_tickers=[],
            inference_mode="meta",
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta")
    def test_meta_mode_emits_inventory_keys(self, mock_meta, mock_write, _v):
        mock_meta.return_value = {
            "trained_date": "2026-05-09",
            "promoted": True,
            "meta_val_ic": 0.132,
            "momentum_test_ic": 0.071,
            "volatility_test_ic": 0.043,
            "research_calibrator_n_samples": 4200,
            "isotonic_ece_before": 0.087,
            "isotonic_ece_after": 0.021,
            "isotonic_n_samples": 1500,
        }
        wo.run(self._ctx())

        assert mock_write.called
        metrics = mock_write.call_args.args[3]
        assert "l1_ic" in metrics
        assert "l2_ic" in metrics
        assert "confidence_calibration" in metrics
        assert metrics["l1_ic"] == {
            "momentum": 0.071,
            "volatility": 0.043,
            "research_calibrator": None,
        }
        assert metrics["l2_ic"] == 0.132
        assert metrics["confidence_calibration"] == {
            "method": "isotonic",
            "ece_before": 0.087,
            "ece_after": 0.021,
            "n_samples": 1500,
        }

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta")
    def test_meta_mode_keys_present_when_manifest_partial(self, mock_meta, mock_write, _v):
        # Older manifests written before isotonic_calibrator landed lack the
        # ECE fields. Substrate check asserts presence (not non-null), so the
        # keys must still appear with None values rather than be omitted.
        mock_meta.return_value = {
            "trained_date": "2026-04-15",
            "promoted": True,
            "meta_val_ic": 0.10,
            "momentum_test_ic": 0.05,
            "volatility_test_ic": 0.03,
        }
        wo.run(self._ctx())

        metrics = mock_write.call_args.args[3]
        assert "l1_ic" in metrics
        assert "l2_ic" in metrics
        assert "confidence_calibration" in metrics
        assert metrics["confidence_calibration"]["ece_before"] is None
        assert metrics["confidence_calibration"]["ece_after"] is None

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_load_gbm_meta", return_value={})
    def test_non_meta_mode_does_not_emit_keys(self, _m, mock_write, _v):
        from inference.pipeline import PipelineContext
        ctx = PipelineContext(
            date_str="2026-05-09",
            bucket="bucket",
            dry_run=True,
            predictions=[],
            signals_data={},
            explicit_tickers=[],
            inference_mode="mse",
        )
        wo.run(ctx)
        metrics = mock_write.call_args.args[3]
        # Keys are scoped to meta mode — legacy modes don't have a Layer-2 IC
        # to report, so they should be absent rather than null.
        assert "l1_ic" not in metrics
        assert "l2_ic" not in metrics
        assert "confidence_calibration" not in metrics


class TestEmailGating:
    """Email is sent on exactly the terminal invocation — first run when
    no coverage-gap re-invoke is expected, or the supplemental re-invoke
    itself. Prevents duplicate morning briefings (two emails on 2026-04-24
    incident)."""

    def _ctx(self, *, explicit_tickers):
        from inference.pipeline import PipelineContext
        return PipelineContext(
            date_str="2026-04-24",
            bucket="bucket",
            dry_run=False,
            predictions=[
                {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
                 "predicted_direction": "UP", "prediction_confidence": 0.55},
                {"ticker": "B", "predicted_alpha": -0.01, "combined_rank": 2,
                 "predicted_direction": "DOWN", "prediction_confidence": 0.55},
            ],
            signals_data={},  # empty so the hard-fail coverage guard passes
            explicit_tickers=explicit_tickers,
        )

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_first_invocation_no_gap_sends(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.return_value = {
            "has_gap": False, "missing_count": 0, "missing_tickers": [],
        }
        wo.run(self._ctx(explicit_tickers=[]))
        assert mock_send.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_first_invocation_with_gap_defers(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.return_value = {
            "has_gap": True, "missing_count": 7,
            "missing_tickers": ["VLO", "LLY", "XEL", "BIIB", "ROST", "SNDK", "WDC"],
        }
        wo.run(self._ctx(explicit_tickers=[]))
        assert not mock_send.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "_read_existing_predictions", return_value=[])
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_supplemental_invocation_always_sends(
        self, mock_delta, mock_send, _r, _w, _m1, _m2,
    ):
        # Even if the coverage-delta would report a gap, the supplemental
        # invocation is terminal — it must send the email regardless.
        mock_delta.return_value = {
            "has_gap": True, "missing_count": 1, "missing_tickers": ["ZZZ"],
        }
        wo.run(self._ctx(explicit_tickers=["VLO", "LLY"]))
        assert mock_send.called
        # And compute_coverage_delta should not be consulted on the
        # supplemental path (we already know we're terminal).
        assert not mock_delta.called

    @patch.object(wo, "get_veto_threshold", return_value=0.65)
    @patch.object(wo, "_load_gbm_meta", return_value={})
    @patch.object(wo, "write_predictions")
    @patch.object(wo, "send_predictor_email", return_value=True)
    @patch("inference.coverage_check.compute_coverage_delta")
    def test_coverage_delta_failure_fails_open(
        self, mock_delta, mock_send, _w, _m1, _m2,
    ):
        mock_delta.side_effect = RuntimeError("S3 transient")
        wo.run(self._ctx(explicit_tickers=[]))
        # If we can't determine whether a re-invoke is coming, send the
        # email rather than risk silencing the morning briefing entirely.
        assert mock_send.called


# ── L234: morning email surfaces effective optimizer params ──────────────


class TestMorningEmailOptimizerParamsBlock:
    """ROADMAP L234 morning-email side — the predictor email surfaces
    the live `config/executor_params.json` block alongside the briefing
    so the operator sees effective min_score_to_enter /
    max_position_pct / atr_multiplier without tailing executor.log.

    Best-effort + degrades gracefully on missing artifact.
    """

    def _build(self, *, bucket=None):
        # Minimal happy-path inputs that exercise _build_predictor_email.
        predictions = [
            {"ticker": "A", "predicted_alpha": 0.01, "combined_rank": 1,
             "predicted_direction": "UP", "prediction_confidence": 0.7,
             "p_up": 0.65, "p_down": 0.35},
        ]
        metrics = {"model_version": "v1", "ic_30d": 0.10, "inference_mode": "meta"}
        return wo._build_predictor_email(
            predictions, metrics, "2026-05-22",
            signals_data=None, veto_threshold=0.6, bucket=bucket,
        )

    def test_no_bucket_skips_block(self):
        subject, html, plain = self._build(bucket=None)
        assert "Effective Optimizer Params" not in html
        assert "EFFECTIVE OPTIMIZER PARAMS" not in plain

    def test_bucket_supplied_but_artifact_missing_silent(self):
        with patch.object(wo, "_load_executor_params_for_email", return_value=None):
            subject, html, plain = self._build(bucket="b")
        assert "Effective Optimizer Params" not in html
        assert "EFFECTIVE OPTIMIZER PARAMS" not in plain

    def test_artifact_present_surfaces_block(self):
        fake_params = {
            "min_score": 65.0,
            "max_position_pct": 0.10,
            "atr_multiplier": 2.5,
            "profit_take_pct": 0.20,
            "updated_at": "2026-05-23",
            "best_sharpe": 1.42,
            "improvement_pct": 0.08,
            "n_combos_tested": 200,
        }
        with patch.object(wo, "_load_executor_params_for_email", return_value=fake_params):
            subject, html, plain = self._build(bucket="b")
        # HTML block markers
        assert "Effective Optimizer Params" in html
        assert "min_score_to_enter" in html
        assert "max_position_pct" in html
        assert "atr_multiplier" in html
        assert "2026-05-23" in html  # updated_at metadata
        assert "1.42" in html         # best_sharpe metadata
        assert "+8.0%" in html        # improvement_pct metadata
        # Plain text mirror
        assert "EFFECTIVE OPTIMIZER PARAMS" in plain
        assert "min_score_to_enter" in plain

    def test_manual_override_flag_surfaces_warning(self):
        fake_params = {
            "min_score": 55.0,
            "manual_override": True,
        }
        with patch.object(wo, "_load_executor_params_for_email", return_value=fake_params):
            subject, html, _plain = self._build(bucket="b")
        # Warning class present
        assert "manual override" in html.lower()

    def test_load_helper_swallows_s3_errors(self):
        """`_load_executor_params_for_email` must not raise — secondary
        observability path; primary briefing path must survive S3
        outage.
        """
        with patch("boto3.client", side_effect=RuntimeError("S3 down")):
            result = wo._load_executor_params_for_email("b")
        assert result is None
