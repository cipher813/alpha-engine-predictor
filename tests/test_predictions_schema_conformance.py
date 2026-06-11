"""Producer-side schema conformance for the predictions slot contract (M0, config#989).

Complements ``test_predictions_producer_contract.py`` (AST source-binding of the
dict literals): that test pins *which keys the source emits*; this one validates
*the assembled artifact* against the versioned Slot M contract in
``alpha_engine_lib.contracts`` (lib v0.59.0) — the same schema executor/backtester
consumer fixtures and external slot implementations validate against.

The payload is captured from the REAL envelope-assembly path
(``write_output.write_predictions`` with the S3 seam patched), not hand-built.
"""

from __future__ import annotations

import ast
import json
import os
from unittest.mock import MagicMock, patch

import pytest

contracts = pytest.importorskip(
    "alpha_engine_lib.contracts",
    reason="needs alpha-engine-lib[contracts] >= 0.59.0",
)

from inference.stages.write_output import write_predictions

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _representative_entry(**overrides) -> dict:
    """A per-ticker entry shaped like run_inference's emission (required core +
    the common optional fields a modern run carries)."""
    entry = {
        "ticker": "TEST",
        "predicted_direction": "UP",
        "prediction_confidence": 0.62,
        "predicted_alpha": 0.031,
        "combined_rank": 1,
        "gbm_veto": False,
        "momentum_veto": False,
        "p_up": 0.81,
        "p_flat": 0.0,
        "p_down": 0.19,
        "momentum_confirmation": 0.4,
        "momentum_20d": 0.02,
        "expected_move": 0.013,
        "research_calibrator_prob": 0.6,
        "predicted_alpha_std": 0.012,
        "barrier_win_prob": 0.55,
        "stance": "momentum",
        "stance_loadings": {"momentum": 0.7, "value": 0.1, "quality": 0.1, "catalyst": 0.1},
        "stance_source": "pillar_based",
        "catalyst_date": None,
        "meta_model_version": "v3.0",
        "price_data_age_days": 0,
        "watchlist_source": "tracked",
    }
    entry.update(overrides)
    return entry


def _capture_written_envelope(predictions: list[dict]) -> dict:
    """Run write_predictions through the real assembly path; return the dated-key payload."""
    with patch.dict("sys.modules", {"boto3": MagicMock()}), patch(
        "inference.stages.write_output._s3_put_json"
    ) as mock_put:
        write_predictions(predictions, "2026-06-11", "test-bucket", {"model_version": "v-test"})
    assert mock_put.call_count >= 1, "producer wrote nothing"
    # First write is the dated predictions key; body is the JSON string (4th arg).
    body = mock_put.call_args_list[0].args[3]
    return json.loads(body)


class TestProducerOutputConformsToSlotContract:
    def test_assembled_envelope_validates(self):
        envelope = _capture_written_envelope(
            [_representative_entry(), _representative_entry(ticker="TST2", combined_rank=2)]
        )
        contracts.validate("predictions", envelope)  # raises ContractViolation on drift

    def test_empty_run_still_conforms(self):
        contracts.validate("predictions", _capture_written_envelope([]))

    def test_broken_payload_fails_loud(self):
        """The red-fixture demo (config#989 closes-when): a dropped load-bearing
        field must produce a non-empty conformance error list, proving the gate
        actually fires."""
        envelope = _capture_written_envelope([_representative_entry()])
        del envelope["predictions"][0]["gbm_veto"]
        errors = contracts.conformance_errors("predictions", envelope)
        assert errors and "gbm_veto" in " ".join(errors)


class TestSchemaSourceParity:
    """Every schema-required per-item field must be emitted somewhere in the
    producer source — either in run_inference.py's prediction dict literal or
    assigned at write time in write_output.py (e.g. gbm_veto). Schema and
    producer can't silently drift apart."""

    @staticmethod
    def _emitted_keys(path: str) -> set[str]:
        src = open(path).read()
        keys: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Dict):
                keys |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
            # entry["key"] = ... write-time assignments
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.slice, ast.Constant)
                        and isinstance(tgt.slice.value, str)
                    ):
                        keys.add(tgt.slice.value)
        return keys

    def test_required_entry_fields_emitted_by_producer_source(self):
        schema = contracts.load_schema("predictions")
        required = set(schema["$defs"]["prediction_entry"]["required"])
        emitted = self._emitted_keys(
            os.path.join(_REPO, "inference", "stages", "run_inference.py")
        ) | self._emitted_keys(os.path.join(_REPO, "inference", "stages", "write_output.py"))
        missing = required - emitted
        assert not missing, (
            f"schema-required entry fields not emitted anywhere in the producer "
            f"source: {sorted(missing)} — additive-only evolution violated"
        )
