"""Pins spot_train.sh to propagate PREDICTOR_DEFER_TRAINING_EMAIL into the
--full-only full-training workload env.

Brian's spec: the Saturday SF exports PREDICTOR_DEFER_TRAINING_EMAIL before
invoking spot_train.sh --full-only so the base champion-arch retrain defers its
per-run training email to the consolidated model-zoo digest. The full-training
heredoc is single-quoted (cannot interpolate), so the script computes an export
line in the dispatcher and prepends it to the heredoc body. This test catches a
future edit that drops the propagation.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def test_script_exists():
    assert _SCRIPT.is_file()


def test_computes_defer_export_from_dispatcher_env():
    text = _SCRIPT.read_text()
    # The dispatcher reads the var into a prependable export line.
    assert "PREDICTOR_DEFER_TRAINING_EMAIL" in text
    assert 'DEFER_EMAIL_EXPORT="export PREDICTOR_DEFER_TRAINING_EMAIL=' in text


def test_full_training_runssm_prepends_defer_export():
    text = _SCRIPT.read_text()
    # The full-training run_ssm body is prefixed with ${DEFER_EMAIL_EXPORT}.
    assert 'run_ssm "full-training" "${DEFER_EMAIL_EXPORT}${SHADOW_EXPORT}$(cat <<\'TRAIN\'' in text


def test_defer_export_empty_when_unset_keeps_bare_run_equivalent():
    # The else branch sets DEFER_EMAIL_EXPORT="" so a bare --full-only run (no
    # SF env) is byte-equivalent to before — no stray export injected.
    text = _SCRIPT.read_text()
    assert 'DEFER_EMAIL_EXPORT=""' in text
