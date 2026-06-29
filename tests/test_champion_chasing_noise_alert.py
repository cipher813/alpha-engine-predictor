"""config#971 / L4539 — the champion realized-edge NOISE monitor must SURFACE its
verdict to the operator, not just write it to leaderboard.json.

Relative-best promotion has no absolute-significance gate and realized-edge auto-
demote is NOT live, so a champion with non-positive realized rank-IC keeps trading
capital until the operator manually remediates (rotate / retrain / hold). These
tests pin that ``chasing_noise == True`` fans out a `warning` alert with the
remediation philosophy + the dated detail key, and that nothing fires otherwise.
"""
from __future__ import annotations

import training.model_zoo as mz


def _capture_alerts(monkeypatch):
    sent: list[dict] = []
    import krepis.alerts as _alerts

    monkeypatch.setattr(_alerts, "publish", lambda **kw: sent.append(kw) or None)
    return sent


def _monitor(ic, n=30, weeks=6):
    return {
        "champion": {
            "realized_rank_ic": ic,
            "n_matured_outcomes": n,
            "n_weeks_coverage": weeks,
        }
    }


def test_chasing_noise_fires_warning_alert_with_remediation_and_detail(monkeypatch):
    sent = _capture_alerts(monkeypatch)
    mz._alert_champion_chasing_noise("alpha-engine-research", "2026-06-27", _monitor(-0.0023))

    assert len(sent) == 1
    a = sent[0]
    assert a["severity"] == "warning"
    # surfaces the remediation philosophy options
    assert "rotate" in a["message"] and "retrain" in a["message"] and "hold" in a["message"]
    # surfaces the realized IC and the dated detail key
    assert "-0.0023" in a["message"]
    assert "observe_leaderboard/2026-06-27.json" in a["message"]
    # deterministic dedup so one episode collapses to one operator alert
    assert a["dedup_key"] == "model_zoo_champion_chasing_noise_2026-06-27"


def test_alert_never_raises_on_publish_failure(monkeypatch):
    import krepis.alerts as _alerts

    def _boom(**kw):
        raise RuntimeError("sns down")

    monkeypatch.setattr(_alerts, "publish", _boom)
    # must swallow — alert is observability off a path that already promoted
    mz._alert_champion_chasing_noise("b", "2026-06-27", _monitor(-0.01))


def test_alert_never_raises_on_malformed_monitor(monkeypatch):
    _capture_alerts(monkeypatch)
    mz._alert_champion_chasing_noise("b", "2026-06-27", None)
    mz._alert_champion_chasing_noise("b", "2026-06-27", {})
