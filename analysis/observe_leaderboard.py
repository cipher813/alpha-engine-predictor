"""
analysis/observe_leaderboard.py — realized-edge measurement for the model zoo.

config#671/#673/#702/#1052/L4539. Originally the SLOWER confirm layer for the
Option-B observe tier; under the RELATIVE-BEST promotion rule (#1052) that tier is
retired (a beats-champion challenger PROMOTES, it does not enter an observe soak).
This module now serves TWO realized-edge reads, both pure MEASUREMENT (never
promote / demote / allocate):

1. ``build_champion_realized_monitor`` (the primary, live read) — the
   NOISE-CHASING MONITOR of the PROMOTED champion. Relative-best promotion accepts
   the best-of-N challenger over the incumbent each week without an absolute
   significance gate; the risk is chasing noise. This scores the LIVE champion's
   own ``predictor/predictions/{date}.json`` vs REALIZED 21d sector-neutral alpha
   over a trailing window. A weak/negative realized rank-IC ⇒ ``chasing_noise``
   True — an OBSERVABILITY/alarm signal that we may be over-fitting the selector,
   NOT a gate. ``training/model_zoo.py::run_rotation_and_select`` calls this each
   rotation after the promote and stashes the verdict on the leaderboard.

2. ``build_observe_leaderboard`` (retained) — scores any still-shadowed versions
   (``predictor/predictions_shadow/{version_id}/{date}.json``; the shadow runner
   still shadows registered CHALLENGER-stage versions) against realized 21d alpha
   alongside the champion. A diagnostic comparison surface; with the observe tier
   retired it no longer GATES any promotion (its ``ready_for_full_promotion`` is
   advisory only). Operator may still read it.

Reuses the realized-alpha machinery from ``analysis/variant_cutover_gate.py``
(``compute_realized_alpha_for_pairs``) and the S3 price / sector_map loaders from
``analysis/triple_barrier_cutover_runner.py`` so there is one realized-alpha
implementation, not a parallel shortcut (SOTA mirror, not re-invent).

S3 layout::

    predictor/model_zoo/observe_leaderboard/
      ├── {trading_day}.json   ← per-run leaderboard / champion monitor (dated)
      └── latest.json          ← single-fetch operator UX (mirror)

Both payloads carry ``calendar_date`` + ``trading_day`` per DATE_CONVENTIONS.md.

Usage (programmatic)::

    from analysis.observe_leaderboard import build_champion_realized_monitor
    mon = build_champion_realized_monitor(bucket="alpha-engine-research")

Usage (CLI)::

    python -m analysis.observe_leaderboard --bucket alpha-engine-research --window 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
from analysis.triple_barrier_cutover_runner import (  # noqa: E402
    _load_prices_from_s3,
    _load_sector_map_from_s3,
)
from analysis.variant_cutover_gate import (  # noqa: E402
    compute_realized_alpha_for_pairs,
)

# Canonical alpha field emitted by both live and shadow predictions
# (inference/stages/run_inference.py:835).
ALPHA_FIELD = "predicted_alpha"

# Forward-realized horizon — the canonical 21d label.
DEFAULT_HORIZON = 21

# Trailing prediction-history window. ~6 weeks gives a 4-week soak a buffer for
# the 21d realized-window close.
DEFAULT_WINDOW_DAYS = 42

# S3 prefixes.
_SHADOW_PREFIX = "predictor/predictions_shadow"
_LIVE_PREDICTIONS_PREFIX = "predictor/predictions"
OUTPUT_PREFIX = "predictor/model_zoo/observe_leaderboard"

# #702 GATE-2 pre-registered soak criteria — the realized-edge promotion floor.
MIN_SOAK_WEEKS = 4
MIN_REALIZED_OUTCOMES = 20
# Minimum matured pairs to compute a meaningful realized rank-IC at all.
_MIN_PAIRS_FOR_IC = 10


def _list_shadow_versions(bucket: str, s3_client=None) -> list[str]:
    """The version_ids under ``predictor/predictions_shadow/`` (one dir per
    shadowed version). Best-effort: returns [] on read failure."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    versions: set[str] = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket, Prefix=f"{_SHADOW_PREFIX}/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                pfx = cp.get("Prefix", "")
                # predictor/predictions_shadow/<vid>/  → <vid>
                vid = pfx[len(_SHADOW_PREFIX) + 1:].rstrip("/")
                if vid:
                    versions.add(vid)
    except Exception:  # noqa: BLE001 — empty leaderboard is a valid (no-data) result
        log.warning("observe_leaderboard: failed to list shadow versions", exc_info=True)
        return []
    return sorted(versions)


def _load_shadow_pairs(bucket: str, version_id: str, n_days: int, s3_client=None) -> list[dict]:
    """Flatten the last ``n_days`` of a version's shadow predictions into
    (date, ticker, predicted_alpha) rows (realized_alpha filled downstream)."""
    import boto3
    import datetime

    s3 = s3_client or boto3.client("s3")
    today = datetime.date.today()
    pairs: list[dict] = []
    for d_offset in range(n_days):
        d = today - datetime.timedelta(days=d_offset)
        key = f"{_SHADOW_PREFIX}/{version_id}/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001 — most days have no file; skip cleanly
            continue
        for p in data.get("predictions", []):
            pairs.append({
                "date": data.get("date") or d.isoformat(),
                "ticker": p.get("ticker"),
                ALPHA_FIELD: p.get(ALPHA_FIELD),
                "realized_alpha": None,
            })
    return pairs


def _load_live_pairs(bucket: str, n_days: int, s3_client=None) -> list[dict]:
    """Flatten the last ``n_days`` of LIVE champion predictions (the baseline the
    observe versions must beat on realized edge)."""
    import boto3
    import datetime

    s3 = s3_client or boto3.client("s3")
    today = datetime.date.today()
    pairs: list[dict] = []
    for d_offset in range(n_days):
        d = today - datetime.timedelta(days=d_offset)
        key = f"{_LIVE_PREDICTIONS_PREFIX}/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001
            continue
        for p in data.get("predictions", []):
            pairs.append({
                "date": data.get("date") or d.isoformat(),
                "ticker": p.get("ticker"),
                ALPHA_FIELD: p.get(ALPHA_FIELD),
                "realized_alpha": None,
            })
    return pairs


def _realized_rank_ic(pairs: list[dict]) -> tuple[float | None, int, int]:
    """Spearman rank-IC of ``predicted_alpha`` vs ``realized_alpha`` over pairs
    whose forward window has CLOSED (realized_alpha is finite). Rank-IC matches
    the canonical-alpha framework's OOS Spearman read (SOTA over raw Pearson on
    small matured samples). Returns ``(ic_or_None, n_matured, n_weeks_spanned)``.
    """
    import numpy as np
    from scipy.stats import spearmanr

    pred = np.array(
        [p.get(ALPHA_FIELD) if p.get(ALPHA_FIELD) is not None else np.nan for p in pairs],
        dtype=float,
    )
    real = np.array(
        [p.get("realized_alpha") if p.get("realized_alpha") is not None else np.nan for p in pairs],
        dtype=float,
    )
    mask = np.isfinite(pred) & np.isfinite(real)
    n = int(mask.sum())
    weeks = _distinct_weeks([p.get("date") for p, m in zip(pairs, mask) if m])
    if n < _MIN_PAIRS_FOR_IC:
        return None, n, weeks
    sp = spearmanr(pred[mask], real[mask])
    ic = float(sp.correlation) if np.isfinite(sp.correlation) else None
    return ic, n, weeks


def _distinct_weeks(dates) -> int:
    """Count distinct ISO (year, week) buckets among the prediction dates — the
    soak-coverage breadth (>= MIN_SOAK_WEEKS gates promotion-readiness)."""
    import datetime

    weeks: set = set()
    for d in dates:
        if d is None:
            continue
        try:
            dt = datetime.date.fromisoformat(str(d)[:10])
        except Exception:  # noqa: BLE001
            continue
        iso = dt.isocalendar()
        weeks.add((iso[0], iso[1]))
    return len(weeks)


def _write_payload(payload: dict, bucket: str, trading_day: str | None,
                   *, write_latest: bool, s3_client=None) -> str:
    """Write a realized-edge payload to ``observe_leaderboard/{trading_day}.json``
    (+ ``latest.json`` mirror). Single write chokepoint shared by the champion
    monitor and the (retained) shadow leaderboard — keeps exactly two PUT sites in
    this file (pinned in tests/test_artifact_registry_coverage.py)."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    key = f"{OUTPUT_PREFIX}/{trading_day or 'latest'}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    log.info("observe_leaderboard written to s3://%s/%s", bucket, key)
    if write_latest:
        latest_key = f"{OUTPUT_PREFIX}/latest.json"
        s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
        log.info("observe_leaderboard mirrored to s3://%s/%s", bucket, latest_key)
    return key


def build_champion_realized_monitor(
    bucket: str | None = None,
    n_days: int = DEFAULT_WINDOW_DAYS,
    horizon_days: int = DEFAULT_HORIZON,
    *,
    date_str: str | None = None,
    write_to_s3: bool = True,
    write_latest: bool = True,
    s3_client=None,
    live_pairs: list[dict] | None = None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
) -> dict:
    """NOISE-CHASING MONITOR of the PROMOTED champion (config#671/#673/#1052).

    Relative-best promotion accepts the best-of-N challenger over the incumbent
    every week with no absolute-significance gate; the risk is chasing noise. This
    scores the LIVE champion's own ``predictor/predictions/{date}.json`` against
    REALIZED 21d sector-neutral alpha over a trailing window and flags
    ``chasing_noise`` when the realized rank-IC is non-positive (or unavailable
    once enough outcomes have matured). Pure OBSERVABILITY — it NEVER promotes,
    demotes, or allocates; the verdict is an alarm/diagnostic signal only.

    Test-injectable: ``live_pairs`` / ``prices_by_ticker`` / ``sector_map`` bypass
    S3 when provided.
    """
    bucket = bucket or cfg.S3_BUCKET
    horizon_days = horizon_days or int(getattr(cfg, "FORWARD_DAYS", 21))

    calendar_date = None
    trading_day = date_str
    try:
        from alpha_engine_lib.dates import now_dual
        dual = now_dual()
        calendar_date = dual.calendar_date
        trading_day = trading_day or (
            dual.trading_day.isoformat() if hasattr(dual.trading_day, "isoformat")
            else str(dual.trading_day)
        )
    except Exception:  # noqa: BLE001 — date resolution must not block the monitor
        log.warning("champion_monitor: now_dual failed; date fields best-effort", exc_info=True)

    if live_pairs is None:
        live_pairs = _load_live_pairs(bucket, n_days, s3_client=s3_client)

    if prices_by_ticker is None or sector_map is None:
        tickers = {p["ticker"] for p in live_pairs if p.get("ticker")}
        sector_map = sector_map or _load_sector_map_from_s3(bucket, s3_client=s3_client)
        bench_symbols = set(sector_map.values()) | {"SPY"}
        prices_by_ticker = prices_by_ticker or _load_prices_from_s3(
            bucket, tickers | bench_symbols, s3_client=s3_client,
        )

    compute_realized_alpha_for_pairs(
        live_pairs, horizon_days=horizon_days,
        prices_by_ticker=prices_by_ticker, sector_map=sector_map,
    )
    champ_ic, champ_n, champ_weeks = _realized_rank_ic(live_pairs)

    # chasing_noise verdict: only assertable once enough outcomes have matured.
    # Non-positive realized rank-IC on >= MIN_REALIZED_OUTCOMES matured pairs ⇒ the
    # promoted champion shows no realized predictive edge → likely chasing noise.
    if champ_ic is None or champ_n < MIN_REALIZED_OUTCOMES:
        chasing_noise = None
        verdict_reason = (
            f"insufficient matured outcomes ({champ_n}; need >= {MIN_REALIZED_OUTCOMES}) "
            f"to assert a realized-edge verdict"
        )
    elif champ_ic <= 0.0:
        chasing_noise = True
        verdict_reason = (
            f"NOISE WATCH: champion realized rank-IC {champ_ic:.4f} <= 0 over "
            f"{champ_n} matured outcomes / {champ_weeks} weeks — relative-best "
            f"promotion may be chasing noise (observability, NOT a gate)"
        )
    else:
        chasing_noise = False
        verdict_reason = (
            f"healthy: champion realized rank-IC {champ_ic:.4f} > 0 over {champ_n} "
            f"matured outcomes / {champ_weeks} weeks"
        )

    payload = {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "date": trading_day,
        "kind": "champion_realized_monitor",
        "window_days": n_days,
        "horizon_days": horizon_days,
        "champion": {
            "realized_rank_ic": champ_ic,
            "n_matured_outcomes": champ_n,
            "n_weeks_coverage": champ_weeks,
        },
        "chasing_noise": chasing_noise,
        "verdict_reason": verdict_reason,
        "note": (
            "Relative-best promotion (config#671/#673/#1052): the best-of-N "
            "challenger beating the champion by margin auto-promotes weekly; DSR is "
            "observability, not a gate. This monitor flags whether that selection is "
            "chasing noise via the PROMOTED champion's realized 21d rank-IC. It never "
            "promotes/demotes/allocates."
        ),
    }

    if write_to_s3:
        payload["s3_key"] = _write_payload(
            payload, bucket, trading_day, write_latest=write_latest, s3_client=s3_client,
        )

    log.info(
        "champion_realized_monitor: realized rank-IC=%s over %d matured / %d weeks; "
        "chasing_noise=%s",
        champ_ic, champ_n, champ_weeks, chasing_noise,
    )
    return payload


def build_observe_leaderboard(
    bucket: str | None = None,
    n_days: int = DEFAULT_WINDOW_DAYS,
    horizon_days: int = DEFAULT_HORIZON,
    *,
    write_to_s3: bool = True,
    write_latest: bool = True,
    s3_client=None,
    shadow_pairs_by_version: dict | None = None,
    live_pairs: list[dict] | None = None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
    date_str: str | None = None,
) -> dict:
    """Score every shadow-run version against realized 21d alpha + emit the
    leaderboard with a per-version ``ready_for_full_promotion`` verdict.

    REALIZED-edge driven (NOT training DSR): a version is promotion-ready iff it
    has >= ``MIN_SOAK_WEEKS`` weeks coverage, >= ``MIN_REALIZED_OUTCOMES`` matured
    outcomes, AND its realized rank-IC >= the champion's over the same window.
    Measurement only — it NEVER promotes or allocates.

    Test-injectable: ``shadow_pairs_by_version`` / ``live_pairs`` /
    ``prices_by_ticker`` / ``sector_map`` bypass S3 when provided.
    """
    bucket = bucket or cfg.S3_BUCKET
    horizon_days = horizon_days or int(getattr(cfg, "FORWARD_DAYS", 21))

    # Resolve dual dates (best-effort; tests pass date_str directly).
    calendar_date = None
    trading_day = date_str
    try:
        from alpha_engine_lib.dates import now_dual
        dual = now_dual()
        calendar_date = dual.calendar_date
        trading_day = trading_day or (
            dual.trading_day.isoformat() if hasattr(dual.trading_day, "isoformat")
            else str(dual.trading_day)
        )
    except Exception:  # noqa: BLE001 — date resolution must not block the leaderboard
        log.warning("observe_leaderboard: now_dual failed; date fields best-effort", exc_info=True)

    # 1) Gather shadow + live prediction rows.
    if shadow_pairs_by_version is None:
        versions = _list_shadow_versions(bucket, s3_client=s3_client)
        shadow_pairs_by_version = {
            vid: _load_shadow_pairs(bucket, vid, n_days, s3_client=s3_client)
            for vid in versions
        }
    if live_pairs is None:
        live_pairs = _load_live_pairs(bucket, n_days, s3_client=s3_client)

    # 2) Load prices + sector_map once for the whole universe (shared across all
    #    versions; realized alpha is version-agnostic given (date, ticker)).
    all_pairs_for_universe = list(live_pairs)
    for _vp in shadow_pairs_by_version.values():
        all_pairs_for_universe.extend(_vp)
    if prices_by_ticker is None or sector_map is None:
        tickers = {p["ticker"] for p in all_pairs_for_universe if p.get("ticker")}
        sector_map = sector_map or _load_sector_map_from_s3(bucket, s3_client=s3_client)
        bench_symbols = set(sector_map.values()) | {"SPY"}
        prices_by_ticker = prices_by_ticker or _load_prices_from_s3(
            bucket, tickers | bench_symbols, s3_client=s3_client,
        )

    # 3) Fill realized alpha for the champion + each version's rows.
    compute_realized_alpha_for_pairs(
        live_pairs, horizon_days=horizon_days,
        prices_by_ticker=prices_by_ticker, sector_map=sector_map,
    )
    champ_ic, champ_n, champ_weeks = _realized_rank_ic(live_pairs)

    entries: list[dict] = []
    for vid, vpairs in shadow_pairs_by_version.items():
        compute_realized_alpha_for_pairs(
            vpairs, horizon_days=horizon_days,
            prices_by_ticker=prices_by_ticker, sector_map=sector_map,
        )
        v_ic, v_n, v_weeks = _realized_rank_ic(vpairs)
        beats = (
            v_ic is not None and champ_ic is not None and v_ic >= champ_ic
        )
        enough_data = (v_weeks >= MIN_SOAK_WEEKS and v_n >= MIN_REALIZED_OUTCOMES)
        ready = bool(enough_data and beats)
        if v_ic is None:
            verdict_reason = (
                f"insufficient matured outcomes: {v_n} (need >= {_MIN_PAIRS_FOR_IC} "
                f"to compute realized IC; >= {MIN_REALIZED_OUTCOMES} to be promotion-ready)"
            )
        elif not enough_data:
            verdict_reason = (
                f"soak too young: {v_weeks} weeks / {v_n} matured outcomes "
                f"(need >= {MIN_SOAK_WEEKS} weeks AND >= {MIN_REALIZED_OUTCOMES})"
            )
        elif not beats:
            verdict_reason = (
                f"realized rank-IC {v_ic:.4f} < champion {champ_ic:.4f} over the soak "
                f"— hold; realized edge not demonstrated"
            )
        else:
            verdict_reason = (
                f"READY: realized rank-IC {v_ic:.4f} >= champion {champ_ic:.4f} over "
                f"{v_weeks} weeks / {v_n} matured outcomes. Operator may promote: "
                f"python -m model.registry --bucket {bucket} --promote {vid}"
            )
        entries.append({
            "version_id": vid,
            "realized_rank_ic": v_ic,
            "n_matured_outcomes": v_n,
            "n_weeks_coverage": v_weeks,
            "beats_champion_realized": beats,
            "ready_for_full_promotion": ready,
            "verdict_reason": verdict_reason,
        })

    # Rank promotion-ready first, then by realized IC desc (None last).
    entries.sort(
        key=lambda e: (
            not e["ready_for_full_promotion"],
            -(e["realized_rank_ic"] if e["realized_rank_ic"] is not None else -1e9),
        )
    )

    payload = {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "date": trading_day,
        "window_days": n_days,
        "horizon_days": horizon_days,
        "soak_criteria": {
            "min_weeks": MIN_SOAK_WEEKS,
            "min_realized_outcomes": MIN_REALIZED_OUTCOMES,
            "gate": "realized rank-IC >= champion realized rank-IC (NOT training DSR)",
        },
        "champion": {
            "realized_rank_ic": champ_ic,
            "n_matured_outcomes": champ_n,
            "n_weeks_coverage": champ_weeks,
        },
        "entries": entries,
        "ready_version_ids": [e["version_id"] for e in entries if e["ready_for_full_promotion"]],
    }

    if write_to_s3:
        payload["s3_key"] = _write_payload(
            payload, bucket, trading_day, write_latest=write_latest, s3_client=s3_client,
        )

    log.info(
        "observe_leaderboard: %d version(s) scored; %d ready-for-full-promotion "
        "(champion realized rank-IC=%s over %d matured / %d weeks)",
        len(entries), len(payload["ready_version_ids"]), champ_ic, champ_n, champ_weeks,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bucket", default=None,
                        help=f"S3 bucket. Default: cfg.S3_BUCKET ({cfg.S3_BUCKET}).")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"Trailing prediction-history window in days. Default: {DEFAULT_WINDOW_DAYS}.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Forward-realized window in trading days. Default: {DEFAULT_HORIZON}.")
    parser.add_argument("--no-write", action="store_true",
                        help="Compute + print the leaderboard but do not write to S3.")
    args = parser.parse_args()

    result = build_observe_leaderboard(
        bucket=args.bucket, n_days=args.window, horizon_days=args.horizon,
        write_to_s3=not args.no_write,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
