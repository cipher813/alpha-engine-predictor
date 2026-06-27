"""training/model_zoo.py — declarative model-variant zoo + train driver (L4488c).

Model-rotation scaffolding (arc L4488). A **model spec** is a config OVERLAY
over the existing training knobs (``FORWARD_DAYS``, ``RESIDUAL_MOMENTUM_ENABLED``,
``XSEC_DEMEAN_ALPHA_ENABLED``, ``MODEL_VERSION_LABEL``, …). Running a spec trains
ONE variant and registers it as a CHALLENGER (via the capture-gap in
``meta_trainer``; challenger-first means it never overwrites the live champion).
The shadow runner then shadow-runs it and the leaderboard + net-of-cost scorer
rank it — so a variety of models can be rotated in/out as experiments are run.

Deliberately a THIN spec-overlay — NOT a generic ML platform. It reuses
``train_handler.main()`` unchanged and applies the spec's overrides around the
call via a save/restore context: the knobs are already module-level ``cfg``
constants read at call time (verified: all reads are ``cfg.X`` attribute access,
not bound-at-import), so mutating them here is how a spec takes effect without
threading a config object through the whole trainer.

Specs are declared in ``predictor.yaml`` under ``model_specs:``. Run one with
``python -m training.model_zoo --bucket B --spec <id>``; every active spec with
``--all-active`` (sequential); or the bounded weekly cadence with
``--weekly-rotation`` (L4488g) — trains only the ``MODEL_ZOO_WEEKLY_BUDGET``
(default 3) STALEST active specs so the zoo refreshes round-robin without
running every spec every week. Each spec is ~one full training run, so the
rotation is how the zoo stays current on a tenable compute budget.

Limitation (documented): the override only affects knobs read via ``cfg.X`` at
call time. A horizon change (``FORWARD_DAYS``) additionally needs any
import-time-DERIVED constant to be read at call time too — verify per-knob when
a spec exercises it (the 60d-target variant, L4488d).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
from pathlib import Path

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all entrypoints; mirrors training/train_handler.py +
# inference/handler.py). Module-top so import-time errors below (config
# resolution, the lazy train_handler/meta_trainer imports that pull in the
# heavy GBM stack) are captured by flow-doctor's ERROR handler.
#
# The model-zoo rotation is its OWN spot entrypoint (spot_train.sh
# --model-zoo-weekly stages /tmp/spot-model-zoo-weekly.py which does
# `from training.model_zoo import run_rotation_and_select`) — importing this
# module IS the wiring point for that path, AND the `python -m
# training.model_zoo` CLI hits the same module-top. Wiring here covers both
# without a per-entrypoint call. Zoo-orchestration crashes (spec resolution /
# selection / leaderboard, before the first per-spec train) are now captured;
# previously they were uncaptured because train_handler (the only wired
# entrypoint) is imported only LAZILY inside train_spec().
#
# Runs on EC2 spot (NOT Lambda) — no LAMBDA_TASK_ROOT, so the yaml path
# resolves via two-dirs-up from this file (training/model_zoo.py → repo root,
# where flow-doctor-model-zoo.yaml lives). The model-zoo yaml mirrors the
# training yaml's email sink AND the inference yaml's S3 sink so crash
# artifacts persist even when SMTP/email delivery is unavailable.
#
# exclude_patterns starts empty by deliberate convention; add patterns only
# after observing real ERROR-level noise during rotation runs.
from krepis.logging import setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = str(
    Path(__file__).resolve().parent.parent / "flow-doctor-model-zoo.yaml"
)
setup_logging(
    "predictor-model-zoo",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

import config as cfg

log = logging.getLogger(__name__)

# S3 prefixes (mirror model.registry defaults; kept local so the selection path
# has no import cycle with the trainer).
_REGISTRY_PREFIX = "predictor/registry"
_LEADERBOARD_PREFIX = "predictor/model_zoo/leaderboard"
# L4582(b): cumulative cross-rotation trial ledger — every candidate the zoo
# has ever evaluated, deduped by version_id. The count is the honest n_trials
# for DSR deflation (the per-run CPCV n_combos proxy understates the true
# search breadth once rotations accumulate).
_TRIAL_LOG_KEY = "predictor/model_zoo/trial_log.json"

# Only these cfg knobs may be overridden by a spec — fail loud on anything else
# so a spec can't set arbitrary attributes on the config module. Extend ONLY
# after confirming the trainer reads the knob via cfg.X at call time.
_ALLOWED_OVERRIDES = {
    "FORWARD_DAYS",
    "RESIDUAL_MOMENTUM_ENABLED",
    "MOMENTUM_L1_IN_META",
    "XSEC_DEMEAN_ALPHA_ENABLED",
    "MODEL_VERSION_LABEL",
    # L4565 SOTA directional-combine levers. Read via cfg.X at fit time in
    # meta_trainer (build_train_meta_features + the meta_model.fit standardize
    # arg). Default-preserving so the champion is byte-identical when unset.
    "EXPECTED_MOVE_IN_META",
    "META_STANDARDIZE_ENABLED",
    # Long-horizon validation: drop the research-derived meta features + skip
    # the signals-join so a 60/90d spec can train past the short signals.json
    # history (validate-only; the horizon filter keeps it out of the 21d slot).
    "RESEARCH_FEATURES_IN_META",
}

_SENTINEL = object()


class ModelSpecError(ValueError):
    """A spec is malformed, retired, missing, or sets a disallowed override."""


def resolve_spec(spec_id: str, specs: list | None = None) -> dict:
    """Return the active spec dict for ``spec_id`` or raise ModelSpecError."""
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    for s in specs:
        if s.get("id") == spec_id:
            status = s.get("status", "active")
            if status != "active":
                raise ModelSpecError(f"spec {spec_id!r} is not active (status={status!r})")
            return s
    raise ModelSpecError(f"spec {spec_id!r} not found in model_specs")


def _validate_overrides(overrides: dict) -> None:
    bad = sorted(set(overrides) - _ALLOWED_OVERRIDES)
    if bad:
        raise ModelSpecError(
            f"spec overrides {bad} not in the allowlist {sorted(_ALLOWED_OVERRIDES)} — "
            "add to _ALLOWED_OVERRIDES only after confirming the trainer reads the "
            "knob via cfg.X at call time."
        )


@contextlib.contextmanager
def spec_overrides(overrides: dict):
    """Temporarily set ``cfg`` attributes for the duration of a train call, then
    restore them (even on exception). Validates against the allowlist first."""
    _validate_overrides(overrides)
    prev = {k: getattr(cfg, k, _SENTINEL) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(cfg, k, v)
        yield
    finally:
        for k, v in prev.items():
            if v is _SENTINEL:
                delattr(cfg, k)  # was absent before → remove the override
            else:
                setattr(cfg, k, v)


def train_spec(
    spec_id: str,
    bucket: str,
    *,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    train_fn=None,
) -> dict:
    """Train the variant for ``spec_id`` with its overrides applied, registering
    it as a challenger. ``train_fn`` is injectable for tests (defaults to
    ``train_handler.main``). Returns the train result dict."""
    spec = resolve_spec(spec_id, specs)
    overrides = dict(spec.get("overrides", {}))
    # Always pin a label so the challenger is identifiable on the leaderboard;
    # default to the spec's declared label, else "spec-<id>".
    overrides.setdefault(
        "MODEL_VERSION_LABEL", spec.get("model_version_label", f"spec-{spec_id}")
    )
    if train_fn is None:
        # The zoo trains CHALLENGERS — each must NOT send its own per-run
        # training email (that was the per-challenger email storm). Bind
        # send_email=False onto the real trainer here; the consolidated zoo
        # digest (send_zoo_digest_email, end of run_rotation_and_select) is the
        # single email for the whole rotation. An injected test train_fn keeps
        # its own (send_email-free) signature — the partial is only the default.
        import functools
        from training.train_handler import main as _train_main
        train_fn = functools.partial(_train_main, send_email=False)
    log.info("model_zoo: training spec %s — overrides %s", spec_id, overrides)
    with spec_overrides(overrides):
        return train_fn(bucket, date_str=date_str, dry_run=dry_run)


def train_one_spec(
    spec_id: str,
    bucket: str,
    *,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    train_fn=None,
    s3=None,
) -> dict:
    """config#1083 — TRAIN exactly ONE spec as a challenger, fully isolated, for
    the SF parallel fan-out (one memory-isolated spot per spec).

    This is the per-spot entrypoint the ``ModelZooTrainMap`` iterations call. It
    applies the SAME G2 contract isolation the sequential rotation applies around
    the whole batch, but scoped to this single spec so the per-spot run is self-
    contained: snapshot+restore the live champion contract around the train (a
    challenger train overwrites the live manifest/feature_list unconditionally;
    restore it after — best-effort, never fatal). Training is UNCONDITIONALLY
    challenger-first since config#1052/#679 (the former G1 force-flag is gone), so
    a spec can never self-promote — selection is the only promoter.

    Raises on a REAL training failure so the Map iteration records THIS spec's
    failure (and only this spec's) without aborting siblings — the key robustness
    property. Returns the train result dict on success.
    """
    # G2 — snapshot the live contract before the challenger train overwrites it.
    # Only for the real trainer (train_fn is None) on a non-dry run; an injected
    # test train_fn does no live writes, so there's nothing to restore.
    saved_contract: dict = {}
    if not dry_run and train_fn is None:
        if s3 is None:
            try:
                import boto3
                s3 = boto3.client("s3")
            except Exception:  # noqa: BLE001 — G2 is best-effort
                log.warning("model_zoo train-spec G2: no S3 client — live-contract restore skipped", exc_info=True)
        if s3 is not None:
            saved_contract = _snapshot_live_contract(s3, bucket)
    try:
        return train_spec(
            spec_id, bucket, date_str=date_str, dry_run=dry_run,
            specs=specs, train_fn=train_fn,
        )
    finally:
        # G2 — always restore, even if the train raised mid-way.
        if saved_contract and s3 is not None:
            _restore_live_contract(s3, bucket, saved_contract)


def train_all_active(
    bucket: str,
    *,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    train_fn=None,
) -> dict:
    """Train every active spec sequentially; one spec's failure never aborts the
    rest (each is captured to the registry before the next runs)."""
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    active = [s["id"] for s in specs if s.get("status", "active") == "active" and s.get("id")]
    log.info("model_zoo: %d active spec(s) to train: %s", len(active), active)
    results: dict = {}
    for sid in active:
        try:
            results[sid] = train_spec(
                sid, bucket, date_str=date_str, dry_run=dry_run,
                specs=specs, train_fn=train_fn,
            )
        except Exception as exc:  # noqa: BLE001 — one variant must not block the zoo
            log.warning("model_zoo: spec %s failed (continuing): %s", sid, exc, exc_info=True)
            results[sid] = {"status": "error", "error": str(exc)}
    return results


def _resolve_label(spec: dict) -> str:
    """The registry ``model_version`` a spec's trained versions carry.

    Mirrors ``train_spec`` EXACTLY: a ``MODEL_VERSION_LABEL`` in the spec's own
    ``overrides`` wins (it's applied verbatim and `setdefault` won't replace it),
    else the spec's declared ``model_version_label``, else ``spec-<id>``. The
    trainer writes this as ``manifest["version"]`` → the registry's
    ``model_version`` (verified meta_trainer.py:3537), so it's how the rotation
    maps a registered version back to its spec.
    """
    ov = spec.get("overrides", {})
    if "MODEL_VERSION_LABEL" in ov:
        return ov["MODEL_VERSION_LABEL"]
    return spec.get("model_version_label", f"spec-{spec.get('id')}")


def select_rotation_specs(
    specs: list, registered_versions: list, budget: int,
) -> list[str]:
    """Pick the ``budget`` STALEST active spec ids for a weekly rotation.

    Staleness = the date of a spec's NEWEST registered version (via the registry
    lineage, grouped by ``model_version``). A spec with NO registered version is
    maximally stale (never trained → trained first). Staleness ALWAYS dominates;
    within an equal-staleness bucket an optional integer ``priority`` (default 0,
    higher = picked first) breaks the tie ahead of the spec id. Priority is a
    WITHIN-bucket tiebreak ONLY — it never overrides staleness, so a high-priority
    spec cannot starve the others out of the round-robin: once it has a fresh
    registered version the never-registered specs are more stale and sort ahead
    of it. Use it to steer the COLD START (all never-registered) toward the one
    promote-eligible / highest-value variant first. Ties after priority break on
    spec id for deterministic, reproducible selection. Pure (registry list
    injected) so the policy is unit-testable without S3.
    """
    active = [s for s in specs
              if s.get("status", "active") == "active" and s.get("id")]
    newest: dict[str, str] = {}
    for v in registered_versions:
        mv, d = v.get("model_version"), v.get("date")
        if mv and d and d > newest.get(mv, ""):
            newest[mv] = d

    def _key(s: dict):
        last = newest.get(_resolve_label(s))
        # never-registered (last is None) sorts first; then oldest date first;
        # then higher `priority` first (negated so larger sorts earlier); then id.
        return (last is not None, last or "", -int(s.get("priority", 0)), s["id"])

    return [s["id"] for s in sorted(active, key=_key)[:max(0, budget)]]


def list_rotation_spec_ids(
    bucket: str,
    *,
    budget: int | None = None,
    specs: list | None = None,
    registered_versions: list | None = None,
) -> list[str]:
    """config#1083 — the spec ids the rotation should train THIS run, as a flat
    list, for the SF ``ResolveZooSpecs`` dispatcher (its output ItemsPath feeds
    the per-spec ``Map``). Thin wrapper over ``select_rotation_specs`` reading the
    budget from cfg + the registry best-effort (a registry read failure → all
    specs look never-trained → first ``budget`` by id, never blocks the run).
    Returns [] if no active specs (the Map then runs zero iterations and select
    handles the empty pool)."""
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    if budget is None:
        budget = int(getattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 4))
    if registered_versions is None:
        registered_versions = _list_registry_versions(bucket)
    selected = select_rotation_specs(specs, registered_versions, budget)
    log.info("model_zoo list-rotation-specs: budget=%d → %s", budget, selected)
    return selected


def _list_registry_versions(bucket: str) -> list:
    """Best-effort registry enumeration for the rotation. A registry read
    failure must not block training — return [] (every spec then looks
    never-trained, so the scheduler still trains the first ``budget`` by id)."""
    try:
        import boto3

        from model.registry import list_versions
        return list_versions(boto3.client("s3"), bucket)
    except Exception:  # noqa: BLE001 — staleness is best-effort, never fatal
        log.warning(
            "model_zoo rotation: could not read registry — falling back to "
            "id-ordered selection (staleness unavailable).", exc_info=True,
        )
        return []


# ── Rotation isolation (L4544 G2) ───────────────────────────────────────────
# A zoo rotation trains CHALLENGER variants — it must NOT disturb the live
# champion. One hazard remains in meta_trainer:
#   • feature_list.json + manifest.json are written to the LIVE keys
#     UNCONDITIONALLY (model WEIGHTS are gated on `promoted`, but the two
#     contract files are not) → a challenger train leaves the live champion's
#     contract describing the challenger.
# G2 restores the live contract after the rotation. (The former G1 — forcing
# TRAINING_AUTO_PROMOTE_ENABLED=False so no spec self-promoted — is GONE: since
# config#1052/#679 training is UNCONDITIONALLY challenger-first, so a spec can
# never self-promote and there is nothing to force. Promotion is decided solely
# by the `select_winner` selection step below.) G2 is localized here (no
# meta_trainer surgery — the champion path is freshly stabilized by #240).


def _live_contract_keys() -> list[str]:
    """The live champion contract objects meta_trainer overwrites unconditionally."""
    return [getattr(cfg, "META_MANIFEST_KEY"), getattr(cfg, "META_FEATURE_LIST_KEY")]


def _snapshot_live_contract(s3, bucket: str) -> dict:
    """G2 — read the live champion contract into memory for post-rotation
    restore. Best-effort: a missing/unreadable key just isn't restored."""
    saved: dict = {}
    for key in _live_contract_keys():
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            saved[key] = (obj["Body"].read(), obj.get("ContentType", "application/json"))
        except Exception:  # noqa: BLE001 — best-effort capture
            log.warning("model_zoo G2: could not snapshot live contract key %s", key, exc_info=True)
    return saved


def _restore_live_contract(s3, bucket: str, saved: dict) -> None:
    """G2 — restore the captured live champion contract after the rotation. A
    restore failure is logged LOUDLY (the live contract may describe a
    challenger), but #240 embeds feature_names in the meta pickle so inference
    stays aligned regardless — so we warn, never raise, on this best-effort path."""
    for key, (body, ctype) in saved.items():
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=ctype)
        except Exception:  # noqa: BLE001
            log.warning(
                "model_zoo G2: FAILED to restore live contract key %s — the live "
                "champion contract may describe a challenger (mitigated by "
                "feature_names-in-pickle #240); investigate.", key, exc_info=True,
            )


def train_weekly_rotation(
    bucket: str,
    *,
    budget: int | None = None,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    train_fn=None,
    registered_versions: list | None = None,
    s3=None,
) -> dict:
    """Train the ``budget`` stalest active specs (the weekly rotation). Bounds
    compute vs ``train_all_active`` while refreshing the whole zoo round-robin.
    ``registered_versions`` is injectable for tests; otherwise read best-effort
    from the registry. One spec's failure never aborts the rest.

    Isolated per L4544: G1 forces challenger-first (no self-promote), G2 restores
    the live champion contract after training. G2 needs an S3 client (injectable;
    auto-created when ``not dry_run`` and ``s3 is None``); it is skipped on a
    dry run (no live writes happen) and best-effort if S3 is unavailable."""
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    if budget is None:
        budget = int(getattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 4))
    if registered_versions is None:
        registered_versions = _list_registry_versions(bucket)
    selected = select_rotation_specs(specs, registered_versions, budget)
    n_active = sum(1 for s in specs
                   if s.get("status", "active") == "active" and s.get("id"))
    log.info(
        "model_zoo rotation: budget=%d → training %d/%d active spec(s): %s",
        budget, len(selected), n_active, selected,
    )

    # G2 — snapshot the live contract before any challenger train overwrites it.
    # Auto-create the client only for the REAL trainer (train_fn is None); a test
    # that injects a fake train_fn does no live writes, so there's nothing to
    # restore and we must not touch S3 (keeps unit tests pure).
    saved_contract: dict = {}
    if not dry_run:
        if s3 is None and train_fn is None:
            try:
                import boto3
                s3 = boto3.client("s3")
            except Exception:  # noqa: BLE001 — G2 is best-effort
                log.warning("model_zoo G2: no S3 client — live-contract restore skipped", exc_info=True)
        if s3 is not None:
            saved_contract = _snapshot_live_contract(s3, bucket)

    # Training is unconditionally challenger-first (config#1052/#679), so no spec
    # can self-promote — the former G1 guard is gone. Promotion is decided only by
    # the `select_winner` selection step downstream.
    results: dict = {}
    try:
        for sid in selected:
            try:
                results[sid] = train_spec(
                    sid, bucket, date_str=date_str, dry_run=dry_run,
                    specs=specs, train_fn=train_fn,
                )
            except Exception as exc:  # noqa: BLE001 — one variant must not block the rest
                log.warning("model_zoo: spec %s failed (continuing): %s", sid, exc, exc_info=True)
                results[sid] = {"status": "error", "error": str(exc)}
    finally:
        # G2 — always restore the live contract, even if the rotation raised mid-way.
        if saved_contract and s3 is not None:
            _restore_live_contract(s3, bucket, saved_contract)
    return results


# ── Immediate CPCV selection (L4544) ────────────────────────────────────────
# After the rotation, rank the freshly-trained challengers by leak-free CPCV
# mean IC (gated by the downside-Sortino + DSR battery) and pick a winner that
# beats the live champion by a margin. The realized-edge leaderboard (L4539) is
# the SLOWER confirm/demote layer — this is the immediate training-time selector.
#
# DEPTH NOTE (config #671, C — Re-exam: 2026-10-15): the FULL promotion gate
# (downside + DSR >= WF_DSR_THRESHOLD, default 0.95) is correct but DATA-STARVED.
# At ~45 dates / 21d horizon there is ≈ 1 independent block, so the 0.95 DSR bar
# needs an IC-IR ≈ 1.0 — not realistically passable until ~late-2026 once enough
# independent 21d blocks accrue. The thresholds are DELIBERATELY left un-weakened
# (operator gate). The principled path while the full gate is too young: a
# challenger that beats the champion + clears the LOWER registry bar is moved to
# the SHADOW-ONLY observe tier (Option B), and its REALIZED edge — not training
# DSR — gates a later observe→champion promotion (Option A / L4539 leaderboard).
# Re-examine the full-gate floor's reachability on 2026-10-15 (recorded on #671).


def _cpcv_mean(manifest: dict | None):
    """Leak-free CPCV mean IC from a manifest, or None if absent/unusable.

    L4565c: a date-starved CPCV run reports ``mean_ic = NaN`` (status
    ``no_valid_combos``). ``float('nan')`` is not None and silently slips
    past an ``ic is None`` eligibility check, then every NaN comparison
    evaluates False — so the candidate is mislabeled ``below_champion_plus_margin``
    instead of ``no_cpcv``. Collapse non-finite to None here (the single
    chokepoint — also used for the incumbent ``champ_ic``) so the caller's
    ``ic is None`` branch catches it correctly.
    """
    cpcv = (manifest or {}).get("meta_model_oos_ic_cpcv") or {}
    v = cpcv.get("mean_ic")
    try:
        f = float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
    if f is not None and not math.isfinite(f):
        return None
    return f


def _gate_pass(manifest: dict | None) -> bool:
    """Whether a manifest clears BOTH the downside-Sortino and DSR/PSR gates."""
    p = (manifest or {}).get("meta_model_promotion_stats") or {}
    downside = bool((p.get("downside") or {}).get("passes_downside_gate"))
    overfit = bool((p.get("overfit") or {}).get("passes_overfit_gate"))
    return downside and overfit


def _manifest_dsr(manifest: dict | None):
    """The training-time DSR from a manifest's promotion stats, or None."""
    p = (manifest or {}).get("meta_model_promotion_stats") or {}
    v = (p.get("overfit") or {}).get("dsr")
    try:
        f = float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
    return f if (f is not None and math.isfinite(f)) else None


def _registry_bar_pass(manifest: dict | None) -> bool:
    """L4582(c) two-threshold discipline — the LOWER registry-entry bar.

    Entry bar = downside gate passes AND DSR >= WF_DSR_REGISTRY_THRESHOLD
    (default 0.80) vs the promotion bar's WF_DSR_THRESHOLD (0.95). A candidate
    between the bars is a NEAR-MISS: ineligible to promote, but labeled so its
    evidence accumulates legibly across rotation leaderboards instead of
    reading as a flat gate_failed."""
    p = (manifest or {}).get("meta_model_promotion_stats") or {}
    downside = bool((p.get("downside") or {}).get("passes_downside_gate"))
    dsr = _manifest_dsr(manifest)
    bar = float(getattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80))
    return downside and dsr is not None and dsr >= bar


def _read_registry_manifest(s3, bucket: str, version_id: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=f"{_REGISTRY_PREFIX}/{version_id}/manifest.json")
    return json.loads(obj["Body"].read())


def _read_live_manifest(s3, bucket: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=getattr(cfg, "META_MANIFEST_KEY"))
    return json.loads(obj["Body"].read())


def _resolve_trained_versions(s3, bucket: str, results: dict, *, specs: list | None = None) -> list[dict]:
    """Map the specs that trained successfully → their freshly-registered
    challenger versions. The newest challenger per spec label (the run we just
    did) is the candidate. Best-effort: a registry read failure → []."""
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    ok_ids = [sid for sid, r in results.items()
              if not (isinstance(r, dict) and r.get("status") == "error")]
    label_to_spec = {}
    for sid in ok_ids:
        try:
            label_to_spec[_resolve_label(resolve_spec(sid, specs))] = sid
        except ModelSpecError:
            continue
    try:
        from model.registry import list_versions
        versions = list_versions(s3, bucket, stage="challenger")
    except Exception:  # noqa: BLE001 — selection is best-effort
        log.warning("model_zoo select: could not list registry challengers", exc_info=True)
        return []
    newest: dict[str, dict] = {}  # label → newest version row
    for v in versions:
        mv = v.get("model_version")
        if mv in label_to_spec:
            cur = newest.get(mv)
            if cur is None or (v.get("date", ""), v.get("created_utc", "")) > (cur.get("date", ""), cur.get("created_utc", "")):
                newest[mv] = v
    return [
        {"spec_id": label_to_spec[mv], "model_version": mv, "version_id": v.get("version_id")}
        for mv, v in newest.items() if v.get("version_id")
    ]


def _resolve_registered_specs_for_date(
    s3, bucket: str, date_str: str | None, *, specs: list | None = None,
) -> list[dict]:
    """config#1083 — the registry-driven pool for the PARALLEL ``select`` step.

    Unlike ``_resolve_trained_versions`` (which keys off an in-process ``results``
    dict from a sequential rotation), the parallel fan-out has NO results dict:
    each spec trained on its OWN spot and registered independently. So we read the
    registry directly and map every ACTIVE spec's label → its newest registered
    challenger FOR ``date_str`` (the version this rotation produced). A spec whose
    Map iteration FAILED simply has no version for this date → it is absent from
    the pool (TOLERATED — logged present/absent, never crashes). Best-effort: a
    registry read failure → [] (select then ranks only the base champion-arch).
    """
    specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    active = [s for s in specs
              if s.get("status", "active") == "active" and s.get("id")]
    label_to_spec = {}
    for s in active:
        try:
            label_to_spec[_resolve_label(s)] = s["id"]
        except Exception:  # noqa: BLE001 — a malformed spec just isn't in the pool
            continue
    try:
        from model.registry import list_versions
        versions = list_versions(s3, bucket, stage="challenger")
    except Exception:  # noqa: BLE001 — selection is best-effort
        log.warning("model_zoo select: could not list registry challengers", exc_info=True)
        return []
    # Newest challenger per label, restricted to today's vintage when date_str is
    # given (the version THIS rotation produced; a stale prior-week version must
    # not masquerade as a fresh candidate). list_versions is newest-first.
    newest: dict[str, dict] = {}
    for v in versions:
        mv = v.get("model_version")
        if mv not in label_to_spec:
            continue
        if date_str and v.get("date") != date_str:
            continue
        cur = newest.get(mv)
        if cur is None or (v.get("date", ""), v.get("created_utc", "")) > (cur.get("date", ""), cur.get("created_utc", "")):
            newest[mv] = v
    pool = [
        {"spec_id": label_to_spec[mv], "model_version": mv, "version_id": v.get("version_id")}
        for mv, v in newest.items() if v.get("version_id")
    ]
    present = sorted(c["spec_id"] for c in pool)
    absent = sorted(sid for sid in label_to_spec.values() if sid not in present)
    log.info(
        "model_zoo select: registered-for-%s pool — present=%s absent=%s "
        "(absent = failed/skipped Map iteration; tolerated)",
        date_str, present, absent,
    )
    return pool


def _resolve_base_champion_version(s3, bucket: str, date_str: str | None) -> dict | None:
    """L4571: the base champion-ARCHITECTURE retrain (the ``--full-only``
    PredictorTraining state that runs BEFORE the zoo on Saturday) registers
    itself as a ``stage="challenger"`` version with ``model_version`` =
    ``cfg.MODEL_VERSION_LABEL`` ("v3.0-meta") when auto-promote is off. It is the
    champion architecture on the CURRENT data vintage — so it must compete in the
    same pool, not sit outside it (else select_winner only ranks rotated variants
    against a stale serving manifest, and a fresh-data retrain can never win).

    Returns the newest such challenger for ``date_str`` (or the newest overall if
    none carries today's date) as a ``trained``-shaped row labelled
    ``spec_id="champion-arch"``, or None if none is registered / on read failure.
    """
    base_label = getattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta")
    try:
        from model.registry import list_versions
        versions = list_versions(s3, bucket, stage="challenger")
    except Exception:  # noqa: BLE001 — best-effort; base just won't be ranked
        log.warning("model_zoo select: could not list challengers for base-arch", exc_info=True)
        return None
    mine = [v for v in versions if v.get("model_version") == base_label]
    if not mine:
        return None
    # Prefer today's vintage; list_versions is already newest-first by (date,
    # created_utc), so the first today-dated row (else the first overall) wins.
    today = [v for v in mine if date_str and v.get("date") == date_str]
    chosen = (today or mine)[0]
    vid = chosen.get("version_id")
    if not vid:
        return None
    return {"spec_id": "champion-arch", "model_version": base_label, "version_id": vid}


def _record_trials(s3, bucket: str, date_str: str | None, trained: list[dict]):
    """L4582(b): append this rotation's candidates to the cumulative trial
    ledger (deduped by version_id) and return ``(n_trials_cumulative, status)``.

    Best-effort like the leaderboard (a ledger failure must never fail the
    rotation) — but the failure is RECORDED, not swallowed: WARN log here +
    the caller writes ``trial_log_status`` into the leaderboard so a broken
    ledger is visible in the artifact every week (per no-silent-fails)."""
    try:
        try:
            obj = s3.get_object(Bucket=bucket, Key=_TRIAL_LOG_KEY)
            ledger = json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001 — first rotation: no ledger yet
            ledger = {"schema_version": 1, "trials": []}
        seen = {t.get("version_id") for t in ledger.get("trials", [])}
        for rec in trained:
            vid = rec.get("version_id")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            ledger["trials"].append({
                "date": date_str,
                "spec_id": rec.get("spec_id"),
                "model_version": rec.get("model_version"),
                "version_id": vid,
            })
        s3.put_object(
            Bucket=bucket, Key=_TRIAL_LOG_KEY,
            Body=json.dumps(ledger, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        n = len(ledger["trials"])
        log.info("model_zoo: trial ledger at %d cumulative trials (%s)", n, _TRIAL_LOG_KEY)
        return n, "ok"
    except Exception as exc:  # noqa: BLE001 — observability artifact, never fatal
        log.warning("model_zoo: trial ledger update FAILED: %s", exc, exc_info=True)
        return None, f"error: {exc}"


def _selection_pbo(candidates: list[dict], manifests: dict) -> dict:
    """L4582(a): CSCV-PBO across this rotation's candidates from their per-combo
    CPCV IC vectors (``meta_model_oos_ic_cpcv.ics``).

    Rows only align across manifests trained on the SAME data vintage with the
    same CPCV shape — true for same-rotation candidates by construction; the
    serving champion's manifest is a stale vintage and is deliberately NOT in
    the matrix. Candidates whose (n_groups, k_test, len(ics)) don't match the
    modal shape are dropped (named in the output, not silently)."""
    rows = []
    for c in candidates:
        m = manifests.get(c.get("version_id")) or {}
        cpcv = m.get("meta_model_oos_ic_cpcv") or {}
        ics = cpcv.get("ics") or []
        if ics:
            rows.append((c.get("spec_id"), (cpcv.get("n_groups"), cpcv.get("k_test"), len(ics)), ics))
    if len(rows) < 2:
        return {"status": "insufficient", "reason": "needs >=2 candidates with CPCV ics",
                "n_specs": len(rows), "pbo": None}
    shapes: dict = {}
    for _, shape, _ics in rows:
        shapes[shape] = shapes.get(shape, 0) + 1
    modal = max(shapes, key=shapes.get)
    aligned = [(sid, ics) for sid, shape, ics in rows if shape == modal]
    dropped = [sid for sid, shape, _ in rows if shape != modal]
    from training.deflated_sharpe import cscv_pbo
    out = cscv_pbo(
        list(map(list, zip(*[ics for _, ics in aligned]))),
        spec_ids=[sid for sid, _ in aligned],
    )
    target = float(getattr(cfg, "MODEL_ZOO_PBO_TARGET", 0.2))
    pbo = out.get("pbo")
    out["pbo_target"] = target
    out["pbo_pass"] = (
        bool(pbo <= target) if isinstance(pbo, float) and math.isfinite(pbo) else None
    )
    if dropped:
        out["dropped_misaligned_specs"] = dropped
    return out


def select_winner(
    s3, bucket: str, *, trained: list[dict], margin: float | None = None,
    n_trials_cumulative: int | None = None,
) -> dict:
    """Rank freshly-trained challengers by leak-free-CPCV mean IC and pick a
    winner that beats the VINTAGE-CONSISTENT baseline by ``margin``. Returns a
    leaderboard dict with three labeled groups (serving_champion / champion_arch /
    challengers); ``winner_version_id`` is the best eligible CHALLENGER or None;
    ``champion_arch_refresh_version_id`` keeps the live model fresh when no
    challenger wins.

    config#671/#673/#1052 + #679(ii) — TRUE FRESH-BEST-WINS promotion.

    BASELINE (#679ii): the promotion comparison is against the FRESH ``champion-arch``
    candidate's CPCV — the champion ARCHITECTURE retrained on THIS data vintage — NOT
    the stale serving manifest (a different, older vintage). This is the apples-to-
    apples same-vintage comparison; we fall back to the serving snapshot ONLY when no
    champion-arch trained this run (logged). The champion-arch row is the baseline, so
    it is NEVER a "challenger" winner (it can't beat itself) — it carries reason
    ``champion_arch_baseline``. But it CAN still refresh the live model when no
    challenger wins (see ``champion_arch_refresh_version_id``), so the deployed 21d
    model stays current weekly.

    The absolute DSR-0.95 hurdle (``_gate_pass``) is NO LONGER a promotion blocker:
    both estimates are equally data-starved on ~1 independent 21d block, so the
    incumbent has no special epistemic claim, and 21d-alpha models go stale —
    fresh-best-wins weekly is correct, not waiting for an absolute-significance gate
    (~late-2026). With ``margin`` = 0 (config) eligibility is simply: right horizon
    AND CPCV IC > the positive floor AND IC >= the champion-arch baseline (the
    highest-CPCV challenger above the floor wins). NO incumbency hurdle beyond the
    configurable margin. The DSR ``_gate_pass`` is still computed and surfaced
    (``dsr_gate_pass``, ``dsr_training``, ``dsr_selection``) for OBSERVABILITY only.

    L4582 observability (none of it changes eligibility): per-candidate
    ``dsr_training`` / ``dsr_selection`` (the latter re-deflated by the
    cumulative trial count when supplied), the ``dsr_gate_pass`` DSR-gate flag, and
    a leaderboard-level ``selection_pbo`` block (CSCV-PBO across the rotation's
    aligned candidates, target <0.2, observe-only)."""
    if margin is None:
        margin = float(getattr(cfg, "MODEL_ZOO_PROMOTE_MARGIN", 0.01))
    min_ic = float(getattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0))

    # ── SERVING champion (the live model that's trading NOW) ──────────────────
    # Its CPCV mean IC is a STALE last-promoted snapshot (a prior vintage), so it
    # is NOT the apples-to-apples promotion baseline — it's surfaced only so the
    # operator sees what the system is currently serving + when it was promoted.
    serving_manifest: dict = {}
    try:
        serving_manifest = _read_live_manifest(s3, bucket)
    except Exception:  # noqa: BLE001 — serving-champion snapshot best-effort
        log.warning("model_zoo select: could not read live champion manifest", exc_info=True)
    champ_fwd = int(serving_manifest.get("forward_days") or getattr(cfg, "FORWARD_DAYS", 21))
    serving_ic = _cpcv_mean(serving_manifest)
    serving_version = serving_manifest.get("served_version") or serving_manifest.get("version")
    serving_date = serving_manifest.get("served_date") or serving_manifest.get("date")

    # ── Read every candidate manifest up front (needed to locate champion-arch
    # BEFORE the eligibility loop, since the fresh champion-arch CPCV is now the
    # promotion baseline — #679(ii) vintage-consistent comparison). ────────────
    manifests: dict[str, dict] = {}
    for rec in trained:
        vid = rec.get("version_id")
        try:
            manifests[vid] = _read_registry_manifest(s3, bucket, vid)
        except Exception:  # noqa: BLE001 — a missing manifest just isn't ranked
            log.warning("model_zoo select: could not read manifest for %s", vid, exc_info=True)

    # #679(ii) — VINTAGE-CONSISTENT promotion baseline. The fresh ``champion-arch``
    # candidate (the base champion-ARCHITECTURE retrained on THIS data vintage) is
    # the correct apples-to-apples reference: a challenger must beat the champion
    # ARCHITECTURE *on the same vintage*, not a stale serving snapshot trained on a
    # different vintage. Fall back to the stale serving manifest ONLY when no
    # champion-arch trained this run (logged either way).
    champ_arch_rec = next(
        (r for r in trained if r.get("spec_id") == "champion-arch"), None)
    champ_arch_vid = champ_arch_rec.get("version_id") if champ_arch_rec else None
    champ_arch_ic = _cpcv_mean(manifests.get(champ_arch_vid)) if champ_arch_vid else None
    if champ_arch_ic is not None:
        baseline_ic = champ_arch_ic
        baseline_source = "champion_arch_fresh"
        log.info(
            "model_zoo select: promotion baseline = FRESH champion-arch CPCV "
            "%.4f (vid=%s) — vintage-consistent (#679ii); serving snapshot CPCV "
            "%s is observability only", baseline_ic, champ_arch_vid, serving_ic,
        )
    else:
        baseline_ic = serving_ic
        baseline_source = "serving_champion_stale"
        log.warning(
            "model_zoo select: no champion-arch candidate this run — promotion "
            "baseline FALLS BACK to the stale serving-champion CPCV %s (a "
            "cross-vintage comparison; #679ii degrades gracefully)", serving_ic,
        )

    candidates: list[dict] = []
    for rec in trained:
        vid = rec.get("version_id")
        is_champ_arch = rec.get("spec_id") == "champion-arch"
        manifest = manifests.get(vid)
        fwd = int((manifest or {}).get("forward_days") or 0)
        ic = _cpcv_mean(manifest)
        # config#671/#673/#1052: the DSR gate is OBSERVABILITY ONLY now — computed
        # and surfaced (dsr_gate_pass), never a promotion blocker. registry_bar is
        # likewise informational. A challenger that beats the baseline's CPCV mean
        # IC by ``margin`` (same horizon) and clears the positive floor PROMOTES.
        gate = _gate_pass(manifest)
        registry_bar = _registry_bar_pass(manifest)
        beats_baseline = (
            ic is not None
            and (baseline_ic is None or ic >= baseline_ic + margin)
        )
        # config#671/#673/#1052 + #679(ii) — RELATIVE-BEST CHALLENGER eligibility
        # chain. The baseline is the FRESH champion-arch CPCV (vintage-consistent),
        # NOT the stale serving snapshot. The champion-arch row itself is the
        # baseline — it is NEVER a "challenger" winner (it can't beat itself); it is
        # marked ``champion_arch_baseline`` and handled by the separate refresh path
        # below (so the live model still stays fresh weekly when no challenger wins).
        #   champion-arch row      → champion_arch_baseline (not a challenger)
        #   wrong horizon          → non_canonical_horizon
        #   no usable CPCV mean IC → no_cpcv
        #   IC <= positive floor   → below_floor (best-of-N noise is not edge)
        #   IC <  baseline+margin  → below_champion_arch_plus_margin
        #   else                   → eligible (PROMOTES — DSR is not consulted)
        group = "champion_arch" if is_champ_arch else "challenger"
        if is_champ_arch:
            eligible, reason = False, "champion_arch_baseline"
        elif fwd != champ_fwd:
            eligible, reason = False, "non_canonical_horizon"
        elif ic is None:
            eligible, reason = False, "no_cpcv"
        elif ic <= min_ic:
            eligible, reason = False, "below_floor"
        elif baseline_ic is not None and ic < baseline_ic + margin:
            eligible, reason = False, "below_champion_arch_plus_margin"
        else:
            eligible, reason = True, "eligible"
        # L4582(b): re-deflate this candidate's CPCV IC series by the CUMULATIVE
        # trial count — the honest multiple-testing N once rotations accumulate
        # (the manifest's training-time DSR used the within-run n_combos proxy).
        # config #671 — A1: feed an EFFECTIVE observation count (n_eff) instead of
        # the raw CPCV combo count. The LdP backtest-path φ = (k/N)·C(N,k)
        # (manifest ``meta_model_oos_ic_cpcv.n_backtest_paths``) is the honest
        # independent-block count on overlapping 21d labels; it TIGHTENS the DSR
        # (more honest), never loosens it. Surfaced via dsr_selection_n_eff +
        # the 'needs ~Y' bar so a 'too young' verdict is explicit, not silent.
        dsr_selection = None
        dsr_selection_n_eff = None
        ic_ir_needed = None
        _cpcv = (manifest or {}).get("meta_model_oos_ic_cpcv") or {}
        _ics = _cpcv.get("ics") or []
        _n_paths = _cpcv.get("n_backtest_paths")
        if _ics and n_trials_cumulative:
            try:
                from training.deflated_sharpe import deflated_sharpe_ratio
                _n_eff = int(_n_paths) if isinstance(_n_paths, (int, float)) and _n_paths else None
                _d = deflated_sharpe_ratio(
                    _ics, n_trials=int(n_trials_cumulative),
                    threshold=float(getattr(cfg, "WF_DSR_THRESHOLD", 0.95)),
                    n_eff=_n_eff,
                )
                dsr_selection = _d.get("dsr")
                dsr_selection_n_eff = _d.get("n_used")
                ic_ir_needed = _d.get("ic_ir_needed_for_threshold")
            except Exception:  # noqa: BLE001 — observe-only stat, never blocks ranking
                log.warning("model_zoo select: dsr_selection failed for %s", vid, exc_info=True)
        candidates.append({
            "spec_id": rec.get("spec_id"), "version_id": vid,
            "model_version": rec.get("model_version"),
            # #679(ii): the disambiguation group — "champion_arch" (fresh retrain,
            # the baseline) vs "challenger" (a rotated spec competing against it).
            "group": group,
            "forward_days": fwd, "cpcv_mean_ic": ic, "passes_gate": gate,
            # config#671/#673/#1052: the DSR/Sortino gate result, surfaced for
            # OBSERVABILITY (is relative-best promotion chasing a model the absolute
            # gate would reject?) — NOT a promotion blocker. registry_bar likewise.
            "dsr_gate_pass": gate,
            "registry_bar_pass": registry_bar,
            # #679(ii): beats the FRESH champion-arch baseline by margin (the
            # vintage-consistent comparison). ``beats_champion_by_margin`` kept as
            # an alias for back-compat with any older consumer.
            "beats_baseline_by_margin": beats_baseline,
            "beats_champion_by_margin": beats_baseline,
            "dsr_training": _manifest_dsr(manifest),
            "dsr_selection": dsr_selection,
            # config #671 A1: the effective-N that fed dsr_selection + the IC-IR a
            # series of that independent length needs to clear the full DSR bar.
            "dsr_selection_n_eff": dsr_selection_n_eff,
            "ic_ir_needed_for_full_bar": ic_ir_needed,
            "eligible": eligible, "reason": reason,
        })

    # The CHALLENGER winner: the highest-CPCV eligible challenger (champion-arch is
    # excluded by construction — it carries reason ``champion_arch_baseline`` and is
    # never ``eligible``, so it can't win against itself).
    eligibles = [c for c in candidates if c["eligible"]]
    winner = max(eligibles, key=lambda c: c["cpcv_mean_ic"], default=None)

    # #679(ii) — CHAMPION-ARCH REFRESH path. When NO challenger wins, the fresh
    # champion-arch (same architecture, retrained on this vintage) should still
    # refresh the LIVE model if it improves on the *serving* champion's CPCV by
    # margin + clears the positive floor — so the deployed 21d model never goes
    # stale waiting for a challenger to win. This is NOT a challenger promotion
    # (no incumbency hurdle is being introduced for challengers); it keeps the
    # serving model on the current vintage. A challenger win always takes
    # precedence (it already beat champion-arch, which beat serving).
    champ_arch_refresh = None
    if winner is None and champ_arch_vid is not None and champ_arch_ic is not None:
        arch_cand = next((c for c in candidates if c["version_id"] == champ_arch_vid), None)
        improves_serving = (
            champ_arch_ic > min_ic
            and (serving_ic is None or champ_arch_ic >= serving_ic + margin)
        )
        right_horizon = arch_cand is not None and arch_cand.get("forward_days") == champ_fwd
        if improves_serving and right_horizon:
            champ_arch_refresh = champ_arch_vid
            log.info(
                "model_zoo select: no challenger won — FRESH champion-arch %s "
                "(CPCV %.4f) refreshes the live model (serving CPCV %s + margin "
                "%s); keeps the deployed 21d model on the current vintage",
                champ_arch_vid, champ_arch_ic, serving_ic, margin,
            )
        else:
            log.info(
                "model_zoo select: no challenger won and champion-arch does NOT "
                "improve on serving (arch CPCV %s vs serving %s + margin %s, "
                "right_horizon=%s) — live model unchanged",
                champ_arch_ic, serving_ic, margin, right_horizon,
            )

    selection_pbo = _selection_pbo(candidates, manifests)
    # config #671 — A1: surface the effective-N + the IC-IR each candidate would
    # need to clear the full DSR bar, so a data-starved 'too young' verdict is
    # explicit rather than a silent gate fail (the full bar is passable only once
    # enough independent 21d blocks accrue; Re-exam: 2026-10-15 — see #671).
    for c in candidates:
        if c.get("dsr_selection_n_eff") is not None:
            log.info(
                "model_zoo A1 (config#671): %s n_eff=%s feeds DSR (needs IC-IR~%s "
                "for the %.2f full bar); dsr_selection=%s, dsr_training=%s",
                c.get("spec_id"), c.get("dsr_selection_n_eff"),
                c.get("ic_ir_needed_for_full_bar"),
                float(getattr(cfg, "WF_DSR_THRESHOLD", 0.95)),
                c.get("dsr_selection"), c.get("dsr_training"),
            )
    log.info(
        "model_zoo selection PBO (OBSERVE, L4582): %s pbo=%s (target<%s, pass=%s) "
        "over %s specs / %s splits; n_trials_cumulative=%s",
        selection_pbo.get("status"), selection_pbo.get("pbo"),
        selection_pbo.get("pbo_target"), selection_pbo.get("pbo_pass"),
        selection_pbo.get("n_specs"), selection_pbo.get("n_splits"),
        n_trials_cumulative,
    )
    return {
        # ── #679(ii): THREE clearly-labeled groups (disambiguated presentation) ──
        # SERVING champion = the live model trading capital NOW. Its CPCV is a
        # STALE last-promoted snapshot (prior vintage) — observability only, NOT
        # the promotion baseline. served/promoted date carried so it reads as old.
        "serving_champion": {
            "forward_days": champ_fwd, "cpcv_mean_ic": serving_ic,
            "cpcv_is_stale_snapshot": True,
            "served_version": serving_version, "served_date": serving_date,
        },
        # CHAMPION-ARCH = the champion ARCHITECTURE retrained THIS run on the
        # current vintage. Its CPCV is the vintage-consistent promotion BASELINE.
        "champion_arch": {
            "version_id": champ_arch_vid, "cpcv_mean_ic": champ_arch_ic,
            "is_promotion_baseline": baseline_source == "champion_arch_fresh",
        },
        # The baseline a challenger must beat by margin + which group it came from.
        "promotion_baseline_ic": baseline_ic,
        "promotion_baseline_source": baseline_source,
        # ``champion`` kept for back-compat (dashboard 7_Predictor reads it). It is
        # the SERVING snapshot — same as serving_champion's CPCV. Older consumers
        # see the old key/shape; new consumers read serving_champion/champion_arch.
        "champion": {"forward_days": champ_fwd, "cpcv_mean_ic": serving_ic},
        "margin": margin,
        "promote_min_ic": min_ic,
        "candidates": candidates,
        # The CHALLENGER winner (a rotated spec that beat the champion-arch
        # baseline). champion-arch is excluded from this set by construction.
        "winner_version_id": winner["version_id"] if winner else None,
        # #679(ii): when no challenger wins, the fresh champion-arch may still
        # refresh the live model (keeps the deployed 21d model on this vintage).
        "champion_arch_refresh_version_id": champ_arch_refresh,
        "selection_pbo": selection_pbo,
        "n_trials_cumulative": n_trials_cumulative,
    }


def _write_leaderboard(s3, bucket: str, date_str: str | None, leaderboard: dict) -> None:
    # config#1083: write BOTH the dated key AND latest.json. The prior code wrote
    # only the dated key (or latest.json when date_str was None) — never both — so
    # consumers reading predictor/model_zoo/leaderboard/latest.json saw a STALE
    # board after every dated rotation. Mirror to latest.json so the latest pointer
    # always reflects the most recent rotation. Each PUT is independent so a
    # dated-key failure doesn't suppress the latest mirror (and vice versa).
    body = json.dumps(leaderboard, indent=2, default=str).encode()
    keys = [f"{_LEADERBOARD_PREFIX}/latest.json"]
    if date_str:
        keys.insert(0, f"{_LEADERBOARD_PREFIX}/{date_str}.json")
    for key in keys:
        try:
            s3.put_object(
                Bucket=bucket, Key=key, Body=body, ContentType="application/json",
            )
            log.info("model_zoo: leaderboard written to s3://%s/%s", bucket, key)
        except Exception:  # noqa: BLE001 — observability artifact, never fatal
            log.warning("model_zoo: could not write leaderboard to %s", key, exc_info=True)


def _current_champion_version_id(s3, bucket: str) -> str | None:
    """The registry version_id of the currently-SERVING champion (the model the
    next auto-promote will demote). Used to build an exact revert command. The
    bundle is only demoted to ``archived`` on promote (never deleted), so this id
    stays valid for `--promote <id>` indefinitely. Best-effort → None."""
    try:
        from model.registry import list_versions
        champs = list_versions(s3, bucket, stage="champion")
    except Exception:  # noqa: BLE001
        return None
    return champs[0].get("version_id") if champs else None


def _candidate_by_vid(leaderboard: dict, vid: str) -> dict:
    for c in leaderboard.get("candidates", []):
        if c.get("version_id") == vid:
            return c
    return {}


def _alert_promotion(bucket, date_str, leaderboard, winner_vid, prior_vid) -> None:
    """L4571: a champion auto-promotion changes the model that trades capital —
    it MUST announce itself (fail-loud; the realized-edge auto-demote L4539 isn't
    live, so revert is MANUAL and depends on the operator SEEING this). Telegram +
    SNS via the fleet alerts chokepoint; never raises (observability off a path
    that already promoted)."""
    try:
        win = _candidate_by_vid(leaderboard, winner_vid)
        champ_ic = (leaderboard.get("champion") or {}).get("cpcv_mean_ic")
        new_ic = win.get("cpcv_mean_ic")
        spec = win.get("spec_id", "?")
        revert = (
            f"python -m model.registry --bucket {bucket} --promote {prior_vid}"
            if prior_vid else "(prior champion version_id unavailable — see predictor/registry/)"
        )
        # config#671/#673/#1052: surface the challenger's DSR (now an OBSERVABILITY
        # number, not a gate) so the operator can see whether relative-best promotion
        # is chasing a model the old absolute DSR-0.95 bar would have rejected.
        dsr = win.get("dsr_selection")
        if dsr is None:
            dsr = win.get("dsr_training")
        dsr_gate = win.get("dsr_gate_pass")
        msg = (
            f"[predictor] Model-zoo AUTO-PROMOTED a new champion ({date_str}).\n"
            f"  {prior_vid or '?'}  →  {winner_vid}  (spec: {spec})\n"
            f"  CPCV mean IC: {champ_ic}  →  {new_ic}  (margin {leaderboard.get('margin')}, "
            f"floor {leaderboard.get('promote_min_ic')})\n"
            f"  Promotion is on RELATIVE-BEST (beats champion + margin), NOT on DSR. "
            f"DSR={dsr} (absolute-gate pass={dsr_gate}) is observability only — "
            f"config#671/#673/#1052.\n"
            f"  Inference serves the new champion from the next run; the #237 turnover "
            f"governor caps the first-day book move.\n"
            f"  REVERT (if it misbehaves): {revert}"
        )
        from krepis import alerts as _alerts
        _alerts.publish(
            message=msg, severity="warning",
            source="alpha-engine-predictor/training/model_zoo.py::run_rotation_and_select",
            dedup_key=f"model_zoo_promote_{date_str}",
        )
    except Exception:  # noqa: BLE001 — alert failure must not fail the SF
        log.warning("model_zoo: promotion alert failed", exc_info=True)


def _emit_challengers_trained_metric(n_trained: int) -> None:
    """config#1051: emit ``AlphaEngine/ModelZooChallengersTrained`` so an inert
    rotation (0 challengers trained) is observable on a dashboard/alarm in
    addition to the SNS alert. Best-effort — a CloudWatch failure WARNs but never
    blocks the rotation (the count is already in the inline log + SNS)."""
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine",
            MetricData=[
                {"MetricName": "ModelZooChallengersTrained",
                 "Dimensions": [{"Name": "Process", "Value": "predictor-model-zoo"}],
                 "Value": float(n_trained), "Unit": "Count"},
            ],
        )
    except Exception:  # noqa: BLE001 — observability, never block the rotation
        log.warning("model_zoo: ModelZooChallengersTrained metric emit failed", exc_info=True)


def _count_challengers_trained(results: dict) -> int:
    """Number of selected specs that produced a usable (non-error) train result.
    config#1051: distinguishes "0 challengers TRAINED" (inert zoo) from
    "trained-but-lost-to-champion" (a winner-less leaderboard with candidates)."""
    return sum(1 for r in (results or {}).values()
               if isinstance(r, dict) and r.get("status") != "error")


def _alert_inert_rotation(bucket, date_str, *, n_active, n_selected, results) -> None:
    """config#1051 (no-silent-fails): the rotation had >= 1 active spec to train
    but trained 0 challengers (empty selection, or every selected spec errored /
    produced no usable result). This is an INERT zoo — distinct from "challengers
    trained but lost to the champion" — and must announce itself (WARN + named CW
    metric + SNS), per the no-silent-fails rule. The prime cause is an empty
    ``MODEL_SPECS`` at runtime on the child spot (config load failure); the alert
    carries the resolved config path so the operator can pin it. Never raises
    (alert delivery is observability off a path that already failed loudly via
    the WARN log + CW metric)."""
    cfg_path = getattr(cfg, "_CONFIG_PATH", "?")
    exp_id = getattr(cfg, "_EXPERIMENT_ID", os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "?"))
    errored = sorted(sid for sid, r in (results or {}).items()
                     if isinstance(r, dict) and r.get("status") == "error")
    msg = (
        f"[predictor] Model-zoo rotation trained 0 CHALLENGERS ({date_str}) — the "
        f"champion/challenger value loop is INERT this run.\n"
        f"  active specs: {n_active}  selected: {n_selected}  errored: {errored or 'none'}\n"
        f"  MODEL_SPECS loaded: {len(getattr(cfg, 'MODEL_SPECS', []))} "
        f"(config: {cfg_path}, ALPHA_ENGINE_EXPERIMENT_ID={exp_id})\n"
        f"  This is NOT 'challengers ran but lost' — nothing trained. Likely an "
        f"empty model_specs roster on the child spot (staged predictor.yaml parse "
        f"/ experiment-package path). See config#1051."
    )
    log.warning(
        "model_zoo config#1051: INERT rotation — %d active spec(s), %d selected, "
        "0 challengers trained (errored=%s). MODEL_SPECS=%d, config=%s, exp_id=%s",
        n_active, n_selected, errored, len(getattr(cfg, "MODEL_SPECS", [])),
        cfg_path, exp_id,
    )
    try:
        from krepis import alerts as _alerts
        _alerts.publish(
            message=msg, severity="warning",
            source="alpha-engine-predictor/training/model_zoo.py::run_rotation_and_select",
            dedup_key=f"model_zoo_inert_{date_str}",
        )
    except Exception:  # noqa: BLE001 — alert failure must not fail the SF
        log.warning("model_zoo: inert-rotation alert failed", exc_info=True)


def _alert_observe_recommendation(bucket, date_str, leaderboard, winner_vid) -> None:
    """Observe-mode: a challenger beat the champion but auto-promote is OFF — a
    low-priority (info) heads-up so the operator reviews the leaderboard + can
    promote manually. Telegram silent-tier; never raises."""
    try:
        win = _candidate_by_vid(leaderboard, winner_vid)
        champ_ic = (leaderboard.get("champion") or {}).get("cpcv_mean_ic")
        msg = (
            f"[predictor] Model-zoo OBSERVE ({date_str}): challenger {winner_vid} "
            f"(spec {win.get('spec_id','?')}) beat the champion "
            f"(CPCV {champ_ic} → {win.get('cpcv_mean_ic')}) but auto-promote is OFF — "
            f"review predictor/model_zoo/leaderboard/{date_str}.json; promote with "
            f"`python -m model.registry --bucket {bucket} --promote {winner_vid}`."
        )
        from krepis import alerts as _alerts
        _alerts.publish(
            message=msg, severity="info",
            source="alpha-engine-predictor/training/model_zoo.py::run_rotation_and_select",
            dedup_key=f"model_zoo_observe_{date_str}",
        )
    except Exception:  # noqa: BLE001
        log.warning("model_zoo: observe alert failed", exc_info=True)


def _read_base_training_summary(s3, bucket: str, date_str: str | None) -> dict:
    """Best-effort read of the base champion-arch retrain's training summary
    (``predictor/metrics/training_summary_{date}.json``, else ``_latest``) to
    enrich the champion section of the digest with the base retrain's IC /
    promotion detail. A read failure just leaves the champion section to the
    leaderboard's CPCV numbers — never fatal."""
    keys = []
    if date_str:
        keys.append(f"predictor/metrics/training_summary_{date_str}.json")
    keys.append("predictor/metrics/training_summary_latest.json")
    for key in keys:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001 — best-effort enrichment
            continue
    log.info("model_zoo digest: no base training_summary found (champion section uses CPCV only)")
    return {}


def _digest_subject(leaderboard: dict, date_str: str | None) -> str:
    cands = leaderboard.get("candidates", []) or []
    # #679(ii): the headline challenger count excludes the champion-arch baseline.
    n_chal = sum(1 for c in cands if c.get("group") != "champion_arch")
    winner = leaderboard.get("winner_version_id")
    refresh = leaderboard.get("champion_arch_refresh_version_id")
    promoted = leaderboard.get("promoted")
    if promoted:
        promo = f"promoted: {promoted}"
    elif winner:
        promo = f"recommended: {winner} (observe)"
    elif refresh:
        promo = f"champion-arch refresh: {refresh} (observe)"
    else:
        promo = "promoted: none"
    return (
        f"Alpha Engine | Model-Zoo Rotation {date_str or 'latest'} | "
        f"{n_chal} challengers | {promo}"
    )


def _fmt_ic(v) -> str:
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def send_zoo_digest_email(leaderboard: dict, bucket: str, date_str: str | None,
                          *, s3=None) -> bool:
    """Send the ONE consolidated model-zoo rotation digest email — the single
    email that replaces both the per-challenger email storm AND the base
    champion-arch retrain's separate email.

    Sections:
      • CHAMPION  — CPCV mean IC + horizon from the leaderboard, enriched with
        the base champion-arch retrain's IC / promotion detail read from
        predictor/metrics/training_summary_{date}.json (if present).
      • CHALLENGERS — per-candidate table (spec_id, cpcv_mean_ic, forward_days,
        passes_gate, eligible, reason).
      • PROMOTION — winner_version_id / promoted: old→new champion if promoted,
        else "no promotion" + why (observe recommendation, or no eligible
        challenger).

    Reuses the train_handler SMTP/SES email chokepoint
    (``krepis.email_sender.send_email`` via train_handler) — NOT a
    hand-rolled mailer. Returns True on send success, False if email is not
    configured. Raising is the caller's concern (see run_rotation_and_select's
    try/except — a digest-send failure is logged + recorded, never aborts the
    rotation, per no-silent-fails)."""
    import config as cfg

    sender = getattr(cfg, "EMAIL_SENDER", None)
    recipients = getattr(cfg, "EMAIL_RECIPIENTS", None)
    if not sender or not recipients:
        log.info("model_zoo digest: EMAIL_SENDER/EMAIL_RECIPIENTS not set — skipping")
        return False

    if s3 is None:
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception:  # noqa: BLE001 — enrichment is best-effort
            s3 = None

    base = _read_base_training_summary(s3, bucket, date_str) if s3 is not None else {}

    # ── #679(ii): the THREE clearly-labeled groups ──────────────────────────────
    # 1) SERVING champion = the live model (its CPCV is a STALE last-promoted
    #    snapshot — explicitly flagged + dated so it never reads as fresh).
    # 2) CHAMPION-ARCH = the fresh retrain this run (the promotion BASELINE).
    # 3) CHALLENGERS = rotated specs competing against the champion-arch baseline.
    serving = leaderboard.get("serving_champion") or leaderboard.get("champion", {}) or {}
    serving_ic = serving.get("cpcv_mean_ic")
    serving_fwd = serving.get("forward_days")
    serving_ver = serving.get("served_version")
    serving_date = serving.get("served_date")
    arch = leaderboard.get("champion_arch", {}) or {}
    arch_ic = arch.get("cpcv_mean_ic")
    arch_vid = arch.get("version_id")
    baseline_ic = leaderboard.get("promotion_baseline_ic")
    baseline_src = leaderboard.get("promotion_baseline_source", "?")
    all_cands = leaderboard.get("candidates", []) or []
    # Split the candidate list by group; champion-arch is shown in its OWN section,
    # never co-mingled with the challengers (eliminates the duplicative look).
    challengers = [c for c in all_cands if c.get("group") != "champion_arch"]
    winner = leaderboard.get("winner_version_id")
    refresh = leaderboard.get("champion_arch_refresh_version_id")
    promoted = leaderboard.get("promoted")
    promoted_kind = leaderboard.get("promoted_kind")
    reverted_from = leaderboard.get("reverted_from")
    mode = leaderboard.get("mode", "?")
    margin = leaderboard.get("margin")
    floor = leaderboard.get("promote_min_ic")

    baseline_label = ("FRESH champion-arch (vintage-consistent)"
                      if baseline_src == "champion_arch_fresh"
                      else "stale serving snapshot (no champion-arch this run)")

    # Base champion-arch enrichment (training_summary fields; all optional).
    base_version = base.get("model_version")
    base_oos = base.get("meta_model_oos_ic")
    base_promoted = base.get("promoted")
    base_passes = base.get("passes_ic_gate")

    # ── PROMOTION line ──
    if promoted:
        kind_txt = ("a CHALLENGER" if promoted_kind == "challenger"
                    else "the FRESH champion-arch (no challenger won — keeping the "
                         "live model on this vintage)")
        promo_plain = (
            f"PROMOTED {kind_txt}: {reverted_from or '?'} -> {promoted}\n"
            f"  (relative-best vs the {baseline_label} CPCV {_fmt_ic(baseline_ic)} "
            f"+ margin {margin}, floor {floor}; revert: python -m model.registry "
            f"--bucket {bucket} --promote {reverted_from or '<prior-id>'})"
        )
        promo_html = (
            f'<p style="color:#2e7d32;"><b>PROMOTED</b> {kind_txt}: '
            f'<code>{reverted_from or "?"}</code> &rarr; <code>{promoted}</code><br>'
            f'<span style="font-size:11px;color:#555;">relative-best vs the '
            f'{baseline_label} CPCV {_fmt_ic(baseline_ic)} + margin {margin}, '
            f'floor {floor}; revert: <code>python -m model.registry --bucket '
            f'{bucket} --promote {reverted_from or "&lt;prior-id&gt;"}</code></span></p>'
        )
    elif winner or refresh:
        rec = winner or refresh
        kind_txt = ("challenger" if winner else
                    "fresh champion-arch refresh (no challenger won)")
        promo_plain = (
            f"No promotion executed (mode={mode}). Recommended {kind_txt}: {rec} "
            f"(beats the {baseline_label} baseline; observe — promote manually if "
            f"confirmed)."
        )
        promo_html = (
            f'<p style="color:#ef6c00;">No promotion executed (mode=<b>{mode}</b>). '
            f'Recommended {kind_txt} <code>{rec}</code> beats the {baseline_label} '
            f'baseline (observe — promote manually if confirmed).</p>'
        )
    else:
        promo_plain = (
            "No promotion — no challenger beat the champion-arch baseline and the "
            "champion-arch does not improve on the serving champion this rotation."
        )
        promo_html = (
            '<p style="color:#555;">No promotion — no challenger beat the '
            'champion-arch baseline and the champion-arch does not improve on the '
            'serving champion this rotation.</p>'
        )

    # ── CHALLENGER table (champion-arch excluded — it's the baseline, shown above) ──
    cand_rows_plain = []
    cand_rows_html = []
    for c in challengers:
        sid = c.get("spec_id", "?")
        cpcv = _fmt_ic(c.get("cpcv_mean_ic"))
        fwd = c.get("forward_days")
        gate = c.get("passes_gate")
        elig = c.get("eligible")
        reason = c.get("reason", "")
        cand_rows_plain.append(
            f"  {str(sid):<18} cpcv={cpcv}  fwd={fwd}  gate={gate}  "
            f"eligible={elig}  ({reason})"
        )
        row_color = "#2e7d32" if elig else "#888"
        cand_rows_html.append(
            f'<tr><td style="padding:2px 8px;font-family:monospace;">{sid}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;">{cpcv}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;">{fwd}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;">{gate}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;color:{row_color};">{elig}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;">{reason}</td></tr>'
        )
    if not challengers:
        cand_rows_plain.append("  (no challenger candidates this rotation)")

    base_plain = ""
    if base:
        base_plain = (
            f"\n  (base retrain training_summary: model={base_version} "
            f"OOS_IC={_fmt_ic(base_oos)} passes_gate={base_passes} promoted={base_promoted})"
        )

    plain_body = (
        f"Alpha Engine — Model-Zoo Rotation Digest ({date_str})\n"
        f"Mode: {mode}\n"
        f"\n=== SERVING CHAMPION (live model now — CPCV is a STALE snapshot) ===\n"
        f"  CPCV mean IC: {_fmt_ic(serving_ic)}  (forward_days={serving_fwd}; "
        f"served_version={serving_ver}; promoted/served on {serving_date})\n"
        f"\n=== CHAMPION-ARCH (fresh retrain THIS run — promotion baseline) ===\n"
        f"  CPCV mean IC: {_fmt_ic(arch_ic)}  (version={arch_vid})"
        f"{base_plain}\n"
        f"  >> PROMOTION BASELINE = {baseline_label}: CPCV {_fmt_ic(baseline_ic)} "
        f"(challengers must beat this + margin {margin})\n"
        f"\n=== CHALLENGERS ({len(challengers)}) ===\n"
        + "\n".join(cand_rows_plain)
        + f"\n\n=== PROMOTION ===\n{promo_plain}\n"
    )

    base_html = ""
    if base:
        base_html = (
            f'<p style="font-size:11px;color:#777;margin:2px 0;">base retrain '
            f'training_summary: model=<b>{base_version}</b> &nbsp; OOS IC '
            f'<b>{_fmt_ic(base_oos)}</b> &nbsp; passes_gate=<b>{base_passes}</b> '
            f'&nbsp; promoted=<b>{base_promoted}</b></p>'
        )
    html_body = (
        f'<html><body style="font-family:sans-serif;font-size:13px;color:#222;max-width:640px;">'
        f'<h2 style="margin-bottom:4px;">Model-Zoo Rotation Digest — {date_str}</h2>'
        f'<p style="color:#555;font-size:12px;margin-top:0;">Mode: <b>{mode}</b></p>'
        f'<h3 style="margin-bottom:2px;">Serving champion '
        f'<span style="font-size:11px;font-weight:normal;color:#999;">'
        f'(live model now — CPCV is a STALE snapshot)</span></h3>'
        f'<p style="font-size:13px;margin:2px 0;">CPCV mean IC: '
        f'<b>{_fmt_ic(serving_ic)}</b> &nbsp;|&nbsp; forward_days: <b>{serving_fwd}</b> '
        f'&nbsp;|&nbsp; served: <code>{serving_ver}</code> on {serving_date}</p>'
        f'<h3 style="margin-top:16px;margin-bottom:2px;">Champion-arch '
        f'<span style="font-size:11px;font-weight:normal;color:#999;">'
        f'(fresh retrain this run — promotion baseline)</span></h3>'
        f'<p style="font-size:13px;margin:2px 0;">CPCV mean IC: '
        f'<b>{_fmt_ic(arch_ic)}</b> &nbsp;|&nbsp; version: <code>{arch_vid}</code></p>'
        f'{base_html}'
        f'<p style="font-size:12px;color:#1565c0;margin:4px 0;"><b>Promotion baseline</b> '
        f'= {baseline_label}: CPCV <b>{_fmt_ic(baseline_ic)}</b> '
        f'(challengers must beat this + margin {margin}).</p>'
        f'<h3 style="margin-top:16px;margin-bottom:4px;">Challengers ({len(challengers)})</h3>'
        f'<table style="border-collapse:collapse;font-size:11px;">'
        f'<tr style="background:#e0e0e0;">'
        f'<th style="padding:3px 8px;">Spec</th>'
        f'<th style="padding:3px 8px;">CPCV IC</th>'
        f'<th style="padding:3px 8px;">Fwd</th>'
        f'<th style="padding:3px 8px;">Gate</th>'
        f'<th style="padding:3px 8px;">Eligible</th>'
        f'<th style="padding:3px 8px;">Reason</th></tr>'
        + ("".join(cand_rows_html) if cand_rows_html
           else '<tr><td colspan="6" style="padding:4px 8px;color:#888;">'
                '(no challenger candidates this rotation)</td></tr>')
        + f'</table>'
        f'<h3 style="margin-top:16px;margin-bottom:4px;">Promotion</h3>'
        f'{promo_html}'
        f'<p style="font-size:10px;color:#aaa;margin-top:20px;">'
        f'Consolidated model-zoo digest — replaces the per-challenger training '
        f'emails and the base --full-only retrain email.</p>'
        f'</body></html>'
    )

    subject = _digest_subject(leaderboard, date_str)
    # REUSE the train_handler SMTP/SES chokepoint (Gmail primary, SES fallback)
    # — same mailer the per-run training email uses. NOT a hand-rolled sender.
    from krepis.email_sender import send_email as _send_email
    return _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=getattr(cfg, "AWS_REGION", None),
    )


def run_rotation_and_select(
    bucket: str,
    *,
    budget: int | None = None,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    train_fn=None,
    registered_versions: list | None = None,
    s3=None,
    auto_promote_winner: bool | None = None,
) -> dict:
    """L4544 entrypoint: run the weekly rotation, then rank the trained variants
    by leak-free CPCV and select a winner. OBSERVE-FIRST — with
    ``auto_promote_winner`` False (default from ``MODEL_ZOO_AUTO_PROMOTE_WINNER``)
    the recommended promotion is logged + written to the leaderboard but NOT
    executed; True promotes the winner via the registry (the turnover governor
    caps the resulting book move). Returns the leaderboard dict."""
    if auto_promote_winner is None:
        auto_promote_winner = bool(getattr(cfg, "MODEL_ZOO_AUTO_PROMOTE_WINNER", False))

    # config#1051: pin a real trading day so the leaderboard / trial_log key on a
    # date, not ``null`` (the 6/13 leaderboard had ``date=null`` because the CLI
    # passed no --date). Backward-looking dual-track per DATE_CONVENTIONS.
    if date_str is None:
        try:
            from krepis.dates import now_dual
            _td = now_dual().trading_day
            date_str = _td.isoformat() if hasattr(_td, "isoformat") else str(_td)
            log.info("model_zoo: no date passed — defaulting to trading_day %s", date_str)
        except Exception:  # noqa: BLE001 — date defaulting must not block the rotation
            log.warning("model_zoo: could not resolve trading_day via now_dual", exc_info=True)

    # config#1051 logging probe + fail-loud config guard: the 6/13 rotation
    # trained 0 challengers because MODEL_SPECS was EMPTY at runtime on the child
    # spot. Log the loaded spec count + resolved config path + experiment id so
    # the next run pins the empty-load, then RAISE before training rather than
    # degrading to a benign "no eligible challenger" INFO downstream.
    _resolved_specs = specs if specs is not None else getattr(cfg, "MODEL_SPECS", [])
    log.info(
        "model_zoo config#1051 probe: MODEL_SPECS=%d, config=%s, "
        "ALPHA_ENGINE_EXPERIMENT_ID=%s",
        len(_resolved_specs), getattr(cfg, "_CONFIG_PATH", "?"),
        getattr(cfg, "_EXPERIMENT_ID", os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "?")),
    )
    # Only enforce the guard when the roster comes from cfg (the real run); a test
    # / caller that injects an explicit ``specs`` list owns its own roster.
    if specs is None and hasattr(cfg, "assert_model_specs_loaded"):
        cfg.assert_model_specs_loaded()

    if s3 is None and not dry_run:
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception:  # noqa: BLE001
            log.warning("model_zoo: no S3 client — selection limited", exc_info=True)

    results = train_weekly_rotation(
        bucket, budget=budget, date_str=date_str, dry_run=dry_run,
        specs=specs, train_fn=train_fn, registered_versions=registered_versions, s3=s3,
    )

    # config#1051 (no-silent-fails): a rotation that had >= 1 active spec to train
    # but trained 0 challengers is INERT — fail loud (WARN + CW metric + SNS),
    # distinct from "challengers ran but lost". This catches the empty-selection /
    # all-errored cases the guard above can't (e.g. every selected spec crashed).
    # Skipped on a dry run (no real training happened — nothing to alert on).
    if not dry_run:
        _n_active = sum(1 for s in _resolved_specs
                        if s.get("status", "active") == "active" and s.get("id"))
        _n_trained = _count_challengers_trained(results)
        _emit_challengers_trained_metric(_n_trained)
        if _n_active >= 1 and _n_trained == 0:
            _alert_inert_rotation(
                bucket, date_str, n_active=_n_active, n_selected=len(results), results=results,
            )

    mode = "cutover" if auto_promote_winner else "observe"
    if dry_run or s3 is None:
        leaderboard = {"date": date_str, "mode": mode, "dry_run": bool(dry_run),
                       "candidates": [], "winner_version_id": None, "promoted": None,
                       "rotation_results": {k: (v.get("status") if isinstance(v, dict) else "ok")
                                            for k, v in results.items()}}
        log.info("model_zoo: rotation complete (dry_run/no-S3) — selection skipped")
        return leaderboard

    trained = _resolve_trained_versions(s3, bucket, results, specs=specs)
    return select_and_finalize(
        s3, bucket, date_str=date_str, mode=mode,
        auto_promote_winner=auto_promote_winner, trained=trained, dry_run=dry_run,
    )


def select_and_finalize(
    s3,
    bucket: str,
    *,
    date_str: str | None = None,
    mode: str | None = None,
    auto_promote_winner: bool | None = None,
    trained: list[dict] | None = None,
    specs: list | None = None,
    dry_run: bool = False,
) -> dict:
    """config#1083 — the SELECTION + finalization tail of a model-zoo rotation,
    factored out so it is reusable by BOTH the sequential ``run_rotation_and_select``
    (which supplies ``trained`` from its in-process results) AND the PARALLEL
    ``select`` entrypoint (which passes ``trained=None`` → the pool is resolved
    from the registry: every active spec's newest challenger registered for
    ``date_str``, TOLERATING specs whose Map iteration failed — they are simply
    absent, never a crash).

    Steps (unchanged from the prior inline tail): add the base champion-arch to
    the pool, record the cumulative trial ledger, capture the prior champion for
    a revert command, ``select_winner``, write the leaderboard to BOTH the dated
    key AND latest.json, promote the winner if ``auto_promote_winner``, run the
    realized-edge monitor, and ALWAYS send the one consolidated digest (the digest
    is best-effort observability — a send failure is logged LOUD + recorded on the
    leaderboard, never aborts the rotation). Returns the leaderboard dict.
    """
    if auto_promote_winner is None:
        auto_promote_winner = bool(getattr(cfg, "MODEL_ZOO_AUTO_PROMOTE_WINNER", False))
    if mode is None:
        mode = "cutover" if auto_promote_winner else "observe"
    if date_str is None:
        try:
            from krepis.dates import now_dual
            _td = now_dual().trading_day
            date_str = _td.isoformat() if hasattr(_td, "isoformat") else str(_td)
            log.info("model_zoo select: no date passed — defaulting to trading_day %s", date_str)
        except Exception:  # noqa: BLE001 — date defaulting must not block selection
            log.warning("model_zoo select: could not resolve trading_day via now_dual", exc_info=True)

    if s3 is None and not dry_run:
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception:  # noqa: BLE001
            log.warning("model_zoo select: no S3 client — selection limited", exc_info=True)

    if dry_run or s3 is None:
        leaderboard = {"date": date_str, "mode": mode, "dry_run": bool(dry_run),
                       "candidates": [], "winner_version_id": None, "promoted": None}
        log.info("model_zoo select: dry_run/no-S3 — selection skipped")
        return leaderboard

    # PARALLEL path: no in-process trained list → resolve the pool from the
    # registry (every active spec's newest challenger for this date; failed Map
    # iterations are absent, tolerated).
    if trained is None:
        trained = _resolve_registered_specs_for_date(s3, bucket, date_str, specs=specs)

    # L4571: the base champion-arch retrain competes in the SAME pool.
    base = _resolve_base_champion_version(s3, bucket, date_str)
    if base and base["version_id"] not in {t.get("version_id") for t in trained}:
        trained.append(base)
        log.info("model_zoo select: base champion-arch %s in the pool", base["version_id"])
    # L4582(b): ledger BEFORE selection so the cumulative count already includes
    # this rotation's candidates (DSR must deflate for the trials just run too).
    _n_trials_cum, _trial_log_status = _record_trials(s3, bucket, date_str, trained)
    # Capture the OUTGOING champion's registry version BEFORE any promote so the
    # operator alert can carry an exact, copy-pasteable revert command.
    _prior_champ_vid = _current_champion_version_id(s3, bucket)

    # Minimal leaderboard so the try/finally digest below ALWAYS has something
    # to report — even if select_winner itself raises (the digest reports
    # whatever candidates exist, never goes silent on the rotation).
    leaderboard: dict = {"date": date_str, "mode": mode, "candidates": [],
                         "winner_version_id": None, "promoted": None}
    try:
        leaderboard = select_winner(
            s3, bucket, trained=trained, n_trials_cumulative=_n_trials_cum,
        )
        leaderboard["date"] = date_str
        leaderboard["mode"] = mode
        leaderboard["trial_log_status"] = _trial_log_status

        winner_vid = leaderboard.get("winner_version_id")
        # #679(ii): the version that will actually become champion. A CHALLENGER win
        # takes precedence; otherwise the fresh champion-arch may refresh the live
        # model (keeps the deployed 21d model on the current vintage). The two are
        # mutually exclusive by construction (refresh is only set when winner is None).
        refresh_vid = leaderboard.get("champion_arch_refresh_version_id")
        promote_vid = winner_vid or refresh_vid
        promote_kind = "challenger" if winner_vid else ("champion-arch-refresh" if refresh_vid else None)
        if promote_vid and auto_promote_winner:
            try:
                from model.registry import promote_to_champion
                promote_to_champion(s3, bucket, promote_vid)
                leaderboard["promoted"] = promote_vid
                leaderboard["promoted_kind"] = promote_kind
                leaderboard["reverted_from"] = _prior_champ_vid
                log.info("model_zoo CUTOVER: promoted %s %s to champion",
                         promote_kind, promote_vid)
                _alert_promotion(bucket, date_str, leaderboard, promote_vid, _prior_champ_vid)
            except Exception as exc:  # noqa: BLE001 — a promote failure must not fail the SF
                leaderboard["promoted"] = None
                leaderboard["promote_error"] = str(exc)
                log.warning("model_zoo CUTOVER: promote of %s FAILED: %s", promote_vid, exc, exc_info=True)
        else:
            leaderboard["promoted"] = None
            if promote_vid:
                log.info(
                    "model_zoo OBSERVE: recommended promotion %s (%s) — NOT executed "
                    "(MODEL_ZOO_AUTO_PROMOTE_WINNER off; flip after soak)",
                    promote_vid, promote_kind,
                )
                _alert_observe_recommendation(bucket, date_str, leaderboard, promote_vid)
            else:
                log.info(
                    "model_zoo: no challenger beat the champion-arch baseline and "
                    "champion-arch does not improve on serving — live model unchanged"
                )

        _write_leaderboard(s3, bucket, date_str, leaderboard)

        # config#671/#673/#1052 — RELATIVE-BEST noise-chasing MONITOR (repurposed from
        # the retired Option-B observe-tier leaderboard). With the relative-best rule a
        # beats-champion challenger PROMOTES rather than entering an observe soak, so the
        # leaderboard no longer gates an observe→champion move. It is repointed to SCORE
        # THE PROMOTED CHAMPION's own predictions vs realized 21d alpha over a trailing
        # window: if relative-best promotion is chasing noise, the live champion's
        # realized rank-IC will be weak/negative — an OBSERVABILITY/alarm signal, NOT a
        # gate (it never promotes, demotes, or allocates). Best-effort (its own crash
        # must never fail the rotation SF — the promote already happened above).
        try:
            from analysis.observe_leaderboard import build_champion_realized_monitor
            _mon = build_champion_realized_monitor(bucket=bucket, date_str=date_str, s3_client=s3)
            leaderboard["champion_realized_monitor"] = {
                "realized_rank_ic": (_mon.get("champion") or {}).get("realized_rank_ic"),
                "n_matured_outcomes": (_mon.get("champion") or {}).get("n_matured_outcomes"),
                "chasing_noise": _mon.get("chasing_noise"),
            }
        except Exception:  # noqa: BLE001 — monitor is observability off a path that already promoted
            log.warning("model_zoo: champion realized-edge monitor failed", exc_info=True)
    finally:
        # ONE consolidated digest email — ALWAYS fires when the real rotation
        # python runs (this finally is reached even if select_winner / promote
        # raised; the minimal leaderboard above guarantees a reportable shape).
        # no-silent-fails carve-out: the digest is SECONDARY OBSERVABILITY on a
        # path whose primary outputs (leaderboard.json, training_summary.json,
        # promoted weights) already persisted to S3, so a send failure is
        # logged LOUD (WARN + a named "model_zoo_digest_email_failed" marker on
        # the returned leaderboard) but does NOT abort the rotation. Recording
        # surface: WARN log + leaderboard["digest_email"] field (visible in the
        # persisted leaderboard.json + the spot log). We do NOT re-raise.
        if not dry_run:
            try:
                _sent = send_zoo_digest_email(leaderboard, bucket, date_str, s3=s3)
                leaderboard["digest_email"] = "sent" if _sent else "skipped_not_configured"
            except Exception as exc:  # noqa: BLE001 — see carve-out comment above
                leaderboard["digest_email"] = f"model_zoo_digest_email_failed: {exc}"
                log.warning(
                    "model_zoo: consolidated digest email FAILED (rotation NOT "
                    "aborted — leaderboard/weights already persisted to S3): %s",
                    exc, exc_info=True,
                )

    return leaderboard


def run_select_only(
    bucket: str,
    *,
    date_str: str | None = None,
    dry_run: bool = False,
    specs: list | None = None,
    s3=None,
    auto_promote_winner: bool | None = None,
) -> dict:
    """config#1083 — the PARALLEL ``select`` entrypoint: run ONLY the selection
    over whatever spec-* challengers registered for this date (the per-spec Map
    iterations trained + registered them on their own spots), plus the base
    champion-arch. Tolerates missing specs, writes the leaderboard to BOTH keys,
    promotes the winner if auto-promote is on, sends the consolidated digest, and
    fires the inert-rotation alert when the pool is empty (no challenger trained
    AND no base) so a totally-failed Map never goes silent. Returns the
    leaderboard dict."""
    if date_str is None:
        try:
            from krepis.dates import now_dual
            _td = now_dual().trading_day
            date_str = _td.isoformat() if hasattr(_td, "isoformat") else str(_td)
            log.info("model_zoo select: no date passed — defaulting to trading_day %s", date_str)
        except Exception:  # noqa: BLE001 — date defaulting must not block selection
            log.warning("model_zoo select: could not resolve trading_day via now_dual", exc_info=True)

    if s3 is None and not dry_run:
        try:
            import boto3
            s3 = boto3.client("s3")
        except Exception:  # noqa: BLE001
            log.warning("model_zoo select: no S3 client — selection limited", exc_info=True)

    leaderboard = select_and_finalize(
        s3, bucket, date_str=date_str, auto_promote_winner=auto_promote_winner,
        trained=None, specs=specs, dry_run=dry_run,
    )

    # config#1051/#1083 (no-silent-fails): an empty pool — zero challenger
    # candidates AND no base champion-arch — means every Map iteration failed /
    # produced nothing. Fail loud (CW metric + SNS), distinct from "challengers
    # ran but lost". Skipped on a dry run. The base champion-arch (if present)
    # is itself a candidate, so the alert fires only on a TRULY inert select.
    if not dry_run and s3 is not None:
        _cands = leaderboard.get("candidates", []) or []
        _n_specs = sum(1 for s in (specs if specs is not None else getattr(cfg, "MODEL_SPECS", []))
                       if s.get("status", "active") == "active" and s.get("id"))
        _n_challengers = sum(1 for c in _cands if c.get("spec_id") != "champion-arch")
        _emit_challengers_trained_metric(_n_challengers)
        if _n_specs >= 1 and len(_cands) == 0:
            _alert_inert_rotation(
                bucket, date_str, n_active=_n_specs, n_selected=0, results={},
            )
    return leaderboard


def _cli_subcommand(argv: list) -> bool:
    """config#1083 — the PARALLEL fan-out subcommands (train-spec / select /
    list-rotation-specs) the Step Function invokes per spot. Dispatched when the
    first argv token is one of them; returns True if it handled the call. The
    legacy flag-based CLI (--spec / --weekly-rotation / --run-rotation-and-select)
    is preserved for the manual / sequential path — these are ADDITIVE."""
    import sys
    sub = argv[0] if argv else None
    if sub not in {"train-spec", "select", "list-rotation-specs"}:
        return False

    p = argparse.ArgumentParser(prog=f"model_zoo {sub}")
    p.add_argument("--bucket", required=True)
    p.add_argument("--date", default=None)
    p.add_argument("--dry-run", action="store_true")
    if sub == "train-spec":
        p.add_argument("--spec", required=True, help="The single spec id to train.")
    if sub == "list-rotation-specs":
        p.add_argument("--budget", type=int, default=None,
                       help="Rotation budget (default MODEL_ZOO_WEEKLY_BUDGET).")
    args = p.parse_args(argv[1:])

    if sub == "list-rotation-specs":
        ids = list_rotation_spec_ids(args.bucket, budget=args.budget)
        print(json.dumps(ids))
        return True
    if sub == "train-spec":
        # Trains exactly ONE spec; exits non-zero ONLY on a real training failure
        # so the SF Map iteration records THIS spec's failure without aborting
        # siblings (the per-spec isolation property). Let the exception propagate
        # to a non-zero exit (no swallow — fail loud per no-silent-fails).
        train_one_spec(args.spec, args.bucket, date_str=args.date, dry_run=args.dry_run)
        return True
    if sub == "select":
        run_select_only(args.bucket, date_str=args.date, dry_run=args.dry_run)
        return True
    return False  # pragma: no cover


def _cli() -> None:
    import sys
    if _cli_subcommand(sys.argv[1:]):
        return
    p = argparse.ArgumentParser(
        description="Model zoo: train a declared variant spec as a challenger (L4488c).\n"
                    "Parallel fan-out subcommands (config#1083): "
                    "`train-spec --spec <id> --bucket B`, `select --bucket B`, "
                    "`list-rotation-specs --bucket B [--budget N]`."
    )
    p.add_argument("--bucket", required=True)
    p.add_argument("--spec", default=None, help="Spec id to train.")
    p.add_argument("--all-active", action="store_true", help="Train every active spec (sequential).")
    p.add_argument("--weekly-rotation", action="store_true",
                   help="Train the N stalest active specs (N=--budget or "
                        "MODEL_ZOO_WEEKLY_BUDGET) — the bounded weekly cadence (L4488g).")
    p.add_argument("--run-rotation-and-select", action="store_true",
                   help="Weekly rotation + immediate CPCV selection (L4544): train "
                        "the rotation, rank variants by leak-free CPCV, write a "
                        "leaderboard, and (if MODEL_ZOO_AUTO_PROMOTE_WINNER) promote "
                        "the winner. Observe-first by default.")
    p.add_argument("--budget", type=int, default=None,
                   help="Override the rotation budget (default MODEL_ZOO_WEEKLY_BUDGET).")
    p.add_argument("--date", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="List declared specs and exit.")
    args = p.parse_args()

    if args.list:
        for s in getattr(cfg, "MODEL_SPECS", []):
            print(f"{s.get('status', 'active'):8s} {s.get('id')}  overrides={s.get('overrides', {})}")
        return
    if args.run_rotation_and_select:
        run_rotation_and_select(
            args.bucket, budget=args.budget, date_str=args.date, dry_run=args.dry_run,
        )
    elif args.weekly_rotation:
        train_weekly_rotation(
            args.bucket, budget=args.budget, date_str=args.date, dry_run=args.dry_run,
        )
    elif args.all_active:
        train_all_active(args.bucket, date_str=args.date, dry_run=args.dry_run)
    elif args.spec:
        train_spec(args.spec, args.bucket, date_str=args.date, dry_run=args.dry_run)
    else:
        p.error("provide --spec <id>, --all-active, --weekly-rotation, "
                "--run-rotation-and-select, or --list")


if __name__ == "__main__":
    _cli()
