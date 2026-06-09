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

import config as cfg

log = logging.getLogger(__name__)

# S3 prefixes (mirror model.registry defaults; kept local so the selection path
# has no import cycle with the trainer).
_REGISTRY_PREFIX = "predictor/registry"
_LEADERBOARD_PREFIX = "predictor/model_zoo/leaderboard"

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
        from training.train_handler import main as train_fn
    log.info("model_zoo: training spec %s — overrides %s", spec_id, overrides)
    with spec_overrides(overrides):
        return train_fn(bucket, date_str=date_str, dry_run=dry_run)


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


# ── Rotation isolation (L4544 G1+G2) ────────────────────────────────────────
# A zoo rotation trains CHALLENGER variants — it must NOT disturb the live
# champion. Two hazards in meta_trainer:
#   • TRAINING_AUTO_PROMOTE_ENABLED is globally ON → a spec whose gate passes
#     would SELF-PROMOTE inside its own train run, bypassing the selection step.
#   • feature_list.json + manifest.json are written to the LIVE keys
#     UNCONDITIONALLY (model WEIGHTS are gated on `promoted`, but the two
#     contract files are not) → a challenger train leaves the live champion's
#     contract describing the challenger.
# G1 forces challenger-first for the whole rotation; G2 restores the live
# contract afterwards. Both are localized here (no meta_trainer surgery — the
# champion path is freshly stabilized by #240).


@contextlib.contextmanager
def _challenger_first():
    """G1 — force ``TRAINING_AUTO_PROMOTE_ENABLED=False`` for the duration so no
    zoo spec self-promotes during a rotation (promotion is decided ONLY by the
    selection step) and every challenger's weights stay archive-only."""
    prev = getattr(cfg, "TRAINING_AUTO_PROMOTE_ENABLED", _SENTINEL)
    cfg.TRAINING_AUTO_PROMOTE_ENABLED = False
    try:
        yield
    finally:
        if prev is _SENTINEL:
            delattr(cfg, "TRAINING_AUTO_PROMOTE_ENABLED")
        else:
            cfg.TRAINING_AUTO_PROMOTE_ENABLED = prev


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
        budget = int(getattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 3))
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

    results: dict = {}
    try:
        # G1 — no spec self-promotes for the whole rotation.
        with _challenger_first():
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
        # G2 — always restore, even if the rotation raised mid-way.
        if saved_contract and s3 is not None:
            _restore_live_contract(s3, bucket, saved_contract)
    return results


# ── Immediate CPCV selection (L4544) ────────────────────────────────────────
# After the rotation, rank the freshly-trained challengers by leak-free CPCV
# mean IC (gated by the downside-Sortino + DSR battery) and pick a winner that
# beats the live champion by a margin. The realized-edge leaderboard (L4539) is
# the SLOWER confirm/demote layer — this is the immediate training-time selector.


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


def select_winner(s3, bucket: str, *, trained: list[dict], margin: float | None = None) -> dict:
    """Rank freshly-trained challengers by gated CPCV mean IC and pick a winner
    that beats the live champion by ``margin``. Promotion-eligible iff
    ``forward_days == champion horizon`` (canonical 21d) — horizon variants are
    observe-only. Returns a leaderboard dict (all candidates with eligibility +
    reason; ``winner_version_id`` is the best eligible challenger or None)."""
    if margin is None:
        margin = float(getattr(cfg, "MODEL_ZOO_PROMOTE_MARGIN", 0.01))
    champ_manifest: dict = {}
    try:
        champ_manifest = _read_live_manifest(s3, bucket)
    except Exception:  # noqa: BLE001 — champion baseline best-effort
        log.warning("model_zoo select: could not read live champion manifest", exc_info=True)
    champ_fwd = int(champ_manifest.get("forward_days") or getattr(cfg, "FORWARD_DAYS", 21))
    champ_ic = _cpcv_mean(champ_manifest)

    candidates: list[dict] = []
    for rec in trained:
        vid = rec.get("version_id")
        manifest: dict | None = None
        try:
            manifest = _read_registry_manifest(s3, bucket, vid)
        except Exception:  # noqa: BLE001 — a missing manifest just isn't ranked
            log.warning("model_zoo select: could not read manifest for %s", vid, exc_info=True)
        fwd = int((manifest or {}).get("forward_days") or 0)
        ic = _cpcv_mean(manifest)
        gate = _gate_pass(manifest)
        if fwd != champ_fwd:
            eligible, reason = False, "non_canonical_horizon"
        elif ic is None:
            eligible, reason = False, "no_cpcv"
        elif not gate:
            eligible, reason = False, "gate_failed"
        elif champ_ic is not None and ic < champ_ic + margin:
            eligible, reason = False, "below_champion_plus_margin"
        else:
            eligible, reason = True, "eligible"
        candidates.append({
            "spec_id": rec.get("spec_id"), "version_id": vid,
            "model_version": rec.get("model_version"),
            "forward_days": fwd, "cpcv_mean_ic": ic, "passes_gate": gate,
            "eligible": eligible, "reason": reason,
        })

    eligibles = [c for c in candidates if c["eligible"]]
    winner = max(eligibles, key=lambda c: c["cpcv_mean_ic"], default=None)
    return {
        "champion": {"forward_days": champ_fwd, "cpcv_mean_ic": champ_ic},
        "margin": margin,
        "candidates": candidates,
        "winner_version_id": winner["version_id"] if winner else None,
    }


def _write_leaderboard(s3, bucket: str, date_str: str | None, leaderboard: dict) -> None:
    key = f"{_LEADERBOARD_PREFIX}/{date_str or 'latest'}.json"
    try:
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(leaderboard, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info("model_zoo: leaderboard written to s3://%s/%s", bucket, key)
    except Exception:  # noqa: BLE001 — observability artifact, never fatal
        log.warning("model_zoo: could not write leaderboard to %s", key, exc_info=True)


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

    mode = "cutover" if auto_promote_winner else "observe"
    if dry_run or s3 is None:
        leaderboard = {"date": date_str, "mode": mode, "dry_run": bool(dry_run),
                       "candidates": [], "winner_version_id": None, "promoted": None,
                       "rotation_results": {k: (v.get("status") if isinstance(v, dict) else "ok")
                                            for k, v in results.items()}}
        log.info("model_zoo: rotation complete (dry_run/no-S3) — selection skipped")
        return leaderboard

    trained = _resolve_trained_versions(s3, bucket, results, specs=specs)
    leaderboard = select_winner(s3, bucket, trained=trained)
    leaderboard["date"] = date_str
    leaderboard["mode"] = mode

    winner_vid = leaderboard.get("winner_version_id")
    if winner_vid and auto_promote_winner:
        try:
            from model.registry import promote_to_champion
            promote_to_champion(s3, bucket, winner_vid)
            leaderboard["promoted"] = winner_vid
            log.info("model_zoo CUTOVER: promoted challenger %s to champion", winner_vid)
        except Exception as exc:  # noqa: BLE001 — a promote failure must not fail the SF
            leaderboard["promoted"] = None
            leaderboard["promote_error"] = str(exc)
            log.warning("model_zoo CUTOVER: promote of %s FAILED: %s", winner_vid, exc, exc_info=True)
    else:
        leaderboard["promoted"] = None
        if winner_vid:
            log.info(
                "model_zoo OBSERVE: recommended promotion %s — NOT executed "
                "(MODEL_ZOO_AUTO_PROMOTE_WINNER off; flip after soak)", winner_vid,
            )
        else:
            log.info("model_zoo: no eligible challenger beat the champion this rotation")

    _write_leaderboard(s3, bucket, date_str, leaderboard)
    return leaderboard


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Model zoo: train a declared variant spec as a challenger (L4488c)."
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
