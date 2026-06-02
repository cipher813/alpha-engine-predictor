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
``python -m training.model_zoo --bucket B --spec <id>`` or every active spec
with ``--all-active`` (sequential — each is ~one full training run, so pace them
as experiments allow rather than all in one Saturday SF).

Limitation (documented): the override only affects knobs read via ``cfg.X`` at
call time. A horizon change (``FORWARD_DAYS``) additionally needs any
import-time-DERIVED constant to be read at call time too — verify per-knob when
a spec exercises it (the 60d-target variant, L4488d).
"""
from __future__ import annotations

import argparse
import contextlib
import logging

import config as cfg

log = logging.getLogger(__name__)

# Only these cfg knobs may be overridden by a spec — fail loud on anything else
# so a spec can't set arbitrary attributes on the config module. Extend ONLY
# after confirming the trainer reads the knob via cfg.X at call time.
_ALLOWED_OVERRIDES = {
    "FORWARD_DAYS",
    "RESIDUAL_MOMENTUM_ENABLED",
    "XSEC_DEMEAN_ALPHA_ENABLED",
    "MODEL_VERSION_LABEL",
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


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Model zoo: train a declared variant spec as a challenger (L4488c)."
    )
    p.add_argument("--bucket", required=True)
    p.add_argument("--spec", default=None, help="Spec id to train.")
    p.add_argument("--all-active", action="store_true", help="Train every active spec (sequential).")
    p.add_argument("--date", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="List declared specs and exit.")
    args = p.parse_args()

    if args.list:
        for s in getattr(cfg, "MODEL_SPECS", []):
            print(f"{s.get('status', 'active'):8s} {s.get('id')}  overrides={s.get('overrides', {})}")
        return
    if args.all_active:
        train_all_active(args.bucket, date_str=args.date, dry_run=args.dry_run)
    elif args.spec:
        train_spec(args.spec, args.bucket, date_str=args.date, dry_run=args.dry_run)
    else:
        p.error("provide --spec <id>, --all-active, or --list")


if __name__ == "__main__":
    _cli()
