"""Pins spot_train.sh INSTANCE_TYPES to a memory-adequate (≥8 GiB) set.

The meta-trainer's full-training pass OOM-killed twice on 4 GiB boxes:

  - 2026-04-28: ~2-3 GB peak RSS OOM'd c5.large; addressed by the
    meta_trainer.py streaming refactor (see test_meta_trainer_streaming.py).
  - 2026-06-06: data/universe growth + the observe-only canonical-alpha
    matrix re-crossed 4 GiB; the Saturday SF rotation picked c5.large
    (4 GiB) and the kernel SIGKILL'd full-training right after regime-data
    load. SSM surfaced this as a confusing ``TimedOut`` (the box thrashed
    and lost its SSM heartbeat), not a clean failure.

The default rotation must therefore exclude the 4 GiB `.large` general
/ compute types (c5/c6i/c5a). This test catches a future edit that
re-introduces a 4 GiB type into the default INSTANCE_TYPES.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"

# 2 vCPU `.large` instance types that ship only 4 GiB RAM. The trainer's
# peak RSS exceeds this, so none of these may appear in the default set.
_FOUR_GIB_LARGE_TYPES = {"c5.large", "c5a.large", "c5d.large", "c6i.large", "c6a.large", "c7i.large"}


def _default_instance_types() -> list[str]:
    text = _SCRIPT.read_text()
    m = re.search(r'INSTANCE_TYPES="\$\{INSTANCE_TYPES:-([^}]*)\}"', text)
    assert m, "INSTANCE_TYPES default assignment not found — spot_train.sh structure changed"
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def test_default_instance_types_present():
    assert _default_instance_types(), "INSTANCE_TYPES default is empty"


def test_no_four_gib_large_types_in_default_rotation():
    types = _default_instance_types()
    offenders = sorted(set(types) & _FOUR_GIB_LARGE_TYPES)
    assert not offenders, (
        f"4 GiB instance type(s) {offenders} in default INSTANCE_TYPES — the "
        f"meta-trainer OOMs on 4 GiB (2026-04-28 + 2026-06-06). Use ≥8 GiB "
        f"types (r5/r5a/r6i.large = 16 GiB, m5.large = 8 GiB)."
    )


def test_lead_instance_type_has_memory_headroom():
    """The first (preferred) type should be a 16 GiB r-family box for headroom."""
    lead = _default_instance_types()[0]
    assert lead.startswith("r"), (
        f"lead INSTANCE_TYPE {lead!r} is not memory-optimized (r-family). The "
        f"trainer's peak RSS re-crossed 4 GiB twice; lead with 16 GiB headroom."
    )
