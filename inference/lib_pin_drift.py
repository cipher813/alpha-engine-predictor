"""Cross-repo `alpha-engine-lib` pin-drift probe (L4517).

Preventive companion to the reactive `spot_backtest.sh` co-install guard:
asserts the cross-repo lib-pin invariant BEFORE the Saturday Step Function
spends on any spot launch, so a drift fails the run in seconds (with a named
offender) instead of breaking a co-install ~12 min into a spot.

The invariant (Brian, L4517):
  (a) **co-install parity** — `backtester pin == predictor pin`. `spot_backtest.sh`
      installs both into ONE venv (backtester first, predictor second); a lagging
      predictor silently downgrades the lib and breaks `quant.stats` (2026-05-12).
      This is the only multi-repo co-install site.
  (b) **floor** — every Saturday-SF repo's pin >= `MIN_LIB_VERSION`. The floor sits
      at the LOWEST INTENTIONAL pin (data v0.39.0) — data/research lag on purpose
      and do NOT co-install, so strict fleet lockstep is explicitly NOT required;
      the floor only catches a regression BELOW today's fleet, not currency.

Exposed as `action=check_lib_pin_drift` on the predictor Lambda handler (mirrors
`check_deploy_drift`) so the SF invokes it as an early Task → Choice gate. Returns
a JSON-serializable dict the SF Choice consumes via `has_drift`.

Fail-open on the checker's own fragility (mirrors `deploy_drift.py` degraded mode):
a GitHub-fetch or parse failure for any needed repo → `has_drift=false` + WARN, so
the probe never false-halts the weekly run. It only halts on a CONFIRMED drift
(all needed pins fetched + parsed + a parity mismatch or below-floor pin).
"""

from __future__ import annotations

import logging
import re
import urllib.request

from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

# Floor = data's current (lowest INTENTIONAL) pin. Raising this above any
# intentional laggard (data v0.39.0 / research v0.42.0) would false-halt the
# Saturday SF. Bump deliberately only when the whole Saturday-SF fleet clears it.
MIN_LIB_VERSION = "0.39.0"

# The co-install pair that MUST match (spot_backtest.sh installs both into one
# venv). Order is (first-installed, second-installed) for the message only.
_CO_INSTALL_PAIR = ("cipher813/alpha-engine-backtester", "cipher813/alpha-engine-predictor")

# Every repo that participates in the Saturday SF must clear the floor.
_FLOOR_REPOS = (
    "cipher813/alpha-engine-data",
    "cipher813/alpha-engine-predictor",
    "cipher813/alpha-engine-backtester",
    "cipher813/alpha-engine-research",
)

# Lifted from {predictor,backtester}/tests/test_lib_pin_lockstep.py (the pin
# format is `alpha-engine-lib[extras] @ git+https://.../alpha-engine-lib@vX.Y.Z`).
# TODO(L4517 follow-up, P3): lift this + `_fetch_repo_pin` into
# `alpha_engine_lib.preflight` alongside `_fetch_origin_main_sha` — the regex is
# already duplicated across repo test suites (chokepoint rule). Deferred: needs
# a lib release + cross-repo re-pin.
_LIB_PIN_RE = re.compile(
    r"alpha-engine-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/"
    r"cipher813/alpha-engine-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)

_RAW_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/{repo}/{branch}/requirements.txt"
)


def _parse_pin(text: str) -> str | None:
    """Extract the `vX.Y.Z` lib pin from a requirements.txt body, or None."""
    match = _LIB_PIN_RE.search(text)
    return match.group(1) if match else None


def _fetch_repo_pin(repo: str, branch: str = "main", timeout: float = 5.0) -> str | None:
    """Fetch `repo@branch`'s pinned `alpha-engine-lib` version from GitHub.

    Reads the raw `requirements.txt` (public, no auth) and parses the pin.
    Returns `None` on any network/parse error — the probe treats that as
    "unknown, proceed with warning" (fail-open), mirroring
    `alpha_engine_lib.preflight._fetch_origin_main_sha`.
    """
    url = _RAW_REQUIREMENTS_URL.format(repo=repo, branch=branch)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except OSError as exc:  # URLError/HTTPError + bare read-phase TimeoutError
        log.warning("Lib-pin drift: requirements.txt unreachable for %s (%s)", repo, exc)
        return None
    pin = _parse_pin(text)
    if pin is None:
        log.warning("Lib-pin drift: no alpha-engine-lib pin found in %s/requirements.txt", repo)
    return pin


def _ge_floor(pin: str) -> bool:
    """True if `pin` (vX.Y.Z) >= MIN_LIB_VERSION. Unparseable → False (caller
    only reaches this with a regex-matched pin, so this is belt-and-suspenders)."""
    try:
        return Version(pin.lstrip("v")) >= Version(MIN_LIB_VERSION)
    except InvalidVersion:
        return False


def check_lib_pin_drift(branch: str = "main") -> dict:
    """Assert the cross-repo lib-pin invariant; return a dict the SF Choice reads.

    `has_drift=true` ONLY when every needed pin was fetched + parsed AND a parity
    mismatch or below-floor pin is confirmed. Any fetch/parse miss → `has_drift=false`
    + `reason=fetch_failed` (fail-open). Shape mirrors `check_deploy_drift`.
    """
    repos = tuple(dict.fromkeys(_CO_INSTALL_PAIR + _FLOOR_REPOS))  # de-dup, ordered
    pins: dict[str, str | None] = {r: _fetch_repo_pin(r, branch=branch) for r in repos}

    # Fail-open: if any needed pin is unknown, do NOT halt the weekly run.
    missing = [r for r, p in pins.items() if p is None]
    if missing:
        log.warning(
            "Lib-pin drift: %d repo pin(s) unresolved %s — proceeding (fail-open)",
            len(missing), missing,
        )
        return {
            "has_drift": False,
            "parity_ok": None,
            "floor_ok": None,
            "min_lib_version": MIN_LIB_VERSION,
            "pins": pins,
            "offenders": [],
            "reason": "fetch_failed",
        }

    bt, pred = _CO_INSTALL_PAIR
    parity_ok = pins[bt] == pins[pred]

    below_floor = [r for r in _FLOOR_REPOS if not _ge_floor(pins[r])]
    floor_ok = not below_floor

    offenders: list[str] = []
    if not parity_ok:
        offenders.append(
            f"co-install parity: {bt}={pins[bt]} != {pred}={pins[pred]}"
        )
    offenders.extend(
        f"below floor: {r}={pins[r]} < {MIN_LIB_VERSION}" for r in below_floor
    )

    has_drift = not (parity_ok and floor_ok)
    if has_drift:
        log.error("Lib-pin drift DETECTED: %s", "; ".join(offenders))

    return {
        "has_drift": has_drift,
        "parity_ok": parity_ok,
        "floor_ok": floor_ok,
        "min_lib_version": MIN_LIB_VERSION,
        "pins": pins,
        "offenders": offenders,
        "reason": "drift_detected" if has_drift else "in_sync",
    }
