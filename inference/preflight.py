"""
Inference preflight — connectivity + freshness checks run at the top of
the Lambda handler before any model load or feature read.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into the sequence that matters for the
predictor inference Lambda.

Runs synchronously at cold start. Any failure raises ``RuntimeError``
up through the Lambda handler, causing the invocation to return an
error, the Step Function ``Catch [States.ALL]`` to fire, and
flow-doctor to dispatch email + issue.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from alpha_engine_lib.preflight import BasePreflight

log = logging.getLogger(__name__)


# ── Deploy-drift helpers ─────────────────────────────────────────────────────

# Where deploy.sh's Docker build stamps the source git SHA. If this file is
# missing or contains "unknown", the image was built without the stamp
# (legacy pre-drift-check build) and we degrade to a warning instead of
# hard-failing — otherwise every existing Lambda version would break on the
# first deploy of this code.
GIT_SHA_FILE = Path("/var/task/GIT_SHA.txt")

# Public-repo branch-HEAD API. No auth required for public repos; 60 req/hr
# unauthenticated rate limit is fine for weekday Lambda cold starts (≤5/day).
_GITHUB_BRANCH_URL = "https://api.github.com/repos/{repo}/branches/{branch}"


def _read_baked_git_sha() -> str | None:
    """Return the SHA baked into the image by `deploy.sh --build-arg GIT_SHA=…`.

    Returns None if the stamp file is missing (legacy image) or holds
    "unknown" (build-arg omitted). Callers decide whether None is
    warn-and-continue or hard-fail.
    """
    try:
        sha = GIT_SHA_FILE.read_text().strip()
    except FileNotFoundError:
        return None
    if not sha or sha == "unknown":
        return None
    return sha


def _fetch_origin_main_sha(repo: str, branch: str = "main", timeout: float = 5.0) -> str | None:
    """Fetch HEAD SHA of `branch` for `repo` via GitHub REST API.

    Returns None on any network/parse error — the drift check treats a
    GitHub outage as "unknown, proceed with warning" rather than blocking
    the Lambda. `repo` is "owner/name" (e.g. "cipher813/alpha-engine-predictor").
    """
    url = _GITHUB_BRANCH_URL.format(repo=repo, branch=branch)
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload.get("commit", {}).get("sha")
    except (OSError, json.JSONDecodeError) as exc:
        # OSError covers urllib.error.URLError/HTTPError plus the bare
        # TimeoutError that urlopen raises on a read-phase timeout (the
        # 2026-05-07 weekday SF failure: read timed out inside getresponse,
        # which is past urllib's OSError→URLError wrap point).
        log.warning("Deploy-drift: GitHub API unreachable (%s) — cannot compare", exc)
        return None


class PredictorPreflight(BasePreflight):
    """Connectivity + freshness checks for the inference Lambda.

    Required env vars:
    - ``AWS_REGION`` — S3 / ArcticDB client region

    Required S3:
    - bucket reachable
    - model weights key present (``predictor/weights/meta/meta_model.pkl``
      or the ``MODEL_WEIGHTS_KEY`` from config)

    Data-freshness assertions (universe + macro/SPY + inference-macro
    symbols) live upstream in ``alpha-engine-data``'s preflight, which
    runs before ``PredictorInference`` in every Step Function. If
    upstream data is stale, the data step hard-fails and the SF never
    reaches inference.
    """

    def run_for_drift_gate(self) -> None:
        """Minimal preflight for ``action=check_deploy_drift`` only.

        The drift-check action is a Step Function gate — its job is to
        compare the deployed image/SF/CF SHAs to ``origin/main`` HEAD,
        nothing more. It has no business validating ticker freshness or
        loading model weights. Running the full preflight here turned a
        ~3s gate into a ~200s gate (the 2026-05-01 SF timeout cascade)
        once PR #68 added the universe scan.

        Strict subset of ``run()``:
          - env vars
          - S3 bucket reachability
          - image-SHA drift
        """
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
        self.check_deploy_drift()

    def run(self) -> None:
        """Full preflight for ``action=predict`` + ``action=check_coverage``."""
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
        self.check_deploy_drift()

        # Model weights must exist for the Lambda to do anything useful.
        # load_model is the next stage — if weights are missing, let
        # preflight fail loudly here rather than hitting a cryptic
        # GBMScorer construction error three stages in.
        self.check_s3_key(
            "predictor/weights/meta/meta_model.pkl",
            max_age_days=None,  # existence check only; staleness handled by training monitor
        )

    # ── Deploy-drift check ───────────────────────────────────────────────────

    def check_deploy_drift(
        self,
        repo: str = "cipher813/alpha-engine-predictor",
        branch: str = "main",
    ) -> None:
        """Hard-fail if the image's baked SHA lags the GitHub branch HEAD.

        The deployed Lambda image is stamped with ``GIT_SHA`` at Docker
        build time (see deploy.sh). This check compares that stamp to the
        current ``branch`` HEAD SHA on GitHub. A mismatch means a merge
        landed on main but the CI deploy workflow either failed, was
        skipped by a paths filter, or hasn't run yet — i.e. the Lambda is
        running a prior commit, which is exactly the deploy-drift mode
        that motivated this check (2026-04-20 coverage-gap session).

        Degraded modes (warn, don't fail) — chosen so a GitHub outage or
        an unstamped legacy image doesn't block a trading-hours Lambda:
        - Stamp file missing / "unknown"  → the image predates drift
          checking; log warn and continue.
        - GitHub API unreachable          → log warn and continue.

        Hard-fail mode — when we have both stamps in hand and they differ.
        """
        baked = _read_baked_git_sha()
        if baked is None:
            log.warning(
                "Deploy-drift: no baked GIT_SHA in image (legacy build or "
                "build-arg omitted). Rebuild via deploy.sh to enable this check."
            )
            return

        upstream = _fetch_origin_main_sha(repo, branch=branch)
        if upstream is None:
            # _fetch_origin_main_sha already logged the reason
            return

        if baked != upstream:
            raise RuntimeError(
                f"Deploy drift: Lambda image was built from {baked[:12]} but "
                f"{repo}@{branch} is now at {upstream[:12]}. The CI deploy "
                f"workflow did not promote the latest commit to the Lambda "
                f"'live' alias. Re-run `.github/workflows/deploy.yml` on main "
                f"(or `bash infrastructure/deploy.sh` locally) before "
                f"resuming. Refusing to proceed — running stale code on new "
                f"signals is how 2026-04-20 happened."
            )

        log.info("Deploy-drift: image at %s matches %s@%s ✓", baked[:12], repo, branch)
