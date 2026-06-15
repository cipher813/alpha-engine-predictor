"""
Inference preflight — connectivity + freshness checks run at the top of
the Lambda handler before any model load or feature read.

Primitives — including ``check_deploy_drift`` — live in
``alpha_engine_lib.preflight.BasePreflight``; this module only composes
them into the sequence that matters for the predictor inference Lambda.

Runs synchronously at cold start. Any failure raises ``RuntimeError``
up through the Lambda handler, causing the invocation to return an
error, the Step Function ``Catch [States.ALL]`` to fire, and
flow-doctor to dispatch email + issue.
"""

from __future__ import annotations

import logging

from alpha_engine_lib.preflight import BasePreflight

log = logging.getLogger(__name__)

_PREDICTOR_REPO = "nousergon/crucible-predictor"


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
        self.check_deploy_drift(_PREDICTOR_REPO)

    def run(self, *, skip_deploy_drift: bool = False) -> None:
        """Full preflight for ``action=predict`` + ``action=check_coverage``.

        ``skip_deploy_drift`` — when True, the image-SHA-vs-``origin/main``
        drift assertion is skipped. Set this for ``dry_run=true`` (canary)
        invocations. Rationale: deploy-drift protection exists to stop the
        Lambda *acting on new signals with stale code* — i.e. writing
        ``predictions/{date}.json`` + sending the morning email. A dry_run
        canary writes nothing and emails nothing, so comparing its image
        against live ``main`` HEAD is the wrong invariant. Worse, it is a
        false-failure source: during a merge burst ``main`` can advance
        *after* a deploy's image is built but *before* its canary runs, so
        the canary trips the drift ``RuntimeError`` on a perfectly good
        freshly-built image and the deploy false-fails (config#1073,
        2026-06-14). Production protection is unaffected — real runs
        (``dry_run=false``) still drift-check here, and the Saturday/weekday
        SF ``DeployDriftCheck`` gate (``action=check_deploy_drift`` →
        ``run_for_drift_gate``) checks unconditionally before every
        pipeline.
        """
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
        if not skip_deploy_drift:
            self.check_deploy_drift(_PREDICTOR_REPO)

        # Model weights must exist for the Lambda to do anything useful.
        # load_model is the next stage — if weights are missing, let
        # preflight fail loudly here rather than hitting a cryptic
        # GBMScorer construction error three stages in.
        self.check_s3_key(
            "predictor/weights/meta/meta_model.pkl",
            max_age_days=None,  # existence check only; staleness handled by training monitor
        )
