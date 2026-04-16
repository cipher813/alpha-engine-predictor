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

import logging

from alpha_engine_lib.preflight import BasePreflight

log = logging.getLogger(__name__)


class PredictorPreflight(BasePreflight):
    """Connectivity + freshness checks for the inference Lambda.

    Required env vars:
    - ``AWS_REGION`` — S3 / ArcticDB client region

    Required S3:
    - bucket reachable
    - model weights key present (``predictor/weights/meta/meta_model.pkl``
      or the ``MODEL_WEIGHTS_KEY`` from config)

    Required ArcticDB:
    - ``macro/SPY`` fresh (≤4 days stale — covers Fri→Tue long weekends).
      If SPY is stale, the upstream data path is broken and inference
      would produce predictions from stale features. Better to fail at
      preflight than to ship bad predictions to the executor.
    - ``macro/VIX``, ``macro/VIX3M``, ``macro/TNX``, ``macro/IRX`` fresh
      (same threshold). These feed regime and vol features; stale readings
      would silently corrupt every prediction in the batch.
    """

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()

        # Model weights must exist for the Lambda to do anything useful.
        # load_model is the next stage — if weights are missing, let
        # preflight fail loudly here rather than hitting a cryptic
        # GBMScorer construction error three stages in.
        self.check_s3_key(
            "predictor/weights/meta/meta_model.pkl",
            max_age_days=None,  # existence check only; staleness handled by training monitor
        )

        # Macro freshness: all five tickers must be current before inference
        # runs. SPY is the canonical liveness probe; VIX/VIX3M/TNX/IRX feed
        # regime and vol features directly. A stale reading on any of them
        # would corrupt the entire prediction batch without any downstream
        # signal — better to abort here and let the Step Function alarm fire.
        for symbol in ("SPY", "VIX", "VIX3M", "TNX", "IRX"):
            self.check_arcticdb_fresh("macro", symbol, max_stale_days=4)
