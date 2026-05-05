"""
Training preflight — connectivity + freshness checks run at the top of
``train_handler.main`` before any download or training work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into a training-specific sequence. Matches
the ``PredictorPreflight`` pattern (inference/preflight.py) but with
training-specific checks.

Runs on EC2 spot instance at training start. Raises ``RuntimeError``
up through ``train_handler.main()`` → non-zero exit → spot_train.sh
fails visibly → flow-doctor dispatches.
"""

from __future__ import annotations

import logging

from alpha_engine_lib.preflight import BasePreflight

log = logging.getLogger(__name__)


class TrainingPreflight(BasePreflight):
    """Connectivity + freshness checks for the weekly training run.

    Required env vars:
    - ``AWS_REGION`` — S3 / ArcticDB client region

    Required S3:
    - bucket reachable

    Data-freshness assertion now lives upstream in ``alpha-engine-data``'s
    DataPhase1 preflight, which runs before ``PredictorTraining`` in the
    Saturday Step Function. If macro/universe data is stale, the data
    step hard-fails and the SF never reaches training.
    """

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
