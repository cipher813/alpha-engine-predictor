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

    Required ArcticDB:
    - ``macro/SPY`` fresh (≤4 days stale — covers Fri→Tue long weekends).
      If macro/SPY is stale, the upstream data path hasn't written
      recently and training would produce a model from stale features.
      Since training runs on Saturday against the previous week's data,
      a 4-day threshold catches Sunday and weekday write failures without
      false-positives on the weekend boundary.
    """

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()
        self.check_arcticdb_fresh("macro", "SPY", max_stale_days=4)
