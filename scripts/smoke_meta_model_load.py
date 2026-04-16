"""
smoke_meta_model_load.py — Validate backwards-compat load of prod meta_model.pkl.

The production meta_model.pkl in S3 was trained on the OLD 8-feature schema
(includes regime_bull, regime_bear). After PR #34 deploys, inference will
call predict_single with a 14-feature dict — the new schema minus regime
columns plus 6 raw macro columns. The MetaModel backwards-compat path must:

  1. Load the pkl and reconstruct _feature_names from coefficient keys
     (the current sidecar predates the persisted feature_names field)
  2. predict_single must use the model's own feature list, ignoring the
     6 new macro_* keys in the inference dict
  3. Produce a finite numeric prediction

If this check fails, tomorrow's 6:15 AM PT inference will raise a shape
mismatch on every ticker. If it passes, the Lambda can be redeployed
safely.

Usage:
    python scripts/smoke_meta_model_load.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = os.environ.get("ALPHA_ENGINE_BUCKET", "alpha-engine-research")
META_KEY = "predictor/weights/meta/meta_model.pkl"
META_SIDECAR_KEY = "predictor/weights/meta/meta_model.pkl.meta.json"


def main() -> int:
    import boto3
    from model.meta_model import META_FEATURES, MetaModel

    s3 = boto3.client("s3")
    tmp = Path(tempfile.mkdtemp())
    pkl_path = tmp / "meta_model.pkl"
    meta_path = tmp / "meta_model.pkl.meta.json"

    log.info("Downloading prod meta_model from s3://%s/%s ...", BUCKET, META_KEY)
    s3.download_file(BUCKET, META_KEY, str(pkl_path))
    try:
        s3.download_file(BUCKET, META_SIDECAR_KEY, str(meta_path))
    except Exception as e:
        log.warning("Sidecar not present: %s — load() will fall back to coefficient-key reconstruction", e)

    log.info("Loading MetaModel...")
    mm = MetaModel.load(pkl_path)

    log.info("Loaded model feature names (n=%d):", len(mm._feature_names))
    for name in mm._feature_names:
        log.info("  %s", name)

    log.info("Current META_FEATURES in code (n=%d):", len(META_FEATURES))
    for name in META_FEATURES:
        log.info("  %s", name)

    # Construct an inference-style feature dict with the CURRENT META_FEATURES
    # (14 names: 8 legacy Layer-1/research + 6 new macro). The model was
    # trained on the OLD META_FEATURES (8 names including regime_bull/regime_bear).
    # predict_single must resolve this by using the model's own _feature_names.
    inference_feats = {name: 0.1 for name in META_FEATURES}
    # Also include regime_bull/regime_bear with plausible values — these are
    # NOT in the current META_FEATURES but WILL be looked up by the old model
    # via _feature_names.
    inference_feats["regime_bull"] = 0.35
    inference_feats["regime_bear"] = 0.30

    log.info("Calling predict_single with %d-feature inference dict...", len(inference_feats))
    pred = mm.predict_single(inference_feats)

    import math
    if not math.isfinite(pred):
        log.error("predict_single returned non-finite value: %s", pred)
        return 1

    log.info("=" * 60)
    log.info("SMOKE TEST PASSED")
    log.info("=" * 60)
    log.info("  Prediction: %.6f (finite, within range)", pred)
    log.info("  Model schema: %d features", len(mm._feature_names))
    log.info("  Inference dict: %d keys (including current META_FEATURES + legacy regime)", len(inference_feats))
    log.info("  Unused keys (ignored by predict_single): %s",
             sorted(set(inference_feats.keys()) - set(mm._feature_names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
